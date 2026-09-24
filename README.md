<div align="center">

# ltm-recon-try

## 面向流式 3D 重建的潜变量预测先验 —— 研究方案仓库

基于 [ABot-Recon](https://github.com/amap-cvlab/ABot-Recon) 的改进研究：在不动基座模型的前提下，外挂 V-JEPA 式的下一帧潜变量预测器

</div>

> **一句话**：保留 ABot-Recon 12 帧局部上下文的流式框架，旁挂一个 action 条件化的潜变量预测器——用历史帧的 trunk 表征预测下一帧表征，经零初始化门控注意力融合回感知通路，为 depth / pose 提供时序先验，同时产出免费的新颖性（novelty）信号。

---

## 1. 方案概述

流式稠密重建中，当前帧的 trunk 表征在运动模糊、遮挡、弱纹理、曝光突变时会退化，而这些恰是历史信息不受影响的时刻。本方案在帧 k 处理**之前**就固化一份"对第 k 帧的期望"（由 ≤ k-1 帧信息生成），在帧 k 到来后与实际表征做门控融合，并用预测误差标记新内容：

```
帧 k-1 forward 末尾（生成先验）:
  motion history ──→ PoseExtrapolator ──→ T̂_{k-1→k} ──→ action token ─┐
  previous_descriptor（pose head 状态字典，已 detach）──────────────────┤
  ring buffer（trunk 输出快照，默认=KV 窗口 12 帧）─────────────────────┴→ LatentPredictor → ẑ_k

帧 k forward（消费先验）:
  trunk（encoder + 36 块 decoder）→ h_k ──→ GatedFusion(h_k, ẑ_k) ──→ point / camera heads
```

四个要点：

1. **Ring buffer**：与 KV cache 并行地保留最近帧的 trunk 最终输出快照（每帧 ~3MB，容量默认自动跟随 `local_window_frames`）；
2. **旁挂 LatentPredictor**（~45M，6L/768）：输入历史快照 + action 条件 token，输出下一帧 trunk 表征预测；在帧 k-1 结束时生成、帧 k 融合时消费；
3. **GatedFusion**（Flamingo 式零初始化门控注意力）：主表征 query 预测表征，融合后再进 depth / pose heads；
4. **action 条件六档**（`action_source`）：`none` / `const_velocity`（滞后一步恒速）/ `previous_descriptor`（视角摘要）/ 二者组合（默认）/ `extrapolator`（小型位姿外推器，复用 TemporalRotationRefiner 的滚动窗口+age embedding+门控深度卷积模式）/ 外推器+描述符。

**架构不变量**（有测试锁定）：

- 融合点在 trunk 之后 → 融合结果**不进入**后续帧的 KV cache，流式因果性不被污染；
- ring 只存**未融合**的 trunk 输出 → 预测目标一致；
- 零初始化门控 → **未训练时输出与基座模型逐位相等**，劣质预测器只会被"关门"，下界伤害 ≈ 0；
- 逐 token 预测误差（`last_prediction_error`）是免费的 novelty / 动态物体信号，可馈入置信度头与 `sparse_loop` 关键帧选择。

## 2. 与相关工作的差异定位

| 工作 | 关系 | 差异 |
|---|---|---|
| V-JEPA 2 / 2-AC (arXiv:2506.09985) | latent 级下一帧预测 + action 条件 | 其预测器（300M，block-causal）用于规划与在线适应，**不回流感知**；我们将预测先验融合回感知通路 |
| PredNet 等预测编码 | 误差驱动的分层感知 | 其在监督信号层级传播误差；我们在 token 层级做先验融合 |
| Flamingo (arXiv:2204.14198) | 零初始化门控跨注意力注入 | 其融合外部模态；我们融合"时间上的自我" |
| CUT3R / Spann3R / StreamVGGT | 流式 3D 感知的记忆机制 | 它们改主干为递归/记忆式，破坏 checkpoint 兼容；我们旁挂式，基座权重不失效、可整体关闭 |

## 3. 实现状态

- [x] 全部新模块（`abot_recon/modeling/streaming/latent_prediction.py`）：`LatentPredictor` / `PoseExtrapolator` / `ActionTokenEncoder` / `GatedFusion` / `LatentPredictionManager` / JEPA 损失
- [x] 三条逐帧推理路径集成（SDPA 流式 / FlashInfer paged / camera-only），`infer_mode=full` 下自动禁用并告警
- [x] 配置开关：`InferenceConfig.latent_prediction` / CLI `--latent-prediction '<json>'` / 构造器 kwarg
- [x] released checkpoint 兼容（严格加载 + 新模块键宽容）
- [x] 28 项测试，含"开启未训练 == 关闭"的逐位一致回归、分辨率变化自动 reset、六种 action_source 协议测试
- [ ] Stage 1：predictor / extrapolator 预训练（JEPA 损失 + GT 位姿回归）
- [ ] Stage 2：融合门控微调（任务损失）
- [ ] 基准评测与消融（见 §5）

实现细节、逐文件改动说明与训练路线：见 **[plam.md](plam.md)**。

## 4. 使用方式

基座推理用法与上游一致（见 §6）。开启潜变量预测：

```python
from abot_recon import ABotRecon

model = ABotRecon.from_pretrained(
    "acvlab/ABot-Recon",
    device="cuda",
    loop_closure=False,
    latent_prediction={
        "enabled": True,
        "action_source": "extrapolator_and_descriptor",  # 见 plam.md 六档说明
        # 其余项走默认：ring 自动跟随 KV 窗口(12)，predictor 6L/768，融合开启
    },
)
result = model.infer(images)
```

```bash
python -m abot_recon.cli --image-dir examples/images \
  --latent-prediction '{"enabled": true, "action_source": "extrapolator_and_descriptor"}'
```

注意：新模块当前为随机初始化（未训练时门控保持关闭、输出与基座一致），需按 §5 训练后才有增益。诊断接口：`model.model.network.latent_prediction.diagnostics()`（门控幅度、预测误差统计）。

## 5. 训练与实验计划

**三阶段**：

1. **Stage 1 — 预训练**：冻结主干，`latent_prediction_loss(ẑ, sg(z))` 训 predictor；extrapolator 用 GT 相对位姿做 next-pose 回归（可独立预训练）。
   *缓存技巧*：Stage 1 无需重跑 1B 主干——先缓存 (z_k, descriptor, rel_poses, pos)，predictor 在缓存特征上训练，容量扫描近乎免费。
2. **Stage 2 — 融合微调**：只训门控（+ predictor 小 lr），走现有 depth/pose/conf 任务损失；监控门控幅度（塌 0 = 预测无价值，全开 = 过信任/拖影）。
3. **Stage 3（可选）— 联合微调**：预测目标换 EMA target trunk，防止主干学成"好预测"的表征。

**消融顺序**（核心假设检验）：`none` → `const_velocity` → `previous_descriptor` → `extrapolator_and_descriptor`；外加 `enable_fusion=False`（纯误差监控）与 trivial-prior 基线（ẑ_k = z_{k-1} 复制）对照。

**容量消融**：2L/512 → 6L/768 → 12L/1024，以 held-out **场景**（非帧）的 JEPA loss 判定欠拟合/过拟合。

**评测**：7-Scenes / TUM-RGBD / ScanNet / KITTI 流式协议（沿用上游 `eval` 分支），外加门控激活统计与预测误差-遮挡相关性分析。

## 6. 基座模型：ABot-Recon

ABot-Recon（[arXiv:2608.27529](https://arxiv.org/abs/2608.27529)）以 12 帧局部上下文做长时程流式重建：缓存前 11 帧 KV，逐帧预测当前相机系点图与相邻相对位姿，经序贯位姿合成恢复全局轨迹与点云；轻量旋转修正器与合成感知位姿损失抑制漂移。本仓库在其发布代码基础上做上述扩展，基座行为可通过开关完全还原。

### 安装

```bash
conda create -n abot-recon python=3.11 -y
conda activate abot-recon
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# 可选加速
pip install flashinfer-python
cd abot_recon/modeling/pi3/models/curope && pip install ninja && python setup.py build_ext --inplace && cd -
```

### 检查点

发布权重在 [Hugging Face](https://huggingface.co/acvlab/ABot-Recon) 与 [ModelScope](https://modelscope.cn/models/amap_cvlab/ABot-Recon)，API 自动下载；离线使用放置于 `checkpoints/abot_recon.safetensors`。

### 基座快速上手

```bash
python demo.py --image-dir examples/images --output-dir outputs/demo --attention-backend auto --no-loop-closure
```

输出轨迹、相邻相对位姿、局部点图与置信度图；可视化用 `scripts/export_reconstruction_ply.py`；可选回环闭合（`pip install -e ".[loop]"` + `scripts/download_loop_assets.py`）。完整用法、输出项与评测协议见[上游 README](https://github.com/amap-cvlab/ABot-Recon)。

### 测试

```bash
pytest -q
```

## 7. 引用与致谢

本仓库基于 ABot-Recon 构建，基座模型引用：

```bibtex
@misc{han2026revisitinglocalcontextlonghorizon,
      title={Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction},
      author={Jiarong Han and Jincheng Xiong and Yuzhou Liu and Linzhe Shi and Changjie Wu and Ning Guo and Mu Xu and Hang Zhang and Ming Qian},
      year={2026},
      eprint={2608.27529},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.27529},
}
```

潜变量预测方案的思想来源：V-JEPA 2 / V-JEPA 2-AC（latent 预测与 action 条件化）、Flamingo（零初始化门控跨注意力）、PredNet（预测编码）；基座中的时序模块模式参考其 `TemporalRotationRefiner`。

## 8. 许可

源代码遵循 [Apache License 2.0](LICENSE)；模型权重受 [MODEL_LICENSE.md](MODEL_LICENSE.md) 约束，第三方组件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，使用模型前请阅读 [Model Usage Guidelines](MODEL_USAGE_GUIDELINES.md)。
