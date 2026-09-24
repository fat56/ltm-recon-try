# plam.md — Latent Prediction（V-JEPA 式下一帧先验）改动说明

## 0. 方案回顾（对应讨论的四点）

| 方案要点 | 实现情况 |
|---|---|
| 1. 保留历史 KV 的同时，保留 trunk（AA 层后）输出的 img token | `LatentPredictionManager` 内 ring buffer，只存**未融合**的 trunk 输出；容量 `num_history_frames`，**默认 None = 自动跟随 `local_window_frames`（即 KV 窗口 12 帧）**，显式指定则作为延迟/消融旋钮 |
| 2. 主模块外挂 predictor，在处理第 k-1 帧结束时预测第 k 帧 latent | `LatentPredictor`（时序 transformer + 帧龄嵌入 + 2D Fourier 位置特征），`observe_frame()` 在每帧 forward 末尾生成并 stash 待用先验 |
| 3. 门控注意力融合（主 token query 预测 token）后再进 head | `GatedFusion`（Flamingo 式零初始化 tanh 门 + 可选逐 token sigmoid 门），挂在 trunk 输出与 point/camera decoder 之间 |
| 4. action 输入：previous_descriptor 或小型位姿外推器，config 切换 | `action_source` 六选一（见下），`PoseExtrapolator` 复用 `TemporalRotationRefiner` 的滚动窗口 + age embedding + 门控深度卷积模式 |

**架构不变量（重要的设计决策）**：
- 融合点在 trunk 之后 → 融合结果**不会**进入后续帧的 KV cache，流式因果性不被污染；
- ring buffer 存未融合 trunk 输出 → 预测目标一致；
- `GatedFusion` 零初始化 → 未训练时输出与原模型**逐位相等**（有测试保证）；
- 全部模块挂载为 `network.latent_prediction` 子模块 → `.to()`/`state_dict` 自动跟随。

## 1. 新增文件

### `abot_recon/modeling/streaming/latent_prediction.py`（全部新逻辑都在这里）

| 组件 | 职责 |
|---|---|
| `LatentPredictionConfig` | frozen dataclass 配置 + 校验；`from_mapping` 接受 dict / OmegaConf DictConfig / None |
| `se3_to_vec` | 4×4 位姿 → 9 维（6D 旋转表示 + 平移），连续无奇异点 |
| `PoseExtrapolator` | 运动历史 → 预测 T_{k-1→k}；零初始化输出头 → 未训练时预测恒等变换（"静止相机"先验）；复用 `AdjacentPoseHead._rotvec_to_mat` |
| `LatentPredictor` | ring buffer（+action token）→ 下一帧 trunk token 预测；取最新帧槽位输出 |
| `ActionTokenEncoder` | SE(3) 位姿 / previous_descriptor → 条件 token |
| `GatedFusion` | `h + tanh(α) ⊙ XAttn(LN(h), LN(ẑ))`，α 零初始化；`gate_magnitude()` 诊断 |
| `LatentPredictionManager` | 持有全部模块与流式状态；`fuse_frame` / `observe_frame` 两个协议入口；`last_prediction_error`（逐 token L1，novelty 信号，可喂 conf/关键帧选择）；`diagnostics()`；分辨率变化自动 reset |
| `latent_prediction_loss` | JEPA 训练损失（SmoothL1 + 可选 cosine），内部 detach target |
| `build_latent_prediction_manager` | 工厂：`enabled=False`/`None` → 返回 None（零开销） |

## 2. 修改的现有文件（全部是"加法"，开关关闭时行为逐位不变）

### `abot_recon/modeling/streaming/network.py`（4 个函数，7 处挂钩）

| 位置 | 改动 |
|---|---|
| `__init__`（gate_layers 后） | ① `kwargs.pop("latent_prediction")`（在 `super().__init__` 前）；② 构建 `self.latent_prediction`（None 则不注册任何东西）；`descriptor_dim` 自动读自 `camera_head.hidden_dim` |
| `forward()` | ③ decode 之后：`fuse_frame`（仅 N==1 且 stream 路径激活，`trunk_hidden` 保留未融合版本）；④ 帧 forward 末尾（pose state 就绪后）：`observe_frame` 生成下一帧先验 |
| `_forward_frame_paged()` | ⑤ `_decode_paged` 后 fuse；⑥ 帧 end 处 observe（`frame_idx==0` 触发 reset） |
| `_forward_frame_camera_only()` | ⑦ 同上（paged 用 `frame_idx==0`、sdpa 用 `past_key_values is None` 判定首帧；该路径 camera_state 会 pop 辅助键，但 `previous_descriptor`/`raw_adjacent_rel_poses` 保留，predictor 正常工作） |

覆盖的推理入口：`inference_stream_iter`（demo/api 主路径）、`_inference_stream_sdpa`、`_inference_stream_paged`、`stream_use_cache` 单帧路径。`infer_mode="full"` 一次性整段推理不激活（构造时打印警告）。

### `abot_recon/config.py`

- `InferenceConfig` 新增字段 `latent_prediction: dict | None = None`（默认关闭）+ 类型校验。

### `abot_recon/model.py`

- 构建网络时下发 `latent_prediction=dict(config.latent_prediction)`；
- **checkpoint 宽容加载**：开启 LP 时，临时把子模块置 None（nn.Module 支持的注销方式）→ 对 released checkpoint 严格加载 → 恢复。released 权重不含 `latent_prediction.*` 键，这是唯一被允许缺失的键；训练后保存的 checkpoint 则完整包含 LP 权重，可正常严格加载；
- `reset()` 同时调用 `latent_prediction.reset()`。

### `abot_recon/cli.py`

- 新增 `--latent-prediction '<json>'` 开关。

## 3. 配置开关用法

```python
# Python API
model = ABotRecon.from_pretrained(
    ...,
    latent_prediction={
        "enabled": True,                          # 总开关（默认 False）
        "action_source": "extrapolator_and_descriptor",  # 见下表
        "num_history_frames": null,               # 默认自动跟随 local_window_frames（KV 窗口，12）；显式 int 为延迟/消融旋钮
        "predictor_dim": 768, "predictor_depth": 6, "predictor_heads": 12,   # ~45M 参数（精度优先默认）
        "enable_fusion": True,                    # False = 仅监控预测误差，不融合
        "fusion_dim": 512, "fusion_heads": 8, "fusion_token_gate": True,
        "extrapolator_hidden_dim": 256,
        "extrapolator_kernel_size": 10,           # 运动历史窗口 K
        "extrapolator_max_rot_deg": 30.0,
    },
)
```

```bash
# CLI
python -m abot_recon.cli --image-dir ... \
  --latent-prediction '{"enabled": true, "action_source": "extrapolator_and_descriptor"}'
```

**`action_source` 六选一**（对应讨论中的 A/B/C/D 档）：

| 值 | action 信号 | 对应讨论 |
|---|---|---|
| `none` | 无（消融基线） | — |
| `const_velocity` | 最新已知 T_{k-2→k-1}（恒速先验） | 方案 A |
| `previous_descriptor` | camera_state 中已 detach 的 512 维视角摘要 | 方案 B（用户选定之一） |
| `const_velocity_and_descriptor` | 两者拼接（**默认**） | 方案 B 完整版 |
| `extrapolator` | 外推器预测的 T̂_{k-1→k} | 方案 C（用户选定之一） |
| `extrapolator_and_descriptor` | 外推器 + descriptor | 方案 C 完整版（推荐 v2） |

## 4. 时序说明（prediction 何时生成、何时消费）

```
帧 k-1 forward:  decode → fuse(ẑ_{k-1}) → heads → pose state 更新
                                                  └→ observe: push z_{k-1} 入 ring，
                                                     从 motion_history + descriptor 生成
                                                     ẑ_k（此刻 T_{k-1→k} 未知，由所选
                                                     action_source 近似/外推）
帧 k forward:    decode → fuse(ẑ_k)（消费上一帧产物）→ heads → ...
```

- 恒速档用的运动是 `raw_adjacent_rel_poses` 最新条目（滞后一步，恒速近似）；
- 外推器维护自己的运动特征滚动窗口（模式同 `TemporalRotationRefiner.feature_buffer`）；
- `previous_descriptor` / `raw_adjacent_rel_poses` 均来自 pose head 已 detach 的状态字典，**零梯度耦合、零新增依赖边**。

## 5. 训练路线（模块已就绪，训练代码在训练仓）

1. **Stage 1 — predictor/extrapolator 预训练**：冻结主干，用 `manager.predictor(ring, action_tokens, pos)` + `latent_prediction_loss(ẑ, sg(z_target))`；extrapolator 用 GT 相对位姿做 next-pose 回归（可完全独立预训练）。
   - **缓存技巧**：Stage 1 不需要重跑主干——对每个数据 pass 把 trunk 输出 z_k、camera_state（descriptor / rel poses）、pos 缓存下来，predictor 直接在缓存特征上训练。这使得容量扫描（2L/512 → 6L/768 → 12L/1024）几乎免费，无需每次过 1B 主干。
   - **容量与过拟合**：容量太小→欠拟合（学不动 warp+disocclusion 补全）；过拟合风险主要来自数据多样性不足而非参数量。判定标准：held-out **场景**（非帧）上的 JEPA loss 曲线——train/val 都高且 gap 小 = 欠拟合，升容量；train 低 val 高 = 过拟合，加数据/正则（`predictor_dropout`、weight decay）。零初始化门控保证劣质 predictor 只会被关门、不会伤害主精度，因此容量宁可放大。
2. **Stage 2 — 融合微调**：只训 `fusion`（+ predictor 小 lr），任务损失走现有 depth/pose/conf 损失；监控 `diagnostics()["fusion_gate_magnitude"]`（塌到 0 = 预测无价值；全开 = 过信任/时序拖影）。
3. **Stage 3 —（可选）联合微调**：预测目标必须换 EMA target trunk，防止主干把自己学成"好预测"的表征。

消融建议顺序：`none` → `const_velocity` → `previous_descriptor` → `extrapolator_and_descriptor`；以及 `enable_fusion=False`（纯误差监控）对照。

## 6. 测试

`tests/test_latent_prediction.py`（16 个用例）：

- 配置校验 / from_mapping 各变体 / 总开关；
- `se3_to_vec`、外推器零初始化恒等、外推输出 SE(3) 有效性（RᵀR=I, det=1）；
- 融合零初始化恒等、开门后生效；
- predictor 形状 / 位置特征 / 混合分辨率报错；
- manager 全部六种 action_source 的流式协议、误差记录、reset、分辨率变化 reset、fusion-only 模式；
- 损失 target detach；
- **tiny 网络集成**：state_dict 注册开关（验证 model.py 的宽容加载技巧）；`decoder_size="large" + decoder_depth_override=2` 小网络上 5 帧流式推理，**开启未训练 LP 与关闭 LP 输出逐位一致**（atol 1e-5/1e-4）。

运行：`pytest tests/test_latent_prediction.py`（16 passed；存量测试无回归。`test_loop_closure`/部分 `test_api` 因环境缺 `pypose` 失败，与本次改动无关）。

## 7. 已知边界

- `infer_mode="full"` 不激活（整段一次前向无法逐帧穿插先验），构造时打印警告；
- token 数随分辨率变化时自动 reset 预测状态（与 paged manager 重建行为一致）；
- `num_history_frames` 与 KV 窗口是两种内存：KV 是逐层中间表征（会 prune/summary），ring 是每帧一份 trunk 最终输出快照（~3MB/帧）；快照 z_j 已含其窗口内 KV 的信息，故 predictor 的有效感受野大于 ring 长度，允许独立调小控算力（predictor 自注意力 ~ (m×tokens)²）。运行时 `set_local_window_frames()` 改窗口不会联动 ring 容量（构建时解析）；
- 训练路径（training runtime）需在训练仓显式调用 predictor + loss；本仓只保证推理集成。
