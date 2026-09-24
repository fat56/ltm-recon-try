"""JEPA-style next-frame latent prediction for ABot-Recon streaming.

This module implements the "predictive prior" extension for ABot-Recon:

1. While streaming, keep a ring buffer of trunk outputs (the fused hidden
   that feeds the point/camera heads) for the last ``num_history_frames``
   frames, alongside the existing KV carry.
2. An external ``LatentPredictor`` maps that ring buffer (plus optional
   action tokens) to a prediction of the *next* frame's trunk output.  The
   prediction is generated at the end of frame ``k-1`` and stashed as the
   pending prior.
3. At frame ``k`` the prior is fused back into the trunk output with a
   Flamingo-style, zero-initialised gated cross-attention.  Because the gate
   is an exact identity at initialisation, an untrained add-on never changes
   the released checkpoint's outputs.

The action conditioning of the predictor is selected via ``action_source``:

* ``none``                          – unconditional prediction (ablation)
* ``const_velocity``                – embed the newest known relative pose
                                      T_{k-2->k-1} (constant-velocity prior)
* ``previous_descriptor``           – project the pose head's detached
                                      ``previous_descriptor`` (viewpoint
                                      summary carried in ``camera_state``)
* ``const_velocity_and_descriptor`` – both tokens (recommended v1)
* ``extrapolator``                  – small pose extrapolator predicts
                                      T_{k-1->k} from the motion history
* ``extrapolator_and_descriptor``   – extrapolator + descriptor (recommended v2)

``PoseExtrapolator`` mirrors the ``TemporalRotationRefiner`` recipe from
``abot_recon/modeling/pi3/models/layers/adjacent_pose_head.py`` (per-step
features -> rolling window with age embeddings -> gated depthwise temporal
convolution -> bounded heads), but its zero-initialised output heads predict
the identity transform before training ("static camera" prior).

Everything here is an add-on: with ``enabled=False`` (the default) no module
is created and the network behaves exactly as before.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from abot_recon.modeling.pi3.models.layers.adjacent_pose_head import AdjacentPoseHead


ACTION_SOURCES = (
    "none",
    "const_velocity",
    "previous_descriptor",
    "const_velocity_and_descriptor",
    "extrapolator",
    "extrapolator_and_descriptor",
)
DEFAULT_ACTION_SOURCE = "const_velocity_and_descriptor"


def se3_to_vec(poses: torch.Tensor) -> torch.Tensor:
    """(…, 4, 4) SE(3) -> (…, 9): 6D rotation rows + translation.

    The 6D rotation parameterisation (first two rows of R) is continuous and
    singularity-free, which keeps the small MLP encoders well-behaved.
    """
    if poses.dim() < 2 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"expected (..., 4, 4) poses, got {tuple(poses.shape)}")
    rot = poses[..., :2, :3].reshape(*poses.shape[:-2], 6)
    trans = poses[..., :3, 3]
    return torch.cat([rot, trans], dim=-1).float()


@dataclass(frozen=True)
class LatentPredictionConfig:
    """Configuration for the add-on latent-prediction head."""

    enabled: bool = False
    action_source: str = DEFAULT_ACTION_SOURCE
    # None (default) = auto: follow the streaming KV window (local_window_frames)
    # of the network that builds this manager.  An explicit int is a latency /
    # ablation knob: predictor self-attention scales with (frames x tokens)^2,
    # and each ring entry is a full trunk-output snapshot that already saw its
    # own KV window, so shorter contexts remain meaningful.
    num_history_frames: Optional[int] = None
    # Accuracy-first defaults (~45M params, ~+5% trunk FLOPs at ring=12).
    # Latency-first alternative: dim=512, depth=3, heads=8 (~9M).
    # Max-capacity ablation point: dim=1024, depth=12, heads=16 (~160M).
    predictor_dim: int = 768
    predictor_depth: int = 6
    predictor_heads: int = 12
    predictor_dropout: float = 0.0
    enable_fusion: bool = True
    fusion_dim: int = 512
    fusion_heads: int = 8
    fusion_token_gate: bool = True
    extrapolator_hidden_dim: int = 256
    extrapolator_kernel_size: int = 10
    extrapolator_max_rot_deg: float = 30.0

    def __post_init__(self) -> None:
        if self.action_source not in ACTION_SOURCES:
            raise ValueError(
                f"action_source must be one of {ACTION_SOURCES}, got {self.action_source!r}"
            )
        positive_ints = (
            "predictor_dim",
            "predictor_depth",
            "predictor_heads",
            "fusion_dim",
            "fusion_heads",
            "extrapolator_hidden_dim",
            "extrapolator_kernel_size",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.num_history_frames is not None and (
            type(self.num_history_frames) is not int or self.num_history_frames <= 0
        ):
            raise ValueError(
                f"num_history_frames must be a positive int or None (auto), got {self.num_history_frames!r}"
            )
        if self.predictor_dim % self.predictor_heads != 0:
            raise ValueError("predictor_dim must be divisible by predictor_heads")
        if self.fusion_dim % self.fusion_heads != 0:
            raise ValueError("fusion_dim must be divisible by fusion_heads")
        if self.predictor_dim % 4 != 0:
            raise ValueError("predictor_dim must be divisible by 4 (2D Fourier features)")
        if not 0.0 <= self.predictor_dropout <= 1.0:
            raise ValueError("predictor_dropout must be in [0, 1]")
        if self.extrapolator_kernel_size < 2:
            raise ValueError("extrapolator_kernel_size must be >= 2")
        if self.extrapolator_max_rot_deg <= 0.0:
            raise ValueError("extrapolator_max_rot_deg must be positive")

    @classmethod
    def from_mapping(cls, value: Any) -> Optional["LatentPredictionConfig"]:
        """Accepts None / dict / Mapping / OmegaConf DictConfig / dataclass."""
        if value is None:
            return None
        if isinstance(value, LatentPredictionConfig):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(value, DictConfig):
                return cls(**OmegaConf.to_container(value, resolve=True))
        except ImportError:
            pass
        raise TypeError(
            "latent_prediction config must be None, a mapping or a "
            f"LatentPredictionConfig, got {type(value).__name__}"
        )


def latent_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    """JEPA-style latent prediction loss; the target is detached inside."""
    pred = prediction.float()
    tgt = target.detach().float()
    loss = F.smooth_l1_loss(pred, tgt)
    if cosine_weight > 0.0:
        loss = loss + cosine_weight * (1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean())
    return loss


class PoseExtrapolator(nn.Module):
    """Predicts the next adjacent relative pose T_{k-1->k} from motion history.

    Follows the ``TemporalRotationRefiner`` pattern: per-step pose features are
    stacked into a rolling window with age embeddings, a gated depthwise
    temporal convolution aggregates them, and bounded heads emit a rotation
    vector plus translation.  Output heads are zero-initialised, so the
    untrained extrapolator predicts the identity transform.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        kernel_size: int = 10,
        max_rot_deg: float = 30.0,
        descriptor_dim: int = 512,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        self.kernel_size = int(kernel_size)
        self.max_rad = float(max_rot_deg) * math.pi / 180.0

        self.pose_mlp = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.desc_proj: Optional[nn.Sequential] = None
        if descriptor_dim > 0:
            self.desc_proj = nn.Sequential(
                nn.LayerNorm(descriptor_dim),
                nn.Linear(descriptor_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        self.age_embed = nn.Embedding(self.kernel_size, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, self.kernel_size, groups=hidden_dim)
        self.gate_conv = nn.Conv1d(hidden_dim, hidden_dim, self.kernel_size, groups=hidden_dim)
        self.out_rot = nn.Linear(hidden_dim, 3)
        self.out_trans = nn.Linear(hidden_dim, 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for sequence in (self.pose_mlp, self.desc_proj):
            if sequence is None:
                continue
            for module in sequence:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.age_embed.weight, std=0.02)
        for conv in (self.conv, self.gate_conv):
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            nn.init.zeros_(conv.bias)
        for head in (self.out_rot, self.out_trans):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        motion_history: Sequence[torch.Tensor],
        descriptor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """motion_history: (B, 4, 4) tensors, oldest -> newest (non-empty).

        Returns the predicted next relative pose (B, 4, 4), float32.
        """
        if not motion_history:
            raise ValueError("motion_history must contain at least one pose")
        feats = [self.pose_mlp(se3_to_vec(pose)) for pose in motion_history]
        window = torch.stack(feats, dim=1)  # (B, n, H)
        batch, num_steps, hidden = window.shape
        pad = self.kernel_size - num_steps
        valid = window.new_zeros(self.kernel_size)
        if pad > 0:
            window = torch.cat([window.new_zeros(batch, pad, hidden), window], dim=1)
        valid[self.kernel_size - num_steps :] = 1.0

        age_ids = torch.arange(self.kernel_size - 1, -1, -1, device=window.device)
        age = self.age_embed(age_ids).to(window.dtype)
        window = window + age.unsqueeze(0) * valid.view(1, -1, 1)

        temporal = window.transpose(1, 2).float()  # (B, H, K)
        fused = (self.conv(temporal) * torch.sigmoid(self.gate_conv(temporal))).squeeze(-1)

        if descriptor is not None and self.desc_proj is not None:
            desc = descriptor.reshape(descriptor.shape[0], -1).float()
            fused = fused + self.desc_proj(desc).reshape(batch, -1)

        rotvec = self.max_rad * torch.tanh(self.out_rot(fused))
        trans = self.out_trans(fused)
        rot = AdjacentPoseHead._rotvec_to_mat(rotvec)

        delta = torch.zeros((batch, 4, 4), dtype=torch.float32, device=rot.device)
        delta[:, :3, :3] = rot
        delta[:, :3, 3] = trans
        delta[:, 3, 3] = 1.0
        return delta


def sinusoidal_2d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """(B, T, 2) grid coordinates -> (B, T, dim) Fourier features."""
    if pos.dim() != 3 or pos.shape[-1] != 2:
        raise ValueError(f"expected (B, T, 2) positions, got {tuple(pos.shape)}")
    if dim % 4 != 0:
        raise ValueError("dim must be divisible by 4")
    coords = pos.float()
    num_freq = dim // 4
    freqs = torch.exp(
        torch.linspace(0.0, math.log(10000.0), num_freq, device=pos.device, dtype=torch.float32)
    )

    def _one(coord: torch.Tensor) -> torch.Tensor:
        angles = coord.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    return torch.cat([_one(coords[..., 0]), _one(coords[..., 1])], dim=-1)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LatentPredictor(nn.Module):
    """Predicts the next frame's trunk tokens from a ring buffer of past ones."""

    def __init__(
        self,
        *,
        token_dim: int,
        dim: int = 512,
        depth: int = 3,
        num_heads: int = 8,
        num_history_frames: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.dim = int(dim)
        self.num_history_frames = int(num_history_frames)
        self.in_proj = nn.Linear(token_dim, dim)
        self.out_proj = nn.Linear(dim, token_dim)
        self.age_embed = nn.Embedding(num_history_frames, dim)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(dim, num_heads, dropout=dropout) for _ in range(depth)]
        )

    def forward(
        self,
        ring: Sequence[torch.Tensor],
        action_tokens: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ring: (B, T, token_dim) trunk outputs, oldest -> newest (non-empty).

        action_tokens: optional (B, A, dim) conditioning tokens.
        pos: optional (B, T, 2) spatial grid for Fourier features.
        Returns the predicted next-frame tokens (B, T, token_dim).
        """
        if not ring:
            raise ValueError("ring must contain at least one frame")
        token_counts = {frame.shape[1] for frame in ring}
        if len(token_counts) != 1:
            raise ValueError("mixed token counts in ring; reset the streaming state")
        num_tokens = ring[-1].shape[1]
        if pos is not None and pos.shape[1] != num_tokens:
            raise ValueError(
                f"pos has {pos.shape[1]} tokens but ring frames have {num_tokens}"
            )

        spatial = sinusoidal_2d(pos, self.dim) if pos is not None else None
        num_frames = len(ring)
        frames: List[torch.Tensor] = []
        for index, frame in enumerate(ring):
            x = self.in_proj(frame)
            age = num_frames - 1 - index  # 0 = newest
            x = x + self.age_embed.weight[age].to(x.dtype)
            if spatial is not None:
                x = x + spatial.to(x.dtype)
            frames.append(x)

        if action_tokens is not None and action_tokens.shape[1] > 0:
            tokens = torch.cat(frames + [action_tokens.to(frames[-1].dtype)], dim=1)
        else:
            tokens = torch.cat(frames, dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        newest = tokens[:, (num_frames - 1) * num_tokens : num_frames * num_tokens]
        return self.out_proj(newest)


class ActionTokenEncoder(nn.Module):
    """Encodes SE(3) poses and pose-head descriptors as conditioning tokens."""

    def __init__(self, *, dim: int, descriptor_dim: int = 512) -> None:
        super().__init__()
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.desc_proj: Optional[nn.Sequential] = None
        if descriptor_dim > 0:
            self.desc_proj = nn.Sequential(
                nn.LayerNorm(descriptor_dim),
                nn.Linear(descriptor_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

    def pose_token(self, pose: torch.Tensor) -> torch.Tensor:
        """(B, 4, 4) -> (B, 1, dim)."""
        return self.pose_proj(se3_to_vec(pose)).unsqueeze(1)

    def descriptor_token(self, descriptor: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, 1, dim)."""
        if self.desc_proj is None:
            raise RuntimeError("ActionTokenEncoder built without descriptor support")
        desc = descriptor.reshape(descriptor.shape[0], -1).float()
        return self.desc_proj(desc).unsqueeze(1)


class GatedFusion(nn.Module):
    """Flamingo-style gated cross-attention; exact identity at initialisation.

    ``hidden`` is the query path (may carry gradients for task-loss training);
    ``pred`` is treated as detached memory by the caller.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        dim: int = 512,
        num_heads: int = 8,
        token_gate: bool = True,
    ) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(token_dim)
        self.norm_pred = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, dim)
        self.kv_proj = nn.Linear(token_dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim, token_dim)
        self.master_gate = nn.Parameter(torch.zeros(token_dim))
        self.token_gate_proj: Optional[nn.Linear] = None
        if token_gate:
            self.token_gate_proj = nn.Linear(3 * token_dim, 1)
            nn.init.zeros_(self.token_gate_proj.weight)
            nn.init.zeros_(self.token_gate_proj.bias)

    def forward(self, hidden: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        pred = pred.to(dtype=hidden.dtype)
        query = self.q_proj(self.norm_query(hidden))
        memory = self.kv_proj(self.norm_pred(pred))
        delta = self.out_proj(self.attn(query, memory, memory, need_weights=False)[0])
        if self.token_gate_proj is not None:
            gate_input = torch.cat([hidden, pred, hidden - pred], dim=-1)
            delta = delta * torch.sigmoid(self.token_gate_proj(gate_input))
        return hidden + torch.tanh(self.master_gate).to(hidden.dtype) * delta.to(hidden.dtype)

    def gate_magnitude(self) -> torch.Tensor:
        return torch.tanh(self.master_gate).abs().mean().detach()


class LatentPredictionManager(nn.Module):
    """Owns the add-on modules and the streaming prediction state.

    Inference protocol (called from ``ABotReconNetwork``):

    * ``fuse_frame(trunk_hidden, first_frame=...)`` right after the decoder
      trunk, before the head decoders;
    * ``observe_frame(trunk_hidden, camera_state, first_frame=..., pos=...)``
      at the very end of the frame forward with the *unfused* trunk output
      and the pose head's updated camera state.

    The ring buffer only ever stores unfused trunk outputs, so the prediction
    target stays consistent and the fused representation never leaks into the
    KV carry of future frames.
    """

    def __init__(
        self,
        config: LatentPredictionConfig,
        *,
        token_dim: int,
        descriptor_dim: int = 512,
    ) -> None:
        super().__init__()
        if not isinstance(config, LatentPredictionConfig):
            raise TypeError("config must be a LatentPredictionConfig")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if config.num_history_frames is None:
            # Direct construction bypasses the factory: fall back to 8.  The
            # factory resolves None against the network's KV window instead.
            config = replace(config, num_history_frames=8)
        self.config = config
        self.token_dim = int(token_dim)
        source = config.action_source
        self._use_const_velocity = source in ("const_velocity", "const_velocity_and_descriptor")
        self._use_extrapolator = source in ("extrapolator", "extrapolator_and_descriptor")
        self._use_descriptor = source in (
            "previous_descriptor",
            "const_velocity_and_descriptor",
            "extrapolator_and_descriptor",
        )

        self.action_encoder = ActionTokenEncoder(
            dim=config.predictor_dim,
            descriptor_dim=descriptor_dim if self._use_descriptor else 0,
        )
        self.extrapolator: Optional[PoseExtrapolator] = None
        if self._use_extrapolator:
            self.extrapolator = PoseExtrapolator(
                hidden_dim=config.extrapolator_hidden_dim,
                kernel_size=config.extrapolator_kernel_size,
                max_rot_deg=config.extrapolator_max_rot_deg,
                descriptor_dim=descriptor_dim,
            )
        self.predictor = LatentPredictor(
            token_dim=token_dim,
            dim=config.predictor_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            num_history_frames=config.num_history_frames,
            dropout=config.predictor_dropout,
        )
        self.fusion: Optional[GatedFusion] = None
        if config.enable_fusion:
            self.fusion = GatedFusion(
                token_dim=token_dim,
                dim=config.fusion_dim,
                num_heads=config.fusion_heads,
                token_gate=config.fusion_token_gate,
            )

        self._ring: List[torch.Tensor] = []
        self._motion_history: List[torch.Tensor] = []
        self._pending: Optional[torch.Tensor] = None
        self._frame_count = 0
        self._last_error: Optional[torch.Tensor] = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_prediction_error(self) -> Optional[torch.Tensor]:
        """Per-token L1 error of the most recent fused prediction, (B, T)."""
        return self._last_error

    @property
    def pending(self) -> Optional[torch.Tensor]:
        return self._pending

    def reset(self) -> None:
        self._ring = []
        self._motion_history = []
        self._pending = None
        self._frame_count = 0
        self._last_error = None

    def _motion_history_cap(self) -> int:
        if self._use_extrapolator:
            return self.config.extrapolator_kernel_size
        return 2

    def fuse_frame(self, trunk_hidden: torch.Tensor, *, first_frame: bool = False) -> torch.Tensor:
        """Fuse the pending prior into the trunk output (identity if none)."""
        if first_frame:
            self.reset()
        if (
            self._pending is not None
            and self._pending.shape != trunk_hidden.shape
        ):
            # Resolution changed mid-stream: start a fresh prediction state.
            self.reset()
        self._last_error = None
        pred = self._pending
        if pred is None:
            return trunk_hidden
        error = (trunk_hidden.detach().float() - pred.float()).abs().mean(dim=-1)
        self._last_error = error
        if self.fusion is None:
            return trunk_hidden
        return self.fusion(trunk_hidden, pred)

    def observe_frame(
        self,
        trunk_hidden: torch.Tensor,
        camera_state: Optional[Dict[str, torch.Tensor]],
        *,
        first_frame: bool = False,
        pos: Optional[torch.Tensor] = None,
    ) -> None:
        """Store the unfused trunk output and generate the next-frame prior."""
        if first_frame:
            self.reset()
        if camera_state is None:
            return
        if self._ring and self._ring[-1].shape[1] != trunk_hidden.shape[1]:
            self.reset()

        self._ring.append(trunk_hidden.detach())
        if len(self._ring) > self.config.num_history_frames:
            self._ring.pop(0)

        rel = camera_state.get("raw_adjacent_rel_poses")
        if (
            torch.is_tensor(rel)
            and rel.dim() == 3
            and rel.shape[-2:] == (4, 4)
            and rel.shape[1] > 0
        ):
            self._motion_history.append(rel[:, -1].detach().float())
            cap = self._motion_history_cap()
            if len(self._motion_history) > cap:
                self._motion_history.pop(0)

        descriptor = None
        if self._use_descriptor:
            candidate = camera_state.get("previous_descriptor")
            if torch.is_tensor(candidate):
                descriptor = candidate

        action_tokens = self._build_action_tokens(descriptor)
        prediction = self.predictor(self._ring, action_tokens=action_tokens, pos=pos)
        self._pending = prediction.detach()
        self._frame_count += 1

    def _build_action_tokens(self, descriptor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        tokens: List[torch.Tensor] = []
        if self._use_const_velocity and self._motion_history:
            tokens.append(self.action_encoder.pose_token(self._motion_history[-1]))
        if (
            self._use_extrapolator
            and self.extrapolator is not None
            and self._motion_history
        ):
            predicted_pose = self.extrapolator(self._motion_history, descriptor)
            tokens.append(self.action_encoder.pose_token(predicted_pose))
        if self._use_descriptor and descriptor is not None:
            tokens.append(self.action_encoder.descriptor_token(descriptor))
        if not tokens:
            return None
        return torch.cat(tokens, dim=1)

    def diagnostics(self) -> Dict[str, Any]:
        error = self._last_error
        return {
            "frame_count": self._frame_count,
            "ring_size": len(self._ring),
            "has_pending": self._pending is not None,
            "last_prediction_error_mean": (
                float(error.mean()) if error is not None else None
            ),
            "fusion_gate_magnitude": (
                float(self.fusion.gate_magnitude()) if self.fusion is not None else None
            ),
        }


def build_latent_prediction_manager(
    config: Any,
    *,
    token_dim: int,
    descriptor_dim: int = 512,
    num_history_frames: Optional[int] = None,
) -> Optional[LatentPredictionManager]:
    """Factory: returns None unless the config enables the add-on.

    ``num_history_frames`` is the auto-fallback used when the config leaves
    the ring size unset: the caller (the network) passes its streaming KV
    window so the predictor context follows it by default.  Standalone
    construction without a window falls back to 8.
    """
    parsed = LatentPredictionConfig.from_mapping(config)
    if parsed is None or not parsed.enabled:
        return None
    if parsed.num_history_frames is None:
        if num_history_frames is not None and num_history_frames > 0:
            parsed = replace(parsed, num_history_frames=int(num_history_frames))
        else:
            parsed = replace(parsed, num_history_frames=8)
    return LatentPredictionManager(parsed, token_dim=token_dim, descriptor_dim=descriptor_dim)
