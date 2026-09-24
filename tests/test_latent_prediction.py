import math

import pytest
import torch

from abot_recon.modeling.streaming.latent_prediction import (
    ACTION_SOURCES,
    DEFAULT_ACTION_SOURCE,
    GatedFusion,
    LatentPredictionConfig,
    LatentPredictionManager,
    LatentPredictor,
    PoseExtrapolator,
    build_latent_prediction_manager,
    latent_prediction_loss,
    se3_to_vec,
    sinusoidal_2d,
)


def test_config_validates_action_source():
    for source in ACTION_SOURCES:
        LatentPredictionConfig(enabled=True, action_source=source)
    with pytest.raises(ValueError, match="action_source"):
        LatentPredictionConfig(enabled=True, action_source="warp_gates")


def test_config_validates_numeric_fields():
    with pytest.raises(ValueError, match="predictor_dim"):
        LatentPredictionConfig(enabled=True, predictor_dim=100, predictor_heads=8)
    with pytest.raises(ValueError, match="extrapolator_kernel_size"):
        LatentPredictionConfig(enabled=True, extrapolator_kernel_size=1)
    with pytest.raises(ValueError, match="predictor_dropout"):
        LatentPredictionConfig(enabled=True, predictor_dropout=1.5)


def test_config_from_mapping_variants():
    assert LatentPredictionConfig.from_mapping(None) is None
    parsed = LatentPredictionConfig.from_mapping({"enabled": True})
    assert isinstance(parsed, LatentPredictionConfig)
    assert parsed.enabled is True
    assert parsed.action_source == DEFAULT_ACTION_SOURCE
    assert LatentPredictionConfig.from_mapping(parsed) is parsed
    with pytest.raises(TypeError):
        LatentPredictionConfig.from_mapping(42)


def test_build_manager_respects_master_switch():
    assert build_latent_prediction_manager(None, token_dim=64) is None
    assert build_latent_prediction_manager({"enabled": False}, token_dim=64) is None
    manager = build_latent_prediction_manager(
        {"enabled": True}, token_dim=64, descriptor_dim=16
    )
    assert isinstance(manager, LatentPredictionManager)


def test_build_manager_auto_history_follows_kv_window():
    # Auto: ring size follows the caller-provided streaming KV window.
    manager = build_latent_prediction_manager(
        {"enabled": True}, token_dim=64, descriptor_dim=16, num_history_frames=12
    )
    assert manager.config.num_history_frames == 12
    # Explicit config wins over the network window (latency / ablation knob).
    manager = build_latent_prediction_manager(
        {"enabled": True, "num_history_frames": 3},
        token_dim=64,
        descriptor_dim=16,
        num_history_frames=12,
    )
    assert manager.config.num_history_frames == 3
    # Standalone construction without a window falls back to 8.
    manager = build_latent_prediction_manager(
        {"enabled": True}, token_dim=64, descriptor_dim=16
    )
    assert manager.config.num_history_frames == 8
    with pytest.raises(ValueError, match="num_history_frames"):
        LatentPredictionConfig(enabled=True, num_history_frames=0)


def test_se3_to_vec_identity():
    identity = torch.eye(4).unsqueeze(0)
    vec = se3_to_vec(identity)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(vec, expected)
    batched = se3_to_vec(torch.eye(4).expand(3, 2, 4, 4))
    assert batched.shape == (3, 2, 9)


def test_extrapolator_is_identity_before_training():
    extrapolator = PoseExtrapolator(hidden_dim=32, kernel_size=4, max_rot_deg=30.0)
    history = [torch.eye(4).unsqueeze(0) for _ in range(3)]
    predicted = extrapolator(history)
    assert predicted.shape == (1, 4, 4)
    assert torch.allclose(predicted, torch.eye(4).unsqueeze(0), atol=1e-6)


def test_extrapolator_output_is_valid_se3():
    torch.manual_seed(0)
    extrapolator = PoseExtrapolator(
        hidden_dim=32, kernel_size=4, max_rot_deg=30.0, descriptor_dim=16
    )
    for head in (extrapolator.out_rot, extrapolator.out_trans):
        torch.nn.init.normal_(head.weight, std=0.5)
        torch.nn.init.normal_(head.bias, std=0.5)
    history = [torch.eye(4).unsqueeze(0) + 0.01 * torch.randn(1, 4, 4) for _ in range(4)]
    predicted = extrapolator(history, descriptor=torch.randn(1, 16))
    rot = predicted[:, :3, :3]
    assert torch.allclose(rot.transpose(1, 2) @ rot, torch.eye(3), atol=1e-4)
    assert torch.allclose(torch.linalg.det(rot), torch.ones(1), atol=1e-4)
    assert predicted[:, 3, 3].allclose(torch.ones(1))


def test_gated_fusion_is_identity_at_init():
    torch.manual_seed(0)
    fusion = GatedFusion(token_dim=64, dim=16, num_heads=2, token_gate=True)
    hidden = torch.randn(2, 10, 64)
    pred = torch.randn(2, 10, 64)
    fused = fusion(hidden, pred)
    assert torch.allclose(fused, hidden, atol=1e-6)


def test_gated_fusion_opens_with_gate():
    torch.manual_seed(0)
    fusion = GatedFusion(token_dim=64, dim=16, num_heads=2, token_gate=True)
    with torch.no_grad():
        fusion.master_gate.fill_(3.0)
    hidden = torch.randn(2, 10, 64)
    pred = torch.randn(2, 10, 64)
    fused = fusion(hidden, pred)
    assert not torch.allclose(fused, hidden)
    assert fusion.gate_magnitude().item() > 0.9


def test_predictor_shapes_and_position_features():
    torch.manual_seed(0)
    predictor = LatentPredictor(token_dim=32, dim=16, depth=1, num_heads=2, num_history_frames=4)
    ring = [torch.randn(2, 7, 32) for _ in range(3)]
    action = torch.randn(2, 2, 16)
    pos = torch.randint(0, 8, (2, 7, 2)).float()
    prediction = predictor(ring, action_tokens=action, pos=pos)
    assert prediction.shape == (2, 7, 32)
    assert sinusoidal_2d(pos, 16).shape == (2, 7, 16)
    with pytest.raises(ValueError, match="mixed token counts"):
        predictor([torch.randn(2, 7, 32), torch.randn(2, 9, 32)])


def test_manager_streaming_protocol_all_action_sources():
    torch.manual_seed(0)
    batch, tokens, token_dim, descriptor_dim = 2, 12, 32, 16
    for source in ACTION_SOURCES:
        config = LatentPredictionConfig(
            enabled=True,
            action_source=source,
            num_history_frames=3,
            predictor_dim=16,
            predictor_depth=1,
            predictor_heads=2,
            fusion_dim=16,
            fusion_heads=2,
            extrapolator_hidden_dim=8,
            extrapolator_kernel_size=3,
        )
        manager = LatentPredictionManager(
            config, token_dim=token_dim, descriptor_dim=descriptor_dim
        )
        identity = torch.eye(4)
        for frame in range(4):
            trunk = torch.randn(batch, tokens, token_dim)
            fused = manager.fuse_frame(trunk, first_frame=frame == 0)
            if frame == 0:
                assert torch.equal(fused, trunk)
            else:
                # Zero-init gate keeps the fusion an exact identity until trained.
                assert torch.allclose(fused, trunk, atol=1e-6)
                assert manager.last_prediction_error is not None
                assert manager.last_prediction_error.shape == (batch, tokens)
            camera_state = {
                "previous_descriptor": torch.randn(batch, descriptor_dim),
                "raw_adjacent_rel_poses": identity.expand(batch, 1, 4, 4).clone(),
            }
            manager.observe_frame(
                trunk, camera_state, first_frame=frame == 0, pos=torch.arange(tokens).float().view(1, -1, 1).expand(batch, tokens, 2) / 7
            )
            assert manager.pending is not None
            assert manager.frame_count == frame + 1
        assert len(manager._ring) == 3  # capped at num_history_frames
        manager.reset()
        assert manager.frame_count == 0
        assert manager.pending is None
        assert manager.diagnostics()["frame_count"] == 0


def test_manager_fusion_only_mode_records_error():
    torch.manual_seed(0)
    config = LatentPredictionConfig(
        enabled=True,
        action_source="none",
        enable_fusion=False,
        predictor_dim=16,
        predictor_depth=1,
        predictor_heads=2,
    )
    manager = LatentPredictionManager(config, token_dim=32, descriptor_dim=16)
    camera_state = {
        "previous_descriptor": torch.randn(1, 16),
        "raw_adjacent_rel_poses": torch.eye(4).expand(1, 1, 4, 4),
    }
    first = torch.randn(1, 8, 32)
    assert torch.equal(manager.fuse_frame(first, first_frame=True), first)
    manager.observe_frame(first, camera_state, first_frame=True)
    second = torch.randn(1, 8, 32)
    assert torch.equal(manager.fuse_frame(second), second)  # fusion disabled
    assert manager.last_prediction_error is not None


def test_manager_resets_on_resolution_change():
    torch.manual_seed(0)
    config = LatentPredictionConfig(
        enabled=True, action_source="none", predictor_dim=16, predictor_depth=1, predictor_heads=2
    )
    manager = LatentPredictionManager(config, token_dim=32, descriptor_dim=16)
    camera_state = {"raw_adjacent_rel_poses": torch.eye(4).expand(1, 1, 4, 4)}
    manager.observe_frame(torch.randn(1, 8, 32), camera_state, first_frame=True)
    assert manager.pending is not None
    other_resolution = torch.randn(1, 6, 32)
    assert torch.equal(manager.fuse_frame(other_resolution), other_resolution)
    assert manager.pending is None  # state was reset


def test_latent_prediction_loss_detaches_target():
    prediction = torch.randn(2, 5, 8, requires_grad=True)
    target = torch.randn(2, 5, 8, requires_grad=True)
    loss = latent_prediction_loss(prediction, target, cosine_weight=0.1)
    assert loss.ndim == 0
    loss.backward()
    assert prediction.grad is not None
    assert target.grad is None


def _build_tiny_network(latent_prediction=None, seed: int = 0):
    torch.manual_seed(seed)
    from abot_recon.modeling.streaming.network import ABotReconNetwork

    return ABotReconNetwork(
        pos_type="rope100",
        decoder_size="large",
        decoder_depth_override=2,
        load_vggt=False,
        freeze_encoder=False,
        freeze_prediction_heads=False,
        use_global_points=False,
        train_conf=False,
        causal_global_attn=True,
        use_packaged_flash_attn=False,
        camera_pose_mode="relative_adjacent",
        relative_camera_head_cfg={
            "head_type": "token_pair",
            "hidden_dim": 64,
            "pair_hidden_dim": 64,
            "num_pose_tokens": 5,
            "rot_correction_kernel": 4,
            "rot_correction_max_deg": 2.0,
        },
        point_z_log_max=10.0,
        ckpt=None,
        use_paged_kv=False,
        global_pos_encoding="pi3_2d",
        local_window_frames=4,
        infer_mode="stream",
        gate_layers=[],
        latent_prediction=latent_prediction,
    )


def test_tiny_network_latent_prediction_module_registration():
    network = _build_tiny_network(
        {
            "enabled": True,
            "action_source": "extrapolator_and_descriptor",
            "predictor_dim": 32,
            "predictor_depth": 1,
            "predictor_heads": 2,
            "fusion_dim": 32,
            "fusion_heads": 2,
            "num_history_frames": 3,
            "extrapolator_hidden_dim": 16,
            "extrapolator_kernel_size": 3,
        }
    )
    assert network.latent_prediction is not None
    keys_with = [k for k in network.state_dict() if k.startswith("latent_prediction.")]
    assert keys_with
    # The strict released-checkpoint load relies on this deregistration trick.
    module = network.latent_prediction
    network.latent_prediction = None
    assert not [k for k in network.state_dict() if k.startswith("latent_prediction.")]
    network.latent_prediction = module
    assert [k for k in network.state_dict() if k.startswith("latent_prediction.")] == keys_with


def test_tiny_network_auto_history_follows_local_window():
    network = _build_tiny_network({"enabled": True, "action_source": "none"})
    assert network.local_window_frames == 4
    assert network.latent_prediction.config.num_history_frames == 4


def test_tiny_streaming_network_untrained_addon_is_output_identical():
    lp_config = {
        "enabled": True,
        "action_source": "extrapolator_and_descriptor",
        "predictor_dim": 32,
        "predictor_depth": 1,
        "predictor_heads": 2,
        "fusion_dim": 32,
        "fusion_heads": 2,
        "num_history_frames": 3,
        "extrapolator_hidden_dim": 16,
        "extrapolator_kernel_size": 3,
    }
    torch.manual_seed(123)
    imgs = torch.randn(1, 5, 3, 56, 56)

    network_off = _build_tiny_network(None, seed=123)
    network_off.eval()
    network_on = _build_tiny_network(lp_config, seed=123)
    network_on.eval()

    with torch.no_grad():
        out_off = network_off.inference_stream(imgs)
        out_on = network_on.inference_stream(imgs)

    assert torch.allclose(out_off["camera_poses"], out_on["camera_poses"], atol=1e-5)
    assert torch.allclose(out_off["local_points"], out_on["local_points"], atol=1e-4)
    assert network_on.latent_prediction.frame_count == 5
    assert network_on.latent_prediction.pending is not None
    assert network_on.latent_prediction.last_prediction_error is not None
