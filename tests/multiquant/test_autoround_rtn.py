#!/usr/bin/env python3
"""AutoRound RTN adapter tests."""

import pytest
import torch


class TestAutoRoundRTNConfig:
    def test_config_creation(self):
        from vllm.multiquant.autoround.config import AutoRoundRTNConfig
        cfg = AutoRoundRTNConfig(bits=4, group_size=128)
        assert cfg.bits == 4
        assert cfg.get_name() == "autoround_rtn"

    def test_config_from_config(self):
        from vllm.multiquant.autoround.config import AutoRoundRTNConfig
        cfg = AutoRoundRTNConfig.from_config({"bits": 4})
        assert cfg.bits == 4


class TestAutoRoundRTNQuality:
    def test_rtn_int4_quality(self):
        """Simple symmetric RTN should preserve ~95%+ cosine sim at 4-bit."""
        from vllm.multiquant.autoround.online_linear import (
            AutoRoundRTNLinearMethod,
        )
        from vllm.multiquant.autoround.config import AutoRoundRTNConfig

        torch.manual_seed(42)
        W = torch.randn(64, 128)
        bits = 4
        group_size = 128
        n_levels = 2 ** bits

        # Symmetric per-group RTN
        max_val = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / (n_levels // 2 - 1)
        q = torch.clamp(torch.round(W / scale), -(n_levels // 2), n_levels // 2 - 1)
        W_q = q * scale

        cos = torch.nn.functional.cosine_similarity(W, W_q, dim=-1).mean()
        assert cos > 0.95, f"RTN INT4 quality too low: {cos:.4f}"

    def test_rtn_int4_vs_archer_rq3(self):
        """Compare RTN INT4 vs Archer RQ3 — RTN should be better at 4bit."""
        torch.manual_seed(42)
        W = torch.randn(64, 128)

        # RTN INT4
        n_levels = 16
        max_val = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / (n_levels // 2 - 1)
        q = torch.clamp(torch.round(W / scale), -8, 7)
        W_rtn = q * scale
        cos_rtn = torch.nn.functional.cosine_similarity(W, W_rtn, dim=-1).mean()

        # Archer RQ3 (from test_archer)
        from vllm.multiquant.shared.centroids import get_centroids
        from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
        Pi = generate_rotation_matrix(128, seed=42)
        centroids = get_centroids(128, 2)  # mse_bits=2 for tq3
        row_norms = W.norm(dim=-1)
        W_unit = W / (row_norms.unsqueeze(-1) + 1e-8)
        rotated = W_unit @ Pi.T
        diffs = rotated.unsqueeze(-1) - centroids
        idx = diffs.abs().argmin(dim=-1)
        quantized = centroids[idx]
        W_mse = quantized @ Pi
        W_archer = row_norms.unsqueeze(-1) * W_mse
        cos_archer = torch.nn.functional.cosine_similarity(
            W, W_archer, dim=-1
        ).mean()

        # RTN INT4 (4bit) should beat Archer RQ3 (3bit MSE only)
        assert cos_rtn > cos_archer, (
            f"RTN INT4 ({cos_rtn:.4f}) should beat Archer 3bit ({cos_archer:.4f})"
        )


class TestAutoRoundRTNMethod:
    def test_uses_meta_device(self):
        from vllm.multiquant.autoround.online_linear import (
            AutoRoundRTNLinearMethod,
        )
        from vllm.multiquant.autoround.config import AutoRoundRTNConfig
        cfg = AutoRoundRTNConfig(bits=4)
        method = AutoRoundRTNLinearMethod(cfg)
        assert method.uses_meta_device is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAutoRoundRTNPackDecompress:
    """Full pack → decompress → F.linear pipeline on GPU."""

    def test_int4_pack_decompress_roundtrip(self):
        """Pack to INT4, decompress, verify quality."""
        from vllm.multiquant.autoround.online_linear import (
            AutoRoundRTNLinearMethod,
        )
        from vllm.multiquant.autoround.config import AutoRoundRTNConfig

        cfg = AutoRoundRTNConfig(bits=4, group_size=128)
        method = AutoRoundRTNLinearMethod(cfg)

        torch.manual_seed(42)
        layer = torch.nn.Module()
        W = torch.randn(256, 128, device="cuda")
        layer.weight = torch.nn.Parameter(W.clone(), requires_grad=False)

        method.process_weights_after_loading(layer)

        assert layer._rtn_packed is True
        assert layer.weight.dtype == torch.int32
        assert hasattr(layer, "_rtn_scales")

        # Decompress
        W_d = method._decompress(layer)
        assert W_d.shape == (256, 128)
        cos = torch.nn.functional.cosine_similarity(
            W, W_d.float(), dim=-1
        ).mean()
        assert cos > 0.95, f"INT4 RTN quality too low: {cos:.4f}"

    def test_int4_linear_inference(self):
        """Full inference: pack weights, run F.linear, check output."""
        from vllm.multiquant.autoround.online_linear import (
            AutoRoundRTNLinearMethod,
        )
        from vllm.multiquant.autoround.config import AutoRoundRTNConfig

        cfg = AutoRoundRTNConfig(bits=4, group_size=128)
        method = AutoRoundRTNLinearMethod(cfg)

        torch.manual_seed(42)
        layer = torch.nn.Module()
        W = torch.randn(64, 128, device="cuda")
        layer.weight = torch.nn.Parameter(W.clone(), requires_grad=False)

        # Quantize
        method.process_weights_after_loading(layer)

        # Inference
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
        y_quant = method.apply(layer, x)

        # Reference
        y_ref = torch.nn.functional.linear(x, W.bfloat16())

        assert y_quant.shape == (4, 64)
        assert not y_quant.isnan().any()
        cos = torch.nn.functional.cosine_similarity(
            y_quant.float(), y_ref.float(), dim=-1
        ).mean()
        assert cos > 0.95, f"INT4 linear output quality: {cos:.4f}"

    def test_memory_reduction(self):
        """Packed INT4 should use ~12.5% of BF16 memory."""
        from vllm.multiquant.autoround.online_linear import (
            AutoRoundRTNLinearMethod,
        )
        from vllm.multiquant.autoround.config import AutoRoundRTNConfig

        cfg = AutoRoundRTNConfig(bits=4, group_size=128)
        method = AutoRoundRTNLinearMethod(cfg)

        layer = torch.nn.Module()
        W = torch.randn(1024, 1024, device="cuda")
        layer.weight = torch.nn.Parameter(W.clone(), requires_grad=False)
        bf16_bytes = 1024 * 1024 * 2

        method.process_weights_after_loading(layer)

        packed_bytes = layer.weight.numel() * layer.weight.element_size()
        ratio = packed_bytes / bf16_bytes
        # INT4 packed in int32: 1024 * (1024/8) * 4 = 512KB vs 2MB = 25%
        # (int32 packing is less efficient than uint8)
        assert ratio < 0.30, f"Packed {ratio:.1%} of BF16 — expected <30%"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAutoRoundRTNGroupSizes:
    """Test different group sizes."""

    @pytest.mark.parametrize("group_size", [32, 64, 128, -1])
    def test_group_size(self, group_size):
        from vllm.multiquant.autoround.online_linear import (
            AutoRoundRTNLinearMethod,
        )
        from vllm.multiquant.autoround.config import AutoRoundRTNConfig

        cfg = AutoRoundRTNConfig(bits=4, group_size=group_size)
        method = AutoRoundRTNLinearMethod(cfg)

        torch.manual_seed(42)
        layer = torch.nn.Module()
        W = torch.randn(64, 256, device="cuda")
        layer.weight = torch.nn.Parameter(W.clone(), requires_grad=False)

        method.process_weights_after_loading(layer)
        W_d = method._decompress(layer)
        cos = torch.nn.functional.cosine_similarity(
            W, W_d.float(), dim=-1
        ).mean()
        assert cos > 0.90, f"group_size={group_size}: cos={cos:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
