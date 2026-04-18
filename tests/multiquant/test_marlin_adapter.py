#!/usr/bin/env python3
"""Marlin MultiQuant adapter tests."""

import pytest
import torch


class TestMarlinAdapter:
    def test_config_creation(self):
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig
        cfg = MarlinMultiQuantConfig(bits=4, group_size=128)
        assert cfg.bits == 4
        assert cfg.get_name() == "marlin_mq"

    def test_config_from_config(self):
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig
        cfg = MarlinMultiQuantConfig.from_config({"bits": 4})
        assert cfg.bits == 4

    def test_min_capability(self):
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig
        assert MarlinMultiQuantConfig.get_min_capability() == 80

    def test_act_dtypes(self):
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig
        import torch
        dtypes = MarlinMultiQuantConfig.get_supported_act_dtypes()
        assert torch.bfloat16 in dtypes
        assert torch.float16 in dtypes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMarlinAdapterDelegation:
    """Test that MarlinMultiQuantConfig delegates to GPTQMarlinLinearMethod."""

    def test_get_quant_method_for_linear(self):
        """Delegation should return a GPTQMarlin method for LinearBase."""
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig

        cfg = MarlinMultiQuantConfig(bits=4, group_size=128)

        # Create a fake LinearBase layer
        try:
            from vllm.model_executor.layers.linear import LinearBase
            layer = LinearBase.__new__(LinearBase)
            method = cfg.get_quant_method(layer, prefix="model.layers.0.mlp")
            # Should return a method (GPTQMarlinLinearMethod) or None if
            # Marlin not available on this platform
            if method is not None:
                assert hasattr(method, "apply"), "Method should have apply()"
                assert hasattr(method, "create_weights"), \
                    "Method should have create_weights()"
        except ImportError:
            pytest.skip("vllm.model_executor.layers.linear not available")

    def test_get_quant_method_for_non_linear(self):
        """Non-linear layers should return None."""
        from vllm.multiquant.marlin.config import MarlinMultiQuantConfig
        import torch.nn as nn

        cfg = MarlinMultiQuantConfig(bits=4, group_size=128)
        layer = nn.LayerNorm(128)
        result = cfg.get_quant_method(layer, prefix="model.norm")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
