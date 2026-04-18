#!/usr/bin/env python3
"""Archer memory management tests.

Verifies that per-layer quantization actually frees BF16 memory.
"""

import gc

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestArcherMemory:
    def test_zero_trick_frees_memory(self):
        """old.data = empty(0) frees CUDA allocation despite external refs."""
        W = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
        external_ref = W  # Simulates params_dict reference

        before = torch.cuda.memory_allocated()
        W.data = torch.empty(0, device="cuda", dtype=W.dtype)
        del W
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()

        assert after < before, (
            f"Memory not freed: before={before}, after={after}"
        )
        # external_ref now points to empty tensor
        assert external_ref.numel() == 0

    def test_per_layer_quantize_frees_bf16(self):
        """Simulates Archer: load BF16, quantize, replace, check memory."""
        import torch.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(10):
                    self.register_parameter(
                        f"w{i}",
                        nn.Parameter(torch.empty(0), requires_grad=False),
                    )

        model = FakeModel().cuda()
        params_dict = dict(model.named_parameters())

        # Load BF16
        for i in range(10):
            w = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
            setattr(model, f"w{i}", nn.Parameter(w, requires_grad=False))

        bf16_mem = torch.cuda.memory_allocated()

        # Quantize (simulate Archer zero trick)
        for i in range(10):
            packed = nn.Parameter(
                torch.zeros(2048, 772, dtype=torch.uint8, device="cuda"),
                requires_grad=False,
            )
            old = getattr(model, f"w{i}")
            old.data = torch.empty(0, device="cuda", dtype=old.dtype)
            setattr(model, f"w{i}", packed)
            del old

        gc.collect()
        torch.cuda.empty_cache()
        packed_mem = torch.cuda.memory_allocated()

        ratio = packed_mem / bf16_mem
        assert ratio < 0.5, (
            f"Packed should be <50% of BF16: {packed_mem/1e6:.0f}MB vs "
            f"{bf16_mem/1e6:.0f}MB = {ratio:.1%}"
        )

    def test_unified_memory_nvidia_smi_caveat(self):
        """On GB10 Unified Memory, nvidia-smi != cuda.memory_allocated."""
        if "GB10" not in torch.cuda.get_device_name(0):
            pytest.skip("Only relevant on GB10 Unified Memory")

        W = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        allocated = torch.cuda.memory_allocated()
        free, total = torch.cuda.mem_get_info()
        nvsmi_used = total - free

        # On Unified Memory, nvsmi_used >> allocated because it includes
        # process RSS (mmap'd files, Python heap, etc.)
        assert nvsmi_used > allocated, (
            "Expected nvidia-smi to show more than cuda.memory_allocated "
            "on Unified Memory"
        )
        del W


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
