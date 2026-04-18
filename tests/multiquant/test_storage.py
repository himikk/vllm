#!/usr/bin/env python3
"""Storage format abstraction tests."""

import pytest


class TestStorageSpec:
    def test_int4_symmetric(self):
        from vllm.multiquant.storage import INT4_SYMMETRIC, ComputeKernel
        assert INT4_SYMMETRIC.bits == 4
        assert INT4_SYMMETRIC.symmetric
        assert ComputeKernel.MARLIN in INT4_SYMMETRIC.compatible_kernels

    def test_mq3_packed(self):
        from vllm.multiquant.storage import MQ3_PACKED, ComputeKernel
        assert MQ3_PACKED.bits == 3
        assert ComputeKernel.ARCHER in MQ3_PACKED.compatible_kernels

    def test_best_kernel(self):
        from vllm.multiquant.storage import INT4_SYMMETRIC, ComputeKernel
        assert INT4_SYMMETRIC.best_kernel() == ComputeKernel.MARLIN

    def test_best_kernel_fallback(self):
        from vllm.multiquant.storage import StorageSpec, StorageFormat, ComputeKernel
        empty = StorageSpec(format=StorageFormat.BF16, bits=16,
                            compatible_kernels=[])
        assert empty.best_kernel() == ComputeKernel.TORCH_LINEAR


class TestRegistry:
    def test_rtn_produces_int4(self):
        from vllm.multiquant.storage import (
            get_storage_spec, StorageFormat,
        )
        spec = get_storage_spec("autoround_rtn")
        assert spec.format == StorageFormat.INT4_PACKED

    def test_archer_produces_mq(self):
        from vllm.multiquant.storage import (
            get_storage_spec, StorageFormat,
        )
        spec = get_storage_spec("multiquant_rq3")
        assert spec.format == StorageFormat.MQ_PACKED

    def test_gptq_and_rtn_same_format(self):
        """GPTQ and RTN produce the same INT4 format → same kernel."""
        from vllm.multiquant.storage import get_storage_spec
        gptq = get_storage_spec("gptq")
        rtn = get_storage_spec("autoround_rtn")
        assert gptq.format == rtn.format

    def test_compatible_kernels(self):
        from vllm.multiquant.storage import (
            get_compatible_kernels, StorageFormat, ComputeKernel,
        )
        kernels = get_compatible_kernels(StorageFormat.INT4_PACKED)
        assert ComputeKernel.MARLIN in kernels
        assert ComputeKernel.TORCH_LINEAR in kernels

    def test_unknown_raises(self):
        from vllm.multiquant.storage import get_storage_spec
        with pytest.raises(ValueError):
            get_storage_spec("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
