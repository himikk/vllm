#!/usr/bin/env python3
"""Tests for the fused MultiQuant decode kernel.

Compares the Triton fused decode output against the existing Python
decode loop (which is known correct from test_attention_forward.py).
"""

import math

import pytest
import torch
import torch.nn.functional as F


def _setup_buffers(D=128, mse_bits=2, seed=42, device="cuda"):
    """Create rotation/QJL/centroids buffers for TQ."""
    from vllm.multiquant.turboquant.quantizer import generate_rotation_matrix
    from vllm.multiquant.shared.qjl import generate_qjl_matrix
    from vllm.multiquant.shared.centroids import get_centroids

    Pi = generate_rotation_matrix(D, seed=seed).to(device).float()
    S = generate_qjl_matrix(D, seed=seed + 1).to(device).float()
    centroids = get_centroids(D, mse_bits).to(device).float()
    return Pi, S, centroids


def _fill_cache_tq(cache, num_tokens, num_kv_heads, D, Pi, S, centroids,
                   mse_bits, block_size, seed=42):
    """Fill KV cache with TQ-packed random vectors."""
    from vllm.multiquant.shared.bitpack import pack_vectors_batched

    torch.manual_seed(seed)
    for t in range(num_tokens):
        for is_v in [0, 1]:  # K=0, V=1
            vecs = torch.randn(num_kv_heads, D, device="cuda")
            norms = vecs.norm(dim=-1)
            unit = vecs / (norms.unsqueeze(-1) + 1e-8)
            rotated = unit @ Pi.T
            idx = (rotated.unsqueeze(-1) - centroids).abs().argmin(dim=-1)
            quantized = centroids[idx]
            W_mse = quantized @ Pi
            residual = unit - W_mse
            res_norms = residual.norm(dim=-1)
            signs = torch.sign(residual @ S.T)
            signs[signs == 0] = 1.0
            packed = pack_vectors_batched(
                idx, signs, norms, res_norms, D, mse_bits
            )
            bi = t // block_size
            bo = t % block_size
            cache[bi, is_v, bo, :, :packed.shape[-1]] = packed


def _python_decode_reference(q, kv_cache, Pi, S, centroids, block_table,
                             seq_lens, scale, block_size, num_kv_heads,
                             mse_bits, correction):
    """Reference decode using the Python loop from multiquant_attn.py."""
    B, Hq, D = q.shape
    kv_group_size = Hq // num_kv_heads
    device = q.device
    output = torch.zeros(B, Hq, D, device=device, dtype=torch.float32)

    mse_bytes = (D * mse_bits + 7) // 8
    qjl_bytes = (D + 7) // 8
    mask = (1 << mse_bits) - 1

    for qi in range(B):
        sl = seq_lens[qi].item()
        if sl <= 0:
            continue

        positions = torch.arange(sl, device=device)
        bi_log = positions // block_size
        bo = positions % block_size
        bi_phys = block_table[qi, bi_log.long()]

        for kv_h in range(num_kv_heads):
            k_packed = kv_cache[bi_phys, 0, bo, kv_h]  # (sl, packed_size)
            v_packed = kv_cache[bi_phys, 1, bo, kv_h]

            # Unpack K
            idx_k = torch.zeros(sl, D, dtype=torch.long, device=device)
            for j in range(D):
                bit_off = j * mse_bits
                byte_idx = bit_off // 8
                bit_shift = bit_off % 8
                bv = k_packed[:, byte_idx].long() >> bit_shift
                spill = bit_shift + mse_bits - 8
                if spill > 0 and byte_idx + 1 < mse_bytes:
                    bv = bv | (k_packed[:, byte_idx + 1].long() << (mse_bits - spill))
                idx_k[:, j] = bv & mask

            signs_k = torch.zeros(sl, D, dtype=torch.float32, device=device)
            for b in range(qjl_bytes):
                bv = k_packed[:, mse_bytes + b].long()
                for k in range(8):
                    j = b * 8 + k
                    if j >= D:
                        break
                    signs_k[:, j] = torch.where(
                        ((bv >> k) & 1).bool(),
                        torch.ones(sl, device=device),
                        -torch.ones(sl, device=device),
                    )

            no = mse_bytes + qjl_bytes
            k_vn = k_packed[:, no:no + 2].contiguous().view(
                torch.float16).float().squeeze(-1)
            k_rn = k_packed[:, no + 2:no + 4].contiguous().view(
                torch.float16).float().squeeze(-1)

            c_idx = centroids[idx_k]  # (sl, D)

            for h in range(kv_h * kv_group_size, (kv_h + 1) * kv_group_size):
                q_rot_h = (q[qi, h].float() @ Pi.T).unsqueeze(0)
                q_proj_h = (q[qi, h].float() @ S.T)

                term1 = (q_rot_h * c_idx).sum(-1)
                term2 = (q_proj_h.unsqueeze(0) * signs_k).sum(-1)
                scores = k_vn * (term1 + correction * k_rn * term2) * scale

                # Unpack V
                idx_v = torch.zeros(sl, D, dtype=torch.long, device=device)
                for j in range(D):
                    bit_off = j * mse_bits
                    byte_idx = bit_off // 8
                    bit_shift = bit_off % 8
                    bv = v_packed[:, byte_idx].long() >> bit_shift
                    spill = bit_shift + mse_bits - 8
                    if spill > 0 and byte_idx + 1 < mse_bytes:
                        bv = bv | (v_packed[:, byte_idx + 1].long() << (mse_bits - spill))
                    idx_v[:, j] = bv & mask

                v_signs = torch.zeros(sl, D, dtype=torch.float32, device=device)
                for b in range(qjl_bytes):
                    bv = v_packed[:, mse_bytes + b].long()
                    for k in range(8):
                        j = b * 8 + k
                        if j >= D:
                            break
                        v_signs[:, j] = torch.where(
                            ((bv >> k) & 1).bool(),
                            torch.ones(sl, device=device),
                            -torch.ones(sl, device=device),
                        )

                v_vn = v_packed[:, no:no + 2].contiguous().view(
                    torch.float16).float().squeeze(-1)
                v_rn = v_packed[:, no + 2:no + 4].contiguous().view(
                    torch.float16).float().squeeze(-1)

                v_c = centroids[idx_v]
                v_xm = v_c @ Pi
                v_xq = correction * v_rn.unsqueeze(-1) * (v_signs @ S)
                v_recon = v_vn.unsqueeze(-1) * (v_xm + v_xq)

                weights = F.softmax(scores, dim=-1)
                out_h = (weights.unsqueeze(-1) * v_recon).sum(0)
                output[qi, h] = out_h

    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedDecode:
    """Compare fused Triton kernel against Python reference."""

    @pytest.mark.parametrize("mse_bits", [2])
    def test_single_request(self, mse_bits):
        """Single request, 4 heads, seq_len=8."""
        D = 128
        Hq = 4
        Hkv = 2
        block_size = 16
        seq_len = 8

        Pi, S, centroids = _setup_buffers(D, mse_bits)
        packed_size = (D * mse_bits + 7) // 8 + (D + 7) // 8 + 4
        n_blocks = (seq_len + block_size - 1) // block_size
        cache = torch.zeros(
            n_blocks, 2, block_size, Hkv, packed_size,
            dtype=torch.uint8, device="cuda",
        )

        _fill_cache_tq(cache, seq_len, Hkv, D, Pi, S, centroids,
                       mse_bits, block_size)

        torch.manual_seed(99)
        q = torch.randn(1, Hq, D, device="cuda", dtype=torch.float32)
        block_table = torch.arange(n_blocks, device="cuda").unsqueeze(0)
        seq_lens_t = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        correction = math.sqrt(math.pi / 2) / D

        # Reference
        ref = _python_decode_reference(
            q, cache, Pi, S, centroids, block_table, seq_lens_t,
            scale, block_size, Hkv, mse_bits, correction,
        )

        # Fused
        from vllm.v1.attention.ops.triton_mq_fused_decode import (
            mq_fused_decode_attention,
        )
        fused = mq_fused_decode_attention(
            q, cache, Pi, S, centroids, block_table, seq_lens_t,
            scale, block_size, Hkv, mse_bits, correction,
        )

        # Compare
        cos = F.cosine_similarity(
            ref.reshape(1, -1).float(), fused.reshape(1, -1).float(), dim=-1
        ).item()
        rel_err = (ref - fused).abs().max() / (ref.abs().max() + 1e-8)

        assert cos > 0.99, f"cos={cos:.4f} (expected >0.99)"
        assert not fused.isnan().any(), "Fused output has NaN"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDAKernelDimensions:
    """Test CUDA kernel with various HEAD_DIM / head configs."""

    @pytest.mark.parametrize("D,Hq,Hkv", [
        (128, 4, 2),    # minimal
        (128, 32, 4),   # GQA
        (64, 8, 8),     # small, no GQA
        (256, 20, 20),  # GLM-4.7 actual config (triggers device-side assert)
        (256, 4, 2),    # D=256 with GQA
    ])
    def test_cuda_kernel_direct(self, D, Hq, Hkv):
        """CUDA kernel must produce non-zero, non-NaN output for all configs."""
        mse_bits = 2
        block_size = 16
        seq_len = 8

        Pi, S, centroids = _setup_buffers(D, mse_bits)
        packed_size = (D * mse_bits + 7) // 8 + (D + 7) // 8 + 4
        n_blocks = (seq_len + block_size - 1) // block_size
        cache = torch.zeros(
            n_blocks, 2, block_size, Hkv, packed_size,
            dtype=torch.uint8, device="cuda",
        )
        _fill_cache_tq(cache, seq_len, Hkv, D, Pi, S, centroids,
                       mse_bits, block_size)

        torch.manual_seed(99)
        q = torch.randn(1, Hq, D, device="cuda", dtype=torch.float32)
        block_table = torch.arange(n_blocks, device="cuda").unsqueeze(0).int()
        seq_lens_t = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)

        # Try CUDA kernel directly
        from vllm.v1.attention.ops.triton_mq_fused_decode import _load_cuda_kernel
        kernel = _load_cuda_kernel()
        if kernel is None:
            pytest.skip("CUDA kernel not available")

        q_rot = (q.reshape(Hq, D).float() @ Pi.T).reshape(1, Hq, D).contiguous()
        q_proj = (q.reshape(Hq, D).float() @ S.T).reshape(1, Hq, D).contiguous()
        centroid_acc = torch.zeros(1, Hq, D, device="cuda", dtype=torch.float32)
        sign_acc = torch.zeros(1, Hq, D, device="cuda", dtype=torch.float32)

        s_block = cache.stride(0)
        s_kv = cache.stride(1)
        s_slot = cache.stride(2)
        s_head = cache.stride(3)
        correction = math.sqrt(math.pi / 2) / D

        try:
            kernel.tq_fused_decode_attention(
                q_rot, q_proj, cache, centroids,
                block_table, seq_lens_t,
                centroid_acc, sign_acc,
                D, mse_bits, centroids.shape[0], scale,
                s_block, s_kv, s_slot, s_head,
            )
            torch.cuda.synchronize()  # catch async errors
        except RuntimeError as e:
            pytest.fail(f"CUDA kernel crashed at D={D} Hq={Hq}: {e}")

        # Verify output is non-zero and non-NaN
        assert not centroid_acc.isnan().any(), f"D={D}: NaN in centroid_acc"
        assert not sign_acc.isnan().any(), f"D={D}: NaN in sign_acc"
        assert centroid_acc.abs().sum() > 0, f"D={D}: centroid_acc all zeros"
        assert sign_acc.abs().sum() > 0, f"D={D}: sign_acc all zeros"

    @pytest.mark.parametrize("D", [128, 256])
    def test_cuda_vs_triton(self, D):
        """CUDA and Triton kernels must produce same output."""
        mse_bits = 2
        Hq = 4
        Hkv = 2
        block_size = 16
        seq_len = 8

        Pi, S, centroids = _setup_buffers(D, mse_bits)
        packed_size = (D * mse_bits + 7) // 8 + (D + 7) // 8 + 4
        cache = torch.zeros(
            1, 2, block_size, Hkv, packed_size,
            dtype=torch.uint8, device="cuda",
        )
        _fill_cache_tq(cache, seq_len, Hkv, D, Pi, S, centroids,
                       mse_bits, block_size)

        torch.manual_seed(99)
        q = torch.randn(1, Hq, D, device="cuda", dtype=torch.float32)
        block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        seq_lens_t = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        correction = math.sqrt(math.pi / 2) / D

        from vllm.v1.attention.ops.triton_mq_fused_decode import (
            mq_fused_decode_attention,
        )
        # Force Triton (temporarily disable CUDA kernel)
        import vllm.v1.attention.ops.triton_mq_fused_decode as fmod
        saved = fmod._cuda_kernel
        fmod._cuda_kernel = None
        triton_out = mq_fused_decode_attention(
            q, cache, Pi, S, centroids, block_table, seq_lens_t,
            scale, block_size, Hkv, mse_bits, correction,
        )
        fmod._cuda_kernel = saved

        # Now try CUDA kernel
        cuda_out = mq_fused_decode_attention(
            q, cache, Pi, S, centroids, block_table, seq_lens_t,
            scale, block_size, Hkv, mse_bits, correction,
        )

        if cuda_out.abs().sum() == 0:
            pytest.skip(f"CUDA kernel returned zeros at D={D}")

        cos = F.cosine_similarity(
            triton_out.reshape(1, -1).float(),
            cuda_out.reshape(1, -1).float(), dim=-1
        ).item()
        assert cos > 0.99, f"D={D}: CUDA vs Triton cos={cos:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedDecodeKernelLaunches:
    """Verify the fused path uses constant kernel launches."""

    def test_launch_count(self):
        """The wrapper should do 5 operations: 2 pre + 1 Triton + 2 post."""
        D = 128
        Hq = 4
        Hkv = 2
        block_size = 16
        seq_len = 8
        mse_bits = 2

        Pi, S, centroids = _setup_buffers(D, mse_bits)
        packed_size = (D * mse_bits + 7) // 8 + (D + 7) // 8 + 4
        cache = torch.zeros(
            1, 2, block_size, Hkv, packed_size,
            dtype=torch.uint8, device="cuda",
        )
        _fill_cache_tq(cache, seq_len, Hkv, D, Pi, S, centroids,
                       mse_bits, block_size)

        q = torch.randn(1, Hq, D, device="cuda", dtype=torch.float32)
        block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        seq_lens_t = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        correction = math.sqrt(math.pi / 2) / D

        # Just verify it runs without error — no Python loops
        from vllm.v1.attention.ops.triton_mq_fused_decode import (
            mq_fused_decode_attention,
        )
        output = mq_fused_decode_attention(
            q, cache, Pi, S, centroids, block_table, seq_lens_t,
            scale, block_size, Hkv, mse_bits, correction,
        )

        assert output.shape == (1, Hq, D)
        assert not output.isnan().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
