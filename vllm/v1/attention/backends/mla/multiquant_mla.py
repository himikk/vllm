# SPDX-License-Identifier: Apache-2.0
"""MultiQuant MLA Backend — compressed KV-cache for MLA models.

Wraps Triton MLA: compress latent vectors on write, decompress on read.
Cache is stored as packed uint8, saving ~2.6× vs FP8.

Prefill: uses raw kv_c/k_pe directly (no cache read).
Decode: decompress from packed cache → BF16 → Triton MLA kernel.
"""

import math
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MultipleOf,
)

logger = init_logger(__name__)


class MultiQuantMLABackend(MLACommonBackend):
    """MLA backend with MultiQuant compressed KV-cache."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "tq3", "tq4", "rq3", "rq4", "rq2",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []  # Accept any (packed_size varies)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "MULTIQUANT_MLA"

    @staticmethod
    def get_impl_cls() -> type["MultiQuantMLAImpl"]:
        return MultiQuantMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class MultiQuantMLAImpl(MLACommonImpl[MLACommonMetadata]):
    """MLA attention with MultiQuant compressed KV-cache.

    Write path: compress [kv_c | k_pe] → packed uint8, write to cache.
    Prefill: use live kv_c/k_pe (no cache read), delegate to parent.
    Decode: decompress cache → BF16, call Triton MLA decode kernel.
    """

    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **mla_args,
    ) -> None:
        # Store original MQ dtype before passing to parent
        self._mq_kv_cache_dtype = kv_cache_dtype
        self._original_head_size = head_size

        # Get MQ config for compression params
        from vllm.multiquant.registry import get_kv_quantizer_config
        self._mq_config = get_kv_quantizer_config(
            kv_cache_dtype, head_size
        )

        # Parent sees "bfloat16" — the Triton kernel works on BF16 data
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            "bfloat16",  # Inner dtype for Triton kernel
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        self.supports_quant_query_input = False
        self._packed_size = self._mq_config.key_packed_size

    @torch.compiler.disable
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Compress [kv_c | k_pe] → packed uint8 → write to cache."""
        if kv_cache.numel() == 0:
            return

        # Concat latent + positional: (num_tokens, head_size)
        k_pe_flat = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
        latent = torch.cat([kv_c_normed, k_pe_flat], dim=-1)
        num_tokens = latent.shape[0]
        head_size = latent.shape[-1]
        device = latent.device

        # Get compression buffers from the attention layer
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        layer_name = None
        for name, layer in ctx.no_compile_layers.items():
            if hasattr(layer, '_tq_Pi'):
                # Find matching layer by checking buffer device
                if layer._tq_Pi.device == device:
                    layer_name = name
                    break

        if layer_name is None:
            # Fallback: write uncompressed (should not happen in production)
            logger.warning_once("MQ-MLA: no compression buffers found, "
                                "writing uncompressed")
            flat_slot = slot_mapping.flatten()
            page_size = kv_cache.shape[1]
            pages = flat_slot // page_size
            offsets = flat_slot % page_size
            # Truncate or pad to cache dim
            cache_dim = kv_cache.shape[-1]
            if head_size <= cache_dim:
                kv_cache[pages, offsets, :head_size] = latent.to(
                    kv_cache.dtype)
            return

        attn_layer = ctx.no_compile_layers[layer_name]
        Pi = attn_layer._tq_Pi.to(device).float()
        S = attn_layer._tq_S.to(device).float()
        centroids = attn_layer._tq_centroids.to(device).float()
        mse_bits = self._mq_config.mse_bits
        mse_bytes = math.ceil(head_size * mse_bits / 8)
        qjl_bytes = math.ceil(head_size / 8)
        mask = (1 << mse_bits) - 1
        coords_per_byte = 8 // mse_bits

        # Vectorized compression
        flat = latent.float()
        vec_norm = flat.norm(dim=-1, keepdim=True)
        x_hat = flat / (vec_norm + 1e-8)

        # Rotation (TQ: dense Pi, RQ: rotor sandwich)
        if self._mq_kv_cache_dtype.startswith("rq"):
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors,
                extract_vectors_from_multivectors,
                reverse, rotor_sandwich,
            )
            mv = embed_vectors_as_multivectors(x_hat)
            mv_rot = rotor_sandwich(Pi, mv)
            for gi in [1, 2, 3, 7]:
                comp = mv_rot[..., gi]
                diffs = comp.unsqueeze(-1) - centroids
                indices = diffs.abs().argmin(dim=-1)
                mv_rot[..., gi] = centroids[indices]
            rotor_rev = reverse(Pi)
            mv_recon = rotor_sandwich(rotor_rev, mv_rot)
            x_mse = extract_vectors_from_multivectors(mv_recon, head_size)
        else:
            rotated = x_hat @ Pi.T
            diffs = rotated.unsqueeze(-1) - centroids
            indices = diffs.abs().argmin(dim=-1)
            quantized = centroids[indices]
            x_mse = quantized @ Pi

        # QJL residual
        residual = x_hat - x_mse
        res_norm = residual.norm(dim=-1, keepdim=True)
        projected = residual @ S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0

        # Pack into uint8
        packed = torch.zeros(num_tokens, self._packed_size,
                             dtype=torch.uint8, device=device)

        # MSE indices from rotated space
        if self._mq_kv_cache_dtype.startswith("rq"):
            # For RQ, re-extract indices from the quantized multivector
            # (already computed above during the sandwich)
            rotated = x_hat  # placeholder — indices already applied in-place
            # Re-compute indices for packing
            mv2 = embed_vectors_as_multivectors(x_hat)
            mv2_rot = rotor_sandwich(Pi, mv2)
            all_coords = []
            for gi in [1, 2, 3, 7]:
                all_coords.append(mv2_rot[..., gi])
            flat_coords = torch.cat([c.reshape(num_tokens, -1)
                                     for c in all_coords], dim=-1)
            diffs = flat_coords.unsqueeze(-1) - centroids
            idx_flat = diffs.abs().argmin(dim=-1)
        else:
            rotated = x_hat @ Pi.T
            diffs = rotated.unsqueeze(-1) - centroids
            idx_flat = diffs.abs().argmin(dim=-1)

        # Pack MSE indices
        for b in range(mse_bytes):
            val = torch.zeros(num_tokens, dtype=torch.uint8, device=device)
            for k in range(coords_per_byte):
                j = b * coords_per_byte + k
                if j >= head_size:
                    break
                val = val | ((idx_flat[:, j].byte() & mask) << (k * mse_bits))
            packed[:, b] = val

        # Pack QJL signs
        for b in range(qjl_bytes):
            val = torch.zeros(num_tokens, dtype=torch.uint8, device=device)
            for k in range(8):
                j = b * 8 + k
                if j >= head_size:
                    break
                val = val | ((signs[:, j] > 0).byte() << k)
            packed[:, mse_bytes + b] = val

        # Pack norms
        no = mse_bytes + qjl_bytes
        vn_u8 = vec_norm.squeeze(-1).half().view(torch.uint8).reshape(
            num_tokens, 2)
        rn_u8 = res_norm.squeeze(-1).half().view(torch.uint8).reshape(
            num_tokens, 2)
        packed[:, no:no + 2] = vn_u8
        packed[:, no + 2:no + 4] = rn_u8

        # Write to cache
        flat_slot = slot_mapping.flatten()
        page_size = kv_cache.shape[1]
        pages = flat_slot // page_size
        offsets = flat_slot % page_size
        kv_cache[pages, offsets, :self._packed_size] = packed

    @torch.compiler.disable
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode: fused MQ attention directly on packed cache."""
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        # Try fused Triton kernel (no decompress step)
        try:
            from vllm.v1.attention.ops.triton_mq_decode import (
                mq_mla_decode_attention,
            )
            Pi = layer._tq_Pi.to(q.device).float()
            S = layer._tq_S.to(q.device).float()
            centroids = layer._tq_centroids.to(q.device).float()

            o, lse = mq_mla_decode_attention(
                q=q,
                kv_cache=kv_c_and_k_pe_cache,
                centroids=centroids,
                Pi=Pi,
                S=S,
                block_table=attn_metadata.decode.block_table,
                seq_lens=attn_metadata.decode.seq_lens,
                scale=self.scale,
                page_size=kv_c_and_k_pe_cache.shape[1],
                head_size=self._original_head_size,
                kv_lora_rank=self.kv_lora_rank,
                mse_bits=self._mq_config.mse_bits,
            )
            return o, lse
        except Exception as e:
            logger.debug("Fused MQ-MLA decode failed, using decompress: %s", e)

        device = q.device
        head_size = self._original_head_size
        mse_bits = self._mq_config.mse_bits
        mse_bytes = math.ceil(head_size * mse_bits / 8)
        qjl_bytes = math.ceil(head_size / 8)
        mask = (1 << mse_bits) - 1
        coords_per_byte = 8 // mse_bits

        # Get buffers
        Pi = layer._tq_Pi.to(device).float()
        S = layer._tq_S.to(device).float()
        centroids = layer._tq_centroids.to(device).float()

        # Decompress ONLY pages used by current batch (not all 1M+ cache)
        # kv_c_and_k_pe_cache shape: [num_pages, page_size, packed_size]
        num_pages, page_size, cache_dim = kv_c_and_k_pe_cache.shape
        block_table = attn_metadata.decode.block_table  # [B, max_blocks]
        seq_lens = attn_metadata.decode.seq_lens  # [B]

        # Gather only the pages we need
        used_pages = block_table.flatten().unique()
        used_pages = used_pages[used_pages >= 0]  # filter padding

        # Decompress used pages only
        packed = kv_c_and_k_pe_cache[used_pages, :, :self._packed_size]
        flat_packed = packed.reshape(-1, self._packed_size)
        N = flat_packed.shape[0]

        # Build decompressed cache with full shape (sparse — only used pages filled)
        decompressed = torch.zeros(
            num_pages, page_size, head_size,
            dtype=q.dtype, device=device,
        )

        if N > 0:
            # CUDA unpack kernel — same as Archer, shared code
            unpack_ok = False
            try:
                from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
                result = cuda_unpack(flat_packed, head_size, mse_bits)
                if result is not None:
                    idx_all, signs_all, vn, rn = result
                    idx_all = idx_all.long()
                    unpack_ok = True
            except Exception:
                pass

            if not unpack_ok:
                # Python fallback
                idx_all = torch.zeros(N, head_size, dtype=torch.long,
                                      device=device)
                for b in range(mse_bytes):
                    bv = flat_packed[:, b].long()
                    for k in range(coords_per_byte):
                        j = b * coords_per_byte + k
                        if j >= head_size:
                            break
                        idx_all[:, j] = (bv >> (k * mse_bits)) & mask

                signs_all = torch.zeros(N, head_size, dtype=torch.float32,
                                        device=device)
                for b in range(qjl_bytes):
                    bv = flat_packed[:, mse_bytes + b].long()
                    for k in range(8):
                        j = b * 8 + k
                        if j >= head_size:
                            break
                        signs_all[:, j] = torch.where(
                            ((bv >> k) & 1).bool(),
                            torch.ones(N, device=device),
                            -torch.ones(N, device=device),
                        )

                no = mse_bytes + qjl_bytes
                vn = flat_packed[:, no:no + 2].contiguous().view(
                    torch.float16).float().squeeze(-1)
                rn = flat_packed[:, no + 2:no + 4].contiguous().view(
                    torch.float16).float().squeeze(-1)

            # Reconstruct: inverse rotation + QJL correction
            c_vals = centroids[idx_all]

            if self._mq_kv_cache_dtype.startswith("rq"):
                from vllm.multiquant.rotorquant.clifford import (
                    embed_vectors_as_multivectors,
                    extract_vectors_from_multivectors,
                    reverse, rotor_sandwich,
                )
                # Build multivector from quantized coords
                # (simplified: use vector grades only for now)
                mv_q = embed_vectors_as_multivectors(c_vals)
                rotor_rev = reverse(Pi)
                mv_recon = rotor_sandwich(rotor_rev, mv_q)
                x_mse = extract_vectors_from_multivectors(mv_recon, head_size)
            else:
                x_mse = c_vals @ Pi

            correction = math.sqrt(math.pi / 2) / head_size
            x_qjl = correction * rn.unsqueeze(-1) * (signs_all @ S)
            x_recon = vn.unsqueeze(-1) * (x_mse + x_qjl)

            # Write decompressed data back to the correct pages
            x_recon_pages = x_recon.reshape(
                len(used_pages), page_size, head_size
            ).to(q.dtype)
            decompressed[used_pages] = x_recon_pages

        # Now delegate to Triton MLA decode with decompressed BF16 cache
        from vllm.model_executor.layers.batch_invariant import (
            vllm_is_batch_invariant,
        )
        from vllm.v1.attention.ops.triton_decode_attention import (
            decode_attention_fwd,
        )

        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=device)
        num_kv_splits = 1 if vllm_is_batch_invariant() else 4
        attn_logits = torch.empty(
            (B, q_num_heads, num_kv_splits, self.kv_lora_rank + 1),
            dtype=torch.float32, device=device,
        )

        decompressed_4d = decompressed.unsqueeze(2)
        kv_c_cache = decompressed_4d[..., :self.kv_lora_rank]

        decode_attention_fwd(
            q,
            decompressed_4d,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            page_size,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
        )

        return o, lse
