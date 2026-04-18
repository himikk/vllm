# SPDX-License-Identifier: Apache-2.0
"""Archer online linear method — BF16/FP8 → MultiQuant packed weights.

Per-layer quantization: each layer is quantized as soon as all its shards
arrive, then BF16 is replaced by packed uint8. Peak memory = 1 layer BF16.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter

if TYPE_CHECKING:
    from vllm.multiquant.weight_quant.config import ArcherConfig

logger = init_logger(__name__)


# ── Custom Op for torch.compile compatibility ─────────────────────
# Dynamo can't trace _decompress (Python control flow, CUDA kernel
# try/except, lazy imports). Register as custom op with fake_impl.

def _archer_apply_impl(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    rotation: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    in_features: int,
    mse_bits: int,
    is_rq: bool,
) -> torch.Tensor:
    """Real impl: decompress packed uint8 → F.linear.

    Graph-safe: uses CUDA kernels (no Python loops, no host→device).
    Falls back to Python only when no CUDA kernel available.
    """
    out_features = packed_weight.shape[0]
    device = packed_weight.device

    # Try CUDA full decompress (TQ only, D≤256) — graph-safe
    if in_features <= 256 and not is_rq:
        try:
            from vllm.multiquant.weight_quant.archer_ops import cuda_decompress
            result = cuda_decompress(
                packed_weight, rotation, S, centroids, in_features,
                out_dtype=torch.bfloat16,
            )
            if result is not None:
                return F.linear(x, result.to(x.dtype))
        except Exception:
            pass

    # Try CUDA unpack (TQ only — RQ uses Python fallback for safety)
    result = None
    if not is_rq:
        try:
            from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
            result = cuda_unpack(packed_weight, in_features, mse_bits)
            if result is not None:
                idx, signs, row_norms, res_norms = result
                idx = idx.long()
        except Exception:
            result = None

    if result is None:
        # Python fallback — NOT graph-safe (Python loops).
        # This path should not be reached if CUDA kernels are available.
        mse_bytes = math.ceil(in_features * mse_bits / 8)
        qjl_bytes = math.ceil(in_features / 8)
        mask = (1 << mse_bits) - 1
        idx = torch.empty(out_features, in_features,
                          dtype=torch.long, device=device)
        for j in range(in_features):
            bo = j * mse_bits
            bi = bo // 8
            bs = bo % 8
            bv = packed_weight[:, bi].long() >> bs
            spill = bs + mse_bits - 8
            if spill > 0 and bi + 1 < mse_bytes:
                bv = bv | (packed_weight[:, bi + 1].long() << (mse_bits - spill))
            idx[:, j] = bv & mask
        # Vectorized sign unpack (graph-safer, no torch.ones/torch.where)
        sign_bytes = packed_weight[:, mse_bytes:mse_bytes + qjl_bytes]
        signs = torch.empty(out_features, in_features,
                            dtype=torch.float32, device=device)
        for j in range(in_features):
            b = j // 8
            k = j % 8
            bit = (sign_bytes[:, b].long() >> k) & 1
            signs[:, j] = bit.float() * 2.0 - 1.0  # 0→-1, 1→+1
        no = mse_bytes + qjl_bytes
        row_norms = packed_weight[:, no:no + 2].contiguous().view(
            torch.float16).float().squeeze(-1)
        res_norms = packed_weight[:, no + 2:no + 4].contiguous().view(
            torch.float16).float().squeeze(-1)

    c_vals = centroids[idx]
    if is_rq:
        from vllm.multiquant.rotorquant.clifford import (
            embed_vectors_as_multivectors,
            extract_vectors_from_multivectors,
            reverse, rotor_sandwich,
        )
        mv_q = embed_vectors_as_multivectors(c_vals)
        rotor_rev = reverse(rotation)
        mv_recon = rotor_sandwich(rotor_rev, mv_q)
        W_mse = extract_vectors_from_multivectors(mv_recon, in_features)
    else:
        W_mse = c_vals @ rotation

    correction = math.sqrt(math.pi / 2) / in_features
    W_qjl = correction * res_norms.unsqueeze(-1) * (signs @ S)
    W = (row_norms.unsqueeze(-1) * (W_mse + W_qjl)).to(x.dtype).contiguous()
    return F.linear(x, W)


def _archer_apply_fake(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    rotation: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    in_features: int,
    mse_bits: int,
    is_rq: bool,
) -> torch.Tensor:
    """Fake impl for torch.compile graph tracing."""
    return torch.empty(x.shape[0], packed_weight.shape[0],
                       dtype=x.dtype, device=x.device)


try:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="archer_apply",
        op_func=_archer_apply_impl,
        fake_impl=_archer_apply_fake,
    )
    _archer_op = torch.ops.vllm.archer_apply
    logger.info("Archer custom op registered (torch.compile safe)")
except Exception as e:
    logger.warning("Archer custom op registration failed: %s — "
                   "torch.compile with --weights may not work", e)
    _archer_op = _archer_apply_impl


class ArcherOnlineLinearMethod(QuantizeMethodBase):
    """BF16/FP8 → MultiQuant packed, quantized per-layer on arrival.

    uses_meta_device=True: weight starts on meta (0 bytes).
    patched_weight_loader: materializes on first shard, quantizes when done.
    Peak memory = 1 BF16 layer + accumulated packed layers.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: ArcherConfig):
        self.quant_config = quant_config
        self.bits = quant_config.bits
        self.method = quant_config.method

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        out_size = sum(output_partition_sizes)
        orig_loader = extra_weight_attrs.get("weight_loader")
        archer_method = self  # closure reference

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            # First shard: materialize from meta → real device
            if not hasattr(layer, "_archer_loaded"):
                layer._archer_loaded = 0
                real = ModelWeightParameter(
                    data=torch.empty(
                        out_size, input_size_per_partition,
                        dtype=params_dtype,
                        device=layer._archer_device,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                layer.register_parameter("weight", real)
                del layer._archer_device

            # Load this shard via original loader
            param = layer.weight
            if orig_loader is not None:
                orig_loader(param, loaded_weight, *args, **kwargs)
            else:
                param.data.copy_(loaded_weight)
            # Track by loaded_weight size (not copy_ ops — those vary)
            layer._archer_loaded += loaded_weight.numel()

            # All shards for this layer arrived? → quantize + free BF16
            if layer._archer_loaded >= param.data.numel():
                archer_method._quantize_layer(layer)
                layer._already_called_process_weights_after_loading = True

        # Create on meta device — zero memory
        weight = ModelWeightParameter(
            data=torch.empty(
                out_size, input_size_per_partition,
                device="meta", dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        layer._archer_device = torch.get_default_device()
        layer.register_parameter("weight", weight)

    def _quantize_layer(self, layer: nn.Module) -> None:
        """Compress one layer: BF16 → packed uint8."""
        W = layer.weight.data.float()
        out_features, in_features = W.shape
        device = W.device

        cache_dtype = f"{self.method}{self.bits}"
        from vllm.multiquant.registry import get_kv_quantizer_config
        mq_config = get_kv_quantizer_config(cache_dtype, in_features)
        mse_bits = mq_config.mse_bits
        packed_size = mq_config.key_packed_size

        from vllm.multiquant.shared.centroids import get_centroids
        centroids = get_centroids(in_features, mse_bits).to(device)

        seed = self.quant_config.seed
        if self.method == "rq":
            from vllm.multiquant.rotorquant.quantizer import generate_rotors
            rotation = generate_rotors(in_features, seed=seed).to(device)
        else:
            from vllm.multiquant.turboquant.quantizer import (
                generate_rotation_matrix,
            )
            rotation = generate_rotation_matrix(
                in_features, seed=seed
            ).to(device)

        from vllm.multiquant.shared.qjl import generate_qjl_matrix
        S = generate_qjl_matrix(in_features, seed=seed + 1).to(device)

        # Per-row quantization (PyTorch — runs once at load time)
        row_norms = W.norm(dim=-1)
        W_unit = W / (row_norms.unsqueeze(-1) + 1e-8)

        # Quantize: rotate → nearest-centroid → inverse rotate → residual + QJL signs
        # NOTE: Must compute signs from pre-QJL residual (W_unit - W_mse), NOT from
        # a round-trip that includes QJL correction. Using tq_round_trip_keys here
        # would corrupt signs (16% match → cos drops from 0.92 to 0.66).
        if self.method == "rq":
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors,
                extract_vectors_from_multivectors,
                reverse, rotor_sandwich,
            )
            # Forward rotation via Clifford rotor sandwich
            mv = embed_vectors_as_multivectors(W_unit)
            mv_rot = rotor_sandwich(rotation, mv)
            # Extract as flat vector (group-by-group order matches embed)
            rotated = extract_vectors_from_multivectors(mv_rot, in_features)
            # Per-coordinate quantization (same as TQ)
            diffs = rotated.unsqueeze(-1) - centroids
            mse_indices = diffs.abs().argmin(dim=-1)
            # Reconstruct W_mse via inverse rotation
            quantized = centroids[mse_indices]
            mv_q = embed_vectors_as_multivectors(quantized)
            rotor_rev = reverse(rotation)
            mv_recon = rotor_sandwich(rotor_rev, mv_q)
            W_mse = extract_vectors_from_multivectors(mv_recon, in_features)
        else:
            rotated = W_unit @ rotation.T
            diffs = rotated.unsqueeze(-1) - centroids
            mse_indices = diffs.abs().argmin(dim=-1)
            quantized = centroids[mse_indices]
            W_mse = quantized @ rotation

        residual = W_unit - W_mse
        res_norms = residual.norm(dim=-1)
        projected = residual @ S.T
        signs = torch.sign(projected)
        signs[signs == 0] = 1.0

        from vllm.multiquant.shared.bitpack import pack_vectors_batched
        packed_W = pack_vectors_batched(
            mse_indices, signs, row_norms, res_norms,
            in_features, mse_bits,
        )

        # Zero the BF16 data first — releases actual GPU memory even if
        # params_dict in model.load_weights() still holds a reference.
        # Then resize to 0 to minimize the dangling allocation.
        old_weight = layer.weight
        old_weight.data = torch.empty(0, device=device, dtype=old_weight.dtype)

        # Set new packed weight
        layer.weight = nn.Parameter(packed_W, requires_grad=False)

        # Force GC to release the old allocation
        del old_weight
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Decompression metadata
        layer.register_buffer("_archer_rotation", rotation, persistent=False)
        layer.register_buffer("_archer_S", S, persistent=False)
        layer.register_buffer("_archer_centroids", centroids, persistent=False)
        layer._archer_in_features = in_features
        layer._archer_mse_bits = mse_bits
        layer._archer_method = self.method
        layer._archer_packed = True

        logger.info(
            "Archer %s%d: (%dx%d) → (%dx%d) uint8, %.1f%% of BF16",
            self.method.upper(), mse_bits,
            out_features, in_features,
            out_features, packed_size,
            100.0 * packed_size / (in_features * 2),
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Fallback — normally patched_weight_loader already quantized."""
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if getattr(layer, "_archer_packed", False):
            return
        if layer.weight.device != torch.device("meta"):
            self._quantize_layer(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_archer_packed", False):
            out = _archer_op(
                x, layer.weight.data,
                layer._archer_rotation, layer._archer_S,
                layer._archer_centroids,
                layer._archer_in_features, layer._archer_mse_bits,
                layer._archer_method == "rq",
            )
            if bias is not None:
                out = out + bias
            return out
        return F.linear(x, layer.weight, bias)

    def _decompress(self, layer: nn.Module) -> torch.Tensor:
        packed = layer.weight.data
        out_features = packed.shape[0]
        in_features = layer._archer_in_features
        mse_bits = layer._archer_mse_bits
        device = packed.device

        rotation = layer._archer_rotation
        S = layer._archer_S
        centroids = layer._archer_centroids

        # CUDA full decompress kernel for TQ only (uses matrix rotation)
        # RQ uses Clifford rotors — must go through unpack + Python rotation
        if in_features <= 256 and layer._archer_method != "rq":
            try:
                from vllm.multiquant.weight_quant.archer_ops import cuda_decompress
                result = cuda_decompress(
                    packed, rotation, S, centroids, in_features,
                    out_dtype=torch.bfloat16,
                )
                if result is not None:
                    return result
            except Exception as e:
                logger.debug("Archer cuda_decompress fallback: %s", e)

        # Try CUDA unpack kernel (eliminates Python bit-loops)
        # Only for dimensions that have been tested (128-2048)
        centroids = layer._archer_centroids
        # CUDA unpack kernel — verified correct in isolated + repeated tests.
        # Device-side assert in serving is from MLA/FLA, not unpack.
        result = None
        try:
            from vllm.multiquant.weight_quant.archer_ops import cuda_unpack
            result = cuda_unpack(packed, in_features, mse_bits)
            if result is not None:
                idx, signs, row_norms, res_norms = result
                idx = idx.long()
        except Exception as e:
            logger.debug("Archer cuda_unpack fallback: %s", e)
            result = None

        if result is None:
            # Python fallback
            mse_bytes = math.ceil(in_features * mse_bits / 8)
            qjl_bytes = math.ceil(in_features / 8)
            mask = (1 << mse_bits) - 1

            idx = torch.zeros(out_features, in_features,
                              dtype=torch.long, device=device)
            for j in range(in_features):
                bit_offset = j * mse_bits
                byte_idx = bit_offset // 8
                bit_shift = bit_offset % 8
                bv = packed[:, byte_idx].long() >> bit_shift
                spill = bit_shift + mse_bits - 8
                if spill > 0 and byte_idx + 1 < mse_bytes:
                    bv = bv | (packed[:, byte_idx + 1].long() << (mse_bits - spill))
                idx[:, j] = bv & mask

            signs = torch.zeros(out_features, in_features,
                                dtype=torch.float32, device=device)
            for b in range(qjl_bytes):
                bv = packed[:, mse_bytes + b].long()
                for k in range(8):
                    j = b * 8 + k
                    if j >= in_features:
                        break
                    signs[:, j] = torch.where(
                        ((bv >> k) & 1).bool(),
                        torch.ones(out_features, device=device),
                        -torch.ones(out_features, device=device),
                    )

            no = mse_bytes + qjl_bytes
            row_norms = packed[:, no:no + 2].contiguous().view(
                torch.float16).float().squeeze(-1)
            res_norms = packed[:, no + 2:no + 4].contiguous().view(
                torch.float16).float().squeeze(-1)

        # Centroid lookup (GPU gather) + inverse rotation (cuBLAS GEMM)
        c_vals = centroids[idx]  # (out, in) — fast GPU gather
        if layer._archer_method == "rq":
            from vllm.multiquant.rotorquant.clifford import (
                embed_vectors_as_multivectors,
                extract_vectors_from_multivectors,
                reverse, rotor_sandwich,
            )
            mv_q = embed_vectors_as_multivectors(c_vals)
            rotor_rev = reverse(rotation)
            mv_recon = rotor_sandwich(rotor_rev, mv_q)
            W_mse = extract_vectors_from_multivectors(mv_recon, in_features)
        else:
            W_mse = c_vals @ rotation  # cuBLAS GEMM — fast!

        # QJL correction (cuBLAS GEMM)
        correction = math.sqrt(math.pi / 2) / in_features
        W_qjl = correction * res_norms.unsqueeze(-1) * (signs @ S)
        return row_norms.unsqueeze(-1) * (W_mse + W_qjl)
