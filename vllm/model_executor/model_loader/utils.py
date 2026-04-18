# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""

import inspect
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.model_loader.reload import (
    record_metadata_for_reloading,
    set_torchao_reload_attrs,
)
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.tracing import instrument
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


@instrument(span_name="Initialize model")
def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    if model_config is None:
        model_config = vllm_config.model_config
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            record_metadata_for_reloading(model)
            return model

    msg = (
        "vLLM model class should accept `vllm_config` and `prefix` as "
        "input arguments. Possibly you have an old-style model class"
        " registered from out of tree and it is used for new vLLM version. "
        "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
        "for the design and update the model class accordingly."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
    # try to be compatible with old-style model class
    kwargs: dict[str, Any] = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        model = model_class(**kwargs)
        record_metadata_for_reloading(model)

    return model


def _mem_snapshot(tag: str) -> str:
    """Layered memory breakdown. On GB10 Unified Memory, CUDA allocs and CPU
    tensors share the same physical DRAM, so we track every layer:
    - VmRSS   : resident set (process-owned CPU pages)
    - VmHWM   : peak RSS ever seen by this process
    - VmData  : data+heap segment (Python+libc malloc)
    - PssAnon : proportional share of anonymous mem (smaps_rollup)
    - CUDA alloc/reserved : PyTorch allocator view
    - MemAvail: kernel-reported free+reclaimable
    Differences between these tell us:
      * RSS << MemUsed → CUDA/UVM pages not in RSS (unified mem cdev)
      * VmData - RSS   → mmap/cache pages swapped out or unmapped
      * reserved - alloc → PyTorch caching-allocator overhead
    """
    vm_rss = vm_hwm = vm_data = vm_peak = pss = -1
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    vm_rss = int(line.split()[1]) // 1024
                elif line.startswith("VmHWM:"):
                    vm_hwm = int(line.split()[1]) // 1024
                elif line.startswith("VmData:"):
                    vm_data = int(line.split()[1]) // 1024
                elif line.startswith("VmPeak:"):
                    vm_peak = int(line.split()[1]) // 1024
    except Exception:
        pass
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    pss = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass
    ram_total = ram_used = ram_avail = mem_free = cached = -1
    shmem = slab = sreclaim = sunreclaim = kreclaim = anon = ptbl = -1
    try:
        with open("/proc/meminfo") as f:
            mi = {}
            for line in f:
                k, _, v = line.partition(":")
                mi[k] = int(v.strip().split()[0])
            ram_total = mi.get("MemTotal", 0) // 1024
            ram_avail = mi.get("MemAvailable", 0) // 1024
            mem_free = mi.get("MemFree", 0) // 1024
            cached = mi.get("Cached", 0) // 1024
            shmem = mi.get("Shmem", 0) // 1024
            slab = mi.get("Slab", 0) // 1024
            sreclaim = mi.get("SReclaimable", 0) // 1024
            sunreclaim = mi.get("SUnreclaim", 0) // 1024
            kreclaim = mi.get("KReclaimable", 0) // 1024
            anon = mi.get("AnonPages", 0) // 1024
            ptbl = mi.get("PageTables", 0) // 1024
            ram_used = ram_total - ram_avail
    except Exception:
        pass
    gpu = "GPU=n/a"
    gc_info = ""
    driver_free = driver_total = -1
    try:
        import torch as _t
        if _t.cuda.is_available():
            alloc = _t.cuda.memory_allocated() // (1024 * 1024)
            reserv = _t.cuda.memory_reserved() // (1024 * 1024)
            stats = _t.cuda.memory_stats()
            active_mib = stats.get("active_bytes.all.current", 0) // (1024 * 1024)
            inactive_mib = stats.get("inactive_split_bytes.all.current", 0) // (1024 * 1024)
            # Driver-level view: mem_get_info returns (free, total) from
            # cudaMemGetInfo. On GB10 Unified Memory this includes the
            # driver's own reserved regions (UVM page tables, framebuffer,
            # CUDA-runtime bookkeeping), not just PyTorch's allocator.
            try:
                df, dt = _t.cuda.mem_get_info()
                driver_free = df // (1024 * 1024)
                driver_total = dt // (1024 * 1024)
            except Exception:
                pass
            gpu = (f"CUDA alloc={alloc} resv={reserv} "
                   f"active={active_mib} inactive={inactive_mib} "
                   f"driver_free={driver_free}/{driver_total}")
    except Exception:
        gpu = "GPU=err"
    # GC-walk: independent of PyTorch allocator. Shows what tensor objects
    # are actually alive in Python. If GC-walk << CUDA alloc, the gap is
    # caching-allocator pool or non-Python-owned storage.
    try:
        import gc as _gc
        import torch as _t
        live_cuda = 0
        live_meta = 0
        total_cuda_bytes = 0
        for obj in _gc.get_objects():
            if isinstance(obj, _t.Tensor):
                dev = obj.device.type
                if dev == "cuda":
                    live_cuda += 1
                    total_cuda_bytes += obj.numel() * obj.element_size()
                elif dev == "meta":
                    live_meta += 1
        gc_info = (f" | GC: cuda={live_cuda} tensors "
                   f"{total_cuda_bytes // (1024*1024)}MiB, meta={live_meta}")
    except Exception:
        gc_info = ""
    # Accounting check: MemUsed - Pss - Cached - Slab - Shmem - PageTables
    # tells us how much memory is "missing" from our picture (likely NVIDIA
    # driver reservations on GB10 Unified Memory). Note we ignore Anon
    # here because it's already part of RSS/Pss.
    try:
        accounted = (pss if pss > 0 else 0) + (shmem if shmem > 0 else 0) \
                    + (slab if slab > 0 else 0) + (ptbl if ptbl > 0 else 0)
        unaccounted = (ram_used - accounted - (cached if cached > 0 else 0)) \
                      if ram_used > 0 else -1
    except Exception:
        unaccounted = -1
    return (f"{tag}: "
            f"RSS={vm_rss} HWM={vm_hwm} Data={vm_data} Pss={pss} Anon={anon} "
            f"| MemUsed={ram_used}/{ram_total} Avail={ram_avail} "
            f"Cached={cached} Free={mem_free} Shmem={shmem} "
            f"Slab={slab}({sreclaim}r+{sunreclaim}u) PgTbl={ptbl} "
            f"Unaccounted≈{unaccounted} "
            f"| {gpu} [MiB]{gc_info}")


def _top_cuda_params(model, top_n: int = 12) -> str:
    """Dump the top-N largest CUDA parameters with their names, shapes, dtypes.
    Useful when [stream-mem] reports high CUDA alloc — we see exactly which
    parameters are holding it. Called sparingly (once at LMHead #1)."""
    try:
        import torch as _t
        items = []
        for name, p in model.named_parameters():
            if p.device.type == "cuda" and p.numel() > 0:
                items.append((p.numel() * p.element_size(),
                              name, tuple(p.shape), str(p.dtype)))
        items.sort(reverse=True)
        total = sum(b for b, _, _, _ in items)
        lines = [f"    total cuda params: {total // (1024*1024)} MiB, "
                 f"{len(items)} params"]
        for b, n, s, d in items[:top_n]:
            lines.append(f"    {b // (1024*1024):>6} MiB  {n}  {s}  {d}")
        return "\n".join(lines)
    except Exception as e:
        return f"    (top-cuda-params failed: {e})"


# Thread-local flag: when True, MoE create_weights methods should allocate
# their (huge) expert tensors on the meta device rather than CUDA. Set by
# base_loader around initialize_model() so the streaming-quant-on-load
# path can materialize them per-layer instead of OOM-ing during init.
import threading as _threading
_moe_meta_flag = _threading.local()


def _moe_meta_active() -> bool:
    return getattr(_moe_meta_flag, "active", False)


def _set_moe_meta_flag(value: bool) -> None:
    _moe_meta_flag.active = value


def initialize_streaming_quantload(model: nn.Module) -> None:
    """Wrap weight loaders to trigger per-layer quantization as weights arrive.

    Two strategies depending on layer type:
    - LinearBase: replace params with meta device, materialize on weight_loader
    - FusedMoE (has load_weights): wrap load_weights to trigger quant after
      all experts loaded. Params stay real (FusedMoE manages its own memory).

    Peak memory: ~1 layer BF16 + all previously quantized layers.
    """
    logger.info("[stream-mem] %s", _mem_snapshot("streaming-quantload entry"))
    # Stash the model reference so streaming_loader can dump top-params when
    # triggered — lets us see which params actually hold CUDA memory when
    # the allocator reports a high peak.
    global _sq_model_ref
    _sq_model_ref = model
    linear_count = 0
    moe_count = 0

    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, QuantizeMethodBase):
            continue

        module._sq_processed = False

        # FusedMoE expert tensors were allocated on meta in create_weights
        # (when base_loader set the moe_meta flag). Swap them to real CUDA
        # tensors here — they are plain torch.nn.Parameter, so the subclass
        # __torch_function__ compat check that bites Linear doesn't apply.
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
        except ImportError:
            FusedMoE = None
        if FusedMoE is not None and isinstance(module, FusedMoE):
            _target = torch.device("cuda:0") if torch.cuda.is_available() \
                else torch.device("cpu")
            for _pname in list(module._parameters.keys()):
                _p = module._parameters[_pname]
                if _p is None or _p.device.type != "meta":
                    continue
                _real = torch.empty(_p.shape, dtype=_p.dtype, device=_target)
                _new = torch.nn.Parameter(_real, requires_grad=False)
                # Preserve loader hooks and routing metadata
                for _attr in ("weight_loader", "expert_mapping",
                              "output_dim", "input_dim", "packed_dim"):
                    if hasattr(_p, _attr):
                        try:
                            setattr(_new, _attr, getattr(_p, _attr))
                        except Exception:
                            pass
                module._parameters[_pname] = _new

        # Unified path: wrap param.weight_loader for every param that has
        # one. This covers LinearBase, FusedMoE (model-level load_weights in
        # e.g. qwen3_5 calls param.weight_loader directly — bypasses any
        # module.load_weights wrapping), embeddings, etc.
        total_numel = sum(
            p.numel() for p in module.parameters(recurse=False)
            if p is not None
        )
        if total_numel == 0:
            continue

        module._sq_load_numel = 0
        module._sq_load_total = total_numel

        for name in list(module._parameters.keys()):
            param = module._parameters[name]
            if param is None:
                continue
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue

            shape = tuple(param.shape)
            dtype = param.dtype
            device = param.device

            # Free the underlying storage while keeping the parameter object
            # (and its subclass — ModelWeightParameter etc.) alive. We set
            # a zero-size tensor on the same device, which passes vLLM's
            # subclass `__torch_function__` compat check (a meta-tensor
            # here fails with "incompatible tensor type"). The streaming
            # loader re-allocates full shape on first weight_loader call.
            param.data = torch.empty(0, dtype=dtype, device=device)
            param._streaming_shape = shape
            param._streaming_dtype = dtype
            param.weight_loader = _make_streaming_loader(
                module, name, original_loader, shape, dtype)
            linear_count += 1

    if linear_count > 0:
        logger.info(
            "Streaming quant-on-load: %d params swapped to meta",
            linear_count,
        )
        logger.info("[stream-mem] %s", _mem_snapshot("after meta-swap"))


def _move_params_to_device(layer: nn.Module) -> None:
    """Move all direct parameters to CUDA after CPU quantization."""
    target = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device("cpu")
    for name in list(layer._parameters.keys()):
        p = layer._parameters[name]
        if p is not None and p.device != target:
            layer._parameters[name] = torch.nn.Parameter(
                p.data.to(target), requires_grad=False)


def _make_moe_streaming_loader(layer, original_load_weights, meta_specs):
    """Wrap FusedMoE.load_weights to materialize meta params, then quant.

    meta_specs: dict[name -> (shape, dtype, attrs)] for params swapped to meta
    in initialize_streaming_quantload. On first call, allocate real tensors
    on CUDA (pre-quantization BF16 storage), run the original load_weights so
    FusedMoE.weight_loader can copy into them, then immediately quantize and
    free the BF16 storage via process_weights_after_loading.
    """
    import functools

    target = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device("cpu")

    @functools.wraps(original_load_weights)
    def streaming_load_weights(weights):
        # Materialize meta params before FusedMoE loads experts into them.
        materialized_bytes = 0
        for name, (shape, dtype, attrs) in meta_specs.items():
            p = layer._parameters.get(name)
            if p is None or p.device.type != "meta":
                continue
            real = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype, device=target),
                requires_grad=False,
            )
            for k, v in attrs.items():
                try:
                    setattr(real, k, v)
                except Exception:
                    pass
            layer._parameters[name] = real
            materialized_bytes += real.data.numel() * real.data.element_size()
        if materialized_bytes:
            logger.info(
                "[stream-mem] %s",
                _mem_snapshot(
                    f"MoE materialize +{materialized_bytes // (1024*1024)} MiB "
                    f"({layer.__class__.__name__})"),
            )

        result = original_load_weights(weights)
        # load_weights is a generator — drain it so the experts actually load.
        import types
        if isinstance(result, types.GeneratorType):
            result = list(result)

        if not layer._sq_processed:
            layer._sq_processed = True
            logger.info("[stream-mem] %s",
                        _mem_snapshot(f"MoE pre-quant ({layer.__class__.__name__})"))
            quant_method = getattr(layer, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(layer)
                _move_params_to_device(layer)
                logger.info("[stream-mem] %s",
                            _mem_snapshot(f"MoE post-quant ({layer.__class__.__name__})"))

        return result
    return streaming_load_weights


def _make_streaming_loader(layer, param_name, original_loader,
                           orig_shape, orig_dtype):
    """Wrap a weight_loader to materialize meta params and trigger quantization."""
    import functools

    target = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device("cpu")

    @functools.wraps(original_loader)
    def streaming_loader(*args, **kwargs):
        # Materialize parameter data on first load — mutate .data in place
        # so the subclass object (and its custom methods) survive.
        param = getattr(layer, param_name)
        if param.data.numel() == 0 and getattr(param, "_streaming_shape", None):
            param.data = torch.empty(orig_shape, dtype=orig_dtype,
                                     device=target)

        # Update args to use the (now-materialized) parameter
        if len(args) > 0 and hasattr(args[0], "device"):
            args = (param,) + args[1:]
        elif "param" in kwargs:
            kwargs["param"] = param
        ret = original_loader(*args, **kwargs)

        # Track how many elements have been loaded for this layer
        loaded_weight = args[1] if len(args) > 1 else kwargs.get(
            "loaded_weight", None)
        if loaded_weight is not None and hasattr(loaded_weight, "numel"):
            layer._sq_load_numel += loaded_weight.numel()

        # When all weights for this layer are loaded, quantize immediately
        if (layer._sq_load_numel >= layer._sq_load_total
                and not layer._sq_processed):
            layer._sq_processed = True
            quant_method = getattr(layer, "quant_method", None)
            if quant_method is not None:
                # Count + log every N-th layer to avoid log spam but still
                # give a per-layer memory trace — critical for validating
                # that streaming actually frees BF16 as we advance.
                import gc as _gc
                cls_counter = _mem_stream_counters
                key = layer.__class__.__name__
                cls_counter[key] = cls_counter.get(key, 0) + 1
                n = cls_counter[key]
                log_this = (n <= 3 or n % 20 == 0)
                if log_this:
                    logger.info("[stream-mem] %s",
                                _mem_snapshot(
                                    f"pre-quant {key} #{n} "
                                    f"({layer._sq_load_total} params)"))
                # On the very first pre-quant event, dump the top largest
                # CUDA params so we can see exactly what's holding the
                # memory at the "everything loaded, nothing packed" peak.
                if not _first_inventory_dumped[0]:
                    _first_inventory_dumped[0] = True
                    model_ref = globals().get("_sq_model_ref")
                    if model_ref is not None:
                        logger.info("[stream-inventory] top CUDA params "
                                    "at first pre-quant (%s #%d):\n%s",
                                    key, n, _top_cuda_params(model_ref, 15))
                quant_method.process_weights_after_loading(layer)
                _move_params_to_device(layer)
                _gc.collect()
                # Release caching-allocator blocks so reserved ~= allocated.
                # Critical for UMA devices (GB10) where reserved-but-unused
                # pool still counts against the system MemUsed budget.
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                if log_this:
                    logger.info("[stream-mem] %s",
                                _mem_snapshot(f"post-quant {key} #{n}"))

        return ret
    return streaming_loader


_mem_stream_counters: dict[str, int] = {}

# Flag to ensure the top-cuda-params inventory dump fires only once (at
# the first pre-quant event). Using a list for mutable closure access.
_first_inventory_dumped = [False]

# Populated by initialize_streaming_quantload so streaming_loader can dump
# a full top-N-params listing on the first pack event.
_sq_model_ref = None


def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # Skip modules already processed by streaming quant-on-load
            if getattr(module, "_sq_processed", False):
                continue
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    # NOTE: Happens after other modules so we can easily decompress weights.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)

    # Needed for torchao model reloading via model.reload_weights
    # @kylesayrs @jerryzh168 this can be removed if callers move to `reload_weights`
    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}
    uva_offloaded_parameters: list[str] = []

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        use_pin_memory = (
            is_pin_memory_available()
            and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
        )
        # Restore parameters to their original devices, ignoring new parameters
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                p.data = p.data.to(original_device)

            # parameter is UVA offloaded, but was replaced with a new device tensor
            # re-offload it to CPU using UVA
            if name in uva_offloaded_parameters and not getattr(
                p, "_vllm_is_uva_offloaded", False
            ):
                cpu_data = p.data.to(device="cpu")
                if use_pin_memory:
                    cpu_data = cpu_data.pin_memory()
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True


_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""Caches the outputs of `_get_model_architecture`."""


def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    from vllm.model_executor.models.adapters import as_embedding_model, as_seq_cls_model

    architectures = getattr(model_config.hf_config, "architectures", None) or []

    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )

    if arch == model_config._get_transformers_backend_cls():
        assert model_config.model_impl != "vllm"
        if model_config.model_impl == "auto":
            logger.warning_once(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",
                arch,
            )

    convert_type = model_config.convert_type
    if convert_type == "none":
        pass
    elif convert_type == "embed":
        logger.debug_once("Converting to embedding model.")
        model_cls = as_embedding_model(model_cls)
    elif convert_type == "classify":
        logger.debug_once("Converting to sequence classification model.")
        model_cls = as_seq_cls_model(model_cls)
    else:
        assert_never(convert_type)

    return model_cls, arch


def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    key = hash(
        (
            model_config.model,
            model_config.convert_type,
            model_config.runner_type,
            model_config.trust_remote_code,
            model_config.model_impl,
            tuple(getattr(model_config.hf_config, "architectures", None) or []),
        )
    )
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    model_arch = _get_model_architecture(model_config)
    _MODEL_ARCH_BY_HASH[key] = model_arch
    return model_arch


def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    return get_model_architecture(model_config)[0]


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their
    constituent parts.
    """

    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        for packed_name, sub_params in self.packed_mapping.items():
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )

    def get_sub_modules(self, module_name: str) -> tuple[str, list[str]] | None:
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None


def configure_quant_config(
    quant_config: QuantizationConfig, model_class: type[nn.Module]
):
    """
    Pass packed_modules_mapping by reference to quant_config so that
    quant_config can properly match fused modules

    Note that model attributes are passed by reference to quant_config,
    enabling them to be updated by model_class.__new__ (ex. chatglm, qwen)

    Once the `SupportsQuant` mixin has been added to all models, this
    function can be removed
    """
    if not issubclass(model_class, SupportsQuant):
        hf_to_vllm_mapper = getattr(model_class, "hf_to_vllm_mapper", None)
        packed_mapping = getattr(model_class, "packed_modules_mapping", None)

        # pass mappings by reference to quant_config
        if hf_to_vllm_mapper is not None:
            quant_config.apply_vllm_mapper(hf_to_vllm_mapper)
        if packed_mapping is not None:
            quant_config.packed_modules_mapping = packed_mapping
