# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        streaming_quant = model_config.quantization in ("autoround_rtn",)
        with set_default_torch_dtype(model_config.dtype):
            # For streaming quant-on-load, tell MoE create_weights to allocate
            # its (huge) expert tensors on the meta device. Without this the
            # `with target_device:` context below triggers CUDA allocation of
            # all MoE BF16 weights before any weight loading can start — OOM
            # on models where the unquantized BF16 doesn't fit in host memory.
            if streaming_quant:
                from vllm.model_executor.model_loader.utils import (
                    _set_moe_meta_flag,
                )
                _set_moe_meta_flag(True)
            try:
                with target_device:
                    model = initialize_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        prefix=prefix,
                    )
            finally:
                if streaming_quant:
                    _set_moe_meta_flag(False)

            log_model_inspection(model)

            logger.debug("Loading weights on %s ...", load_device)
            # Streaming quant-on-load: wrap weight loaders to quantize
            # per-layer as weights arrive (saves memory for large models)
            if streaming_quant:
                from vllm.model_executor.model_loader.utils import (
                    initialize_streaming_quantload,
                )
                initialize_streaming_quantload(model)
                # Tell DefaultModelLoader to iterate safetensors grouped
                # by layer — otherwise shard-interleaved keys defeat the
                # per-layer quant trigger.
                self._streaming_quant_active = True

                # Initialize the MultiQuant disk cache if enabled. Each
                # quant method (XFP, Archer/TQ/RQ, AutoRound RTN, …) reads
                # from / writes to it in their process_weights_after_loading.
                _initialize_multiquant_cache(vllm_config, model_config)

            # Quantization does not happen in `load_weights` but after it
            # (unless streaming quant-on-load processes it per-layer)
            self.load_weights(model, model_config)

            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda():
                peak_memory = torch.accelerator.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                    scope="local",
                )

            process_weights_after_loading(model, model_config, target_device)

            # Finalize cache (write manifest + log summary) after all
            # per-layer process_weights_after_loading calls have fired.
            _finalize_multiquant_cache()

        return model.eval()


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))


def _initialize_multiquant_cache(
    vllm_config: VllmConfig, model_config: ModelConfig,
) -> None:
    """Set up the MultiQuant weight cache as a singleton for this process.

    Silently no-ops when ``MULTIQUANT_CACHE_DIR`` (or legacy
    ``XFP_CACHE_DIR``) is unset — the common case. Any hard failure
    (unreadable model path, bad env) is logged and swallowed: a broken
    cache must never prevent model loading.
    """
    try:
        from vllm.multiquant.policy import MultiQuantPolicyRegistry
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
    except ImportError:
        return

    registry = MultiQuantPolicyRegistry.get_active()
    model_path = getattr(model_config, "model", "") or ""
    if not model_path:
        return

    tp_size = 1
    ep_size = 1
    try:
        par = getattr(vllm_config, "parallel_config", None)
        if par is not None:
            tp_size = int(getattr(par, "tensor_parallel_size", 1) or 1)
            ep_size = int(getattr(par, "expert_parallel_size", 1) or 1)
    except Exception:
        pass

    try:
        cache = MultiQuantWeightCache.from_env(
            model_path=model_path,
            registry=registry,
            hf_config=getattr(model_config, "hf_config", None),
            tp_size=tp_size,
            ep_size=ep_size,
        )
    except Exception as e:
        logger.warning(
            "MultiQuant cache: init failed (%s) — running without cache",
            e,
        )
        return

    if cache is None:
        return

    # verify_manifest logs its own INFO/WARNING messages. False just means
    # the cache is cold (first run); save path still runs.
    cache.verify_manifest()
    MultiQuantWeightCache.set_active(cache)


def _finalize_multiquant_cache() -> None:
    """Write manifest + summary log at the end of model load."""
    try:
        from vllm.multiquant.policy import MultiQuantPolicyRegistry
        from vllm.multiquant.weight_cache import MultiQuantWeightCache
    except ImportError:
        return

    cache = MultiQuantWeightCache.get_active()
    if cache is None:
        return
    try:
        registry = MultiQuantPolicyRegistry.get_active()
        inventory = [p.stem for p in cache.cache_dir.glob("*.safetensors")]
        cache.write_manifest(registry, inventory)
        cache.log_summary()
    except Exception as e:
        logger.warning("MultiQuant cache: finalize failed (%s)", e)
