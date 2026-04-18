# SPDX-License-Identifier: Apache-2.0
"""TurboQuant — compatibility shim, delegates to vllm.multiquant.turboquant."""

from vllm.multiquant.turboquant.config import TurboQuantConfig
from vllm.multiquant.turboquant.quantizer import TurboQuantizer

__all__ = ["TurboQuantConfig", "TurboQuantizer"]
