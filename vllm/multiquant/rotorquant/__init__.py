# SPDX-License-Identifier: Apache-2.0
"""RotorQuant: Clifford-algebra KV-cache quantization (MSE + QJL).

Uses Cl(3,0) rotor sandwiches instead of dense rotation matrices.
44× fewer parameters, 10-19× faster kernels, identical quality to TurboQuant.
"""

from vllm.multiquant.rotorquant.config import RotorQuantConfig
from vllm.multiquant.rotorquant.quantizer import RotorQuantizer

__all__ = ["RotorQuantConfig", "RotorQuantizer"]
