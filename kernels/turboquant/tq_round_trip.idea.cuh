// SPDX-License-Identifier: Apache-2.0
// TurboQuant v2 header — optimized fused round-trip kernel.
#pragma once

#include <torch/extension.h>

namespace turboquant {

void turboquant_round_trip(
    torch::Tensor& key,
    torch::Tensor& Pi,
    torch::Tensor& S,
    torch::Tensor& centroids,
    torch::Tensor& output,
    int head_size,
    int n_centroids);

}  // namespace turboquant
