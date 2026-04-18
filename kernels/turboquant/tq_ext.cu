#include <torch/extension.h>
#include "tq_round_trip.cuh"

namespace turboquant {
void turboquant_round_trip(
    torch::Tensor& key, torch::Tensor& Pi, torch::Tensor& S,
    torch::Tensor& centroids, torch::Tensor& output,
    int head_size, int n_centroids);
}

void tq_round_trip_wrapper(
    torch::Tensor key, torch::Tensor Pi, torch::Tensor S,
    torch::Tensor centroids, torch::Tensor output,
    int head_size, int n_centroids) {
    turboquant::turboquant_round_trip(key, Pi, S, centroids, output, head_size, n_centroids);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("round_trip", &tq_round_trip_wrapper, "TurboQuant fused round-trip");
}
