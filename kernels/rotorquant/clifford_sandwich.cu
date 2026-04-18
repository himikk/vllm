// SPDX-License-Identifier: Apache-2.0
// Fused Clifford Cl(3,0) Rotor Sandwich: embed → R·x·R̃ → extract
//
// Replaces ~70 PyTorch kernel launches with 1 CUDA kernel.
// CUDA Graph safe by design (no allocations, no Python control flow).
//
// Grid: (batch,), Block: (n_groups,) — one thread per group of 3 dimensions.
// For D=128: n_groups=43, one warp + 11 threads.

#include <torch/extension.h>

namespace rotorquant {

// Cl(3,0) geometric product: c = a * b
// Hardcoded multiplication table for (+,+,+) signature.
// Components: [scalar, e1, e2, e3, e12, e13, e23, e123]
__device__ __forceinline__ void geo_prod(
    const float a[8], const float b[8], float c[8]
) {
    c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
          - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    c[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[4]*b[2]
          - a[3]*b[5] + a[5]*b[3] + a[6]*b[7] + a[7]*b[6];
    c[2] = a[0]*b[2] + a[2]*b[0] + a[1]*b[4] - a[4]*b[1]
          - a[3]*b[6] + a[6]*b[3] - a[5]*b[7] - a[7]*b[5];
    c[3] = a[0]*b[3] + a[3]*b[0] + a[1]*b[5] - a[5]*b[1]
          + a[2]*b[6] - a[6]*b[2] + a[4]*b[7] + a[7]*b[4];
    c[4] = a[0]*b[4] + a[4]*b[0] + a[1]*b[2] - a[2]*b[1]
          + a[5]*b[6] - a[6]*b[5] + a[3]*b[7] - a[7]*b[3];
    c[5] = a[0]*b[5] + a[5]*b[0] + a[1]*b[3] - a[3]*b[1]
          - a[4]*b[6] + a[6]*b[4] - a[2]*b[7] + a[7]*b[2];
    c[6] = a[0]*b[6] + a[6]*b[0] + a[2]*b[3] - a[3]*b[2]
          + a[4]*b[5] - a[5]*b[4] + a[1]*b[7] - a[7]*b[1];
    c[7] = a[0]*b[7] + a[7]*b[0] + a[1]*b[6] - a[6]*b[1]
          - a[2]*b[5] + a[5]*b[2] + a[3]*b[4] - a[4]*b[3];
}

// Forward rotation: embed x → R·x·R̃ → extract
__global__ void clifford_sandwich_forward_kernel(
    const float* __restrict__ x,        // [batch, D]
    const float* __restrict__ rotors,    // [n_groups, 8]
    float* __restrict__ out,             // [batch, D]
    int D, int n_groups
) {
    const int b = blockIdx.x;
    const int g = threadIdx.x;
    if (g >= n_groups) return;

    // 1. Embed: x → pure vector multivector (0, x0, x1, x2, 0, 0, 0, 0)
    float mv[8] = {0};
    mv[1] = (g*3   < D) ? x[b*D + g*3]   : 0.0f;  // e1
    mv[2] = (g*3+1 < D) ? x[b*D + g*3+1] : 0.0f;  // e2
    mv[3] = (g*3+2 < D) ? x[b*D + g*3+2] : 0.0f;  // e3

    // 2. Load rotor R
    float R[8];
    for (int i = 0; i < 8; i++) R[i] = rotors[g*8 + i];

    // 3. Compute R̃ (reverse: negate grade-2 and grade-3)
    float Rr[8];
    Rr[0] = R[0]; Rr[1] = R[1]; Rr[2] = R[2]; Rr[3] = R[3];
    Rr[4] = -R[4]; Rr[5] = -R[5]; Rr[6] = -R[6]; Rr[7] = -R[7];

    // 4. Sandwich: result = R * mv * R̃
    float tmp[8], result[8];
    geo_prod(R, mv, tmp);
    geo_prod(tmp, Rr, result);

    // 5. Extract: grade-1 components → output
    if (g*3   < D) out[b*D + g*3]   = result[1];  // e1
    if (g*3+1 < D) out[b*D + g*3+1] = result[2];  // e2
    if (g*3+2 < D) out[b*D + g*3+2] = result[3];  // e3
}

// Inverse rotation: embed x → R̃·x·R → extract
__global__ void clifford_sandwich_inverse_kernel(
    const float* __restrict__ x,        // [batch, D]
    const float* __restrict__ rotors,    // [n_groups, 8]
    float* __restrict__ out,             // [batch, D]
    int D, int n_groups
) {
    const int b = blockIdx.x;
    const int g = threadIdx.x;
    if (g >= n_groups) return;

    float mv[8] = {0};
    mv[1] = (g*3   < D) ? x[b*D + g*3]   : 0.0f;
    mv[2] = (g*3+1 < D) ? x[b*D + g*3+1] : 0.0f;
    mv[3] = (g*3+2 < D) ? x[b*D + g*3+2] : 0.0f;

    float R[8];
    for (int i = 0; i < 8; i++) R[i] = rotors[g*8 + i];

    // Inverse: R̃ * mv * R (reverse order)
    float Rr[8];
    Rr[0] = R[0]; Rr[1] = R[1]; Rr[2] = R[2]; Rr[3] = R[3];
    Rr[4] = -R[4]; Rr[5] = -R[5]; Rr[6] = -R[6]; Rr[7] = -R[7];

    float tmp[8], result[8];
    geo_prod(Rr, mv, tmp);
    geo_prod(tmp, R, result);

    if (g*3   < D) out[b*D + g*3]   = result[1];
    if (g*3+1 < D) out[b*D + g*3+1] = result[2];
    if (g*3+2 < D) out[b*D + g*3+2] = result[3];
}

// Python bindings
void clifford_sandwich_forward(
    torch::Tensor x, torch::Tensor rotors, torch::Tensor out, int D
) {
    int batch = x.size(0);
    int n_groups = rotors.size(0);
    int threads = ((n_groups + 31) / 32) * 32;  // round to warp
    clifford_sandwich_forward_kernel<<<batch, threads>>>(
        x.data_ptr<float>(), rotors.data_ptr<float>(),
        out.data_ptr<float>(), D, n_groups);
}

void clifford_sandwich_inverse(
    torch::Tensor x, torch::Tensor rotors, torch::Tensor out, int D
) {
    int batch = x.size(0);
    int n_groups = rotors.size(0);
    int threads = ((n_groups + 31) / 32) * 32;
    clifford_sandwich_inverse_kernel<<<batch, threads>>>(
        x.data_ptr<float>(), rotors.data_ptr<float>(),
        out.data_ptr<float>(), D, n_groups);
}

}  // namespace rotorquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("clifford_sandwich_forward", &rotorquant::clifford_sandwich_forward,
          "Fused Clifford rotor sandwich: embed → R·x·R̃ → extract");
    m.def("clifford_sandwich_inverse", &rotorquant::clifford_sandwich_inverse,
          "Fused Clifford inverse sandwich: embed → R̃·x·R → extract");
}
