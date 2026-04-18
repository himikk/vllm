// SPDX-License-Identifier: Apache-2.0
// Archer bit-unpack kernel — extract MSE indices + QJL signs from packed uint8.
//
// Runtime D (no template) — works for any dimension.
// Grid: (num_rows,), Block: (256,)

#include <torch/extension.h>
#include <cuda_fp16.h>

namespace archer {

constexpr int BLOCK_SIZE = 256;

template <int MSE_BITS>
__global__ void archer_unpack_kernel(
    const uint8_t* __restrict__ packed,
    int32_t* __restrict__ indices,
    float* __restrict__ signs,
    float* __restrict__ row_norms,
    float* __restrict__ res_norms,
    int num_rows,
    int packed_size,
    int D          // runtime dimension
) {
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const int tid = threadIdx.x;

    const int MSE_BYTES = (D * MSE_BITS + 7) / 8;
    const int QJL_BYTES = (D + 7) / 8;
    constexpr int MASK = (1 << MSE_BITS) - 1;

    const uint8_t* row_packed = packed + row * packed_size;
    int32_t* row_indices = indices + row * D;
    float* row_signs = signs + row * D;

    // Unpack MSE indices from bitstream (handles non-power-of-2 bit widths)
    for (int j = tid; j < D; j += BLOCK_SIZE) {
        int bit_offset = j * MSE_BITS;
        int byte_idx = bit_offset / 8;
        int bit_shift = bit_offset % 8;
        int val = row_packed[byte_idx] >> bit_shift;
        int spill = bit_shift + MSE_BITS - 8;
        if (spill > 0 && byte_idx + 1 < MSE_BYTES) {
            val |= row_packed[byte_idx + 1] << (MSE_BITS - spill);
        }
        row_indices[j] = val & MASK;
    }

    // Unpack QJL signs
    for (int j = tid; j < D; j += BLOCK_SIZE) {
        int byte_idx = MSE_BYTES + j / 8;
        int bit_pos = j % 8;
        row_signs[j] = ((row_packed[byte_idx] >> bit_pos) & 1) ? 1.0f : -1.0f;
    }

    // Thread 0 unpacks norms
    if (tid == 0) {
        int no = MSE_BYTES + QJL_BYTES;
        uint16_t vn_bits = row_packed[no] | (row_packed[no + 1] << 8);
        uint16_t rn_bits = row_packed[no + 2] | (row_packed[no + 3] << 8);
        row_norms[row] = __half2float(*reinterpret_cast<const __half*>(&vn_bits));
        res_norms[row] = __half2float(*reinterpret_cast<const __half*>(&rn_bits));
    }
}

void archer_unpack(
    torch::Tensor packed,
    torch::Tensor indices,
    torch::Tensor signs,
    torch::Tensor row_norms,
    torch::Tensor res_norms,
    int head_size,
    int mse_bits
) {
    int num_rows = packed.size(0);
    int packed_size = packed.size(1);

    if (mse_bits == 1) {
        archer_unpack_kernel<1><<<num_rows, BLOCK_SIZE>>>(
            packed.data_ptr<uint8_t>(),
            indices.data_ptr<int32_t>(),
            signs.data_ptr<float>(),
            row_norms.data_ptr<float>(),
            res_norms.data_ptr<float>(),
            num_rows, packed_size, head_size);
    } else if (mse_bits == 2) {
        archer_unpack_kernel<2><<<num_rows, BLOCK_SIZE>>>(
            packed.data_ptr<uint8_t>(),
            indices.data_ptr<int32_t>(),
            signs.data_ptr<float>(),
            row_norms.data_ptr<float>(),
            res_norms.data_ptr<float>(),
            num_rows, packed_size, head_size);
    } else if (mse_bits == 3) {
        archer_unpack_kernel<3><<<num_rows, BLOCK_SIZE>>>(
            packed.data_ptr<uint8_t>(),
            indices.data_ptr<int32_t>(),
            signs.data_ptr<float>(),
            row_norms.data_ptr<float>(),
            res_norms.data_ptr<float>(),
            num_rows, packed_size, head_size);
    } else {
        TORCH_CHECK(false, "Unsupported mse_bits: ", mse_bits);
    }
}

}  // namespace archer

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("archer_unpack", &archer::archer_unpack,
          "Archer unpack: packed uint8 → indices + signs + norms");
}
