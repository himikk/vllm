#!/usr/bin/env python3
"""Verify mq_gemm_int2 and mq_gemm_int3 fused GEMM kernels.

Compares CUDA fused dequant+GEMM against Python dequant + torch.mm.
Run inside container with GPU.
"""
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def load_kernel(name, src):
    return load(name=name, sources=[src],
        extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math',
            '-gencode=arch=compute_120,code=sm_120',
            '-gencode=arch=compute_121,code=sm_121'],
        verbose=False)

def test_int2():
    print("=== mq_gemm_int2 ===")
    k = load_kernel("mq_gemm_int2", "/opt/mq/mq_gemm_int2.cu")

    M, K, N, gs = 4, 2048, 1536, 128
    torch.manual_seed(42)

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    # Create INT2 packed weights with known values
    # Pack random 2-bit values
    raw = torch.randint(0, 4, (K, N), dtype=torch.int32, device="cuda")
    # Pack into int32: 16 per word
    K_packed = K // 16
    qw = torch.zeros(K_packed, N, dtype=torch.int32, device="cuda")
    for i in range(16):
        qw |= (raw[torch.arange(K_packed)*16 + i] & 0x3) << (i * 2)

    n_groups = K // gs
    scales = torch.randn(n_groups, N, dtype=torch.float16, device="cuda") * 0.05
    N_zp = (N + 15) // 16
    qz = torch.zeros(n_groups, N_zp, dtype=torch.int32, device="cuda")  # zp=0

    # CUDA fused GEMM
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    k.mq_gemm_int2(A, qw, scales, qz, C, gs)
    torch.cuda.synchronize()

    # Reference: dequant then matmul
    gi = torch.arange(K, device="cuda") // gs
    W_fp = (scales[gi] * raw.float()).to(torch.float16)  # zp=0
    C_ref = A @ W_fp

    cos = F.cosine_similarity(C.float().flatten().unsqueeze(0),
                               C_ref.float().flatten().unsqueeze(0)).item()
    max_diff = (C.float() - C_ref.float()).abs().max().item()
    print(f"  cos={cos:.6f}  max_diff={max_diff:.4f}")
    print(f"  {'PASS' if cos > 0.99 else 'FAIL'}")

def test_int3():
    print("=== mq_gemm_int3 ===")
    k = load_kernel("mq_gemm_int3", "/opt/mq/mq_gemm_int3.cu")

    M, K, N, gs = 4, 2048, 1536, 128
    torch.manual_seed(42)

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    # Create INT3 packed weights
    raw = torch.randint(0, 8, (K, N), dtype=torch.int32, device="cuda")
    # Bit-stream pack: expand to bits, pack 32 bits per int32
    K_packed = (K * 3 + 31) // 32
    bits = torch.zeros(K * 3, N, dtype=torch.int32, device="cuda")
    for i in range(3):
        bits[torch.arange(K)*3 + i] = (raw >> i) & 1
    # Pack bits into int32
    qw = torch.zeros(K_packed, N, dtype=torch.int32, device="cuda")
    for b in range(min(K_packed * 32, K * 3)):
        word = b // 32
        bit_pos = b % 32
        if b < bits.shape[0]:
            qw[word] |= bits[b] << bit_pos

    n_groups = K // gs
    scales = torch.randn(n_groups, N, dtype=torch.float16, device="cuda") * 0.05
    N_zp = (N * 3 + 31) // 32
    qz = torch.zeros(n_groups, N_zp, dtype=torch.int32, device="cuda")

    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    k.mq_gemm_int3(A, qw, scales, qz, C, K, gs)
    torch.cuda.synchronize()

    gi = torch.arange(K, device="cuda") // gs
    W_fp = (scales[gi] * raw.float()).to(torch.float16)
    C_ref = A @ W_fp

    cos = F.cosine_similarity(C.float().flatten().unsqueeze(0),
                               C_ref.float().flatten().unsqueeze(0)).item()
    max_diff = (C.float() - C_ref.float()).abs().max().item()
    print(f"  cos={cos:.6f}  max_diff={max_diff:.4f}")
    print(f"  {'PASS' if cos > 0.99 else 'FAIL'}")

if __name__ == "__main__":
    test_int2()
    print()
    test_int3()
