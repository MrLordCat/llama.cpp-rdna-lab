// D138 probe: isolated gfx1201 WMMA throughput, 16x16x16, f16 vs fp8.
//
// Question this answers: is the fp8 MMA itself (same accumulator precision,
// fp32) intrinsically faster than the f16 MMA on RDNA4, or does FP8 only win
// on bytes read? The FA kernel spends its time in this instruction, so the
// raw instruction cost bounds any kernel-level optimisation.
//
// Forms taken from ROCm 7.2 rocWMMA (include/rocwmma/internal/wmma_impl.hpp):
//   f16 -> f32 acc : A/B = 4 x i32, C/D = 8 x float (w32)
//   fp8 -> f32 acc : A/B = 2 x i32, C/D = 8 x float (w32)
//   f16 -> f16 acc : A/B = 4 x i32, C/D = 4 x i32   (w32, this is what the
//                    f16 FA path uses for KQ and VKQ)
//
// Usage: d138_fp8_wmma_density.exe [iters] [blocks]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

using v2i32 = int __attribute__((ext_vector_type(2)));
using v4i32 = int __attribute__((ext_vector_type(4)));
using v8f32 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8f32 mma_f16(v4i32 a, v4i32 b, v8f32 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ v8f32 mma_fp8(v2i32 a, v2i32 b, v8f32 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

__device__ __forceinline__ v4i32 mma_f16_acc16(v4i32 a, v4i32 b, v4i32 c) {
    return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, c);
}

// Shared-memory operand variants: every lane fetches its own 8 B (fp8) or
// 16 B (f16) operand slice out of LDS on every step, which is what
// load_matrix_sync costs in the FA kernel. The shared tile is small enough to
// stay resident, so this measures LDS bandwidth/latency plus the MMA, not DRAM.
template<int KIND>
__global__ void bench_wmma_lds(unsigned long long * clocks, int iters, float * sink) {
    extern __shared__ unsigned char smem[];
    // Each lane owns a fixed 16 B window; consecutive lanes hit neighbouring
    // banks, matching a row-major WMMA operand layout.
    const int lane_off = (threadIdx.x * 16) & 4095;
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        smem[i] = (unsigned char)(i * 7 + 13);
    }
    __syncthreads();

    v8f32 acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        acc[k] = (v8f32)0.0f;
    }

    unsigned long long t0 = 0, t1 = 0;
    if (threadIdx.x == 0) {
        t0 = clock64();
    }
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int off = (lane_off + (i + k) * 64) & 4095;
            if constexpr (KIND == 3) {
                const v2i32 a = *reinterpret_cast<const v2i32 *>(smem + off);
                const v2i32 b = *reinterpret_cast<const v2i32 *>(smem + ((off + 2048) & 4095));
                acc[k] = mma_fp8(a, b, acc[k]);
            } else {
                const v4i32 a = *reinterpret_cast<const v4i32 *>(smem + off);
                const v4i32 b = *reinterpret_cast<const v4i32 *>(smem + ((off + 2048) & 4095));
                acc[k] = mma_f16(a, b, acc[k]);
            }
        }
    }
    if (threadIdx.x == 0) {
        t1 = clock64();
        clocks[blockIdx.x] = t1 - t0;
    }

    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s += acc[k][j];
        }
    }
    if (s == 12345.678f) {
        sink[0] = s;
    }
}

// KIND 0 = f16 operand MMA with fp32 accumulators,
// KIND 1 = fp8 operand MMA with fp32 accumulators,
// KIND 2 = f16 operand MMA with f16 accumulators (the f16 FA path).
template<int KIND>
__global__ void bench_wmma(unsigned long long * clocks, int iters, float * sink) {
    v8f32 acc[8];
    v4i32 acc16[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        acc[k] = (v8f32)0.0f;
        acc16[k] = (v4i32)0;
    }
    const v4i32 a16 = {1, 2, 3, 4};
    const v4i32 b16 = {5, 6, 7, 8};
    const v2i32 a8  = {1, 2};
    const v2i32 b8  = {3, 4};

    __syncthreads();
    const unsigned long long t0 = clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if constexpr (KIND == 0) {
                acc[k] = mma_f16(a16, b16, acc[k]);
            } else if constexpr (KIND == 1) {
                acc[k] = mma_fp8(a8, b8, acc[k]);
            } else {
                acc16[k] = mma_f16_acc16(a16, b16, acc16[k]);
            }
        }
    }
    const unsigned long long t1 = clock64();

    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s += acc[k][j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s += float(acc16[k][j]);
        }
    }
    if (threadIdx.x == 0) {
        clocks[blockIdx.x] = t1 - t0;
    }
    if (s == 12345.678f) { // never true; keeps the accumulators alive
        sink[0] = s;
    }
}

static void run_case(const char * name, int kind, int iters, int blocks, int threads) {
    unsigned long long * d_clocks = nullptr;
    float * d_sink = nullptr;
    if (hipMalloc(&d_clocks, blocks * sizeof(unsigned long long)) != hipSuccess ||
        hipMalloc(&d_sink, sizeof(float)) != hipSuccess) {
        printf("%s: hipMalloc failed\n", name);
        return;
    }

    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0);
    if (kind == 0) {
        bench_wmma<0><<<blocks, threads>>>(d_clocks, iters, d_sink);
    } else if (kind == 1) {
        bench_wmma<1><<<blocks, threads>>>(d_clocks, iters, d_sink);
    } else if (kind == 2) {
        bench_wmma<2><<<blocks, threads>>>(d_clocks, iters, d_sink);
    } else if (kind == 3) {
        bench_wmma_lds<3><<<blocks, threads, 4096>>>(d_clocks, iters, d_sink);
    } else {
        bench_wmma_lds<4><<<blocks, threads, 4096>>>(d_clocks, iters, d_sink);
    }
    hipEventRecord(e1);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("%s: launch failed: %s\n", name, hipGetErrorString(err));
        return;
    }
    float ms = 0.0f;
    hipEventElapsedTime(&ms, e0, e1);

    unsigned long long h_clocks[4096];
    const int ncopy = blocks < 4096 ? blocks : 4096;
    hipMemcpy(h_clocks, d_clocks, ncopy * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    // Peak (min) and median-ish (block 0) clock deltas across blocks.
    unsigned long long best = ~0ull;
    for (int i = 0; i < ncopy; ++i) {
        if (h_clocks[i] != 0 && h_clocks[i] < best) {
            best = h_clocks[i];
        }
    }

    const double mma_per_block = double(iters) * 8.0;
    const double clk_per_mma = double(best) / mma_per_block;
    const double flop = 2.0 * 16.0 * 16.0 * 16.0 * mma_per_block * blocks;
    printf("%-10s blocks=%4d threads=%3d iters=%6d | %8.3f ms | total MMA %10.0f | "
           "tflops %8.2f | clk/MMA (best block) %7.3f\n",
           name, blocks, threads, iters, ms, mma_per_block * blocks,
           flop / (double(ms) * 1.0e9), clk_per_mma);

    hipFree(d_clocks);
    hipFree(d_sink);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
}

int main(int argc, char ** argv) {
    const int iters  = argc > 1 ? atoi(argv[1]) : 8192;
    const int blocks = argc > 2 ? atoi(argv[2]) : 64;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("device: %s (%s), CU=%d, clk=%d kHz, warp=%d\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount,
           prop.clockRate, prop.warpSize);

    const int threads = 256;
    printf("\n-- single block (64 lanes x 8 independent accumulators) --\n");
    run_case("f16-f32a", 0, iters, 1, threads);
    run_case("fp8-f32a", 1, iters, 1, threads);
    run_case("f16-f16a", 2, iters, 1, threads);

    printf("\n-- %d blocks (full device) --\n", blocks);
    run_case("f16-f32a", 0, iters, blocks, threads);
    run_case("fp8-f32a", 1, iters, blocks, threads);
    run_case("f16-f16a", 2, iters, blocks, threads);

    printf("\n-- LDS operand path, %d blocks (as in FA load_matrix_sync) --\n", blocks);
    run_case("fp8-lds", 3, iters, blocks, threads);
    run_case("f16-lds", 4, iters, blocks, threads);
    return 0;
}
