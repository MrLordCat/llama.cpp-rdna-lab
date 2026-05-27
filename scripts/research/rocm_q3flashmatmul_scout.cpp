// Standalone ROCm Q3FlashMatmul scout for llama.cpp ROCm route-body research.
// This is not part of normal builds. It compares direct tiled Q3_K matmul
// variants against the current dequantize-to-f16 + rocBLAS GEMM contract.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define QK_K 256

struct block_q3_K_padded {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    __half d;
    uint8_t pad[2];
};
static_assert(sizeof(block_q3_K_padded) == 112, "wrong padded q3_K block size");

#define HIP_CHECK(expr)                                                                         \
    do {                                                                                        \
        hipError_t status__ = (expr);                                                           \
        if (status__ != hipSuccess) {                                                           \
            std::cerr << "HIP error: " << hipGetErrorString(status__) << " at line "          \
                      << __LINE__ << std::endl;                                                 \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

#define ROCBLAS_CHECK(expr)                                                                     \
    do {                                                                                        \
        rocblas_status status__ = (expr);                                                       \
        if (status__ != rocblas_status_success) {                                               \
            std::cerr << "rocBLAS error: " << rocblas_status_to_string(status__)               \
                      << " at line " << __LINE__ << std::endl;                                  \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

struct args_t {
    int m = 512;
    int n = 64;
    int k = 512;
    int warmup = 2;
    int iters = 6;
    int device = 0;
    int pipe_chunk = 2048;
    size_t max_check_elems = 1u << 20;
    std::string csv;
};

struct compare_stats_t {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rmse = 0.0;
    size_t elems = 0;
};

static void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0
        << " [--m M --n N --k K] [--warmup N] [--iters N] [--device N]"
        << " [--max-check-elems N] [--csv PATH]\n";
}

static bool parse_int(const char * text, int & out) {
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

static bool parse_size(const char * text, size_t & out) {
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<size_t>(value);
    return true;
}

static bool parse_args(int argc, char ** argv, args_t & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_int = [&](int & value) -> bool {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return false;
            }
            return parse_int(argv[++i], value);
        };

        if (key == "--m") {
            if (!need_int(args.m)) return false;
        } else if (key == "--n") {
            if (!need_int(args.n)) return false;
        } else if (key == "--k") {
            if (!need_int(args.k)) return false;
        } else if (key == "--warmup") {
            if (!need_int(args.warmup)) return false;
        } else if (key == "--iters") {
            if (!need_int(args.iters)) return false;
        } else if (key == "--device") {
            if (!need_int(args.device)) return false;
        } else if (key == "--pipe-chunk") {
            if (!need_int(args.pipe_chunk)) return false;
        } else if (key == "--max-check-elems") {
            if (i + 1 >= argc || !parse_size(argv[++i], args.max_check_elems)) {
                usage(argv[0]);
                return false;
            }
        } else if (key == "--csv") {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return false;
            }
            args.csv = argv[++i];
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            return false;
        }
    }

    return args.m > 0 && args.n > 0 && args.k > 0 && args.k % QK_K == 0 &&
        args.warmup >= 0 && args.iters > 0 && args.pipe_chunk > 0;
}

static uint32_t lcg(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static void fill_q3(std::vector<block_q3_K_padded> & blocks) {
    uint32_t state = 0x12345678u;
    for (block_q3_K_padded & block : blocks) {
        for (uint8_t & v : block.hmask) {
            v = static_cast<uint8_t>(lcg(state) >> 24);
        }
        for (uint8_t & v : block.qs) {
            v = static_cast<uint8_t>(lcg(state) >> 24);
        }
        for (uint8_t & v : block.scales) {
            v = static_cast<uint8_t>(lcg(state) >> 24);
        }
        block.d = __float2half(0.001f * static_cast<float>((lcg(state) % 7u) + 1u));
        block.pad[0] = 0;
        block.pad[1] = 0;
    }
}

static void fill_half(std::vector<__half> & values) {
    uint32_t state = 0x87654321u;
    for (__half & value : values) {
        const int centered = static_cast<int>(lcg(state) % 31u) - 15;
        value = __float2half(0.01f * static_cast<float>(centered));
    }
}

static __host__ __device__ inline int q3_scale(const block_q3_K_padded & block, const int is) {
    return is <  4 ? (block.scales[is - 0] & 0x0F) | (((block.scales[is + 8] >> 0) & 3) << 4) :
           is <  8 ? (block.scales[is - 0] & 0x0F) | (((block.scales[is + 4] >> 2) & 3) << 4) :
           is < 12 ? (block.scales[is - 8] >> 4)   | (((block.scales[is + 0] >> 4) & 3) << 4) :
                     (block.scales[is - 8] >> 4)   | (((block.scales[is - 4] >> 6) & 3) << 4);
}

static __host__ __device__ inline float q3_dequant_one(const block_q3_K_padded & block, const int element) {
    const int n = element / 128;
    const int rem = element - 128 * n;
    const int j = rem / 32;
    const int l = rem - 32 * j;
    const int is0 = l / 16;
    const int is = 8 * n + 2 * j + is0;
    const int shift = 2 * j;
    const uint8_t mask = static_cast<uint8_t>(1u << (4 * n + j));

    const int us = q3_scale(block, is);
    const float dl = __half2float(block.d) * static_cast<float>(us - 32);
    const int q = (block.qs[32 * n + l] >> shift) & 3;
    const int high = (block.hmask[l] & mask) ? 0 : 4;
    return dl * static_cast<float>(q - high);
}

__global__ static void dequant_q3_to_f16_kernel(
        const block_q3_K_padded * __restrict__ q3,
        __half * __restrict__ a,
        const int m,
        const int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * k;
    if (idx >= total) {
        return;
    }

    const int row = idx / k;
    const int col = idx - row * k;
    const int blocks_per_row = k / QK_K;
    const int block_index = row * blocks_per_row + col / QK_K;
    const int element = col % QK_K;
    a[idx] = __float2half(q3_dequant_one(q3[block_index], element));
}

__global__ static void dequant_q3_to_f16_chunk_kernel(
        const block_q3_K_padded * __restrict__ q3,
        __half * __restrict__ a,
        const int row_start,
        const int rows,
        const int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * k;
    if (idx >= total) {
        return;
    }

    const int local_row = idx / k;
    const int col = idx - local_row * k;
    const int row = row_start + local_row;
    const int blocks_per_row = k / QK_K;
    const int block_index = row * blocks_per_row + col / QK_K;
    const int element = col % QK_K;
    a[idx] = __float2half(q3_dequant_one(q3[block_index], element));
}

template<int tile_m, int tile_n, int tile_k>
__global__ static void q3flash_p0_kernel(
        const block_q3_K_padded * __restrict__ q3,
        const __half * __restrict__ b,
        float * __restrict__ d,
        const int m,
        const int n,
        const int k) {
    extern __shared__ __align__(sizeof(__half)) unsigned char shared_raw[];
    __half * sh_a = reinterpret_cast<__half *>(shared_raw);
    __half * sh_b = sh_a + tile_m * tile_k;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * tile_m + ty;
    const int col = blockIdx.x * tile_n + tx;
    const int linear = ty * tile_n + tx;
    const int threads = tile_m * tile_n;
    const int blocks_per_row = k / QK_K;

    float acc = 0.0f;

    for (int k0 = 0; k0 < k; k0 += tile_k) {
        for (int index = linear; index < tile_m * tile_k; index += threads) {
            const int local_row = index / tile_k;
            const int local_k = index - local_row * tile_k;
            const int global_row = blockIdx.y * tile_m + local_row;
            const int global_k = k0 + local_k;
            float value = 0.0f;
            if (global_row < m && global_k < k) {
                const int block_index = global_row * blocks_per_row + global_k / QK_K;
                value = q3_dequant_one(q3[block_index], global_k % QK_K);
            }
            sh_a[index] = __float2half(value);
        }

        for (int index = linear; index < tile_k * tile_n; index += threads) {
            const int local_k = index / tile_n;
            const int local_col = index - local_k * tile_n;
            const int global_k = k0 + local_k;
            const int global_col = blockIdx.x * tile_n + local_col;
            sh_b[index] = (global_k < k && global_col < n) ? b[global_k + global_col * k] : __float2half(0.0f);
        }

        __syncthreads();

#pragma unroll 4
        for (int local_k = 0; local_k < tile_k; ++local_k) {
            const float av = __half2float(sh_a[ty * tile_k + local_k]);
            const float bv = __half2float(sh_b[local_k * tile_n + tx]);
            acc += av * bv;
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        d[row + col * m] = acc;
    }
}

__global__ static void q3flash_wmma_p1_kernel(
        const block_q3_K_padded * __restrict__ q3,
        const __half * __restrict__ b,
        float * __restrict__ d,
        const int m,
        const int n,
        const int k) {
    using halfx8_t = __attribute__((ext_vector_type(8))) _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8))) float;

    const int lane = threadIdx.x;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * 16;
    const int blocks_per_row = k / QK_K;

    floatx8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < k; k0 += 16) {
        halfx8_t a_frag;
        halfx8_t b_frag;

        const int frag_row = lane % 16;
        const int frag_col0 = 8 * (lane / 16);
        const int a_row = tile_row + frag_row;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int a_col = k0 + frag_col0 + i;
            const int b_row = k0 + frag_col0 + i;
            const int b_col = tile_col + frag_row;

            float av = 0.0f;
            if (a_row < m && a_col < k) {
                const int block_index = a_row * blocks_per_row + a_col / QK_K;
                av = q3_dequant_one(q3[block_index], a_col % QK_K);
            }

            float bv = 0.0f;
            if (b_row < k && b_col < n) {
                bv = __half2float(b[b_row + b_col * k]);
            }

            a_frag[i] = static_cast<_Float16>(av);
            b_frag[i] = static_cast<_Float16>(bv);
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, acc);
    }

    const int out_col = tile_col + (lane % 16);
    const int out_row0 = tile_row + 8 * (lane / 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = out_row0 + i;
        if (out_row < m && out_col < n) {
            d[out_row + out_col * m] = acc[i];
        }
    }
}

template<int nwaves_n>
__global__ static void q3flash_wmma_reuse_p2_kernel(
        const block_q3_K_padded * __restrict__ q3,
        const __half * __restrict__ b,
        float * __restrict__ d,
        const int m,
        const int n,
        const int k) {
    using halfx8_t = __attribute__((ext_vector_type(8))) _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8))) float;

    __shared__ __half sh_a[16 * 16];

    const int lane = threadIdx.x;
    const int wave_n = threadIdx.y;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * (16 * nwaves_n) + wave_n * 16;
    const int blocks_per_row = k / QK_K;

    const int frag_row = lane % 16;
    const int frag_col0 = 8 * (lane / 16);
    floatx8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < k; k0 += 16) {
        if (wave_n == 0) {
            const int a_row = tile_row + frag_row;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int a_col = k0 + frag_col0 + i;
                float av = 0.0f;
                if (a_row < m && a_col < k) {
                    const int block_index = a_row * blocks_per_row + a_col / QK_K;
                    av = q3_dequant_one(q3[block_index], a_col % QK_K);
                }
                sh_a[frag_row * 16 + frag_col0 + i] = __float2half(av);
            }
        }

        __syncthreads();

        halfx8_t a_frag;
        halfx8_t b_frag;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int b_row = k0 + frag_col0 + i;
            const int b_col = tile_col + frag_row;
            const float av = __half2float(sh_a[frag_row * 16 + frag_col0 + i]);
            const float bv = (b_row < k && b_col < n) ? __half2float(b[b_row + b_col * k]) : 0.0f;
            a_frag[i] = static_cast<_Float16>(av);
            b_frag[i] = static_cast<_Float16>(bv);
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, acc);
        __syncthreads();
    }

    const int out_col = tile_col + (lane % 16);
    const int out_row0 = tile_row + 8 * (lane / 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = out_row0 + i;
        if (out_row < m && out_col < n) {
            d[out_row + out_col * m] = acc[i];
        }
    }
}

template<int nwaves_n, int tile_k>
__global__ static void q3flash_wmma_kstage_p3_kernel(
        const block_q3_K_padded * __restrict__ q3,
        const __half * __restrict__ b,
        float * __restrict__ d,
        const int m,
        const int n,
        const int k) {
    using halfx8_t = __attribute__((ext_vector_type(8))) _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8))) float;

    static_assert(tile_k % 16 == 0, "tile_k must be a WMMA K multiple");

    __shared__ __half sh_a[16 * tile_k];

    const int lane = threadIdx.x;
    const int wave_n = threadIdx.y;
    const int linear = wave_n * 32 + lane;
    const int threads = 32 * nwaves_n;
    const int tile_row = blockIdx.y * 16;
    const int tile_col = blockIdx.x * (16 * nwaves_n) + wave_n * 16;
    const int blocks_per_row = k / QK_K;

    const int frag_row = lane % 16;
    const int frag_col0 = 8 * (lane / 16);
    floatx8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < k; k0 += tile_k) {
        for (int index = linear; index < 16 * tile_k; index += threads) {
            const int local_row = index / tile_k;
            const int local_k = index - local_row * tile_k;
            const int a_row = tile_row + local_row;
            const int a_col = k0 + local_k;
            float av = 0.0f;
            if (a_row < m && a_col < k) {
                const int block_index = a_row * blocks_per_row + a_col / QK_K;
                av = q3_dequant_one(q3[block_index], a_col % QK_K);
            }
            sh_a[index] = __float2half(av);
        }

        __syncthreads();

        for (int kk = 0; kk < tile_k; kk += 16) {
            halfx8_t a_frag;
            halfx8_t b_frag;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int local_k = kk + frag_col0 + i;
                const int b_row = k0 + local_k;
                const int b_col = tile_col + frag_row;
                const float av = __half2float(sh_a[frag_row * tile_k + local_k]);
                const float bv = (b_row < k && b_col < n) ? __half2float(b[b_row + b_col * k]) : 0.0f;
                a_frag[i] = static_cast<_Float16>(av);
                b_frag[i] = static_cast<_Float16>(bv);
            }

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    const int out_col = tile_col + (lane % 16);
    const int out_row0 = tile_row + 8 * (lane / 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = out_row0 + i;
        if (out_row < m && out_col < n) {
            d[out_row + out_col * m] = acc[i];
        }
    }
}

template<int nwaves_m, int nwaves_n, int tile_k>
__global__ static void q3flash_wmma_multim_p4_kernel(
        const block_q3_K_padded * __restrict__ q3,
        const __half * __restrict__ b,
        float * __restrict__ d,
        const int m,
        const int n,
        const int k) {
    using halfx8_t = __attribute__((ext_vector_type(8))) _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8))) float;

    static_assert(tile_k % 16 == 0, "tile_k must be a WMMA K multiple");

    constexpr int tile_m = 16 * nwaves_m;
    constexpr int tile_n = 16 * nwaves_n;
    __shared__ __half sh_a[tile_m * tile_k];
    __shared__ __half sh_b[tile_n * tile_k];

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int wave_m = wave / nwaves_n;
    const int wave_n = wave - wave_m * nwaves_n;
    const int linear = wave * 32 + lane;
    const int threads = 32 * nwaves_m * nwaves_n;
    const int tile_row = blockIdx.y * tile_m;
    const int tile_col = blockIdx.x * tile_n;
    const int blocks_per_row = k / QK_K;

    const int frag_row = lane % 16;
    const int frag_col0 = 8 * (lane / 16);
    floatx8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < k; k0 += tile_k) {
        for (int index = linear; index < tile_m * tile_k; index += threads) {
            const int local_row = index / tile_k;
            const int local_k = index - local_row * tile_k;
            const int a_row = tile_row + local_row;
            const int a_col = k0 + local_k;
            float av = 0.0f;
            if (a_row < m && a_col < k) {
                const int block_index = a_row * blocks_per_row + a_col / QK_K;
                av = q3_dequant_one(q3[block_index], a_col % QK_K);
            }
            sh_a[index] = __float2half(av);
        }

        for (int index = linear; index < tile_n * tile_k; index += threads) {
            const int local_n = index / tile_k;
            const int local_k = index - local_n * tile_k;
            const int b_row = k0 + local_k;
            const int b_col = tile_col + local_n;
            float bv = 0.0f;
            if (b_row < k && b_col < n) {
                bv = __half2float(b[b_row + b_col * k]);
            }
            sh_b[index] = __float2half(bv);
        }

        __syncthreads();

        for (int kk = 0; kk < tile_k; kk += 16) {
            halfx8_t a_frag;
            halfx8_t b_frag;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int local_k = kk + frag_col0 + i;
                const int local_row = wave_m * 16 + frag_row;
                const int local_col = wave_n * 16 + frag_row;
                const float av = __half2float(sh_a[local_row * tile_k + local_k]);
                const float bv = __half2float(sh_b[local_col * tile_k + local_k]);
                a_frag[i] = static_cast<_Float16>(av);
                b_frag[i] = static_cast<_Float16>(bv);
            }

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    const int out_col = tile_col + wave_n * 16 + (lane % 16);
    const int out_row0 = tile_row + wave_m * 16 + 8 * (lane / 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = out_row0 + i;
        if (out_row < m && out_col < n) {
            d[out_row + out_col * m] = acc[i];
        }
    }
}

static rocblas_status gemm(
        rocblas_handle handle,
        const args_t & args,
        const __half * a,
        const __half * b,
        float * d) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        args.m, args.n, args.k,
        &alpha, a, rocblas_datatype_f16_r, args.k,
        b, rocblas_datatype_f16_r, args.k,
        &beta, d, rocblas_datatype_f32_r, args.m,
        d, rocblas_datatype_f32_r, args.m,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}

    static rocblas_status gemm_chunk(
        rocblas_handle handle,
        const args_t & args,
        const int row_start,
        const int rows,
        const __half * a,
        const __half * b,
        float * d) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        return rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        rows, args.n, args.k,
        &alpha, a, rocblas_datatype_f16_r, args.k,
        b, rocblas_datatype_f16_r, args.k,
        &beta, d + row_start, rocblas_datatype_f32_r, args.m,
        d + row_start, rocblas_datatype_f32_r, args.m,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    }

static int time_baseline(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        __half * a,
        const __half * b,
        float * d,
        double & avg_ms) {
    const int total_a = args.m * args.k;
    const dim3 dequant_block(256);
    const dim3 dequant_grid((total_a + dequant_block.x - 1) / dequant_block.x);

    for (int i = 0; i < args.warmup; ++i) {
        dequant_q3_to_f16_kernel<<<dequant_grid, dequant_block, 0, stream>>>(q3, a, args.m, args.k);
        HIP_CHECK(hipGetLastError());
        ROCBLAS_CHECK(gemm(handle, args, a, b, d));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        dequant_q3_to_f16_kernel<<<dequant_grid, dequant_block, 0, stream>>>(q3, a, args.m, args.k);
        HIP_CHECK(hipGetLastError());
        ROCBLAS_CHECK(gemm(handle, args, a, b, d));
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
    return 0;
}

template<int tile_m, int tile_n, int tile_k>
static int time_q3flash_tiled(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    const dim3 block(tile_n, tile_m);
    const dim3 grid((args.n + tile_n - 1) / tile_n, (args.m + tile_m - 1) / tile_m);
    const size_t shared_bytes = (tile_m * tile_k + tile_k * tile_n) * sizeof(__half);

    for (int i = 0; i < args.warmup; ++i) {
        q3flash_p0_kernel<tile_m, tile_n, tile_k><<<grid, block, shared_bytes, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        q3flash_p0_kernel<tile_m, tile_n, tile_k><<<grid, block, shared_bytes, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
    return 0;
}

static int time_q3flash(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    return time_q3flash_tiled<16, 16, QK_K>(stream, args, q3, b, d, avg_ms);
}

static int time_q3flash_wide32(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    return time_q3flash_tiled<16, 32, QK_K>(stream, args, q3, b, d, avg_ms);
}

static int time_q3flash_wide64(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    return time_q3flash_tiled<16, 64, QK_K>(stream, args, q3, b, d, avg_ms);
}

static int time_q3flash_wmma(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    const dim3 block(32);
    const dim3 grid((args.n + 15) / 16, (args.m + 15) / 16);

    for (int i = 0; i < args.warmup; ++i) {
        q3flash_wmma_p1_kernel<<<grid, block, 0, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        q3flash_wmma_p1_kernel<<<grid, block, 0, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
    return 0;
}

static int time_q3flash_wmma_reuse(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    constexpr int nwaves_n = 16;
    const dim3 block(32, nwaves_n);
    const dim3 grid((args.n + 16 * nwaves_n - 1) / (16 * nwaves_n), (args.m + 15) / 16);

    for (int i = 0; i < args.warmup; ++i) {
        q3flash_wmma_reuse_p2_kernel<nwaves_n><<<grid, block, 0, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        q3flash_wmma_reuse_p2_kernel<nwaves_n><<<grid, block, 0, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
    return 0;
}

static int time_q3flash_wmma_kstage(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    constexpr int nwaves_n = 8;
    constexpr int tile_k = 128;
    const dim3 block(32, nwaves_n);
    const dim3 grid((args.n + 16 * nwaves_n - 1) / (16 * nwaves_n), (args.m + 15) / 16);

    for (int i = 0; i < args.warmup; ++i) {
        q3flash_wmma_kstage_p3_kernel<nwaves_n, tile_k><<<grid, block, 0, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        q3flash_wmma_kstage_p3_kernel<nwaves_n, tile_k><<<grid, block, 0, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
    return 0;
}

static int time_q3flash_wmma_multim(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    constexpr int nwaves_m = 4;
    constexpr int nwaves_n = 4;
    constexpr int tile_k = 128;
    const dim3 block(32, nwaves_m * nwaves_n);
    const dim3 grid((args.n + 16 * nwaves_n - 1) / (16 * nwaves_n), (args.m + 16 * nwaves_m - 1) / (16 * nwaves_m));

    for (int i = 0; i < args.warmup; ++i) {
        q3flash_wmma_multim_p4_kernel<nwaves_m, nwaves_n, tile_k><<<grid, block, 0, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        q3flash_wmma_multim_p4_kernel<nwaves_m, nwaves_n, tile_k><<<grid, block, 0, stream>>>(q3, b, d, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
    return 0;
}

static int time_streaming_pipeline(
        const args_t & args,
        const block_q3_K_padded * q3,
        const __half * b,
        float * d,
        double & avg_ms) {
    constexpr int buffers = 2;
    const int chunk_rows = std::min(args.pipe_chunk, args.m);
    const size_t chunk_elems = static_cast<size_t>(chunk_rows) * static_cast<size_t>(args.k);

    __half * a_chunks[buffers] = {nullptr, nullptr};
    hipStream_t dequant_streams[buffers] = {nullptr, nullptr};
    hipStream_t gemm_streams[buffers] = {nullptr, nullptr};
    hipEvent_t ready_events[buffers] = {nullptr, nullptr};
    rocblas_handle handles[buffers] = {nullptr, nullptr};

    for (int i = 0; i < buffers; ++i) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&a_chunks[i]), chunk_elems * sizeof(__half)));
        HIP_CHECK(hipStreamCreateWithFlags(&dequant_streams[i], hipStreamNonBlocking));
        HIP_CHECK(hipStreamCreateWithFlags(&gemm_streams[i], hipStreamNonBlocking));
        HIP_CHECK(hipEventCreateWithFlags(&ready_events[i], hipEventDisableTiming));
        ROCBLAS_CHECK(rocblas_create_handle(&handles[i]));
        ROCBLAS_CHECK(rocblas_set_stream(handles[i], gemm_streams[i]));
        ROCBLAS_CHECK(rocblas_set_pointer_mode(handles[i], rocblas_pointer_mode_host));
    }

    auto run_once = [&]() -> int {
        const dim3 block(256);
        for (int row_start = 0, chunk = 0; row_start < args.m; row_start += chunk_rows, ++chunk) {
            const int rows = std::min(chunk_rows, args.m - row_start);
            const int buf = chunk % buffers;
            if (chunk >= buffers) {
                HIP_CHECK(hipStreamSynchronize(gemm_streams[buf]));
            }

            const int total = rows * args.k;
            const dim3 grid((total + block.x - 1) / block.x);
            dequant_q3_to_f16_chunk_kernel<<<grid, block, 0, dequant_streams[buf]>>>(q3, a_chunks[buf], row_start, rows, args.k);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipEventRecord(ready_events[buf], dequant_streams[buf]));
            HIP_CHECK(hipStreamWaitEvent(gemm_streams[buf], ready_events[buf], 0));
            ROCBLAS_CHECK(gemm_chunk(handles[buf], args, row_start, rows, a_chunks[buf], b, d));
        }

        for (int i = 0; i < buffers; ++i) {
            HIP_CHECK(hipStreamSynchronize(gemm_streams[i]));
        }
        return 0;
    };

    for (int i = 0; i < args.warmup; ++i) {
        if (run_once() != 0) {
            return 1;
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.iters; ++i) {
        if (run_once() != 0) {
            return 1;
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::high_resolution_clock::now();
    avg_ms = std::chrono::duration<double, std::milli>(stop - start).count() / static_cast<double>(args.iters);

    for (int i = 0; i < buffers; ++i) {
        ROCBLAS_CHECK(rocblas_destroy_handle(handles[i]));
        HIP_CHECK(hipEventDestroy(ready_events[i]));
        HIP_CHECK(hipStreamDestroy(gemm_streams[i]));
        HIP_CHECK(hipStreamDestroy(dequant_streams[i]));
        HIP_CHECK(hipFree(a_chunks[i]));
    }
    return 0;
}

static compare_stats_t compare_outputs(const std::vector<float> & baseline, const std::vector<float> & candidate) {
    compare_stats_t stats;
    stats.elems = baseline.size();
    double sq = 0.0;
    for (size_t i = 0; i < baseline.size(); ++i) {
        const double ref = static_cast<double>(baseline[i]);
        const double got = static_cast<double>(candidate[i]);
        const double abs_err = std::abs(ref - got);
        const double rel_err = abs_err / std::max(1e-6, std::abs(ref));
        stats.max_abs = std::max(stats.max_abs, abs_err);
        stats.max_rel = std::max(stats.max_rel, rel_err);
        sq += abs_err * abs_err;
    }
    stats.rmse = std::sqrt(sq / std::max<size_t>(1, baseline.size()));
    return stats;
}

int main(int argc, char ** argv) {
    args_t args;
    if (!parse_args(argc, argv, args)) {
        return 2;
    }

    HIP_CHECK(hipSetDevice(args.device));

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    rocblas_handle handle = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    const size_t blocks_per_row = static_cast<size_t>(args.k) / QK_K;
    const size_t q3_blocks = static_cast<size_t>(args.m) * blocks_per_row;
    const size_t a_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.k);
    const size_t b_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.n);
    const size_t d_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.n);

    std::vector<block_q3_K_padded> h_q3(q3_blocks);
    std::vector<__half> h_b(b_elems);
    fill_q3(h_q3);
    fill_half(h_b);

    block_q3_K_padded * d_q3 = nullptr;
    __half * d_a = nullptr;
    __half * d_b = nullptr;
    float * d_baseline = nullptr;
    float * d_q3flash = nullptr;
    float * d_q3flash_wide32 = nullptr;
    float * d_q3flash_wide64 = nullptr;
    float * d_wmma = nullptr;
    float * d_wmma_reuse = nullptr;
    float * d_wmma_kstage = nullptr;
    float * d_wmma_multim = nullptr;
    float * d_pipeline = nullptr;

    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_q3), q3_blocks * sizeof(block_q3_K_padded)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_a), a_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_b), b_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_baseline), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_q3flash), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_q3flash_wide32), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_q3flash_wide64), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_wmma), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_wmma_reuse), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_wmma_kstage), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_wmma_multim), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_pipeline), d_elems * sizeof(float)));

    HIP_CHECK(hipMemcpyAsync(d_q3, h_q3.data(), h_q3.size() * sizeof(block_q3_K_padded), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(), h_b.size() * sizeof(__half), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double baseline_ms = 0.0;
    double q3flash_ms = 0.0;
    double q3flash_wide32_ms = 0.0;
    double q3flash_wide64_ms = 0.0;
    double wmma_ms = 0.0;
    double wmma_reuse_ms = 0.0;
    double wmma_kstage_ms = 0.0;
    double wmma_multim_ms = 0.0;
    double pipeline_ms = 0.0;
    if (time_baseline(handle, stream, args, d_q3, d_a, d_b, d_baseline, baseline_ms) != 0) {
        return 1;
    }
    if (time_q3flash(stream, args, d_q3, d_b, d_q3flash, q3flash_ms) != 0) {
        return 1;
    }
    if (time_q3flash_wide32(stream, args, d_q3, d_b, d_q3flash_wide32, q3flash_wide32_ms) != 0) {
        return 1;
    }
    if (time_q3flash_wide64(stream, args, d_q3, d_b, d_q3flash_wide64, q3flash_wide64_ms) != 0) {
        return 1;
    }
    if (time_q3flash_wmma(stream, args, d_q3, d_b, d_wmma, wmma_ms) != 0) {
        return 1;
    }
    if (time_q3flash_wmma_reuse(stream, args, d_q3, d_b, d_wmma_reuse, wmma_reuse_ms) != 0) {
        return 1;
    }
    if (time_q3flash_wmma_kstage(stream, args, d_q3, d_b, d_wmma_kstage, wmma_kstage_ms) != 0) {
        return 1;
    }
    if (time_q3flash_wmma_multim(stream, args, d_q3, d_b, d_wmma_multim, wmma_multim_ms) != 0) {
        return 1;
    }
    if (time_streaming_pipeline(args, d_q3, d_b, d_pipeline, pipeline_ms) != 0) {
        return 1;
    }

    compare_stats_t stats;
    compare_stats_t wide32_stats;
    compare_stats_t wide64_stats;
    compare_stats_t wmma_stats;
    compare_stats_t wmma_reuse_stats;
    compare_stats_t wmma_kstage_stats;
    compare_stats_t wmma_multim_stats;
    compare_stats_t pipeline_stats;
    if (d_elems <= args.max_check_elems) {
        std::vector<float> h_baseline(d_elems);
        std::vector<float> h_q3flash(d_elems);
        std::vector<float> h_q3flash_wide32(d_elems);
        std::vector<float> h_q3flash_wide64(d_elems);
        std::vector<float> h_wmma(d_elems);
        std::vector<float> h_wmma_reuse(d_elems);
        std::vector<float> h_wmma_kstage(d_elems);
        std::vector<float> h_wmma_multim(d_elems);
        std::vector<float> h_pipeline(d_elems);
        HIP_CHECK(hipMemcpy(h_baseline.data(), d_baseline, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_q3flash.data(), d_q3flash, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_q3flash_wide32.data(), d_q3flash_wide32, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_q3flash_wide64.data(), d_q3flash_wide64, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_wmma.data(), d_wmma, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_wmma_reuse.data(), d_wmma_reuse, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_wmma_kstage.data(), d_wmma_kstage, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_wmma_multim.data(), d_wmma_multim, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_pipeline.data(), d_pipeline, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        stats = compare_outputs(h_baseline, h_q3flash);
        wide32_stats = compare_outputs(h_baseline, h_q3flash_wide32);
        wide64_stats = compare_outputs(h_baseline, h_q3flash_wide64);
        wmma_stats = compare_outputs(h_baseline, h_wmma);
        wmma_reuse_stats = compare_outputs(h_baseline, h_wmma_reuse);
        wmma_kstage_stats = compare_outputs(h_baseline, h_wmma_kstage);
        wmma_multim_stats = compare_outputs(h_baseline, h_wmma_multim);
        pipeline_stats = compare_outputs(h_baseline, h_pipeline);
    }

    const double speedup = q3flash_ms > 0.0 ? baseline_ms / q3flash_ms : 0.0;
    const double wide32_speedup = q3flash_wide32_ms > 0.0 ? baseline_ms / q3flash_wide32_ms : 0.0;
    const double wide64_speedup = q3flash_wide64_ms > 0.0 ? baseline_ms / q3flash_wide64_ms : 0.0;
    const double wmma_speedup = wmma_ms > 0.0 ? baseline_ms / wmma_ms : 0.0;
    const double wmma_reuse_speedup = wmma_reuse_ms > 0.0 ? baseline_ms / wmma_reuse_ms : 0.0;
    const double wmma_kstage_speedup = wmma_kstage_ms > 0.0 ? baseline_ms / wmma_kstage_ms : 0.0;
    const double wmma_multim_speedup = wmma_multim_ms > 0.0 ? baseline_ms / wmma_multim_ms : 0.0;
    const double pipeline_speedup = pipeline_ms > 0.0 ? baseline_ms / pipeline_ms : 0.0;
    std::cout << std::fixed << std::setprecision(4)
              << "m=" << args.m << " n=" << args.n << " k=" << args.k
              << " baseline_ms=" << baseline_ms
              << " q3flash_ms=" << q3flash_ms
              << " speedup=" << speedup
              << " q3flash_wide32_ms=" << q3flash_wide32_ms
              << " q3flash_wide32_speedup=" << wide32_speedup
              << " q3flash_wide64_ms=" << q3flash_wide64_ms
              << " q3flash_wide64_speedup=" << wide64_speedup
              << " wmma_ms=" << wmma_ms
              << " wmma_speedup=" << wmma_speedup
              << " wmma_reuse_ms=" << wmma_reuse_ms
              << " wmma_reuse_speedup=" << wmma_reuse_speedup
              << " wmma_kstage_ms=" << wmma_kstage_ms
              << " wmma_kstage_speedup=" << wmma_kstage_speedup
              << " wmma_multim_ms=" << wmma_multim_ms
              << " wmma_multim_speedup=" << wmma_multim_speedup
              << " pipeline_ms=" << pipeline_ms
              << " pipeline_speedup=" << pipeline_speedup
              << " pipe_chunk=" << args.pipe_chunk
              << " checked=" << stats.elems
              << " max_abs=" << stats.max_abs
              << " max_rel=" << stats.max_rel
              << " rmse=" << stats.rmse
              << " wide32_checked=" << wide32_stats.elems
              << " wide32_max_abs=" << wide32_stats.max_abs
              << " wide32_max_rel=" << wide32_stats.max_rel
              << " wide32_rmse=" << wide32_stats.rmse
              << " wide64_checked=" << wide64_stats.elems
              << " wide64_max_abs=" << wide64_stats.max_abs
              << " wide64_max_rel=" << wide64_stats.max_rel
              << " wide64_rmse=" << wide64_stats.rmse
              << " wmma_checked=" << wmma_stats.elems
              << " wmma_max_abs=" << wmma_stats.max_abs
              << " wmma_max_rel=" << wmma_stats.max_rel
              << " wmma_rmse=" << wmma_stats.rmse
              << " wmma_reuse_checked=" << wmma_reuse_stats.elems
              << " wmma_reuse_max_abs=" << wmma_reuse_stats.max_abs
              << " wmma_reuse_max_rel=" << wmma_reuse_stats.max_rel
              << " wmma_reuse_rmse=" << wmma_reuse_stats.rmse
              << " wmma_kstage_checked=" << wmma_kstage_stats.elems
              << " wmma_kstage_max_abs=" << wmma_kstage_stats.max_abs
              << " wmma_kstage_max_rel=" << wmma_kstage_stats.max_rel
              << " wmma_kstage_rmse=" << wmma_kstage_stats.rmse
              << " wmma_multim_checked=" << wmma_multim_stats.elems
              << " wmma_multim_max_abs=" << wmma_multim_stats.max_abs
              << " wmma_multim_max_rel=" << wmma_multim_stats.max_rel
              << " wmma_multim_rmse=" << wmma_multim_stats.rmse
              << " pipeline_checked=" << pipeline_stats.elems
              << " pipeline_max_abs=" << pipeline_stats.max_abs
              << " pipeline_max_rel=" << pipeline_stats.max_rel
              << " pipeline_rmse=" << pipeline_stats.rmse << std::endl;

    if (!args.csv.empty()) {
        FILE * file = std::fopen(args.csv.c_str(), "w");
        if (file == nullptr) {
            std::cerr << "failed to open csv: " << args.csv << std::endl;
            return 1;
        }
        std::fprintf(file, "m,n,k,warmup,iters,pipe_chunk,baseline_ms,q3flash_ms,speedup,q3flash_wide32_ms,q3flash_wide32_speedup,q3flash_wide64_ms,q3flash_wide64_speedup,wmma_ms,wmma_speedup,wmma_reuse_ms,wmma_reuse_speedup,wmma_kstage_ms,wmma_kstage_speedup,wmma_multim_ms,wmma_multim_speedup,pipeline_ms,pipeline_speedup,checked_elems,max_abs,max_rel,rmse,wide32_checked_elems,wide32_max_abs,wide32_max_rel,wide32_rmse,wide64_checked_elems,wide64_max_abs,wide64_max_rel,wide64_rmse,wmma_checked_elems,wmma_max_abs,wmma_max_rel,wmma_rmse,wmma_reuse_checked_elems,wmma_reuse_max_abs,wmma_reuse_max_rel,wmma_reuse_rmse,wmma_kstage_checked_elems,wmma_kstage_max_abs,wmma_kstage_max_rel,wmma_kstage_rmse,wmma_multim_checked_elems,wmma_multim_max_abs,wmma_multim_max_rel,wmma_multim_rmse,pipeline_checked_elems,pipeline_max_abs,pipeline_max_rel,pipeline_rmse\n");
        std::fprintf(file, "%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%zu,%.9f,%.9f,%.9f,%zu,%.9f,%.9f,%.9f,%zu,%.9f,%.9f,%.9f,%zu,%.9f,%.9f,%.9f,%zu,%.9f,%.9f,%.9f,%zu,%.9f,%.9f,%.9f,%zu,%.9f,%.9f,%.9f,%zu,%.9f,%.9f,%.9f\n",
            args.m, args.n, args.k, args.warmup, args.iters,
            args.pipe_chunk,
            baseline_ms, q3flash_ms, speedup,
            q3flash_wide32_ms, wide32_speedup,
            q3flash_wide64_ms, wide64_speedup,
            wmma_ms, wmma_speedup,
            wmma_reuse_ms, wmma_reuse_speedup, wmma_kstage_ms, wmma_kstage_speedup,
            wmma_multim_ms, wmma_multim_speedup,
            pipeline_ms, pipeline_speedup,
            stats.elems, stats.max_abs, stats.max_rel, stats.rmse,
            wide32_stats.elems, wide32_stats.max_abs, wide32_stats.max_rel, wide32_stats.rmse,
            wide64_stats.elems, wide64_stats.max_abs, wide64_stats.max_rel, wide64_stats.rmse,
            wmma_stats.elems, wmma_stats.max_abs, wmma_stats.max_rel, wmma_stats.rmse,
            wmma_reuse_stats.elems, wmma_reuse_stats.max_abs, wmma_reuse_stats.max_rel, wmma_reuse_stats.rmse,
            wmma_kstage_stats.elems, wmma_kstage_stats.max_abs, wmma_kstage_stats.max_rel, wmma_kstage_stats.rmse,
            wmma_multim_stats.elems, wmma_multim_stats.max_abs, wmma_multim_stats.max_rel, wmma_multim_stats.rmse,
            pipeline_stats.elems, pipeline_stats.max_abs, pipeline_stats.max_rel, pipeline_stats.rmse);
        std::fclose(file);
    }

    HIP_CHECK(hipFree(d_q3));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_baseline));
    HIP_CHECK(hipFree(d_q3flash));
    HIP_CHECK(hipFree(d_q3flash_wide32));
    HIP_CHECK(hipFree(d_q3flash_wide64));
    HIP_CHECK(hipFree(d_wmma));
    HIP_CHECK(hipFree(d_wmma_reuse));
    HIP_CHECK(hipFree(d_wmma_kstage));
    HIP_CHECK(hipFree(d_wmma_multim));
    HIP_CHECK(hipFree(d_pipeline));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
