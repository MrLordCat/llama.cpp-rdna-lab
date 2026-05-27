// Standalone ROCm FFN pair+SwiGLU scout for P002 130k ROCm route research.
// This is a diagnostic tool only: it compares a fused gate/up Q3_K WMMA body
// against two dequantize-to-f16 + rocBLAS GEMMs followed by SwiGLU.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
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
    int n = 128;
    int k = 512;
    int warmup = 2;
    int iters = 6;
    int device = 0;
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

    return args.m > 0 && args.n > 0 && args.k > 0 && args.k % QK_K == 0 && args.warmup >= 0 && args.iters > 0;
}

static uint32_t lcg(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static void fill_q3(std::vector<block_q3_K_padded> & blocks, uint32_t seed) {
    uint32_t state = seed;
    for (block_q3_K_padded & block : blocks) {
        for (uint8_t & value : block.hmask) {
            value = static_cast<uint8_t>(lcg(state) >> 24);
        }
        for (uint8_t & value : block.qs) {
            value = static_cast<uint8_t>(lcg(state) >> 24);
        }
        for (uint8_t & value : block.scales) {
            value = static_cast<uint8_t>(lcg(state) >> 24);
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

static __device__ __forceinline__ float silu_device(float x) {
    return x / (1.0f + expf(-x));
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
    a[idx] = __float2half(q3_dequant_one(q3[block_index], col % QK_K));
}

__global__ static void swiglu_kernel(
        const float * __restrict__ up,
        const float * __restrict__ gate,
        float * __restrict__ out,
        const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float gate_value = gate[idx];
    out[idx] = up[idx] * silu_device(gate_value);
}

template<int nwaves_m, int nwaves_n, int tile_k>
__global__ static void q3_pairglu_wmma_kernel(
        const block_q3_K_padded * __restrict__ q3_up,
        const block_q3_K_padded * __restrict__ q3_gate,
        const __half * __restrict__ b,
        float * __restrict__ out,
        const int m,
        const int n,
        const int k) {
    using halfx8_t = __attribute__((ext_vector_type(8))) _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8))) float;

    static_assert(tile_k % 16 == 0, "tile_k must be a WMMA K multiple");

    constexpr int tile_m = 16 * nwaves_m;
    constexpr int tile_n = 16 * nwaves_n;
    __shared__ __half sh_up[tile_m * tile_k];
    __shared__ __half sh_gate[tile_m * tile_k];
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
    floatx8_t acc_up = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    floatx8_t acc_gate = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < k; k0 += tile_k) {
        for (int index = linear; index < tile_m * tile_k; index += threads) {
            const int local_row = index / tile_k;
            const int local_k = index - local_row * tile_k;
            const int a_row = tile_row + local_row;
            const int a_col = k0 + local_k;
            float up_value = 0.0f;
            float gate_value = 0.0f;
            if (a_row < m && a_col < k) {
                const int block_index = a_row * blocks_per_row + a_col / QK_K;
                const int element = a_col % QK_K;
                up_value = q3_dequant_one(q3_up[block_index], element);
                gate_value = q3_dequant_one(q3_gate[block_index], element);
            }
            sh_up[index] = __float2half(up_value);
            sh_gate[index] = __float2half(gate_value);
        }

        for (int index = linear; index < tile_n * tile_k; index += threads) {
            const int local_n = index / tile_k;
            const int local_k = index - local_n * tile_k;
            const int b_row = k0 + local_k;
            const int b_col = tile_col + local_n;
            float b_value = 0.0f;
            if (b_row < k && b_col < n) {
                b_value = __half2float(b[b_row + b_col * k]);
            }
            sh_b[index] = __float2half(b_value);
        }

        __syncthreads();

        for (int kk = 0; kk < tile_k; kk += 16) {
            halfx8_t up_frag;
            halfx8_t gate_frag;
            halfx8_t b_frag;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int local_k = kk + frag_col0 + i;
                const int local_row = wave_m * 16 + frag_row;
                const int local_col = wave_n * 16 + frag_row;
                up_frag[i] = static_cast<_Float16>(__half2float(sh_up[local_row * tile_k + local_k]));
                gate_frag[i] = static_cast<_Float16>(__half2float(sh_gate[local_row * tile_k + local_k]));
                b_frag[i] = static_cast<_Float16>(__half2float(sh_b[local_col * tile_k + local_k]));
            }

            acc_up = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(up_frag, b_frag, acc_up);
            acc_gate = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(gate_frag, b_frag, acc_gate);
        }

        __syncthreads();
    }

    const int out_col = tile_col + wave_n * 16 + (lane % 16);
    const int out_row0 = tile_row + wave_m * 16 + 8 * (lane / 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int out_row = out_row0 + i;
        if (out_row < m && out_col < n) {
            const float gate_value = acc_gate[i];
            out[out_row + out_col * m] = acc_up[i] * silu_device(gate_value);
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

static int time_baseline_pair(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3_up,
        const block_q3_K_padded * q3_gate,
        __half * a_up,
        __half * a_gate,
        const __half * b,
        float * d_up,
        float * d_gate,
        float * d_out,
        double & avg_ms) {
    const int total_a = args.m * args.k;
    const int total_out = args.m * args.n;
    const dim3 dequant_block(256);
    const dim3 dequant_grid((total_a + dequant_block.x - 1) / dequant_block.x);
    const dim3 glu_block(256);
    const dim3 glu_grid((total_out + glu_block.x - 1) / glu_block.x);

    auto run_once = [&]() -> int {
        dequant_q3_to_f16_kernel<<<dequant_grid, dequant_block, 0, stream>>>(q3_up, a_up, args.m, args.k);
        HIP_CHECK(hipGetLastError());
        dequant_q3_to_f16_kernel<<<dequant_grid, dequant_block, 0, stream>>>(q3_gate, a_gate, args.m, args.k);
        HIP_CHECK(hipGetLastError());
        ROCBLAS_CHECK(gemm(handle, args, a_up, b, d_up));
        ROCBLAS_CHECK(gemm(handle, args, a_gate, b, d_gate));
        swiglu_kernel<<<glu_grid, glu_block, 0, stream>>>(d_up, d_gate, d_out, total_out);
        HIP_CHECK(hipGetLastError());
        return 0;
    };

    for (int i = 0; i < args.warmup; ++i) {
        if (run_once() != 0) {
            return 1;
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        if (run_once() != 0) {
            return 1;
        }
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

template<int nwaves_m, int nwaves_n, int tile_k>
static int time_pairglu_wmma(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * q3_up,
        const block_q3_K_padded * q3_gate,
        const __half * b,
        float * d_out,
        double & avg_ms) {
    const dim3 block(32, nwaves_m * nwaves_n);
    const dim3 grid((args.n + 16 * nwaves_n - 1) / (16 * nwaves_n), (args.m + 16 * nwaves_m - 1) / (16 * nwaves_m));

    for (int i = 0; i < args.warmup; ++i) {
        q3_pairglu_wmma_kernel<nwaves_m, nwaves_n, tile_k><<<grid, block, 0, stream>>>(
            q3_up, q3_gate, b, d_out, args.m, args.n, args.k);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        q3_pairglu_wmma_kernel<nwaves_m, nwaves_n, tile_k><<<grid, block, 0, stream>>>(
            q3_up, q3_gate, b, d_out, args.m, args.n, args.k);
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

    std::vector<block_q3_K_padded> h_q3_up(q3_blocks);
    std::vector<block_q3_K_padded> h_q3_gate(q3_blocks);
    std::vector<__half> h_b(b_elems);
    fill_q3(h_q3_up, 0x12345678u);
    fill_q3(h_q3_gate, 0x31415926u);
    fill_half(h_b);

    block_q3_K_padded * d_q3_up = nullptr;
    block_q3_K_padded * d_q3_gate = nullptr;
    __half * d_a_up = nullptr;
    __half * d_a_gate = nullptr;
    __half * d_b = nullptr;
    float * d_up = nullptr;
    float * d_gate = nullptr;
    float * d_baseline = nullptr;
    float * d_pair64 = nullptr;

    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_q3_up), q3_blocks * sizeof(block_q3_K_padded)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_q3_gate), q3_blocks * sizeof(block_q3_K_padded)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_a_up), a_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_a_gate), a_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_b), b_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_up), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_gate), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_baseline), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_pair64), d_elems * sizeof(float)));

    HIP_CHECK(hipMemcpyAsync(d_q3_up, h_q3_up.data(), h_q3_up.size() * sizeof(block_q3_K_padded), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_q3_gate, h_q3_gate.data(), h_q3_gate.size() * sizeof(block_q3_K_padded), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(), h_b.size() * sizeof(__half), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double baseline_ms = 0.0;
    double pair64_ms = 0.0;
    if (time_baseline_pair(handle, stream, args, d_q3_up, d_q3_gate, d_a_up, d_a_gate, d_b, d_up, d_gate, d_baseline, baseline_ms) != 0) {
        return 1;
    }
    if (time_pairglu_wmma<4, 4, 128>(stream, args, d_q3_up, d_q3_gate, d_b, d_pair64, pair64_ms) != 0) {
        return 1;
    }

    compare_stats_t pair64_stats;
    if (d_elems <= args.max_check_elems) {
        std::vector<float> h_baseline(d_elems);
        std::vector<float> h_pair64(d_elems);
        HIP_CHECK(hipMemcpy(h_baseline.data(), d_baseline, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_pair64.data(), d_pair64, d_elems * sizeof(float), hipMemcpyDeviceToHost));
        pair64_stats = compare_outputs(h_baseline, h_pair64);
    }

    const double pair64_speedup = pair64_ms > 0.0 ? baseline_ms / pair64_ms : 0.0;
    std::cout << std::fixed << std::setprecision(4)
              << "m=" << args.m << " n=" << args.n << " k=" << args.k
              << " baseline_pair_ms=" << baseline_ms
              << " pair64_ms=" << pair64_ms
              << " pair64_speedup=" << pair64_speedup
              << " pair64_checked=" << pair64_stats.elems
              << " pair64_max_abs=" << pair64_stats.max_abs
              << " pair64_max_rel=" << pair64_stats.max_rel
              << " pair64_rmse=" << pair64_stats.rmse
              << std::endl;

    if (!args.csv.empty()) {
        FILE * file = std::fopen(args.csv.c_str(), "w");
        if (file == nullptr) {
            std::cerr << "failed to open csv: " << args.csv << std::endl;
            return 1;
        }
        std::fprintf(file, "m,n,k,warmup,iters,baseline_pair_ms,pair64_ms,pair64_speedup,pair64_checked_elems,pair64_max_abs,pair64_max_rel,pair64_rmse\n");
        std::fprintf(file, "%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%zu,%.9f,%.9f,%.9f\n",
            args.m, args.n, args.k, args.warmup, args.iters,
            baseline_ms, pair64_ms, pair64_speedup,
            pair64_stats.elems, pair64_stats.max_abs, pair64_stats.max_rel, pair64_stats.rmse);
        std::fclose(file);
    }

    HIP_CHECK(hipFree(d_q3_up));
    HIP_CHECK(hipFree(d_q3_gate));
    HIP_CHECK(hipFree(d_a_up));
    HIP_CHECK(hipFree(d_a_gate));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_up));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_baseline));
    HIP_CHECK(hipFree(d_pair64));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}