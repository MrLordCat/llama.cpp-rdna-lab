// Standalone ROCm Q3_K MMQ unpack scout.
// This is a research harness only; it does not participate in normal builds.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define QK_K 256
#define QR3_K 4
#define QI3_K (QK_K / (4 * QR3_K))

struct block_q3_K_padded {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    __half d;
    uint8_t pad[2];
};
static_assert(sizeof(block_q3_K_padded) == 112, "wrong padded q3_K block size");

#define HIP_CHECK(expr)                                                                          \
    do {                                                                                         \
        hipError_t status__ = (expr);                                                            \
        if (status__ != hipSuccess) {                                                            \
            std::cerr << "HIP error: " << hipGetErrorString(status__) << " at line "          \
                      << __LINE__ << std::endl;                                                  \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

struct args_t {
    int blocks = 262144;
    int warmup = 5;
    int iters = 50;
    int device = 0;
    int check_blocks = 4096;
};

static void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0
        << " [--blocks N] [--warmup N] [--iters N] [--device N] [--check-blocks N]\n";
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

        if (key == "--blocks") {
            if (!need_int(args.blocks)) return false;
        } else if (key == "--warmup") {
            if (!need_int(args.warmup)) return false;
        } else if (key == "--iters") {
            if (!need_int(args.iters)) return false;
        } else if (key == "--device") {
            if (!need_int(args.device)) return false;
        } else if (key == "--check-blocks") {
            if (!need_int(args.check_blocks)) return false;
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            return false;
        }
    }

    return args.blocks > 0 && args.warmup >= 0 && args.iters > 0 && args.check_blocks >= 0;
}

static uint32_t lcg(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static void fill_q3(std::vector<block_q3_K_padded> & blocks) {
    uint32_t state = 0x623d9aefu;
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
        block.d = __float2half(0.0005f * static_cast<float>((lcg(state) % 31u) + 1u));
        block.pad[0] = 0;
        block.pad[1] = 0;
    }
}

static __device__ __forceinline__ int get_int_b2_device(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2 * i32 + 0] << 0;
    x32     |= x16[2 * i32 + 1] << 16;
    return x32;
}

static __device__ __forceinline__ int get_int_b4_device(const void * x, const int i32) {
    return ((const int *) x)[i32];
}

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int vsubss4_device(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t &>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t &>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t vc = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(vc);
#else
    int8x4_t vc;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int tmp = static_cast<int>(va[i]) - static_cast<int>(vb[i]);
        tmp = tmp > 127 ? 127 : tmp;
        tmp = tmp < -128 ? -128 : tmp;
        vc[i] = static_cast<int8_t>(tmp);
    }
    return reinterpret_cast<int &>(vc);
#endif
}

template<bool use_b4_loads>
__global__ static void unpack_q3k_mmq_tile_kernel(
        const block_q3_K_padded * __restrict__ src,
        int * __restrict__ qs_out,
        float * __restrict__ df_out,
        const int nblocks) {
    const int rows_per_cta = 8;
    const int base = blockIdx.x * rows_per_cta;
    const int tid = threadIdx.x;

    if (tid < rows_per_cta * 16) {
        const int row = tid / 16;
        const int kqsx = tid - row * 16;
        const int block_index = base + row;
        if (block_index < nblocks) {
            const block_q3_K_padded * bxi = src + block_index;
            const int x_ql_0 = use_b4_loads ? get_int_b4_device(bxi->qs, kqsx) : get_int_b2_device(bxi->qs, kqsx);
            const int x_qh_raw = use_b4_loads ? get_int_b4_device(bxi->hmask, kqsx % (QI3_K / 2)) :
                                                get_int_b2_device(bxi->hmask, kqsx % (QI3_K / 2));
            const int x_qh_0 = x_qh_raw >> (4 * (kqsx / (QI3_K / 2)));

#pragma unroll
            for (int l = 0; l < QR3_K; ++l) {
                const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
                const int x_ql_k =  (x_ql_0 >> (2 * l))      & 0x03030303;
                const int x_qh_k = ((x_qh_0 >>      l) << 2) & 0x04040404;
                qs_out[block_index * 64 + k] = vsubss4_device(x_ql_k | x_qh_k, 0x04040404);
            }
        }
    }

    if (tid < rows_per_cta * 4) {
        const int row = tid / 4;
        const int ksc = tid - row * 4;
        const int block_index = base + row;
        if (block_index < nblocks) {
            const block_q3_K_padded * bxi = src + block_index;
            const int ksc_low = ksc % (QI3_K / 8);
            const int shift_low = 4 * (ksc / (QI3_K / 8));
            const int sc_low_raw = use_b4_loads ? get_int_b4_device(bxi->scales, ksc_low) :
                                                  get_int_b2_device(bxi->scales, ksc_low);
            const int sc_low = (sc_low_raw >> shift_low) & 0x0F0F0F0F;

            const int ksc_high = QI3_K / 8;
            const int shift_high = 2 * ksc;
            const int sc_high_raw = use_b4_loads ? get_int_b4_device(bxi->scales, ksc_high) :
                                                   get_int_b2_device(bxi->scales, ksc_high);
            const int sc_high = (sc_high_raw >> shift_high << 4) & 0x30303030;
            const int sc = vsubss4_device(sc_low | sc_high, 0x20202020);
            const int8_t * sc8 = (const int8_t *) &sc;
            const float d = __half2float(bxi->d);

#pragma unroll
            for (int l = 0; l < 4; ++l) {
                df_out[block_index * 16 + 4 * ksc + l] = d * static_cast<float>(sc8[l]);
            }
        }
    }
}

template<bool use_b4_loads>
__global__ static void unpack_q3k_mmq_tile_shared_kernel(
        const block_q3_K_padded * __restrict__ src,
        int * __restrict__ sink,
        const int nblocks) {
    __shared__ int qs_tile[8 * 64];
    __shared__ float df_tile[8 * 16];

    const int rows_per_cta = 8;
    const int base = blockIdx.x * rows_per_cta;
    const int tid = threadIdx.x;

    if (tid < rows_per_cta * 16) {
        const int row = tid / 16;
        const int kqsx = tid - row * 16;
        const int block_index = base + row;
        if (block_index < nblocks) {
            const block_q3_K_padded * bxi = src + block_index;
            const int x_ql_0 = use_b4_loads ? get_int_b4_device(bxi->qs, kqsx) : get_int_b2_device(bxi->qs, kqsx);
            const int x_qh_raw = use_b4_loads ? get_int_b4_device(bxi->hmask, kqsx % (QI3_K / 2)) :
                                                get_int_b2_device(bxi->hmask, kqsx % (QI3_K / 2));
            const int x_qh_0 = x_qh_raw >> (4 * (kqsx / (QI3_K / 2)));

#pragma unroll
            for (int l = 0; l < QR3_K; ++l) {
                const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
                const int x_ql_k =  (x_ql_0 >> (2 * l))      & 0x03030303;
                const int x_qh_k = ((x_qh_0 >>      l) << 2) & 0x04040404;
                qs_tile[row * 64 + k] = vsubss4_device(x_ql_k | x_qh_k, 0x04040404);
            }
        }
    }

    if (tid < rows_per_cta * 4) {
        const int row = tid / 4;
        const int ksc = tid - row * 4;
        const int block_index = base + row;
        if (block_index < nblocks) {
            const block_q3_K_padded * bxi = src + block_index;
            const int ksc_low = ksc % (QI3_K / 8);
            const int shift_low = 4 * (ksc / (QI3_K / 8));
            const int sc_low_raw = use_b4_loads ? get_int_b4_device(bxi->scales, ksc_low) :
                                                  get_int_b2_device(bxi->scales, ksc_low);
            const int sc_low = (sc_low_raw >> shift_low) & 0x0F0F0F0F;

            const int ksc_high = QI3_K / 8;
            const int shift_high = 2 * ksc;
            const int sc_high_raw = use_b4_loads ? get_int_b4_device(bxi->scales, ksc_high) :
                                                   get_int_b2_device(bxi->scales, ksc_high);
            const int sc_high = (sc_high_raw >> shift_high << 4) & 0x30303030;
            const int sc = vsubss4_device(sc_low | sc_high, 0x20202020);
            const int8_t * sc8 = (const int8_t *) &sc;
            const float d = __half2float(bxi->d);

#pragma unroll
            for (int l = 0; l < 4; ++l) {
                df_tile[row * 16 + 4 * ksc + l] = d * static_cast<float>(sc8[l]);
            }
        }
    }

    __syncthreads();

    int acc = 0;
    for (int i = tid; i < rows_per_cta * 64; i += blockDim.x) {
        acc ^= qs_tile[i];
    }
    for (int i = tid; i < rows_per_cta * 16; i += blockDim.x) {
        acc ^= __float_as_int(df_tile[i]);
    }
    sink[blockIdx.x * blockDim.x + tid] = acc;
}

template<bool use_b4_loads>
static int time_unpack(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * src,
        int * qs_out,
        float * df_out,
        double & avg_ms) {
    const int rows_per_cta = 8;
    const dim3 block(128);
    const dim3 grid((args.blocks + rows_per_cta - 1) / rows_per_cta);

    for (int i = 0; i < args.warmup; ++i) {
        unpack_q3k_mmq_tile_kernel<use_b4_loads><<<grid, block, 0, stream>>>(src, qs_out, df_out, args.blocks);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        unpack_q3k_mmq_tile_kernel<use_b4_loads><<<grid, block, 0, stream>>>(src, qs_out, df_out, args.blocks);
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

template<bool use_b4_loads>
static int time_unpack_shared(
        hipStream_t stream,
        const args_t & args,
        const block_q3_K_padded * src,
        int * sink,
        double & avg_ms) {
    const int rows_per_cta = 8;
    const dim3 block(128);
    const dim3 grid((args.blocks + rows_per_cta - 1) / rows_per_cta);

    for (int i = 0; i < args.warmup; ++i) {
        unpack_q3k_mmq_tile_shared_kernel<use_b4_loads><<<grid, block, 0, stream>>>(src, sink, args.blocks);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        unpack_q3k_mmq_tile_shared_kernel<use_b4_loads><<<grid, block, 0, stream>>>(src, sink, args.blocks);
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

static int compare_outputs(
        const args_t & args,
        const int * qs_b2,
        const int * qs_b4,
        const float * df_b2,
        const float * df_b4) {
    const int check_blocks = std::min(args.blocks, args.check_blocks);
    int qs_mismatches = 0;
    int df_mismatches = 0;
    double max_abs = 0.0;

    for (int i = 0; i < check_blocks * 64; ++i) {
        if (qs_b2[i] != qs_b4[i]) {
            ++qs_mismatches;
            if (qs_mismatches > 16) break;
        }
    }

    for (int i = 0; i < check_blocks * 16; ++i) {
        const double diff = std::abs(static_cast<double>(df_b2[i]) - static_cast<double>(df_b4[i]));
        max_abs = std::max(max_abs, diff);
        if (diff > 0.0) {
            ++df_mismatches;
            if (df_mismatches > 16) break;
        }
    }

    std::cout << "correctness check_blocks=" << check_blocks
              << " qs_mismatches=" << qs_mismatches
              << " df_mismatches=" << df_mismatches
              << " max_abs=" << max_abs << "\n";

    return qs_mismatches == 0 && df_mismatches == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    args_t args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    HIP_CHECK(hipSetDevice(args.device));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    std::vector<block_q3_K_padded> h_src(args.blocks);
    fill_q3(h_src);

    block_q3_K_padded * d_src = nullptr;
    int * d_qs_b2 = nullptr;
    int * d_qs_b4 = nullptr;
    int * d_sink = nullptr;
    float * d_df_b2 = nullptr;
    float * d_df_b4 = nullptr;
    const size_t src_bytes = h_src.size() * sizeof(block_q3_K_padded);
    const size_t qs_bytes = static_cast<size_t>(args.blocks) * 64 * sizeof(int);
    const size_t df_bytes = static_cast<size_t>(args.blocks) * 16 * sizeof(float);
    const size_t sink_bytes = static_cast<size_t>((args.blocks + 7) / 8) * 128 * sizeof(int);

    HIP_CHECK(hipMalloc(&d_src, src_bytes));
    HIP_CHECK(hipMalloc(&d_qs_b2, qs_bytes));
    HIP_CHECK(hipMalloc(&d_qs_b4, qs_bytes));
    HIP_CHECK(hipMalloc(&d_sink, sink_bytes));
    HIP_CHECK(hipMalloc(&d_df_b2, df_bytes));
    HIP_CHECK(hipMalloc(&d_df_b4, df_bytes));
    HIP_CHECK(hipMemcpyAsync(d_src, h_src.data(), src_bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double b2_ms = 0.0;
    double b4_ms = 0.0;
    double b2_shared_ms = 0.0;
    double b4_shared_ms = 0.0;
    if (time_unpack<false>(stream, args, d_src, d_qs_b2, d_df_b2, b2_ms) != 0) return 1;
    if (time_unpack<true>(stream, args, d_src, d_qs_b4, d_df_b4, b4_ms) != 0) return 1;
    if (time_unpack_shared<false>(stream, args, d_src, d_sink, b2_shared_ms) != 0) return 1;
    if (time_unpack_shared<true>(stream, args, d_src, d_sink, b4_shared_ms) != 0) return 1;

    const int check_blocks = std::min(args.blocks, args.check_blocks);
    std::vector<int> h_qs_b2(static_cast<size_t>(check_blocks) * 64);
    std::vector<int> h_qs_b4(static_cast<size_t>(check_blocks) * 64);
    std::vector<float> h_df_b2(static_cast<size_t>(check_blocks) * 16);
    std::vector<float> h_df_b4(static_cast<size_t>(check_blocks) * 16);
    if (check_blocks > 0) {
        HIP_CHECK(hipMemcpy(h_qs_b2.data(), d_qs_b2, h_qs_b2.size() * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_qs_b4.data(), d_qs_b4, h_qs_b4.size() * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_df_b2.data(), d_df_b2, h_df_b2.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_df_b4.data(), d_df_b4, h_df_b4.size() * sizeof(float), hipMemcpyDeviceToHost));
    }

    const size_t bytes_written = qs_bytes + df_bytes;
    const size_t bytes_read = src_bytes;
    const double total_gb = static_cast<double>(bytes_read + bytes_written) / 1.0e9;
    const double speedup = b4_ms > 0.0 ? b2_ms / b4_ms : 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "blocks=" << args.blocks
              << " warmup=" << args.warmup
              << " iters=" << args.iters
              << " src_mb=" << src_bytes / 1048576.0
              << " out_mb=" << bytes_written / 1048576.0 << "\n";
    std::cout << "b2_loads_ms=" << b2_ms
              << " throughput_gb_s=" << (total_gb / (b2_ms / 1000.0)) << "\n";
    std::cout << "b4_loads_ms=" << b4_ms
              << " throughput_gb_s=" << (total_gb / (b4_ms / 1000.0)) << "\n";
    std::cout << "b4_vs_b2_speedup=" << speedup << "x\n";
    std::cout << "shared_b2_loads_ms=" << b2_shared_ms << "\n";
    std::cout << "shared_b4_loads_ms=" << b4_shared_ms << "\n";
    std::cout << "shared_b4_vs_b2_speedup=" << (b4_shared_ms > 0.0 ? b2_shared_ms / b4_shared_ms : 0.0) << "x\n";

    const int compare_status = check_blocks > 0 ? compare_outputs(args, h_qs_b2.data(), h_qs_b4.data(), h_df_b2.data(), h_df_b4.data()) : 0;

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_qs_b2));
    HIP_CHECK(hipFree(d_qs_b4));
    HIP_CHECK(hipFree(d_sink));
    HIP_CHECK(hipFree(d_df_b2));
    HIP_CHECK(hipFree(d_df_b4));
    HIP_CHECK(hipStreamDestroy(stream));
    return compare_status;
}