// Standalone ROCm compact Q3_K layout scout.
// Research harness only; not part of normal builds.

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

struct block_q3_K_compact160 {
    uint8_t qn[QK_K / 2];
    int8_t scales[16];
    __half d;
    uint8_t pad[14];
};
static_assert(sizeof(block_q3_K_compact160) == 160, "wrong compact q3_K block size");

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
    std::cerr << "usage: " << argv0
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
    uint32_t state = 0x5a17c0deu;
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

static int get_int_b2_host(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2 * i32 + 0] << 0;
    x32     |= x16[2 * i32 + 1] << 16;
    return x32;
}

static int pack_i8x4_host(const int8_t * values) {
    int out = 0;
    for (int i = 0; i < 4; ++i) {
        out |= static_cast<int>(static_cast<uint8_t>(values[i])) << (8 * i);
    }
    return out;
}

static void compact_set_q(uint8_t * qn, const int element, const uint8_t q) {
    uint8_t & byte = qn[element / 2];
    if ((element & 1) == 0) {
        byte = (byte & 0xF0u) | (q & 0x0Fu);
    } else {
        byte = (byte & 0x0Fu) | static_cast<uint8_t>((q & 0x0Fu) << 4);
    }
}

static void pack_to_compact(
        const std::vector<block_q3_K_padded> & raw,
        std::vector<block_q3_K_compact160> & compact) {
    compact.resize(raw.size());
    for (size_t block_index = 0; block_index < raw.size(); ++block_index) {
        const block_q3_K_padded & src = raw[block_index];
        block_q3_K_compact160 & dst = compact[block_index];
        std::memset(&dst, 0, sizeof(dst));
        dst.d = src.d;

        for (int kqsx = 0; kqsx < QI3_K; ++kqsx) {
            const int x_ql_0 = get_int_b2_host(src.qs, kqsx);
            const int x_qh_raw = get_int_b2_host(src.hmask, kqsx % (QI3_K / 2));
            const int x_qh_0 = x_qh_raw >> (4 * (kqsx / (QI3_K / 2)));

            for (int l = 0; l < QR3_K; ++l) {
                const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;
                const int x_ql_k =  (x_ql_0 >> (2 * l))      & 0x03030303;
                const int x_qh_k = ((x_qh_0 >>      l) << 2) & 0x04040404;
                const int q4 = x_ql_k | x_qh_k;
                for (int lane = 0; lane < 4; ++lane) {
                    const uint8_t q = static_cast<uint8_t>((q4 >> (8 * lane)) & 0x07);
                    compact_set_q(dst.qn, 4 * k + lane, q);
                }
            }
        }

        for (int ksc = 0; ksc < QI3_K / 4; ++ksc) {
            const int ksc_low = ksc % (QI3_K / 8);
            const int shift_low = 4 * (ksc / (QI3_K / 8));
            const int sc_low_raw = get_int_b2_host(src.scales, ksc_low);
            const int sc_low = (sc_low_raw >> shift_low) & 0x0F0F0F0F;
            const int ksc_high = QI3_K / 8;
            const int shift_high = 2 * ksc;
            const int sc_high_raw = get_int_b2_host(src.scales, ksc_high);
            const int sc_high = (sc_high_raw >> shift_high << 4) & 0x30303030;
            for (int l = 0; l < 4; ++l) {
                const int packed = (sc_low | sc_high) >> (8 * l);
                dst.scales[4 * ksc + l] = static_cast<int8_t>((packed & 0x3F) - 32);
            }
        }
    }
}

static __device__ __forceinline__ int get_int_b2_device(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2 * i32 + 0] << 0;
    x32     |= x16[2 * i32 + 1] << 16;
    return x32;
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

static __device__ __forceinline__ int pack_compact_q4_device(const uint8_t * qn, const int k) {
    int out = 0;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const int element = 4 * k + lane;
        const uint8_t byte = qn[element / 2];
        const int q = (element & 1) == 0 ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        const int8_t signed_q = static_cast<int8_t>(q - 4);
        out |= static_cast<int>(static_cast<uint8_t>(signed_q)) << (8 * lane);
    }
    return out;
}

__global__ static void unpack_raw_kernel(
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
            const int x_ql_0 = get_int_b2_device(bxi->qs, kqsx);
            const int x_qh_raw = get_int_b2_device(bxi->hmask, kqsx % (QI3_K / 2));
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
            const int sc_low_raw = get_int_b2_device(bxi->scales, ksc_low);
            const int sc_low = (sc_low_raw >> shift_low) & 0x0F0F0F0F;
            const int ksc_high = QI3_K / 8;
            const int shift_high = 2 * ksc;
            const int sc_high_raw = get_int_b2_device(bxi->scales, ksc_high);
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

__global__ static void unpack_compact_kernel(
        const block_q3_K_compact160 * __restrict__ src,
        int * __restrict__ qs_out,
        float * __restrict__ df_out,
        const int nblocks) {
    const int rows_per_cta = 8;
    const int base = blockIdx.x * rows_per_cta;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < rows_per_cta * 64; idx += blockDim.x) {
        const int row = idx / 64;
        const int k = idx - row * 64;
        const int block_index = base + row;
        if (block_index < nblocks) {
            qs_out[block_index * 64 + k] = pack_compact_q4_device(src[block_index].qn, k);
        }
    }

    for (int idx = tid; idx < rows_per_cta * 16; idx += blockDim.x) {
        const int row = idx / 16;
        const int scale_index = idx - row * 16;
        const int block_index = base + row;
        if (block_index < nblocks) {
            const block_q3_K_compact160 * bxi = src + block_index;
            df_out[block_index * 16 + scale_index] = __half2float(bxi->d) * static_cast<float>(bxi->scales[scale_index]);
        }
    }
}

__global__ static void unpack_raw_shared_kernel(
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
            const int x_ql_0 = get_int_b2_device(bxi->qs, kqsx);
            const int x_qh_raw = get_int_b2_device(bxi->hmask, kqsx % (QI3_K / 2));
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
            const int sc_low_raw = get_int_b2_device(bxi->scales, ksc_low);
            const int sc_low = (sc_low_raw >> shift_low) & 0x0F0F0F0F;
            const int ksc_high = QI3_K / 8;
            const int shift_high = 2 * ksc;
            const int sc_high_raw = get_int_b2_device(bxi->scales, ksc_high);
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

__global__ static void unpack_compact_shared_kernel(
        const block_q3_K_compact160 * __restrict__ src,
        int * __restrict__ sink,
        const int nblocks) {
    __shared__ int qs_tile[8 * 64];
    __shared__ float df_tile[8 * 16];
    const int rows_per_cta = 8;
    const int base = blockIdx.x * rows_per_cta;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < rows_per_cta * 64; idx += blockDim.x) {
        const int row = idx / 64;
        const int k = idx - row * 64;
        const int block_index = base + row;
        if (block_index < nblocks) {
            qs_tile[row * 64 + k] = pack_compact_q4_device(src[block_index].qn, k);
        }
    }

    for (int idx = tid; idx < rows_per_cta * 16; idx += blockDim.x) {
        const int row = idx / 16;
        const int scale_index = idx - row * 16;
        const int block_index = base + row;
        if (block_index < nblocks) {
            const block_q3_K_compact160 * bxi = src + block_index;
            df_tile[row * 16 + scale_index] = __half2float(bxi->d) * static_cast<float>(bxi->scales[scale_index]);
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

template<typename Src, typename Kernel>
static int time_unpack(
        hipStream_t stream,
        const args_t & args,
        Kernel kernel,
        const Src * src,
        int * qs_out,
        float * df_out,
        double & avg_ms) {
    const int rows_per_cta = 8;
    const dim3 block(128);
    const dim3 grid((args.blocks + rows_per_cta - 1) / rows_per_cta);
    for (int i = 0; i < args.warmup; ++i) {
        kernel<<<grid, block, 0, stream>>>(src, qs_out, df_out, args.blocks);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        kernel<<<grid, block, 0, stream>>>(src, qs_out, df_out, args.blocks);
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

template<typename Src, typename Kernel>
static int time_shared(
        hipStream_t stream,
        const args_t & args,
        Kernel kernel,
        const Src * src,
        int * sink,
        double & avg_ms) {
    const int rows_per_cta = 8;
    const dim3 block(128);
    const dim3 grid((args.blocks + rows_per_cta - 1) / rows_per_cta);
    for (int i = 0; i < args.warmup; ++i) {
        kernel<<<grid, block, 0, stream>>>(src, sink, args.blocks);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        kernel<<<grid, block, 0, stream>>>(src, sink, args.blocks);
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
        const int * qs_raw,
        const int * qs_compact,
        const float * df_raw,
        const float * df_compact) {
    const int check_blocks = std::min(args.blocks, args.check_blocks);
    int qs_mismatches = 0;
    int df_mismatches = 0;
    double max_abs = 0.0;

    for (int i = 0; i < check_blocks * 64; ++i) {
        if (qs_raw[i] != qs_compact[i]) {
            ++qs_mismatches;
            if (qs_mismatches > 16) break;
        }
    }

    for (int i = 0; i < check_blocks * 16; ++i) {
        const double diff = std::abs(static_cast<double>(df_raw[i]) - static_cast<double>(df_compact[i]));
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

    std::vector<block_q3_K_padded> h_raw(args.blocks);
    std::vector<block_q3_K_compact160> h_compact;
    fill_q3(h_raw);
    pack_to_compact(h_raw, h_compact);

    block_q3_K_padded * d_raw = nullptr;
    block_q3_K_compact160 * d_compact = nullptr;
    int * d_qs_raw = nullptr;
    int * d_qs_compact = nullptr;
    int * d_sink = nullptr;
    float * d_df_raw = nullptr;
    float * d_df_compact = nullptr;

    const size_t raw_bytes = h_raw.size() * sizeof(block_q3_K_padded);
    const size_t compact_bytes = h_compact.size() * sizeof(block_q3_K_compact160);
    const size_t qs_bytes = static_cast<size_t>(args.blocks) * 64 * sizeof(int);
    const size_t df_bytes = static_cast<size_t>(args.blocks) * 16 * sizeof(float);
    const size_t sink_bytes = static_cast<size_t>((args.blocks + 7) / 8) * 128 * sizeof(int);

    HIP_CHECK(hipMalloc(&d_raw, raw_bytes));
    HIP_CHECK(hipMalloc(&d_compact, compact_bytes));
    HIP_CHECK(hipMalloc(&d_qs_raw, qs_bytes));
    HIP_CHECK(hipMalloc(&d_qs_compact, qs_bytes));
    HIP_CHECK(hipMalloc(&d_sink, sink_bytes));
    HIP_CHECK(hipMalloc(&d_df_raw, df_bytes));
    HIP_CHECK(hipMalloc(&d_df_compact, df_bytes));
    HIP_CHECK(hipMemcpyAsync(d_raw, h_raw.data(), raw_bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_compact, h_compact.data(), compact_bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double raw_ms = 0.0;
    double compact_ms = 0.0;
    double raw_shared_ms = 0.0;
    double compact_shared_ms = 0.0;
    if (time_unpack(stream, args, unpack_raw_kernel, d_raw, d_qs_raw, d_df_raw, raw_ms) != 0) return 1;
    if (time_unpack(stream, args, unpack_compact_kernel, d_compact, d_qs_compact, d_df_compact, compact_ms) != 0) return 1;
    if (time_shared(stream, args, unpack_raw_shared_kernel, d_raw, d_sink, raw_shared_ms) != 0) return 1;
    if (time_shared(stream, args, unpack_compact_shared_kernel, d_compact, d_sink, compact_shared_ms) != 0) return 1;

    const int check_blocks = std::min(args.blocks, args.check_blocks);
    std::vector<int> h_qs_raw(static_cast<size_t>(check_blocks) * 64);
    std::vector<int> h_qs_compact(static_cast<size_t>(check_blocks) * 64);
    std::vector<float> h_df_raw(static_cast<size_t>(check_blocks) * 16);
    std::vector<float> h_df_compact(static_cast<size_t>(check_blocks) * 16);
    if (check_blocks > 0) {
        HIP_CHECK(hipMemcpy(h_qs_raw.data(), d_qs_raw, h_qs_raw.size() * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_qs_compact.data(), d_qs_compact, h_qs_compact.size() * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_df_raw.data(), d_df_raw, h_df_raw.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_df_compact.data(), d_df_compact, h_df_compact.size() * sizeof(float), hipMemcpyDeviceToHost));
    }

    const double raw_total_gb = static_cast<double>(raw_bytes + qs_bytes + df_bytes) / 1.0e9;
    const double compact_total_gb = static_cast<double>(compact_bytes + qs_bytes + df_bytes) / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "blocks=" << args.blocks
              << " warmup=" << args.warmup
              << " iters=" << args.iters
              << " raw_src_mb=" << raw_bytes / 1048576.0
              << " compact_src_mb=" << compact_bytes / 1048576.0
              << " out_mb=" << (qs_bytes + df_bytes) / 1048576.0 << "\n";
    std::cout << "raw_unpack_ms=" << raw_ms
              << " throughput_gb_s=" << (raw_total_gb / (raw_ms / 1000.0)) << "\n";
    std::cout << "compact_unpack_ms=" << compact_ms
              << " throughput_gb_s=" << (compact_total_gb / (compact_ms / 1000.0)) << "\n";
    std::cout << "compact_vs_raw_unpack_speedup=" << (compact_ms > 0.0 ? raw_ms / compact_ms : 0.0) << "x\n";
    std::cout << "raw_shared_ms=" << raw_shared_ms << "\n";
    std::cout << "compact_shared_ms=" << compact_shared_ms << "\n";
    std::cout << "compact_vs_raw_shared_speedup=" << (compact_shared_ms > 0.0 ? raw_shared_ms / compact_shared_ms : 0.0) << "x\n";

    const int compare_status = check_blocks > 0 ? compare_outputs(args, h_qs_raw.data(), h_qs_compact.data(), h_df_raw.data(), h_df_compact.data()) : 0;

    HIP_CHECK(hipFree(d_raw));
    HIP_CHECK(hipFree(d_compact));
    HIP_CHECK(hipFree(d_qs_raw));
    HIP_CHECK(hipFree(d_qs_compact));
    HIP_CHECK(hipFree(d_sink));
    HIP_CHECK(hipFree(d_df_raw));
    HIP_CHECK(hipFree(d_df_compact));
    HIP_CHECK(hipStreamDestroy(stream));
    return compare_status;
}