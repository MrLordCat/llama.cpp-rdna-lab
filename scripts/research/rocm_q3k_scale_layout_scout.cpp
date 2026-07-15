// Standalone ROCm Q3_K fused-pair decode layout scout.
// Research harness only; not part of normal builds.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

constexpr int QK_K  = 256;
constexpr int QK8_1 = 32;
constexpr int QI3_K = 16;
constexpr int QI8_1 = 8;
constexpr int QR3_K = 4;

struct block_q3_K_padded {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    __half d;
    uint8_t pad[2];
};
static_assert(sizeof(block_q3_K_padded) == 112, "wrong padded Q3_K block size");

struct block_q3_K_scale16 {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    int8_t scales[16];
    __half d;
    uint8_t pad[2];
};
static_assert(sizeof(block_q3_K_scale16) == 116, "wrong expanded-scale Q3_K block size");

struct block_q8_1 {
    __half d;
    __half s;
    int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "wrong Q8_1 block size");

#define HIP_CHECK(expr)                                                                      \
    do {                                                                                     \
        const hipError_t status__ = (expr);                                                  \
        if (status__ != hipSuccess) {                                                        \
            std::cerr << "HIP error: " << hipGetErrorString(status__) << " at line "       \
                      << __LINE__ << std::endl;                                              \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

struct args_t {
    int rows = 17408;
    int cols = 5120;
    int warmup = 10;
    int rounds = 20;
    int iters = 20;
    int device = 0;
};

static void usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " [--rows N] [--cols N] [--warmup N] [--rounds N] [--iters N] [--device N]\n";
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
        auto need_int = [&](int & value) {
            return i + 1 < argc && parse_int(argv[++i], value);
        };
        if (key == "--rows") {
            if (!need_int(args.rows)) return false;
        } else if (key == "--cols") {
            if (!need_int(args.cols)) return false;
        } else if (key == "--warmup") {
            if (!need_int(args.warmup)) return false;
        } else if (key == "--rounds") {
            if (!need_int(args.rounds)) return false;
        } else if (key == "--iters") {
            if (!need_int(args.iters)) return false;
        } else if (key == "--device") {
            if (!need_int(args.device)) return false;
        } else {
            usage(argv[0]);
            return false;
        }
    }
    return args.rows > 0 && args.cols > 0 && args.cols % QK_K == 0 &&
           args.warmup >= 0 && args.rounds > 0 && args.iters > 0;
}

static uint32_t lcg(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static int q3_scale_host(const uint8_t * scales, const int index) {
    const int low = (scales[index % 8] >> (4 * (index / 8))) & 0x0f;
    const int high = ((scales[8 + index % 4] >> (2 * (index / 4))) & 0x03) << 4;
    return (low | high) - 32;
}

static void fill_q3(std::vector<block_q3_K_padded> & raw) {
    uint32_t state = 0x5a17c0deu;
    for (block_q3_K_padded & block : raw) {
        for (uint8_t & value : block.hmask) value = static_cast<uint8_t>(lcg(state) >> 24);
        for (uint8_t & value : block.qs) value = static_cast<uint8_t>(lcg(state) >> 24);
        for (uint8_t & value : block.scales) value = static_cast<uint8_t>(lcg(state) >> 24);
        block.d = __float2half(0.0005f * static_cast<float>((lcg(state) % 31u) + 1u));
        block.pad[0] = 0;
        block.pad[1] = 0;
    }
}

static void expand_scales(
        const std::vector<block_q3_K_padded> & raw,
        std::vector<block_q3_K_scale16> & expanded) {
    expanded.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        const block_q3_K_padded & src = raw[i];
        block_q3_K_scale16 & dst = expanded[i];
        std::copy(std::begin(src.hmask), std::end(src.hmask), std::begin(dst.hmask));
        std::copy(std::begin(src.qs), std::end(src.qs), std::begin(dst.qs));
        for (int scale = 0; scale < 16; ++scale) {
            dst.scales[scale] = static_cast<int8_t>(q3_scale_host(src.scales, scale));
        }
        dst.d = src.d;
        dst.pad[0] = 0;
        dst.pad[1] = 0;
    }
}

static void fill_q8(std::vector<block_q8_1> & blocks) {
    uint32_t state = 0xc001d00du;
    for (block_q8_1 & block : blocks) {
        block.d = __float2half(0.00075f * static_cast<float>((lcg(state) % 29u) + 1u));
        block.s = __float2half(0.0f);
        for (int8_t & value : block.qs) {
            value = static_cast<int8_t>(static_cast<int>(lcg(state) % 255u) - 127);
        }
    }
}

static __device__ __forceinline__ int get_int_b2(const void * ptr, const int index) {
    const uint16_t * values = static_cast<const uint16_t *>(ptr);
    return int(values[2 * index]) | (int(values[2 * index + 1]) << 16);
}

static __device__ __forceinline__ int get_int_b4(const void * ptr, const int index) {
    return static_cast<const int *>(ptr)[index];
}

static __device__ __forceinline__ int dp4a(const int a, const int b) {
    return __builtin_amdgcn_sudot4(true, a, true, b, 0, false);
}

static __device__ __forceinline__ int sub4_packed(const int vl, const int vh) {
    const uint32_t values = uint32_t(vl) | (uint32_t(vh) ^ 0x04040404u);
    return int(((values ^ 0x80808080u) - 0x04040404u) ^ 0x80808080u);
}

static __device__ __forceinline__ int packed_scale_at(
        const uint8_t * scales, const int scale_offset, const int i) {
    const int index = scale_offset + 2 * i;
    const int low = (scales[index % 8] >> (4 * (index / 8))) & 0x0f;
    const int high = ((scales[8 + index % 4] >> (2 * (index / 4))) & 0x03) << 4;
    return (low | high) - 32;
}

static __device__ __forceinline__ uint32_t packed_scale4(
        const uint8_t * scales, const int scale_offset) {
    const uint32_t low0 = static_cast<uint32_t>(get_int_b4(scales, 0));
    const uint32_t low1 = static_cast<uint32_t>(get_int_b4(scales, 1));
    const uint32_t high = static_cast<uint32_t>(get_int_b4(scales, 2));
    const bool odd = (scale_offset & 1) != 0;
    const bool upper = scale_offset >= 8;

    const uint32_t low_selector = odd ? 0x07050301u : 0x06040200u;
    uint32_t low_packed = __builtin_amdgcn_perm(low1, low0, low_selector);
    low_packed = (low_packed >> (upper ? 4 : 0)) & 0x0f0f0f0fu;

    const uint32_t high_selector = odd ? 0x03010301u : 0x02000200u;
    const uint32_t high_packed = __builtin_amdgcn_perm(high, high, high_selector);
    uint32_t high_bits;
    if (upper) {
        high_bits = ((high_packed & 0x00003030u) >> 4) |
                    ((high_packed & 0xc0c00000u) >> 6);
    } else {
        high_bits =  (high_packed & 0x00000303u) |
                    ((high_packed & 0x0c0c0000u) >> 2);
    }

    const uint32_t unsigned_scales = low_packed | (high_bits << 4);
    return ((unsigned_scales ^ 0x80808080u) - 0x20202020u) ^ 0x80808080u;
}

static __device__ __forceinline__ int packed_scale4_at(const uint32_t scales, const int i) {
    return static_cast<int>(static_cast<int8_t>(scales >> (8 * i)));
}

template<typename Block, bool expanded_scales, bool packed_scale_gather = false, bool u32_block_loads = false>
static __device__ __forceinline__ void pair_dot(
        const Block * x,
        const Block * gate,
        const block_q8_1 * q8,
        const int block_index,
        const int iqs,
        float & dot_x,
        float & dot_gate) {
    const Block * bx = x + block_index;
    const Block * bg = gate + block_index;
    const int q8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const int vl_x = u32_block_loads ? get_int_b4(bx->qs, iqs) : get_int_b2(bx->qs, iqs);
    const int vl_g = u32_block_loads ? get_int_b4(bg->qs, iqs) : get_int_b2(bg->qs, iqs);
    const int vh_x = ~(u32_block_loads ? get_int_b4(bx->hmask, iqs % (QI3_K / 2)) :
                                        get_int_b2(bx->hmask, iqs % (QI3_K / 2))) >> q8_offset;
    const int vh_g = ~(u32_block_loads ? get_int_b4(bg->hmask, iqs % (QI3_K / 2)) :
                                        get_int_b2(bg->hmask, iqs % (QI3_K / 2))) >> q8_offset;
    uint32_t scales_x = 0;
    uint32_t scales_g = 0;
    if constexpr (packed_scale_gather) {
        scales_x = packed_scale4(bx->scales, scale_offset);
        scales_g = packed_scale4(bg->scales, scale_offset);
    }
    float sum_x = 0.0f;
    float sum_g = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int u = get_int_b4(q8[q8_offset + i].qs, iqs % QI8_1);
        const float d8 = __half2float(q8[q8_offset + i].d);
        int sx;
        int sg;
        if constexpr (packed_scale_gather) {
            sx = packed_scale4_at(scales_x, i);
            sg = packed_scale4_at(scales_g, i);
        } else if constexpr (expanded_scales) {
            sx = bx->scales[scale_offset + 2 * i];
            sg = bg->scales[scale_offset + 2 * i];
        } else {
            sx = packed_scale_at(bx->scales, scale_offset, i);
            sg = packed_scale_at(bg->scales, scale_offset, i);
        }
        const int qx = sub4_packed((vl_x >> (2 * i)) & 0x03030303,
                                   ((vh_x >> i) << 2) & 0x04040404);
        const int qg = sub4_packed((vl_g >> (2 * i)) & 0x03030303,
                                   ((vh_g >> i) << 2) & 0x04040404);
        sum_x += d8 * static_cast<float>(dp4a(qx, u) * sx);
        sum_g += d8 * static_cast<float>(dp4a(qg, u) * sg);
    }
    dot_x = __half2float(bx->d) * sum_x;
    dot_gate = __half2float(bg->d) * sum_g;
}

template<typename Block, bool expanded_scales, bool packed_scale_gather = false, bool u32_block_loads = false>
__global__ __launch_bounds__(64, 1) static void q3_pair_kernel(
        const Block * __restrict__ x,
        const Block * __restrict__ gate,
        const block_q8_1 * __restrict__ q8,
        float * __restrict__ output,
        const int blocks_per_row) {
    __shared__ float wave1_x[32];
    __shared__ float wave1_gate[32];
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int tid = wave * 32 + lane;
    const int row = blockIdx.x;
    const int iqs = tid % QI3_K;
    float sum_x = 0.0f;
    float sum_gate = 0.0f;

    for (int block = tid / QI3_K; block < blocks_per_row; block += 4) {
        float dot_x;
        float dot_gate;
        pair_dot<Block, expanded_scales, packed_scale_gather, u32_block_loads>(
            x, gate, q8 + block * (QK_K / QK8_1), row * blocks_per_row + block, iqs, dot_x, dot_gate);
        sum_x += dot_x;
        sum_gate += dot_gate;
    }

    if (wave == 1) {
        wave1_x[lane] = sum_x;
        wave1_gate[lane] = sum_gate;
    }
    __syncthreads();
    if (wave != 0) return;

    sum_x += wave1_x[lane];
    sum_gate += wave1_gate[lane];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_x += __shfl_down(sum_x, offset, 32);
        sum_gate += __shfl_down(sum_gate, offset, 32);
    }
    if (lane == 0) {
        output[row] = sum_x * (sum_gate / (1.0f + expf(-sum_gate)));
    }
}

template<typename Block, bool expanded_scales, bool packed_scale_gather = false, bool u32_block_loads = false>
static int launch_many(
        hipStream_t stream,
        const Block * x,
        const Block * gate,
        const block_q8_1 * q8,
        float * output,
        const args_t & args,
        const int count,
        double & average_ms) {
    const dim3 grid(args.rows);
    const dim3 block(32, 2);
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < count; ++i) {
        q3_pair_kernel<Block, expanded_scales, packed_scale_gather, u32_block_loads><<<grid, block, 0, stream>>>(
            x, gate, q8, output, args.cols / QK_K);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    average_ms = static_cast<double>(elapsed_ms) / count;
    return 0;
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

int main(int argc, char ** argv) {
    args_t args;
    if (!parse_args(argc, argv, args)) return 1;
    HIP_CHECK(hipSetDevice(args.device));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    const size_t matrix_blocks = static_cast<size_t>(args.rows) * (args.cols / QK_K);
    std::vector<block_q3_K_padded> h_x_raw(matrix_blocks);
    std::vector<block_q3_K_padded> h_gate_raw(matrix_blocks);
    std::vector<block_q3_K_scale16> h_x_expanded;
    std::vector<block_q3_K_scale16> h_gate_expanded;
    std::vector<block_q8_1> h_q8(args.cols / QK8_1);
    fill_q3(h_x_raw);
    fill_q3(h_gate_raw);
    expand_scales(h_x_raw, h_x_expanded);
    expand_scales(h_gate_raw, h_gate_expanded);
    fill_q8(h_q8);

    block_q3_K_padded * d_x_raw = nullptr;
    block_q3_K_padded * d_gate_raw = nullptr;
    block_q3_K_scale16 * d_x_expanded = nullptr;
    block_q3_K_scale16 * d_gate_expanded = nullptr;
    block_q8_1 * d_q8 = nullptr;
    float * d_raw_output = nullptr;
    float * d_u32_output = nullptr;
    float * d_packed_output = nullptr;
    float * d_expanded_output = nullptr;
    const size_t raw_bytes = matrix_blocks * sizeof(block_q3_K_padded);
    const size_t expanded_bytes = matrix_blocks * sizeof(block_q3_K_scale16);
    const size_t q8_bytes = h_q8.size() * sizeof(block_q8_1);
    const size_t output_bytes = static_cast<size_t>(args.rows) * sizeof(float);

    HIP_CHECK(hipMalloc(&d_x_raw, raw_bytes));
    HIP_CHECK(hipMalloc(&d_gate_raw, raw_bytes));
    HIP_CHECK(hipMalloc(&d_x_expanded, expanded_bytes));
    HIP_CHECK(hipMalloc(&d_gate_expanded, expanded_bytes));
    HIP_CHECK(hipMalloc(&d_q8, q8_bytes));
    HIP_CHECK(hipMalloc(&d_raw_output, output_bytes));
    HIP_CHECK(hipMalloc(&d_u32_output, output_bytes));
    HIP_CHECK(hipMalloc(&d_packed_output, output_bytes));
    HIP_CHECK(hipMalloc(&d_expanded_output, output_bytes));
    HIP_CHECK(hipMemcpyAsync(d_x_raw, h_x_raw.data(), raw_bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_gate_raw, h_gate_raw.data(), raw_bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_x_expanded, h_x_expanded.data(), expanded_bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_gate_expanded, h_gate_expanded.data(), expanded_bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_q8, h_q8.data(), q8_bytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double ignored = 0.0;
    if (launch_many<block_q3_K_padded, false>(stream, d_x_raw, d_gate_raw, d_q8, d_raw_output, args, args.warmup, ignored) != 0) return 1;
    if (launch_many<block_q3_K_padded, false, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_u32_output, args, args.warmup, ignored) != 0) return 1;
    if (launch_many<block_q3_K_padded, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_packed_output, args, args.warmup, ignored) != 0) return 1;
    if (launch_many<block_q3_K_scale16, true>(stream, d_x_expanded, d_gate_expanded, d_q8, d_expanded_output, args, args.warmup, ignored) != 0) return 1;

    std::vector<float> h_raw_output(args.rows);
    std::vector<float> h_u32_output(args.rows);
    std::vector<float> h_packed_output(args.rows);
    std::vector<float> h_expanded_output(args.rows);
    HIP_CHECK(hipMemcpy(h_raw_output.data(), d_raw_output, output_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_u32_output.data(), d_u32_output, output_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_packed_output.data(), d_packed_output, output_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_expanded_output.data(), d_expanded_output, output_bytes, hipMemcpyDeviceToHost));
    double max_abs = 0.0;
    double max_rel = 0.0;
    double packed_max_abs = 0.0;
    double packed_max_rel = 0.0;
    double u32_max_abs = 0.0;
    double u32_max_rel = 0.0;
    int mismatches = 0;
    int packed_mismatches = 0;
    int u32_mismatches = 0;
    for (int i = 0; i < args.rows; ++i) {
        const double diff = std::abs(double(h_raw_output[i]) - double(h_expanded_output[i]));
        const double rel = diff / std::max(1.0, std::abs(double(h_raw_output[i])));
        const double packed_diff = std::abs(double(h_raw_output[i]) - double(h_packed_output[i]));
        const double packed_rel = packed_diff / std::max(1.0, std::abs(double(h_raw_output[i])));
        const double u32_diff = std::abs(double(h_raw_output[i]) - double(h_u32_output[i]));
        const double u32_rel = u32_diff / std::max(1.0, std::abs(double(h_raw_output[i])));
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        packed_max_abs = std::max(packed_max_abs, packed_diff);
        packed_max_rel = std::max(packed_max_rel, packed_rel);
        u32_max_abs = std::max(u32_max_abs, u32_diff);
        u32_max_rel = std::max(u32_max_rel, u32_rel);
        mismatches += rel > 1.0e-5;
        packed_mismatches += packed_rel > 1.0e-5;
        u32_mismatches += u32_rel > 1.0e-5;
    }

    std::vector<double> raw_samples;
    std::vector<double> u32_samples;
    std::vector<double> packed_samples;
    std::vector<double> expanded_samples;
    raw_samples.reserve(args.rounds);
    u32_samples.reserve(args.rounds);
    packed_samples.reserve(args.rounds);
    expanded_samples.reserve(args.rounds);
    for (int round = 0; round < args.rounds; ++round) {
        double raw_ms = 0.0;
        double u32_ms = 0.0;
        double packed_ms = 0.0;
        double expanded_ms = 0.0;
        if (round % 4 == 0) {
            if (launch_many<block_q3_K_padded, false>(stream, d_x_raw, d_gate_raw, d_q8, d_raw_output, args, args.iters, raw_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_u32_output, args, args.iters, u32_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_packed_output, args, args.iters, packed_ms) != 0) return 1;
            if (launch_many<block_q3_K_scale16, true>(stream, d_x_expanded, d_gate_expanded, d_q8, d_expanded_output, args, args.iters, expanded_ms) != 0) return 1;
        } else if (round % 4 == 1) {
            if (launch_many<block_q3_K_padded, false, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_u32_output, args, args.iters, u32_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_packed_output, args, args.iters, packed_ms) != 0) return 1;
            if (launch_many<block_q3_K_scale16, true>(stream, d_x_expanded, d_gate_expanded, d_q8, d_expanded_output, args, args.iters, expanded_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false>(stream, d_x_raw, d_gate_raw, d_q8, d_raw_output, args, args.iters, raw_ms) != 0) return 1;
        } else if (round % 4 == 2) {
            if (launch_many<block_q3_K_padded, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_packed_output, args, args.iters, packed_ms) != 0) return 1;
            if (launch_many<block_q3_K_scale16, true>(stream, d_x_expanded, d_gate_expanded, d_q8, d_expanded_output, args, args.iters, expanded_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false>(stream, d_x_raw, d_gate_raw, d_q8, d_raw_output, args, args.iters, raw_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_u32_output, args, args.iters, u32_ms) != 0) return 1;
        } else {
            if (launch_many<block_q3_K_scale16, true>(stream, d_x_expanded, d_gate_expanded, d_q8, d_expanded_output, args, args.iters, expanded_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false>(stream, d_x_raw, d_gate_raw, d_q8, d_raw_output, args, args.iters, raw_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_u32_output, args, args.iters, u32_ms) != 0) return 1;
            if (launch_many<block_q3_K_padded, false, true>(stream, d_x_raw, d_gate_raw, d_q8, d_packed_output, args, args.iters, packed_ms) != 0) return 1;
        }
        raw_samples.push_back(raw_ms);
        u32_samples.push_back(u32_ms);
        packed_samples.push_back(packed_ms);
        expanded_samples.push_back(expanded_ms);
    }

    hipFuncAttributes raw_attr{};
    hipFuncAttributes u32_attr{};
    hipFuncAttributes packed_attr{};
    hipFuncAttributes expanded_attr{};
    HIP_CHECK(hipFuncGetAttributes(&raw_attr, reinterpret_cast<const void *>(q3_pair_kernel<block_q3_K_padded, false>)));
    HIP_CHECK(hipFuncGetAttributes(&u32_attr, reinterpret_cast<const void *>(q3_pair_kernel<block_q3_K_padded, false, false, true>)));
    HIP_CHECK(hipFuncGetAttributes(&packed_attr, reinterpret_cast<const void *>(q3_pair_kernel<block_q3_K_padded, false, true>)));
    HIP_CHECK(hipFuncGetAttributes(&expanded_attr, reinterpret_cast<const void *>(q3_pair_kernel<block_q3_K_scale16, true>)));

    const double raw_median = median(raw_samples);
    const double u32_median = median(u32_samples);
    const double packed_median = median(packed_samples);
    const double expanded_median = median(expanded_samples);
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "device=" << args.device << " rows=" << args.rows << " cols=" << args.cols
              << " rounds=" << args.rounds << " iters=" << args.iters << "\n";
    std::cout << "raw_block_bytes=" << sizeof(block_q3_K_padded)
              << " expanded_block_bytes=" << sizeof(block_q3_K_scale16)
              << " storage_ratio=" << double(sizeof(block_q3_K_scale16)) / sizeof(block_q3_K_padded) << "x\n";
    std::cout << "correctness mismatches=" << mismatches << " max_abs=" << max_abs
              << " max_rel=" << max_rel << "\n";
    std::cout << "packed_correctness mismatches=" << packed_mismatches
              << " max_abs=" << packed_max_abs << " max_rel=" << packed_max_rel << "\n";
    std::cout << "u32_correctness mismatches=" << u32_mismatches
              << " max_abs=" << u32_max_abs << " max_rel=" << u32_max_rel << "\n";
    std::cout << "raw_median_ms=" << raw_median << " expanded_median_ms=" << expanded_median
              << " expanded_speedup=" << raw_median / expanded_median << "x\n";
    std::cout << "packed_median_ms=" << packed_median
              << " packed_speedup=" << raw_median / packed_median << "x\n";
    std::cout << "u32_median_ms=" << u32_median
              << " u32_speedup=" << raw_median / u32_median << "x\n";
    std::cout << "raw_regs=" << raw_attr.numRegs << " u32_regs=" << u32_attr.numRegs
              << " packed_regs=" << packed_attr.numRegs
              << " expanded_regs=" << expanded_attr.numRegs
              << " raw_shared=" << raw_attr.sharedSizeBytes
              << " u32_shared=" << u32_attr.sharedSizeBytes
              << " packed_shared=" << packed_attr.sharedSizeBytes
              << " expanded_shared=" << expanded_attr.sharedSizeBytes << "\n";

    HIP_CHECK(hipFree(d_x_raw));
    HIP_CHECK(hipFree(d_gate_raw));
    HIP_CHECK(hipFree(d_x_expanded));
    HIP_CHECK(hipFree(d_gate_expanded));
    HIP_CHECK(hipFree(d_q8));
    HIP_CHECK(hipFree(d_raw_output));
    HIP_CHECK(hipFree(d_u32_output));
    HIP_CHECK(hipFree(d_packed_output));
    HIP_CHECK(hipFree(d_expanded_output));
    HIP_CHECK(hipStreamDestroy(stream));
    return mismatches == 0 && packed_mismatches == 0 && u32_mismatches == 0 ? 0 : 2;
}
