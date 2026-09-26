// D138 step-3 admission probe: isolated MMVQ vec_dot bandwidth on gfx1201.
//
// Question this answers (cheapest possible form of the Step 3 decision): does
// the *current* production vec_dot memory pattern already reach the card's
// bandwidth when it is given a saturated grid, or is it stuck well below it?
//
//   - if the isolated pattern reaches ~600+ GB/s, the remaining decode gap
//     (~457 GB/s measured by the DUP probes inside the real graph) is NOT in
//     the vec_dot load pattern, and Step 3 (128 B K-window rewrite) does not
//     buy anything;
//   - if it stalls at ~300-400 GB/s, the pattern/intruction mix is the limiter
//     and the rewrite is justified.
//
// The kernel loop is a faithful copy of ggml/src/ggml-cuda/mmvq.cu:
// mul_mat_vec_q for ncols_dst == 1, non-fused, table_id == MMVQ_PARAMETERS_RDNA4
// (rows_per_cuda_block == 1), reading the REAL vec_dot_* bodies from vecdotq.cuh
// (same file the production build compiles, so there is no second copy of the
// quant maths to get out of sync):
//
//   tid = warp_size*threadIdx.y + threadIdx.x            (block = 32 x nwarps)
//   row0 = blockIdx.x
//   kbx starts at tid/(qi/vdr), steps vdr*nwarps*warp_size/qi
//   kby  = kbx*(qk/QK8_1)
//   kqs  = vdr*(tid % (qi/vdr))
//   dot  += vec_dot(vx, &y[kby], row0*stride_row_x + kbx, kqs)
//
// Build (same flags as the production ROCm target, minus the ggml-hip glue):
//   "C:\Program Files\AMD\ROCm\7.2\bin\clang++.exe" -x hip --offload-arch=gfx1201
//     -O3 -DNDEBUG -std=gnu++17 -D__HIP_PLATFORM_AMD__=1 -DGGML_USE_HIP
//     -I ggml/src/ggml-cuda -I ggml/src -I ggml/include
//     -isystem "C:/Program Files/AMD/ROCm/7.2/include"
//     -funsafe-math-optimizations -ffast-math -fno-finite-math-only
//     scripts/research/d138_vecdot_probe.hip -o build_logs/d138-vecdot-probe.exe
//
// Usage: d138-vecdot-probe.exe [cfg_idx] [reps]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vecdotq.cuh"

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

using vec_dot_t = float (*)(const void *, const block_q8_1 *, const int &, const int &);

// ---------------------------------------------------------------------------
// (a) production-shaped loop, real vec_dot body
// ---------------------------------------------------------------------------
template <vec_dot_t DOT, int QK, int QI, int VDR, int NWARPS>
__global__ void __launch_bounds__(NWARPS*32, 1) vecdot_loop_kernel(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t row_start,
        float * __restrict__ dst) {
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter = VDR * NWARPS * warp_size / QI;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = (int) row_start + blockIdx.x;
    const int blocks_per_row_x = ncols_x / QK;
    const int kbx_offset = row0 * stride_row_x;

    float acc = 0.0f;
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (QK / QK8_1);
        const int kqs = VDR * (tid % (QI / VDR));
        acc += DOT(vx, &vy[kby], kbx_offset + kbx, kqs);
    }
    if (acc == 12345.678f) { // never true; keeps the dot products alive
        dst[row0] = acc;
    }
}

// ---------------------------------------------------------------------------
// (b) reference: same byte volume read with wide (16 B) linear loads
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 1) stream_kernel(
        const uint4 * __restrict__ vx, const size_t n4, float * __restrict__ dst) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const uint4 v = vx[i];
        acc += (float) (v.x & 0xFFu) + (float) (v.y & 0xFFu) + (float) (v.z & 0xFFu) + (float) (v.w & 0xFFu);
    }
    if (acc == 12345.678f) {
        dst[0] = acc;
    }
}

// L2 churn: the decode reads 15 GB of weights per token, so every form must be
// measured with a cold L2. Without this a 61 MB buffer fits the 64 MB L2 of
// gfx1201 and the probe reports cache bandwidth instead of HBM bandwidth.
static void * g_churn = nullptr;
static size_t g_churn_bytes = 0;

static void churn_l2() {
    if (g_churn == nullptr) {
        return;
    }
    stream_kernel<<<dim3(1088), dim3(256)>>>((const uint4 *) g_churn,
        g_churn_bytes / sizeof(uint4), (float *) g_churn);
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
struct cfg_t {
    const char * name;
    const char * type_name;
    int qk;
    int qi;
    int vdr;
    int block_bytes;
    unsigned ncols_x;   // K
    unsigned nrows;     // N (grid.x)
    int nwarps;
    int kind;           // 0 = q5_K, 1 = q6_K, 2 = iq4_xs
};

static inline uint16_t f32_to_f16_bits(const float f) {
    const __half h = __float2half(f);
    uint16_t out = 0;
    memcpy(&out, &h, 2);
    return out;
}

static void fill_host(void * buf, const size_t bytes, const int block_bytes) {
    // Valid-looking quant blocks: d>0, and byte payload from a cheap LCG so the
    // kernel does real dequant work instead of reading zeros.
    uint8_t * p = (uint8_t *) buf;
    uint32_t s = 12345u;
    for (size_t i = 0; i < bytes; ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = (uint8_t) (s >> 24);
    }
    for (size_t off = 0; off + (size_t) block_bytes <= bytes; off += (size_t) block_bytes) {
        const uint16_t d = f32_to_f16_bits(0.02f + 0.001f * (float) ((off / (size_t) block_bytes) % 17));
        memcpy(p + off, &d, sizeof(d));                       // d / dm
        if (block_bytes >= 4) {
            memcpy(p + off + 2, &d, sizeof(d));               // dmin / (v.d for q6_K unused)
        }
    }
}

template <vec_dot_t DOT, int QK, int QI, int VDR, int NWARPS>
static void run_vecdot(const cfg_t & c, const int reps) {
    const size_t blocks_per_row = c.ncols_x / (unsigned) c.qk;
    const size_t weight_bytes = (size_t) c.nrows * blocks_per_row * (size_t) c.block_bytes;

    void * d_vx = nullptr;
    block_q8_1 * d_vy = nullptr;
    float * d_dst = nullptr;
    HIP_OK(hipMalloc(&d_vx, weight_bytes));
    HIP_OK(hipMalloc(&d_vy, (blocks_per_row * (size_t) (c.qk / QK8_1) + 32) * sizeof(block_q8_1)));
    HIP_OK(hipMalloc(&d_dst, (size_t) c.nrows * sizeof(float)));

    std::vector<uint8_t> host(weight_bytes);
    fill_host(host.data(), weight_bytes, c.block_bytes);
    HIP_OK(hipMemcpy(d_vx, host.data(), weight_bytes, hipMemcpyHostToDevice));

    std::vector<block_q8_1> y(blocks_per_row * (size_t) (c.qk / QK8_1) + 32);
    for (auto & b : y) { b.ds = f32_to_f16_bits(0.5f); memset(b.qs, 1, sizeof(b.qs)); }
    HIP_OK(hipMemcpy(d_vy, y.data(), y.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));

    const dim3 grid(c.nrows);
    const dim3 block(32, NWARPS);

    // warmup + timed reps
    for (int i = 0; i < 3; ++i) {
        vecdot_loop_kernel<DOT, QK, QI, VDR, NWARPS><<<grid, block>>>(d_vx, d_vy, c.ncols_x, (unsigned) blocks_per_row, 0, d_dst);
    }
    HIP_OK(hipDeviceSynchronize());

    float best_ms = 1e9f;
    for (int i = 0; i < reps; ++i) {
        hipEvent_t e0, e1;
        HIP_OK(hipEventCreate(&e0));
        HIP_OK(hipEventCreate(&e1));
        churn_l2();
        HIP_OK(hipEventRecord(e0));
        vecdot_loop_kernel<DOT, QK, QI, VDR, NWARPS><<<grid, block>>>(d_vx, d_vy, c.ncols_x, (unsigned) blocks_per_row, 0, d_dst);
        HIP_OK(hipEventRecord(e1));
        HIP_OK(hipEventSynchronize(e1));
        float ms = 0.0f;
        HIP_OK(hipEventElapsedTime(&ms, e0, e1));
        best_ms = std::min(best_ms, ms);
        HIP_OK(hipEventDestroy(e0));
        HIP_OK(hipEventDestroy(e1));
    }

    printf("%-26s K=%-6u N=%-7u warps=%d  grid=(%u,1,1)  bytes=%7.2f MB  %8.3f ms  %7.1f GB/s  (read-once pattern)\n",
           c.name, c.ncols_x, c.nrows, c.nwarps, c.nrows,
           (double) weight_bytes / 1e6, best_ms,
           (double) weight_bytes / (double) best_ms / 1e6);

    HIP_OK(hipFree(d_vx));
    HIP_OK(hipFree(d_vy));
    HIP_OK(hipFree(d_dst));
}

// Split test: the decode runs ~458 short MMVQ launches per token, one after
// another in a single stream. This measures what that fragmentation costs:
// the same byte volume is read either in one launch or in `splits` launches.
template <vec_dot_t DOT, int QK, int QI, int VDR, int NWARPS>
static void run_split_test(const cfg_t & c, const int splits, const int reps) {
    const size_t blocks_per_row = c.ncols_x / (unsigned) c.qk;
    const size_t weight_bytes = (size_t) c.nrows * blocks_per_row * (size_t) c.block_bytes;

    void * d_vx = nullptr;
    block_q8_1 * d_vy = nullptr;
    float * d_dst = nullptr;
    HIP_OK(hipMalloc(&d_vx, weight_bytes));
    HIP_OK(hipMalloc(&d_vy, (blocks_per_row * (size_t) (c.qk / QK8_1) + 32) * sizeof(block_q8_1)));
    HIP_OK(hipMalloc(&d_dst, (size_t) c.nrows * sizeof(float)));

    std::vector<uint8_t> host(weight_bytes);
    fill_host(host.data(), weight_bytes, c.block_bytes);
    HIP_OK(hipMemcpy(d_vx, host.data(), weight_bytes, hipMemcpyHostToDevice));
    std::vector<block_q8_1> y(blocks_per_row * (size_t) (c.qk / QK8_1) + 32);
    for (auto & b : y) { b.ds = f32_to_f16_bits(0.5f); memset(b.qs, 1, sizeof(b.qs)); }
    HIP_OK(hipMemcpy(d_vy, y.data(), y.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));

    const unsigned rows_per_split = c.nrows / (unsigned) splits;
    const dim3 block(32, NWARPS);

    for (int w = 0; w < 2; ++w) {
        for (int s = 0; s < splits; ++s) {
            vecdot_loop_kernel<DOT, QK, QI, VDR, NWARPS><<<dim3(rows_per_split), block>>>(
                d_vx, d_vy, c.ncols_x, (unsigned) blocks_per_row, s * rows_per_split, d_dst);
        }
    }
    HIP_OK(hipDeviceSynchronize());

    float best_ms = 1e9f;
    for (int i = 0; i < reps; ++i) {
        hipEvent_t e0, e1;
        HIP_OK(hipEventCreate(&e0));
        HIP_OK(hipEventCreate(&e1));
        churn_l2();
        HIP_OK(hipEventRecord(e0));
        for (int s = 0; s < splits; ++s) {
            vecdot_loop_kernel<DOT, QK, QI, VDR, NWARPS><<<dim3(rows_per_split), block>>>(
                d_vx, d_vy, c.ncols_x, (unsigned) blocks_per_row, s * rows_per_split, d_dst);
        }
        HIP_OK(hipEventRecord(e1));
        HIP_OK(hipEventSynchronize(e1));
        float ms = 0.0f;
        HIP_OK(hipEventElapsedTime(&ms, e0, e1));
        best_ms = std::min(best_ms, ms);
        HIP_OK(hipEventDestroy(e0));
        HIP_OK(hipEventDestroy(e1));
    }

    printf("split=%-4d rows/launch=%-7u bytes=%7.2f MB  %8.3f ms  %7.1f GB/s  (per-launch cost visible here)\n",
           splits, rows_per_split, (double) weight_bytes / 1e6, best_ms,
           (double) weight_bytes / (double) best_ms / 1e6);

    HIP_OK(hipFree(d_vx));
    HIP_OK(hipFree(d_vy));
    HIP_OK(hipFree(d_dst));
}

static void run_stream(const size_t bytes, const int reps) {
    void * d_vx = nullptr;
    float * d_dst = nullptr;
    HIP_OK(hipMalloc(&d_vx, bytes));
    HIP_OK(hipMalloc(&d_dst, sizeof(float)));
    std::vector<uint8_t> host(bytes, 0x5a);
    HIP_OK(hipMemcpy(d_vx, host.data(), bytes, hipMemcpyHostToDevice));

    const size_t n4 = bytes / sizeof(uint4);
    const dim3 grid(1088);          // 17 waves of 64 CTAs, like the real grid.x
    const dim3 block(256);
    for (int i = 0; i < 3; ++i) {
        stream_kernel<<<grid, block>>>((const uint4 *) d_vx, n4, d_dst);
    }
    HIP_OK(hipDeviceSynchronize());

    float best_ms = 1e9f;
    for (int i = 0; i < reps; ++i) {
        hipEvent_t e0, e1;
        HIP_OK(hipEventCreate(&e0));
        HIP_OK(hipEventCreate(&e1));
        churn_l2();
        HIP_OK(hipEventRecord(e0));
        stream_kernel<<<grid, block>>>((const uint4 *) d_vx, n4, d_dst);
        HIP_OK(hipEventRecord(e1));
        HIP_OK(hipEventSynchronize(e1));
        float ms = 0.0f;
        HIP_OK(hipEventElapsedTime(&ms, e0, e1));
        best_ms = std::min(best_ms, ms);
        HIP_OK(hipEventDestroy(e0));
        HIP_OK(hipEventDestroy(e1));
    }
    printf("%-26s bytes=%7.2f MB  %8.3f ms  %7.1f GB/s  (wide 16 B linear loads, reference)\n",
           "stream reference", (double) bytes / 1e6, best_ms, (double) bytes / (double) best_ms / 1e6);

    HIP_OK(hipFree(d_vx));
    HIP_OK(hipFree(d_dst));
}

int main(int argc, char ** argv) {
    const int reps = argc > 2 ? atoi(argv[2]) : 10;

    g_churn_bytes = (size_t) 256 * 1024 * 1024;   // 4x the 64 MB L2
    HIP_OK(hipMalloc(&g_churn, g_churn_bytes));
    HIP_OK(hipMemset(g_churn, 0x5a, g_churn_bytes));

    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    printf("device: %s  SMs=%d  warpSize=%d  mem=%.1f GB  clock=%.0f MHz\n\n",
           prop.name, prop.multiProcessorCount, prop.warpSize,
           (double) prop.totalGlobalMem / 1e9, (double) prop.memoryClockRate / 1000.0);

    // Configurations taken from the D138 byte census trace (real decode shapes:
    // q5_K = 40 % of decode bytes, iq4_xs = 36 %, q6_K = 14 %).
    const std::vector<cfg_t> cfgs = {
        // name                       type       qk   qi  vdr  bytes  K      N       warps kind
        {"q5_K  K=5120  N=17408",     "q5_K",    256,  32,  2,   176,   5120,  17408,  8, 0},
        {"q5_K  K=17408 N=5120",      "q5_K",    256,  32,  2,   176,  17408,   5120,  8, 0},
        {"q5_K  K=6144  N=5120",      "q5_K",    256,  32,  2,   176,   6144,   5120,  8, 0},
        {"iq4_xs K=5120  N=17408",    "iq4_xs",  256,  32,  4,   136,   5120,  17408,  8, 2},
        {"iq4_xs K=17408 N=5120",     "iq4_xs",  256,  32,  4,   136,  17408,   5120,  8, 2},
        {"q6_K  K=5120  N=10240",     "q6_K",    256,  32,  1,   210,   5120,  10240,  8, 1},
        {"q6_K  K=5120  N=248320",    "q6_K",    256,  32,  1,   210,   5120, 248320,  8, 1},
        // >300 MB forms: L2 (64 MB) cannot hold them, so these are honest
        // HBM numbers even without churn.
        {"q5_K  K=5120  N=87040",     "q5_K",    256,  32,  2,   176,   5120,  87040,  8, 0},
        {"q5_K  K=17408 N=25600",     "q5_K",    256,  32,  2,   176,  17408,  25600,  8, 0},
        {"iq4_xs K=5120  N=87040",    "iq4_xs",  256,  32,  4,   136,   5120,  87040,  8, 2},
        {"iq4_xs K=17408 N=25600",    "iq4_xs",  256,  32,  4,   136,  17408,  25600,  8, 2},
    };

    const int idx = argc > 1 ? atoi(argv[1]) : -1;
    if (idx == -3) {
        // Fragmentation test on the 306 MB q5_K form: one launch vs many.
        printf("-- split test: q5_K K=5120 N=87040 (306 MB), launches from 1 to 128 --\n");
        cfg_t c = { "q5_K 5120 split", "q5_K", 256, 32, 2, 176, 5120, 87040, 8, 0 };
        for (int s : { 1, 2, 8, 32, 128, 458 }) {
            run_split_test<vec_dot_q5_K_q8_1, 256, 32, 2, 8>(c, s, 4);
        }
        return 0;
    }
    if (idx == -2) {
        // N sweep for one form: isolates fixed per-launch cost from bandwidth.
        // fits t(N) = a + b*N, so a is the launch/ramp floor of the pattern.
        printf("-- sweep: q5_K K=5120, N varies (nwarps=8, rows_per_block=1) --\n");
        for (unsigned n : { 256u, 512u, 1024u, 2176u, 5120u, 17408u }) {
            cfg_t c = { "q5_K 5120 sweep", "q5_K", 256, 32, 2, 176, 5120, n, 8, 0 };
            run_vecdot<vec_dot_q5_K_q8_1, 256, 32, 2, 8>(c, reps);
        }
        printf("\n-- sweep: q5_K K=17408, N varies --\n");
        for (unsigned n : { 256u, 1024u, 5120u }) {
            cfg_t c = { "q5_K 17408 sweep", "q5_K", 256, 32, 2, 176, 17408, n, 8, 0 };
            run_vecdot<vec_dot_q5_K_q8_1, 256, 32, 2, 8>(c, reps);
        }
        run_stream((size_t) 256 * 1024 * 1024, reps);
        return 0;
    }
    for (int i = 0; i < (int) cfgs.size(); ++i) {
        if (idx >= 0 && idx != i) {
            continue;
        }
        const cfg_t & c = cfgs[i];
        switch (c.kind) {
            case 0:
                run_vecdot<vec_dot_q5_K_q8_1, 256, 32, 2, 8>(c, reps);
                break;
            case 1:
                run_vecdot<vec_dot_q6_K_q8_1, 256, 32, 1, 8>(c, reps);
                break;
            case 2:
                run_vecdot<vec_dot_iq4_xs_q8_1, 256, 32, 4, 8>(c, reps);
                break;
            default:
                break;
        }
    }

    run_stream((size_t) 64 * 1024 * 1024, reps);
    return 0;
}
