// Isolated HIP microbenchmark for the paired narrow Q8_0 GDN matvecs
// 5120 -> 48 (ssm_alpha, ssm_beta): two separate MMVQ launches (current
// decode route) versus one fused two-output launch.
//
// Contract (docs/research/decode/README.md G02): 48 GDN layers, 2 narrow
// Q8_0 matvecs per layer, 96 nodes per token. Deterministic output check
// plus kernel resource reporting; no model is loaded, device 0 only.
//
// Build (Windows, ROCm 7.2):
//   hipcc -std=c++17 -O3 --offload-arch=gfx1201 \
//     scripts/research/g02_gdn_pair_mmvq_probe.cpp \
//     -o build-rocm72/bin/g02-gdn-pair-probe.exe

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_CHECK(expr)                                                                           \
    do {                                                                                          \
        hipError_t err__ = (expr);                                                                \
        if (err__ != hipSuccess) {                                                                \
            std::fprintf(stderr, "%s:%d HIP error: %s\n", __FILE__, __LINE__,                   \
                         hipGetErrorString(err__));                                                \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

struct block_q8_0 {
    __half d;      // Q8_0 scale, fp16
    int8_t qs[32]; // QK8_0 = 32 signed bytes
};
static_assert(sizeof(block_q8_0) == 34, "unexpected Q8_0 block size");

// One warp per output row, K blocks looped over lanes, same reduce order
// for the separate and fused launches so outputs are bit-equivalent.
__global__ void mmvq_q8_0_kernel(const block_q8_0 * __restrict__ x,
                                 const block_q8_0 * __restrict__ w,
                                 float * __restrict__ out,
                                 const int nrows,
                                 const int kblocks) {
    const int row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x;
    float acc = 0.0f;
    for (int i = lane; i < kblocks; i += 32) {
        const block_q8_0 & xb = x[i];
        const block_q8_0 & wb = w[(size_t) row * kblocks + i];
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            s += (float) xb.qs[j] * (float) wb.qs[j];
        }
        acc += (float) __half2float(xb.d) * (float) __half2float(wb.d) * s;
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0x00000000ffffffffULL, acc, off);
    }
    if (lane == 0) {
        out[row] = acc;
    }
}

static void fill_block(block_q8_0 & b, std::mt19937 & rng, const float scale) {
    b.d = __float2half_rn(scale);
    for (int j = 0; j < 32; ++j) {
        b.qs[j] = (int8_t) (rng() % 255 - 127);
    }
}

static float cpu_reference(const block_q8_0 & xb, const block_q8_0 & wb) {
    double s = 0.0;
    for (int j = 0; j < 32; ++j) {
        s += (double) xb.qs[j] * (double) wb.qs[j];
    }
    return (float) ((double) __half2float(xb.d) * (double) __half2float(wb.d) * s);
}

template <typename Fn>
static double bench_gpu_ms(const int warmup, const int reps, Fn && fn) {
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
    return (double) ms / (double) reps;
}

template <typename Fn>
static double bench_cpu_ms(const int warmup, const int reps, Fn && fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    HIP_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    HIP_CHECK(hipDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    return (double) std::chrono::duration<double, std::milli>(t1 - t0).count() / (double) reps;
}

int main() {
    constexpr int K = 5120;
    constexpr int KBLOCKS = K / 32;      // 160 Q8_0 blocks
    constexpr int NROWS = 48;            // alpha or beta rows per layer
    constexpr int LAYERS = 48;           // GDN layers in the decode route
    constexpr int WARMUP = 200;
    constexpr int REPS = 2000;

    HIP_CHECK(hipSetDevice(0));
    int dev = 0;
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    std::mt19937 rng(42u);
    std::vector<block_q8_0> h_x(KBLOCKS);
    std::vector<block_q8_0> h_w_a((size_t) NROWS * KBLOCKS);
    std::vector<block_q8_0> h_w_b((size_t) NROWS * KBLOCKS);
    for (auto & b : h_x) {
        fill_block(b, rng, 0.25f);
    }
    for (auto & b : h_w_a) {
        fill_block(b, rng, 0.5f);
    }
    for (auto & b : h_w_b) {
        fill_block(b, rng, 0.75f);
    }
    // Fused weight layout: [alpha rows, beta rows] -> one launch, both outputs.
    std::vector<block_q8_0> h_w_f((size_t) 2 * NROWS * KBLOCKS);
    std::copy(h_w_a.begin(), h_w_a.end(), h_w_f.begin());
    std::copy(h_w_b.begin(), h_w_b.end(), h_w_f.begin() + (size_t) NROWS * KBLOCKS);

    block_q8_0 * d_x = nullptr;
    block_q8_0 * d_wa = nullptr;
    block_q8_0 * d_wb = nullptr;
    block_q8_0 * d_wf = nullptr;
    float * d_oa = nullptr;
    float * d_ob = nullptr;
    float * d_of = nullptr;
    HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_wa, h_w_a.size() * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_wb, h_w_b.size() * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_wf, h_w_f.size() * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_oa, NROWS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ob, NROWS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_of, 2 * NROWS * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wa, h_w_a.data(), h_w_a.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wb, h_w_b.data(), h_w_b.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wf, h_w_f.data(), h_w_f.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));

    const int block = 32; // one warp per row

    // Separate: exactly the current route (two launches per layer).
    auto launch_a = [&]() {
        mmvq_q8_0_kernel<<<NROWS, block>>>(d_x, d_wa, d_oa, NROWS, KBLOCKS);
        mmvq_q8_0_kernel<<<NROWS, block>>>(d_x, d_wb, d_ob, NROWS, KBLOCKS);
    };
    // Fused: one launch, two outputs.
    auto launch_f = [&]() {
        mmvq_q8_0_kernel<<<2 * NROWS, block>>>(d_x, d_wf, d_of, 2 * NROWS, KBLOCKS);
    };

    HIP_CHECK(hipDeviceSynchronize());
    launch_a();
    launch_f();
    HIP_CHECK(hipDeviceSynchronize());

    // Deterministic output equivalence: fused must equal the two separate
    // launches exactly (same kernel, same order) and meet the float reference.
    std::vector<float> h_oa(NROWS), h_ob(NROWS), h_of(2 * NROWS);
    HIP_CHECK(hipMemcpy(h_oa.data(), d_oa, NROWS * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_ob.data(), d_ob, NROWS * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_of.data(), d_of, 2 * NROWS * sizeof(float), hipMemcpyDeviceToHost));

    double max_abs_fused_vs_separate = 0.0;
    double max_rel_cpu_err = 0.0;
    for (int r = 0; r < NROWS; ++r) {
        max_abs_fused_vs_separate = std::max(max_abs_fused_vs_separate,
            (double) std::fabs(h_of[r] - h_oa[r]));
        max_abs_fused_vs_separate = std::max(max_abs_fused_vs_separate,
            (double) std::fabs(h_of[NROWS + r] - h_ob[r]));
        double cpu_a = 0.0, cpu_b = 0.0;
        for (int i = 0; i < KBLOCKS; ++i) {
            cpu_a += cpu_reference(h_x[i], h_w_a[(size_t) r * KBLOCKS + i]);
            cpu_b += cpu_reference(h_x[i], h_w_b[(size_t) r * KBLOCKS + i]);
        }
        max_rel_cpu_err = std::max(max_rel_cpu_err,
            (double) std::fabs(h_of[r] - cpu_a) / std::max(1e-6, std::fabs(cpu_a)));
        max_rel_cpu_err = std::max(max_rel_cpu_err,
            (double) std::fabs(h_of[NROWS + r] - cpu_b) / std::max(1e-6, std::fabs(cpu_b)));
    }

    hipFuncAttributes attr;
    HIP_CHECK(hipFuncGetAttributes(&attr, (const void *) mmvq_q8_0_kernel));

    const double gpu_a_ms = bench_gpu_ms(WARMUP, REPS, launch_a);
    const double gpu_f_ms = bench_gpu_ms(WARMUP, REPS, launch_f);
    const double cpu_a_ms = bench_cpu_ms(WARMUP, REPS, launch_a);
    const double cpu_f_ms = bench_cpu_ms(WARMUP, REPS, launch_f);

    // Per-token projection: 96 separate launches vs 48 fused launches.
    const double token_a_ms = (double) LAYERS * gpu_a_ms;
    const double token_f_ms = (double) LAYERS * gpu_f_ms;
    const double token_cpu_a_ms = (double) LAYERS * cpu_a_ms;
    const double token_cpu_f_ms = (double) LAYERS * cpu_f_ms;

    std::printf("device=%s arch=%s name=%s\n", prop.gcnArchName, prop.gcnArchName, prop.name);
    std::printf("config K=%d KBLOCKS=%d NROWS=%d LAYERS=%d block=%d Q8_0=%zu bytes\n",
                K, KBLOCKS, NROWS, LAYERS, block, sizeof(block_q8_0));
    std::printf("equivalence fused_vs_separate_max_abs_diff=%.9f cpu_max_rel_err=%.3e\n",
                max_abs_fused_vs_separate, max_rel_cpu_err);
    std::printf("kernel regs=%d shared=%zu local=%zu max_threads=%d\n",
                attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes,
                attr.maxThreadsPerBlock);
    std::printf("mode_a two_launches gpu_ms=%.6f cpu_ms=%.6f launches_per_layer=2\n",
                gpu_a_ms, cpu_a_ms);
    std::printf("mode_b fused_one_launch gpu_ms=%.6f cpu_ms=%.6f launches_per_layer=1\n",
                gpu_f_ms, cpu_f_ms);
    std::printf("token_projection a_gpu_ms=%.4f f_gpu_ms=%.4f delta_gpu_ms=%.4f delta_pct=%.3f\n",
                token_a_ms, token_f_ms, token_a_ms - token_f_ms,
                100.0 * (token_a_ms - token_f_ms) / token_a_ms);
    std::printf("token_projection a_cpu_ms=%.4f f_cpu_ms=%.4f delta_cpu_ms=%.4f delta_pct=%.3f\n",
                token_cpu_a_ms, token_cpu_f_ms, token_cpu_a_ms - token_cpu_f_ms,
                100.0 * (token_cpu_a_ms - token_cpu_f_ms) / token_cpu_a_ms);

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_wa));
    HIP_CHECK(hipFree(d_wb));
    HIP_CHECK(hipFree(d_wf));
    HIP_CHECK(hipFree(d_oa));
    HIP_CHECK(hipFree(d_ob));
    HIP_CHECK(hipFree(d_of));
    return 0;
}
