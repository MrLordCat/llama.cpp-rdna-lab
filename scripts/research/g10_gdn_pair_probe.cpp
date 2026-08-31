// Isolated HIP probe for the G10 fused narrow f32 GDN pair kernel:
// two separate mul_mat_vec_f<float> launches (ssm_alpha, ssm_beta,
// 5120 -> 48, n=1) versus the single two-output launch at the same
// block size / reduce order. Verifies bit-exactness on random inputs.
//
// Build (Windows, ROCm 7.2):
//   hipcc -std=c++17 -O3 --offload-arch=gfx1201 \
//     scripts/research/g10_gdn_pair_probe.cpp \
//     -o build-rocm72/bin/g10-gdn-pair-probe.exe
//
// Run: build-rocm72/bin/g10-gdn-pair-probe.exe [ncols nrows iters]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_CHECK(expr)                                                                                \
    do {                                                                                               \
        hipError_t err__ = (expr);                                                                     \
        if (err__ != hipSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
            std::exit(1);                                                                              \
        }                                                                                              \
    } while (0)

__device__ __forceinline__ float ggml_mad(float acc, float a, float b) {
    return fmaf(a, b, acc);
}

__device__ __forceinline__ float warp_reduce_sum_32(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffull, x, offset, 32);
    }
    return x;
}

// Replica of the server mul_mat_vec_f<float,float,1,256> inner loop and
// reduce order (fusion off, ncols_dst=1, block_size=256, warp 32).
__global__ void mmvf_f32_256(const float * __restrict__ x, const float * __restrict__ y,
                             float * __restrict__ dst, const int nrows, const int ncols2) {
    const int row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    constexpr int warp_size = 32;

    const float2 * x2 = (const float2 *) x;
    const float2 * y2 = (const float2 *) y;

    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += 256) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        ggml_mad(sumf, tmpx.x, tmpy.x);
        ggml_mad(sumf, tmpx.y, tmpy.y);
    }

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;

    sumf = warp_reduce_sum_32(sumf);

    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();
    buf_iw[tid/warp_size] = sumf;
    __syncthreads();
    if (tid < warp_size) {
        sumf = warp_reduce_sum_32(buf_iw[tid]);
    }

    if (tid != 0) {
        return;
    }
    dst[row] = sumf;
}

// The fused G10 kernel (same order as the server).
__global__ void mmvf_f32_pair_256(const float * __restrict__ xa, const float * __restrict__ xb,
                                  const float * __restrict__ y,
                                  float * __restrict__ dst_a, float * __restrict__ dst_b,
                                  const int nrows, const int ncols2, const int stride_a, const int stride_b) {
    const int row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    constexpr int warp_size = 32;

    const float2 * a2 = (const float2 *) (xa + (size_t) row*stride_a);
    const float2 * b2 = (const float2 *) (xb + (size_t) row*stride_b);
    const float2 * y2 = (const float2 *) y;

    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += 256) {
        const float2 tmpy = y2[col2];
        const float2 tmpa = a2[col2];
        const float2 tmpb = b2[col2];
        ggml_mad(sum_a, tmpa.x, tmpy.x);
        ggml_mad(sum_a, tmpa.y, tmpy.y);
        ggml_mad(sum_b, tmpb.x, tmpy.x);
        ggml_mad(sum_b, tmpb.y, tmpy.y);
    }

    extern __shared__ char data_mmv[];
    float * buf_a = (float *) data_mmv;
    float * buf_b = buf_a + warp_size;

    sum_a = warp_reduce_sum_32(sum_a);
    sum_b = warp_reduce_sum_32(sum_b);

    if (tid < warp_size) {
        buf_a[tid] = 0.0f;
        buf_b[tid] = 0.0f;
    }
    __syncthreads();
    buf_a[tid/warp_size] = sum_a;
    buf_b[tid/warp_size] = sum_b;
    __syncthreads();
    if (tid < warp_size) {
        sum_a = warp_reduce_sum_32(buf_a[tid]);
        sum_b = warp_reduce_sum_32(buf_b[tid]);
    }

    if (tid != 0) {
        return;
    }
    dst_a[row] = sum_a;
    dst_b[row] = sum_b;
}

int main(int argc, char ** argv) {
    int ncols = argc > 1 ? atoi(argv[1]) : 5120;
    int nrows = argc > 2 ? atoi(argv[2]) : 48;
    int iters = argc > 3 ? atoi(argv[3]) : 100;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_x(ncols), h_y(ncols), h_a((size_t) nrows*ncols), h_b((size_t) nrows*ncols);
    for (int i = 0; i < ncols; ++i) h_x[i] = dist(rng);
    for (int i = 0; i < ncols; ++i) h_y[i] = dist(rng);
    for (size_t i = 0; i < h_a.size(); ++i) { h_a[i] = dist(rng); h_b[i] = dist(rng); }

    float * d_x, * d_y, * d_a, * d_b, * d_out_a, * d_out_b;
    HIP_CHECK(hipMalloc(&d_x, sizeof(float)*ncols));
    HIP_CHECK(hipMalloc(&d_y, sizeof(float)*ncols));
    HIP_CHECK(hipMalloc(&d_a, sizeof(float)*h_a.size()));
    HIP_CHECK(hipMalloc(&d_b, sizeof(float)*h_b.size()));
    HIP_CHECK(hipMalloc(&d_out_a, sizeof(float)*nrows));
    HIP_CHECK(hipMalloc(&d_out_b, sizeof(float)*nrows));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), sizeof(float)*ncols, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), sizeof(float)*ncols, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), sizeof(float)*h_a.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), sizeof(float)*h_b.size(), hipMemcpyHostToDevice));

    std::vector<float> res_a(nrows), res_b(nrows), ref_a(nrows), ref_b(nrows);
    const size_t shmem = 64*sizeof(float);

    // fused
    HIP_CHECK(hipMemset(d_out_a, 0, sizeof(float)*nrows));
    HIP_CHECK(hipMemset(d_out_b, 0, sizeof(float)*nrows));
    mmvf_f32_pair_256<<<nrows, 256, shmem>>>(d_a, d_b, d_y, d_out_a, d_out_b, nrows, ncols/2, ncols, ncols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(res_a.data(), d_out_a, sizeof(float)*nrows, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res_b.data(), d_out_b, sizeof(float)*nrows, hipMemcpyDeviceToHost));

    // separate (x = weights as in the server kernel)
    HIP_CHECK(hipMemset(d_out_a, 0, sizeof(float)*nrows));
    HIP_CHECK(hipMemset(d_out_b, 0, sizeof(float)*nrows));
    mmvf_f32_256<<<nrows, 256, 128>>>(d_a, d_y, d_out_a, nrows, ncols/2);
    HIP_CHECK(hipGetLastError());
    mmvf_f32_256<<<nrows, 256, 128>>>(d_b, d_y, d_out_b, nrows, ncols/2);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(ref_a.data(), d_out_a, sizeof(float)*nrows, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ref_b.data(), d_out_b, sizeof(float)*nrows, hipMemcpyDeviceToHost));

    int bad_a = 0, bad_b = 0;
    for (int r = 0; r < nrows; ++r) {
        if (res_a[r] != ref_a[r]) ++bad_a;
        if (res_b[r] != ref_b[r]) ++bad_b;
    }
    std::printf("bit-exact: alpha %d/%d, beta %d/%d\n", nrows - bad_a, nrows, nrows - bad_b, nrows);
    for (int r = 0; r < nrows && (bad_a || bad_b); ++r) {
        if (res_a[r] != ref_a[r] || res_b[r] != ref_b[r]) {
            std::printf("row %d: fused_a=%.9g ref_a=%.9g fused_b=%.9g ref_b=%.9g\n", r, res_a[r], ref_a[r], res_b[r], ref_b[r]);
        }
    }

    // quick timing
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        mmvf_f32_pair_256<<<nrows, 256, shmem>>>(d_a, d_b, d_y, d_out_a, d_out_b, nrows, ncols/2, ncols, ncols);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double fused_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()/iters;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        mmvf_f32_256<<<nrows, 256, 128>>>(d_a, d_y, d_out_a, nrows, ncols/2);
        mmvf_f32_256<<<nrows, 256, 128>>>(d_b, d_y, d_out_b, nrows, ncols/2);
    }
    HIP_CHECK(hipDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    double sep_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()/iters;

    std::printf("timing: fused=%.4f ms (per pair), separate=%.4f ms (per pair), ratio=%.3f\n",
                fused_ms, sep_ms, fused_ms/sep_ms);

    hipFree(d_x); hipFree(d_y); hipFree(d_a); hipFree(d_b); hipFree(d_out_a); hipFree(d_out_b);
    return (bad_a || bad_b) ? 1 : 0;
}
