#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>

#define HIP_CHECK(call) do { \
    hipError_t err__ = (call); \
    if (err__ != hipSuccess) { \
        std::fprintf(stderr, "%s:%d HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

static __global__ void init_kernel(float * x, float * g, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    x[i] = 0.001f * float((i % 257) - 128);
    g[i] = 0.002f * float((i % 251) - 125);
}

static __device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

static __global__ void glu_kernel_256(const float * x, const float * g, float * dst, int64_t k, int64_t n, int64_t o0, int64_t o1) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
    dst[i] = silu(x[j0]) * g[j1];
}

static __global__ void add_kernel_256(const float * x, const float * g, float * dst, int64_t n) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = x[i] + g[i];
}

template <int block_size>
static __global__ void rms_kernel(const float * x, float * dst, int ncols, int64_t stride_row, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    x += int64_t(row) * stride_row;
    dst += int64_t(row) * ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    __shared__ float s_sum[32];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        tmp += __shfl_down(tmp, offset);
    }
    if constexpr (block_size > 64) {
        const int warp_id = threadIdx.x / warpSize;
        const int lane_id = threadIdx.x % warpSize;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = lane_id < block_size / warpSize ? s_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            tmp += __shfl_down(tmp, offset);
        }
    }

    const float scale = rsqrtf(tmp / ncols + eps);
    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static float elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

static float * ptr_at(char * pool, int slot, size_t slot_stride, size_t mod2m) {
    return reinterpret_cast<float *>(pool + size_t(slot) * slot_stride + mod2m);
}

template <typename Launch>
static float bench_kernel(const char * name, int tokens, int warmup, int reps, Launch launch) {
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    HIP_CHECK(hipDeviceSynchronize());

    float total = 0.0f;
    double wall_total = 0.0;
    float max_ms = 0.0f;
    float min_ms = 1.0e30f;
    double wall_max_ms = 0.0;
    double wall_min_ms = 1.0e300;
    for (int i = 0; i < reps; ++i) {
        const auto wall_start = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipEventRecord(start));
        launch();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        const auto wall_end = std::chrono::high_resolution_clock::now();
        const float ms = elapsed_ms(start, stop);
        const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        total += ms;
        wall_total += wall_ms;
        max_ms = ms > max_ms ? ms : max_ms;
        min_ms = ms < min_ms ? ms : min_ms;
        wall_max_ms = wall_ms > wall_max_ms ? wall_ms : wall_max_ms;
        wall_min_ms = wall_ms < wall_min_ms ? wall_ms : wall_min_ms;
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    const float avg = total / reps;
    const double wall_avg = wall_total / reps;
    std::printf("tokens=%d kernel=%s avg_ms=%.6f min_ms=%.6f max_ms=%.6f wall_avg_ms=%.6f wall_min_ms=%.6f wall_max_ms=%.6f\n", tokens, name, avg, min_ms, max_ms, wall_avg, wall_min_ms, wall_max_ms);
    return avg;
}

static void bench_glu_pool_case(const char * label, int tokens, char * pool, size_t slot_stride, size_t x_mod2m, size_t g_mod2m, size_t dst_mod2m) {
    const int ff = 17408;
    const int warmup = 5;
    const int reps = 30;
    const int64_t ff_elems = int64_t(ff) * tokens;

    float * x = ptr_at(pool, 0, slot_stride, x_mod2m);
    float * g = ptr_at(pool, 1, slot_stride, g_mod2m);
    float * dst = ptr_at(pool, 2, slot_stride, dst_mod2m);

    const int init_block = 256;
    const int init_grid = int((ff_elems + init_block - 1) / init_block);
    init_kernel<<<init_grid, init_block>>>(x, g, ff_elems);
    HIP_CHECK(hipDeviceSynchronize());

    std::printf("pool_case=%s tokens=%d x_mod64k=%zu x_mod2m=%zu g_mod64k=%zu g_mod2m=%zu dst_mod64k=%zu dst_mod2m=%zu\n",
        label,
        tokens,
        uintptr_t(x) & 0xffffu,
        uintptr_t(x) & 0x1fffffu,
        uintptr_t(g) & 0xffffu,
        uintptr_t(g) & 0x1fffffu,
        uintptr_t(dst) & 0xffffu,
        uintptr_t(dst) & 0x1fffffu);

    bench_kernel(label, tokens, warmup, reps, [&]() {
        const int block = 256;
        const int grid = int((ff_elems + block - 1) / block);
        glu_kernel_256<<<grid, block>>>(x, g, dst, ff_elems, ff, ff, ff);
    });
}

static void bench_glu_adjacent_case(const char * label, int tokens, char * pool, size_t x_offset) {
    const int ff = 17408;
    const int warmup = 5;
    const int reps = 30;
    const int64_t ff_elems = int64_t(ff) * tokens;
    const size_t bytes = size_t(ff_elems) * sizeof(float);

    float * x = reinterpret_cast<float *>(pool + x_offset);
    float * g = reinterpret_cast<float *>(pool + x_offset + bytes);
    float * dst = reinterpret_cast<float *>(pool + x_offset + 2 * bytes);

    const int init_block = 256;
    const int init_grid = int((ff_elems + init_block - 1) / init_block);
    init_kernel<<<init_grid, init_block>>>(x, g, ff_elems);
    HIP_CHECK(hipDeviceSynchronize());

    std::printf("adjacent_case=%s tokens=%d x_off=%zu bytes=%zu x_mod64k=%zu x_mod2m=%zu g_mod64k=%zu g_mod2m=%zu dst_mod64k=%zu dst_mod2m=%zu\n",
        label,
        tokens,
        x_offset,
        bytes,
        uintptr_t(x) & 0xffffu,
        uintptr_t(x) & 0x1fffffu,
        uintptr_t(g) & 0xffffu,
        uintptr_t(g) & 0x1fffffu,
        uintptr_t(dst) & 0xffffu,
        uintptr_t(dst) & 0x1fffffu);

    bench_kernel(label, tokens, warmup, reps, [&]() {
        const int block = 256;
        const int grid = int((ff_elems + block - 1) / block);
        glu_kernel_256<<<grid, block>>>(x, g, dst, ff_elems, ff, ff, ff);
    });
}

int main() {
    const int ff = 17408;
    const int embd = 5120;
    const int warmup = 5;
    const int reps = 30;
    const std::vector<int> token_counts = {896, 900, 903, 904, 905, 912, 1024};
    const int max_tokens = 1024;
    const int64_t max_ff_elems = int64_t(ff) * max_tokens;
    const int64_t max_embd_elems = int64_t(embd) * max_tokens;

    float * x = nullptr;
    float * g = nullptr;
    float * dst = nullptr;
    HIP_CHECK(hipMalloc(&x, max_ff_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&g, max_ff_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dst, max_ff_elems * sizeof(float)));

    const size_t slot_stride = 128ull * 1024 * 1024;
    const size_t pool_size = 3 * slot_stride + 4ull * 1024 * 1024 + size_t(max_ff_elems) * sizeof(float);
    char * pool = nullptr;
    HIP_CHECK(hipMalloc(&pool, pool_size));

    const int init_block = 256;
    const int init_grid = int((max_ff_elems + init_block - 1) / init_block);
    init_kernel<<<init_grid, init_block>>>(x, g, max_ff_elems);
    HIP_CHECK(hipDeviceSynchronize());

    for (const int tokens : token_counts) {
        const int64_t ff_elems = int64_t(ff) * tokens;
        const int64_t embd_elems = int64_t(embd) * tokens;
        bench_kernel("glu256_ff", tokens, warmup, reps, [&]() {
            const int block = 256;
            const int grid = int((ff_elems + block - 1) / block);
            glu_kernel_256<<<grid, block>>>(x, g, dst, ff_elems, ff, ff, ff);
        });
        bench_kernel("add256_embd", tokens, warmup, reps, [&]() {
            const int block = 256;
            const int grid = int((embd_elems + block - 1) / block);
            add_kernel_256<<<grid, block>>>(x, g, dst, embd_elems);
        });
        bench_kernel("rms1024_embd", tokens, warmup, reps, [&]() {
            rms_kernel<1024><<<tokens, 1024, 32 * sizeof(float)>>>(x, dst, embd, embd, 1.0e-6f);
        });
    }

    bench_glu_pool_case("pool_trace903_a", 903, pool, slot_stride, 856704, 819840, 782976);
    bench_glu_pool_case("pool_trace903_b", 903, pool, slot_stride, 856704, 320000, 2003584);
    bench_glu_pool_case("pool_trace904_a", 904, pool, slot_stride, 1028736, 1061504, 1094272);
    bench_glu_pool_case("pool_trace904_b", 904, pool, slot_stride, 1028736, 340480, 242304);
    bench_glu_adjacent_case("adjacent_mod904", 904, pool, 1028736);
    bench_glu_adjacent_case("adjacent_real904", 904, pool, 155824768);
    bench_glu_adjacent_case("adjacent_real903_base393216", 903, pool, 393216 + 155652736);
    bench_glu_adjacent_case("adjacent_real904_base393216", 904, pool, 393216 + 155824768);

    HIP_CHECK(hipFree(pool));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipFree(g));
    HIP_CHECK(hipFree(x));
    return 0;
}
