// W21 probe: gfx1201 WMMA-I8 (m16n16k16, int8 x int8 -> i32) vs DP4A.
// 8 independent accumulators per lane to saturate the pipeline.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

using v2i32 = __attribute__((ext_vector_type(2))) int;
using v8i32 = __attribute__((ext_vector_type(8))) int;

__device__ __forceinline__ v8i32 wmma_i8_probe(v2i32 a, v2i32 b, v8i32 c) {
    v8i32 r;
    r = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12((bool)true, a, (bool)true, b, c, (bool)true);
    return r;
}

__device__ __forceinline__ int dp4a_probe(uint32_t a, uint32_t b, int c) {
    int r = 0;
    asm volatile("v_dot4_i32_iu8 %0, %1, %2, %3\n" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}

// WMMA: one 16x16x16 tile per 32-lane wave; each lane processes 8 bytes of A,
// 8 bytes of B, accumulates 8 i32.
__global__ void bench_wmma_i8(int32_t *dst, const int32_t *a, const int32_t *b, int n) {
    v8i32 acc[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 8; ++i) acc[k][i] = 0;
    }
    v2i32 av = { a[0], a[1] };
    v2i32 bv = { b[0], b[1] };
    for (int i = 0; i < n; ++i) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) acc[k] = wmma_i8_probe(av, bv, acc[k]);
    }
    int s = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) for (int i = 0; i < 8; ++i) s += acc[k][i];
    dst[threadIdx.x] = s;
}

__global__ void bench_dp4a(uint32_t *dst, const uint32_t *a, const uint32_t *b, int n) {
    int acc[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 0;
    for (int i = 0; i < n; ++i) {
        const uint32_t x = a[i & 3];
        const uint32_t y = b[i & 3];
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] = dp4a_probe(x, y, acc[k]);
    }
    int s = 0;
    #pragma unroll
    for (int k = 0; k < 16; ++k) s += acc[k];
    dst[threadIdx.x] = s;
}

int main() {
    const int N = 20000000;
    uint32_t *da, *db, *dd;
    hipMalloc(&da, 16); hipMalloc(&db, 16); hipMalloc(&dd, 4096);
    hipMemset(da, 0x7B, 16); hipMemset(db, 0x35, 16);
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);

    bench_wmma_i8<<<128, 256>>>((int32_t*)dd, (const int32_t*)(const uint32_t*)da, (const int32_t*)(const uint32_t*)db, 1024);
    bench_dp4a<<<128, 256>>>(dd, da, db, 1024);
    hipDeviceSynchronize();

    for (int rep = 0; rep < 3; ++rep) {
        hipEventRecord(s);
        bench_dp4a<<<128, 256>>>(dd, da, db, N);
        hipEventRecord(e); hipDeviceSynchronize();
        float ms1; hipEventElapsedTime(&ms1, s, e);

        hipEventRecord(s);
        bench_wmma_i8<<<128, 256>>>((int32_t*)dd, (const int32_t*)(const uint32_t*)da, (const int32_t*)(const uint32_t*)db, N);
        hipEventRecord(e); hipDeviceSynchronize();
        float ms2; hipEventElapsedTime(&ms2, s, e);

        // DP4A: N iter * 16 accum * 4 MAC = 64*N MAC/lane
        // WMMA: N iter * 4 tiles * 4096 MAC/32 lanes = 512*N MAC/lane
        const double mac_dp = (double)N * 16 * 4 * 128 * 256;      // lanes * 4 MAC
        const double mac_wm = (double)N * 4 * (4096.0/32.0) * 128 * 256; // per lane 128 MAC * lanes
        printf("rep %d: dp4a %7.3f ms (%.1f GMAC/s)  wmma_i8 %7.3f ms (%.1f GMAC/s)  ratio %.2fx\n",
               rep, ms1, mac_dp/ms1/1e6, ms2, mac_wm/ms2/1e6, (mac_wm/ms2)/(mac_dp/ms1));
    }
    return 0;
}
