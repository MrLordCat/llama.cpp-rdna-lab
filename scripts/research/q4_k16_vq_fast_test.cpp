// Q4_K16_VQ fast vs ref check + throughput benchmark (research tool).
//   mode 0: compare quantize_row_q4_K16_VQ_fast vs _ref on random data
//           (block index differences, bit-exact dequant after decode)
//   mode 1: benchmark _fast on n values (wall ms + MB/s)
//
// Build (git-bash, Strawberry MinGW in PATH):
//   g++ -std=c++17 -O3 -mavx2 -fopenmp -I ggml/include -I ggml/src -o scripts/research/q4_k16_vq_fast_test.exe \
//       scripts/research/q4_k16_vq_fast_test.cpp build-cpu/ggml/src/ggml-base.a \
//       build-cpu/ggml/src/ggml-cpu.a C:/Strawberry/c/lib/libgomp.dll.a

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint32_t rng_state = 0x9e3779b9u;
static float frand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float) (rng_state >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

static uint32_t unpack_bits_u32_c(const uint8_t * src, int j, int nbits) {
    int bitpos = j * nbits;
    uint32_t v = 0;
    for (int b = 0; b < nbits; ++b) {
        if (src[bitpos >> 3] & (1u << (bitpos & 7))) {
            v |= 1u << b;
        }
        ++bitpos;
    }
    return v;
}

int main(int argc, char ** argv) {
    int mode = 0;
    int64_t n = 512 * 256; // 256 superblocks
    if (argc > 1) mode = atoi(argv[1]);
    if (argc > 2) n = (int64_t) strtoll(argv[2], nullptr, 10);

    std::vector<float> x(n);
    std::vector<float> qw(n);
    for (int64_t i = 0; i < n; ++i) {
        x[i] = frand() * 2.0f;
        qw[i] = 0.5f + frand() * 0.5f; // positive imatrix-like weights
    }

    if (mode == 0) {
        const int64_t nb = n / QK_K16;
        std::vector<block_q4_K16_VQ> ref(nb), fast(nb);
        quantize_row_q4_K16_VQ_ref(&x[0], &ref[0], n);
        quantize_row_q4_K16_VQ_fast(&x[0], &fast[0], n, nullptr); // same input as ref (no weights)

        int64_t d_block_diff = 0, dmin_diff = 0, qs_diff = 0, idx_diff = 0, sub_diff = 0;
        for (int64_t i = 0; i < nb; ++i) {
            if (ref[i].d != fast[i].d) ++d_block_diff;
            if (ref[i].dmin != fast[i].dmin) ++dmin_diff;
            if (memcmp(ref[i].qs, fast[i].qs, 256) != 0) ++qs_diff;
            for (int j = 0; j < 32; ++j) {
                uint32_t kr = unpack_bits_u32_c(ref[i].idx, j, 10);
                uint32_t kf = unpack_bits_u32_c(fast[i].idx, j, 10);
                if (kr != kf) {
                    ++idx_diff;
                    ++sub_diff;
                }
            }
        }
        printf("blocks=%lld  d_diff=%lld  dmin_diff=%lld  qs_diff=%lld  idx_diff=%lld/%lld (%.4f%%)\n",
               (long long) nb, (long long) d_block_diff, (long long) dmin_diff,
               (long long) qs_diff, (long long) idx_diff, (long long) nb * 32,
               100.0 * idx_diff / (double) (nb * 32));

        // dequant both, compare SNR
        std::vector<float> yr(n), yf(n);
        dequantize_row_q4_K16_VQ(&ref[0], &yr[0], n);
        dequantize_row_q4_K16_VQ(&fast[0], &yf[0], n);
        double sig = 0, err = 0;
        for (int64_t i = 0; i < n; ++i) {
            sig += (double) x[i] * x[i];
            double e = (double) yr[i] - yf[i];
            err += e * e;
        }
        printf("dequant(ref) vs dequant(fast): relative MSE = %.6e\n", err / sig);
    } else if (mode == 3) {
        std::vector<block_q4_K16_VQ> y(n / QK_K16);
        double t0 = now_ms();
        quantize_row_q4_K16_VQ_ref(&x[0], &y[0], n);
        double t1 = now_ms();
        double blocks_per_s = (n / QK_K16) / ((t1 - t0) / 1000.0);
        printf("ref: %.0f values -> %.1f blocks/s\n", (double) n, blocks_per_s);
    } else {
        std::vector<block_q4_K16_VQ> y(n / QK_K16);
        double t0 = now_ms();
        quantize_row_q4_K16_VQ_fast(&x[0], &y[0], n, &qw[0]);
        double t1 = now_ms();
        double dt = (t1 - t0) / 1000.0;
        double blocks_per_s = (n / QK_K16) / dt;
        printf("fast: %.0f values in %.3f s -> %.1f blocks/s, %.2f MB/s, est full model (52.7M blocks, 8 threads): %.1f min\n",
               (double) n, dt, blocks_per_s, (n * 4.0 / 1e6) / dt,
               52.7e6 / (blocks_per_s * 8.0) / 60.0);
    }
    return 0;
}
