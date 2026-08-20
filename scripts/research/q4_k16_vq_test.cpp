// Q4_K16_VQ CPU validation harness (subProject_q4/docs/10_VQ_K1024_IMPLEMENTATION.md §8.1).
// Reads f32 input, quantizes with quantize_row_q4_K16_VQ_ref, dequantizes with
// dequantize_row_q4_K16_VQ, and dumps both the raw block bytes and the
// dequantized values for byte-for-byte comparison against the numpy reference
// (subProject_q4/prototype/lab/vq_c_bitcheck.py).
//
// Usage: q4_k16_vq_test.exe <input.f32> <n_values> <out_blocks.bin> <out_deq.bin>
//
// Build (git-bash, Strawberry MinGW in PATH):
//   g++ -std=c++17 -O2 -I ggml/include -I ggml/src -o scripts/research/q4_k16_vq_test.exe \
//       scripts/research/q4_k16_vq_test.cpp build-cpu/ggml/src/ggml-base.a \
//       build-cpu/ggml/src/ggml-cpu.a C:/Strawberry/c/lib/libgomp.dll.a

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <input.f32> <n_values> <out_blocks.bin> <out_deq.bin>\n", argv[0]);
        return 1;
    }

    const char * in_path   = argv[1];
    const int64_t n        = (int64_t) strtoll(argv[2], nullptr, 10);
    const char * blk_path  = argv[3];
    const char * deq_path  = argv[4];

    if (n <= 0 || n % QK_K16 != 0) {
        fprintf(stderr, "n must be a positive multiple of %d\n", QK_K16);
        return 1;
    }

    std::vector<float> x(n);
    FILE * f = fopen(in_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
    if (fread(x.data(), sizeof(float), n, f) != (size_t) n) {
        fprintf(stderr, "short read on %s\n", in_path);
        fclose(f);
        return 1;
    }
    fclose(f);

    const int64_t nb = n / QK_K16;
    std::vector<block_q4_K16_VQ> blocks(nb);
    std::vector<float> y(n);

    quantize_row_q4_K16_VQ_ref(x.data(), blocks.data(), n);
    dequantize_row_q4_K16_VQ(blocks.data(), y.data(), n);

    f = fopen(blk_path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", blk_path); return 1; }
    fwrite(blocks.data(), sizeof(block_q4_K16_VQ), nb, f);
    fclose(f);

    f = fopen(deq_path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", deq_path); return 1; }
    fwrite(y.data(), sizeof(float), n, f);
    fclose(f);

    printf("wrote %d blocks (%zu bytes each) and %lld dequantized values\n",
           (int) nb, sizeof(block_q4_K16_VQ), (long long) n);
    return 0;
}
