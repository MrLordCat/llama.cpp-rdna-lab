// Q4_K16 bit-exactness harness (research/q4-k16-quant, WIP tool).
// Reads f32 x (N x 512) and optional imatrix qw, quantizes with the C++
// implementation via ggml_quantize_chunk, and dumps, per block:
//   d (fp16 u16 LE), dmin (fp16 u16 LE), ls[32], lm[32], qs[256] (raw)
// plus the dequantized f32 row (N x 512).
// Compare against subProject_q4/prototype/quants.py via q4_k16_bitcheck.py.
//
// Build (git-bash, Strawberry MinGW in PATH):
//   g++ -std=c++17 -O2 -I ggml/include -I ggml/src -o /tmp/q4_k16_harness.exe \
//       scripts/research/q4_k16_harness.cpp build-cpu/ggml/src/ggml-base.a

#include "ggml.h"
#include "ggml-common.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint8_t unpack_bits_u8(const uint8_t * src, int j, int nbits) {
    int bitpos = j * nbits;
    uint8_t v = 0;
    for (int b = 0; b < nbits; ++b) {
        if (src[bitpos >> 3] & (1u << (bitpos & 7))) {
            v |= (uint8_t) (1u << b);
        }
        ++bitpos;
    }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s x.f32 cfg out.bin [qw.f32]\n", argv[0]);
        fprintf(stderr, "  cfg: b77 (Q4_K16_M) | b76 (Q4_K16) | e55 (Q4_K16_S)\n");
        return 1;
    }

    ggml_type type;
    int sc_bits, min_bits;
    if (strcmp(argv[2], "b77") == 0) { type = GGML_TYPE_Q4_K16_M; sc_bits = 7; min_bits = 7; }
    else if (strcmp(argv[2], "b76") == 0) { type = GGML_TYPE_Q4_K16; sc_bits = 7; min_bits = 6; }
    else if (strcmp(argv[2], "e55") == 0) { type = GGML_TYPE_Q4_K16_S; sc_bits = 5; min_bits = 5; }
    else {
        fprintf(stderr, "unknown cfg %s\n", argv[2]);
        return 1;
    }

    const int64_t n_per_row = 512;

    FILE * fx = fopen(argv[1], "rb");
    if (!fx) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fseek(fx, 0, SEEK_END);
    const int64_t nbytes = ftell(fx);
    fseek(fx, 0, SEEK_SET);
    if (nbytes % (n_per_row * 4) != 0) {
        fprintf(stderr, "x size not a multiple of 512 f32\n");
        return 1;
    }
    const int64_t nrows = nbytes / (n_per_row * 4);
    std::vector<float> x(nbytes / 4);
    if (fread(x.data(), 1, nbytes, fx) != (size_t) nbytes) { fprintf(stderr, "read x failed\n"); return 1; }
    fclose(fx);

    std::vector<float> qw;
    const float * qw_ptr = nullptr;
    if (argc >= 5) {
        FILE * fq = fopen(argv[4], "rb");
        if (!fq) { fprintf(stderr, "cannot open %s\n", argv[5]); return 1; }
        fseek(fq, 0, SEEK_END);
        const int64_t qbytes = ftell(fq);
        fseek(fq, 0, SEEK_SET);
        if (qbytes != nbytes) { fprintf(stderr, "qw size mismatch\n"); return 1; }
        qw.resize(nbytes / 4);
        if (fread(qw.data(), 1, nbytes, fq) != (size_t) nbytes) { fprintf(stderr, "read qw failed\n"); return 1; }
        fclose(fq);
        qw_ptr = qw.data();
    }

    const size_t row_size = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> dst(nrows * row_size);

    const size_t quantized = ggml_quantize_chunk(type, x.data(), dst.data(), 0, nrows, n_per_row, qw_ptr);
    if (quantized != nrows * row_size) {
        fprintf(stderr, "quantize_chunk returned %zu, expected %zu\n", quantized, nrows * row_size);
        return 1;
    }

    if (argc >= 7 && strcmp(argv[6], "debugw") == 0 && qw_ptr) {
        // debug: dump the imatrix weights as the C++ impl would see them
        // (row 0, sub-blocks 0..2, first 16 entries each), f32 raw to stdout
        const int64_t nb = n_per_row / 512;
        for (int64_t i = 0; i < nb && i < 1; ++i) {
            float sum_x2 = 0;
            for (int l = 0; l < 512; ++l) sum_x2 += x[i*512 + l]*x[i*512 + l];
            const float sigma2 = 2*sum_x2/512;
            fprintf(stdout, "sigma2=%.9g\n", sigma2);
            for (int j = 0; j < 3; ++j) {
                for (int l = 0; l < 16; ++l) {
                    const float w = qw_ptr[i*512 + 16*j + l] * sqrtf(sigma2 + x[i*512 + 16*j + l]*x[i*512 + 16*j + l]);
                    fprintf(stdout, "%.9g%c", w, l == 15 ? '\n' : ' ');
                }
            }
        }
        return 0;
    }

    const int sc_bytes = (32 * sc_bits + 7) / 8;
    const int m_bytes   = (32 * min_bits + 7) / 8;

    FILE * fo = fopen(argv[3], "wb");
    if (!fo) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }

    std::vector<float> deq(nrows * n_per_row);
    for (int64_t i = 0; i < nrows; ++i) {
        const uint8_t * block = dst.data() + i * row_size;
        const ggml_fp16_t d    = *(const ggml_fp16_t *) block;
        const ggml_fp16_t dmin = *(const ggml_fp16_t *) (block + 2);
        const uint8_t * sc_stream = block + 4;
        const uint8_t * m_stream  = sc_stream + sc_bytes;
        const uint8_t * qs        = m_stream + m_bytes;

        fwrite(&d, 2, 1, fo);
        fwrite(&dmin, 2, 1, fo);
        uint8_t ls[32], lm[32];
        for (int j = 0; j < 32; ++j) {
            ls[j] = unpack_bits_u8(sc_stream, j, sc_bits);
            lm[j] = unpack_bits_u8(m_stream, j, min_bits);
        }
        fwrite(ls, 32, 1, fo);
        fwrite(lm, 32, 1, fo);
        fwrite(qs, 256, 1, fo);

        const float d_f    = ggml_fp16_to_fp32(d);
        const float dmin_f = ggml_fp16_to_fp32(dmin);
        for (int j = 0; j < 32; ++j) {
            const float dd = d_f * ls[j];
            const float dm = dmin_f * lm[j];
            for (int ii = 0; ii < 16; ++ii) {
                const uint8_t qv = qs[8*j + (ii >> 1)];
                const uint8_t l = (ii & 1) ? (qv >> 4) : (qv & 0xF);
                deq[i * n_per_row + 16*j + ii] = dd * l - dm;
            }
        }
    }
    fclose(fo);

    FILE * fd = fopen((std::string(argv[3]) + ".deq").c_str(), "wb");
    if (!fd) { fprintf(stderr, "cannot open deq out\n"); return 1; }
    fwrite(deq.data(), 4, deq.size(), fd);
    fclose(fd);

    printf("cfg=%s nrows=%lld row_size=%zu -> %s (+.deq)\n", argv[2], (long long) nrows, row_size, argv[3]);
    return 0;
}
