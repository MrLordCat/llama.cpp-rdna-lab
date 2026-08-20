// Q4_K16_VQ model file builder — C++ fast path (research tool).
// Reads the base GGUF (Q4_K16* tensors to replace), the bf16 source shards
// and the imatrix, quantizes every Q4_K16_M/K/S tensor to Q4_K16_VQ with the
// AVX2 fast kernel (quantize_row_q4_K16_VQ_fast), and dumps the VQ tensors
// into a precomputed binary consumed by requant_vq.py --precomputed.
//
// usage: q4_k16_vq_quant.exe <base.gguf> <bf16a.gguf> <bf16b.gguf> <imatrix.gguf|-> <out.bin> [max_tensors]
//
// Build (git-bash, Strawberry MinGW in PATH):
//   gcc -c -std=gnu11 -O3 -mavx2 -fopenmp -DNDEBUG -I ggml/include -I ggml/src \
//       ggml/src/ggml-quants.c -o /tmp/vq_quants.o
//   g++ -std=c++17 -O3 -mavx2 -fopenmp -I ggml/include -I ggml/src \
//       -o scripts/research/q4_k16_vq_quant.exe scripts/research/q4_k16_vq_quant.cpp \
//       /tmp/vq_quants.o build-cpu/ggml/src/ggml-base.a build-cpu/ggml/src/ggml-cpu.a \
//       C:/Strawberry/c/lib/libgomp.dll.a

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include "gguf.h"

#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static size_t file_read_at(const char * path, uint64_t offset, void * dst, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
#ifdef _WIN32
    if (_fseeki64(f, (int64_t) offset, SEEK_SET) != 0) {
#else
    if (fseek(f, (long) offset, SEEK_SET) != 0) {
#endif
        fprintf(stderr, "fseek failed on %s\n", path);
        exit(1);
    }
    size_t got = fread(dst, 1, n, f);
    fclose(f);
    return got;
}

struct src_tensor {
    std::string name;
    std::string path;
    uint64_t offset = 0;
    uint64_t size = 0; // bytes (bf16)
};

static bool find_src(const std::vector<src_tensor> & tensors, const std::string & name, std::string & path, uint64_t & offset, uint64_t & size) {
    for (const auto & t : tensors) {
        if (t.name == name) {
            path = t.path;
            offset = t.offset;
            size = t.size;
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <base.gguf> <bf16a.gguf> <bf16b.gguf> <imatrix.gguf|-> <out.bin> [max_tensors]\n", argv[0]);
        return 1;
    }
    const char * base_path  = argv[1];
    const char * bf16a_path = argv[2];
    const char * bf16b_path = argv[3];
    const char * imat_path  = strcmp(argv[4], "-") == 0 ? nullptr : argv[4];
    const char * out_path   = argv[5];
    const int max_tensors   = argc > 6 ? atoi(argv[6]) : 0;

    const int n_threads = omp_get_max_threads();
    printf("threads=%d\n", n_threads);
    fflush(stdout);

    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * base = gguf_init_from_file(base_path, params);
    if (!base) {
        fprintf(stderr, "failed to read base %s\n", base_path);
        return 1;
    }

    // index the bf16 shards
    std::vector<src_tensor> src;
    for (const char * p : { bf16a_path, bf16b_path }) {
        gguf_context * shard = gguf_init_from_file(p, params);
        if (!shard) {
            fprintf(stderr, "failed to read shard %s\n", p);
            return 1;
        }
        const int64_t n = gguf_get_n_tensors(shard);
        for (int64_t i = 0; i < n; ++i) {
            src_tensor t;
            t.name = gguf_get_tensor_name(shard, i);
            t.path = p;
            t.offset = (uint64_t) gguf_get_data_offset(shard) + (uint64_t) gguf_get_tensor_offset(shard, i);
            t.size = (uint64_t) gguf_get_tensor_size(shard, i);
            src.push_back(std::move(t));
        }
        gguf_free(shard);
    }

    // imatrix (weights per graph-row column)
    gguf_context * imat = nullptr;
    if (imat_path) {
        imat = gguf_init_from_file(imat_path, params);
        if (!imat) {
            fprintf(stderr, "failed to read imatrix %s\n", imat_path);
            return 1;
        }
    }

    FILE * out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "cannot create %s\n", out_path);
        return 1;
    }

    // header: u64 count (patched after we know it)
    uint64_t count = 0;
    fwrite(&count, sizeof(count), 1, out);

    const int64_t n_tensors = gguf_get_n_tensors(base);
    int processed = 0;

    // warm up the candidate LUT single-threaded
    {
        float x0[512] = { 0 };
        block_q4_K16_VQ b0;
        quantize_row_q4_K16_VQ_fast(x0, &b0, 512, nullptr);
    }

    for (int64_t tid = 0; tid < n_tensors; ++tid) {
        const char * name = gguf_get_tensor_name(base, tid);
        const enum ggml_type type = gguf_get_tensor_type(base, tid);
        if (type != GGML_TYPE_Q4_K16_M && type != GGML_TYPE_Q4_K16 && type != GGML_TYPE_Q4_K16_S) {
            continue;
        }
        if (max_tensors && ++processed > max_tensors) {
            continue;
        }

        uint64_t off = 0, size = 0;
        std::string src_path;
        if (!find_src(src, name, src_path, off, size)) {
            fprintf(stderr, "tensor %s not found in bf16 shards\n", name);
            return 1;
        }
        // bf16 source: 2 bytes per element; this defines n_elements
        const size_t n_el = (size_t) (size / 2);
        if (size % 2 != 0) {
            fprintf(stderr, "odd bf16 size for %s\n", name);
            return 1;
        }

        // sanity check against the base file size (block bytes = full block)
        const size_t block_bytes = type == GGML_TYPE_Q4_K16_M ? sizeof(block_q4_K16_M)
                                 : type == GGML_TYPE_Q4_K16   ? sizeof(block_q4_K16)
                                                               : sizeof(block_q4_K16_S);
        const size_t stored_bytes = gguf_get_tensor_size(base, tid);
        const size_t expect_stored = ((n_el + 511) / 512) * block_bytes;
        if (stored_bytes != expect_stored) {
            fprintf(stderr, "size mismatch for %s: base %llu vs expected %llu (n_el %llu)\n", name,
                    (unsigned long long) stored_bytes, (unsigned long long) expect_stored,
                    (unsigned long long) n_el);
            return 1;
        }

        const double t0 = now_ms();
        std::vector<uint16_t> bf16(n_el);
        if (file_read_at(src_path.c_str(), off, bf16.data(), size) != size) {
            fprintf(stderr, "short read for %s\n", name);
            return 1;
        }

        std::vector<float> x(n_el);
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) bf16.data(), x.data(), (int64_t) n_el);

        // column-wise imatrix weights for this tensor
        std::vector<float> qw_cols;
        int64_t n_weights_per_row = 0;
        if (imat) {
            const std::string sname = std::string(name) + ".in_sum2";
            const int64_t wid = gguf_find_tensor(imat, sname.c_str());
            if (wid >= 0) {
                const size_t sum_size = gguf_get_tensor_size(imat, wid);
                const uint64_t sum_off = (uint64_t) gguf_get_data_offset(imat) + (uint64_t) gguf_get_tensor_offset(imat, wid);
                std::vector<float> sums(sum_size / sizeof(float));
                if (file_read_at(imat_path, sum_off, sums.data(), sum_size) != sum_size) {
                    fprintf(stderr, "short imatrix read for %s\n", name);
                    return 1;
                }
                const std::string cname = std::string(name) + ".counts";
                const int64_t cid = gguf_find_tensor(imat, cname.c_str());
                std::vector<float> counts;
                if (cid >= 0) {
                    const size_t csize = gguf_get_tensor_size(imat, cid);
                    const uint64_t coff = (uint64_t) gguf_get_data_offset(imat) + (uint64_t) gguf_get_tensor_offset(imat, cid);
                    // counts are stored as F32 (one per calibration matrix)
                    counts.resize(csize / sizeof(float));
                    if (file_read_at(imat_path, coff, counts.data(), csize) != csize) {
                        fprintf(stderr, "short counts read for %s\n", name);
                        return 1;
                    }
                }
                const int64_t n_mat = counts.empty() ? 1 : (int64_t) counts.size();
                n_weights_per_row = (int64_t) (sums.size() / (size_t) n_mat);
                qw_cols.resize((size_t) (n_mat * n_weights_per_row));
                for (int64_t r = 0; r < n_mat; ++r) {
                    const float cnt = counts.empty() ? 1.0f : counts[(size_t) r];
                    for (int64_t c = 0; c < n_weights_per_row; ++c) {
                        qw_cols[(size_t) (r * n_weights_per_row + c)] = sums[(size_t) (r * n_weights_per_row + c)] / cnt;
                    }
                }
            }
        }

        const size_t n_blocks = (n_el + 511) / 512;
        if (n_el % 512 != 0) {
            x.resize(n_blocks * 512, 0.0f);
        }
        std::vector<block_q4_K16_VQ> y(n_blocks);

        const bool has_w = !qw_cols.empty();
        const int64_t nprow = n_weights_per_row;

#pragma omp parallel for schedule(dynamic, 16)
        for (int64_t b = 0; b < (int64_t) n_blocks; ++b) {
            const float * qw = nullptr;
            if (has_w) {
                const int64_t c0 = (int64_t) ((b * 512) % nprow);
                // wrap-around column tiling (same as fast_tile in requant)
                static thread_local std::vector<float> qw_buf;
                if (c0 + 512 <= nprow) {
                    qw = qw_cols.data() + c0;
                } else {
                    qw_buf.resize(512);
                    for (int i = 0; i < 512; ++i) {
                        qw_buf[i] = qw_cols[(size_t) ((c0 + i) % nprow)];
                    }
                    qw = qw_buf.data();
                }
            }
            quantize_row_q4_K16_VQ_fast(x.data() + b * 512, y.data() + b, 512, qw);
        }

        const uint64_t data_len = (uint64_t) n_blocks * sizeof(block_q4_K16_VQ);
        const uint64_t name_len = (uint64_t) strlen(name);
        fwrite(&name_len, sizeof(name_len), 1, out);
        fwrite(name, 1, (size_t) name_len, out);
        fwrite(&data_len, sizeof(data_len), 1, out);
        fwrite(y.data(), 1, (size_t) data_len, out);
        ++count;

        printf("tensor %-36s n_el=%12llu blocks=%8llu -> %8.1f MB (%.1f s)\n", name,
               (unsigned long long) n_el, (unsigned long long) n_blocks,
               (double) data_len / 1e6, (now_ms() - t0) / 1000.0);
        fflush(stdout);
    }

    // patch the count
    fseek(out, 0, SEEK_SET);
    fwrite(&count, sizeof(count), 1, out);
    fclose(out);

    if (imat) {
        gguf_free(imat);
    }
    gguf_free(base);
    printf("done: %llu tensors -> %s\n", (unsigned long long) count, out_path);
    return 0;
}
