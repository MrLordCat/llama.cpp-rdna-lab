// D094 c8 diagnostic harness: run ggml MUL_MAT with eh_proj-like shapes on a
// given backend and dump the output for byte-exact backend comparison.
//
// Usage: dump_nextn <backend> <outfile>
//   backend: Vulkan0 | ROCm0 | CPU
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

static void fill_random(float * p, size_t n, uint64_t seed) {
    uint64_t x = seed ? seed : 42;
    for (size_t i = 0; i < n; ++i) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        p[i] = (float) ((int64_t) (x >> 33) % 2000) / 1000.0f - 1.0f;
    }
}

static void run_mul_mat(ggml_backend_t backend, ggml_type type_a, int64_t m, int64_t n, int64_t k,
                        const char * outfile, ggml_type type_b = GGML_TYPE_F32, bool q8_1_d4 = false) {
    struct ggml_init_params ip = { /*.mem_size=*/ 16ull*1024*1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, type_a, k, m); // [k, m]
    ggml_tensor * b = ggml_new_tensor_2d(ctx, type_b, k, n); // [k, n]
    ggml_tensor * out = ggml_mul_mat(ctx, a, b); // [m, n]

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // fill b with random f32, then convert to type_b
    std::vector<float> bf((size_t) k * n);
    fill_random(bf.data(), bf.size(), 7);
    if (type_b == GGML_TYPE_F32) {
        ggml_backend_tensor_set(b, bf.data(), 0, bf.size() * sizeof(float));
    } else if (type_b == GGML_TYPE_F16) {
        std::vector<uint16_t> bh(bf.size());
        for (size_t i = 0; i < bf.size(); ++i) {
            // f32 -> f16 via ggml API
            uint16_t h;
            ggml_fp32_to_fp16_row(&bf[i], &h, 1);
            bh[i] = h;
        }
        ggml_backend_tensor_set(b, bh.data(), 0, bh.size() * sizeof(uint16_t));
    } else if (type_b == GGML_TYPE_Q8_1) {
        // D094 experiment-2: pre-quantize x on CPU in CUDA mmq D4 semantics:
        // 32-elem blocks, amax, d = amax/127, qs = round(x*127/amax) half-away,
        // packed 4 blocks into the 128-elem block_q8_1_mmq layout.
        std::vector<int8_t> qs(bf.size());
        std::vector<uint16_t> ds(bf.size() / 32 * 2); // half2 (d, sum*d) per 32-elem block
        const int nblocks = (int) bf.size() / 32;
        for (int bl = 0; bl < nblocks; ++bl) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::abs(bf[bl*32 + i]));
            }
            const float d = amax / 127.0f;
            const float d_inv = amax != 0.0f ? 127.0f / amax : 0.0f;
            float sum = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float v = bf[bl*32 + i] * d_inv;
                const int q = v >= 0.0f ? (int) std::floor(v + 0.5f) : (int) std::ceil(v - 0.5f);
                qs[bl*32 + i] = (int8_t) q;
                sum += (float) q;
            }
            ggml_fp32_to_fp16_row(&d, &ds[bl*2 + 0], 1);
            const float sd = sum * d;
            ggml_fp32_to_fp16_row(&sd, &ds[bl*2 + 1], 1);
        }
        // block_q8_1_x4_packed128 layout (types.glsl): ds[4] f16vec2 FIRST (16 B), then qs[8] ivec4 (128 B)
        std::vector<uint8_t> packed;
        const int x4 = (int) bf.size() / 128;
        for (int b4 = 0; b4 < x4; ++b4) {
            for (int j = 0; j < 8; ++j) { // 4 half2 = 8 halves
                const uint16_t h = ds[b4*4 + j];
                packed.push_back((uint8_t) (h & 0xFF));
                packed.push_back((uint8_t) (h >> 8));
            }
            for (int i = 0; i < 128; ++i) {
                packed.push_back((uint8_t) qs[b4*128 + i]);
            }
        }
        ggml_backend_tensor_set(b, packed.data(), 0, packed.size());
    } else if (type_b == GGML_TYPE_Q8_0) {
        // D094 exp-2b: block_q8_0-compatible layout (what load_tiles_q8_0 reads
        // from src1): per 32-elem block: d (f16) then qs (32 i8) = 34 B/block.
        std::vector<uint8_t> packed;
        const int nblocks = (int) bf.size() / 32;
        for (int bl = 0; bl < nblocks; ++bl) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::abs(bf[bl*32 + i]));
            }
            const float d = amax / 127.0f;
            const float d_inv = amax != 0.0f ? 127.0f / amax : 0.0f;
            uint16_t dh;
            ggml_fp32_to_fp16_row(&d, &dh, 1);
            packed.push_back((uint8_t) (dh & 0xFF));
            packed.push_back((uint8_t) (dh >> 8));
            for (int i = 0; i < 32; ++i) {
                const float v = bf[bl*32 + i] * d_inv;
                const int q = v >= 0.0f ? (int) std::floor(v + 0.5f) : (int) std::ceil(v - 0.5f);
                packed.push_back((uint8_t) (int8_t) q);
            }
        }
        ggml_backend_tensor_set(b, packed.data(), 0, packed.size());
    } else {
        fprintf(stderr, "unsupported type_b\n");
        return;
    }

    // fill a: random f32 then quantize
    if (type_a == GGML_TYPE_F32) {
        std::vector<float> af((size_t) k * m);
        fill_random(af.data(), af.size(), 3);
        ggml_backend_tensor_set(a, af.data(), 0, af.size() * sizeof(float));
    } else {
        std::vector<float> af((size_t) k * m);
        fill_random(af.data(), af.size(), 3);
        std::vector<uint8_t> aq(ggml_row_size(type_a, k) * m);
        ggml_quantize_chunk(type_a, af.data(), aq.data(), 0, m, k, nullptr);
        ggml_backend_tensor_set(a, aq.data(), 0, aq.size());
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> res((size_t) m * n);
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));

    FILE * f = fopen(outfile, "wb");
    fwrite(res.data(), sizeof(float), res.size(), f);
    fclose(f);

    fprintf(stderr, "%s: type=%s m=%lld n=%lld k=%lld -> %s (%.0f KiB)\n",
            ggml_backend_name(backend), ggml_type_name(type_a), (long long) m, (long long) n, (long long) k,
            outfile, res.size() * sizeof(float) / 1024.0);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <backend> <outfile>\n", argv[0]);
        return 1;
    }
    const std::string backend_name = argv[1];
    const char * outfile = argv[2];

    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (backend_name == ggml_backend_dev_name(dev)) {
            backend = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }
    if (!backend) {
        fprintf(stderr, "backend %s not found; available: %s\n", backend_name.c_str(),
                ggml_backend_dev_get(0) ? ggml_backend_dev_name(ggml_backend_dev_get(0)) : "none");
        return 1;
    }

    // eh_proj shapes: q8_0 K=10240, M=320, N=2 and N=1; plus a small q8_0 control
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".q8n2.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 64, 10240, (std::string(outfile) + ".q8n64.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 64, 512,   (std::string(outfile) + ".q8n64s.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 1, 10240, (std::string(outfile) + ".q8n1.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 4, 512,   (std::string(outfile) + ".q8c.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 1024,  (std::string(outfile) + ".k1k.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 2048,  (std::string(outfile) + ".k2k.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 4096,  (std::string(outfile) + ".k4k.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 1280,  (std::string(outfile) + ".k1280.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 1536,  (std::string(outfile) + ".k1536.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 1792,  (std::string(outfile) + ".k1792.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 1856,  (std::string(outfile) + ".k1856.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 1920,  (std::string(outfile) + ".k1920.bin").c_str());
    run_mul_mat(backend, GGML_TYPE_Q8_0, 64, 2, 1984,  (std::string(outfile) + ".k1984.bin").c_str());
    // b already f16: both backends skip q8_1 quantization of x -> isolates dot/dequant
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".q8n2f16.bin").c_str(), GGML_TYPE_F16);
    // b pre-quantized q8_1 on CPU (D4 semantics) -> bypasses backend x-quantize entirely
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".q8n2q1.bin").c_str(), GGML_TYPE_Q8_1);          // VK x4 layout
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".q8n2q1d4.bin").c_str(), GGML_TYPE_Q8_0, true); // ROCm 34B layout
    // b=f16 variant: skips the q8_1 x-quantization, isolates dot/dequant path
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".f16n2.bin").c_str(), GGML_TYPE_F16);
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 1, 10240, (std::string(outfile) + ".f16n1.bin").c_str(), GGML_TYPE_F16);
    // same eh_proj shape but with pre-quantized f16 x (no q8_1 path)
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".q8n2f16.bin").c_str(), GGML_TYPE_F16);
    // Same shape with f16 x: separates x-quantization (q8_1) from the dot path
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".q8n2f16.bin").c_str(), GGML_TYPE_F16);
    // same eh_proj shape but b already f16 (no q8_1 quantization of x): isolates quantize vs dot
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".f16n2.bin").c_str(), GGML_TYPE_F16);
    // b=f16 variant: x is NOT quantized to q8_1 -> isolates x-quantization vs dot
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".q8n2f16.bin").c_str(), GGML_TYPE_F16);
    // same eh_proj shape but b already f16 (no q8_1 quantization of x)
    run_mul_mat(backend, GGML_TYPE_Q8_0, 320, 2, 10240, (std::string(outfile) + ".q8n2f16.bin").c_str(), GGML_TYPE_F16);

    ggml_backend_free(backend);
    return 0;
}
