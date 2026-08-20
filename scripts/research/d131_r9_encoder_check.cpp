// D131 R9 encoder math check (CPU, no GPU). Verifies k_norm * scale == k_cur
// with correct ggml flat indexing: 2D [ne0,ne1] flat = i0 + i1*ne0.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    const int64_t n_embd_head = 256;
    const int64_t n_head       = 4;
    const int64_t n_tokens     = 2;
    const int64_t n_embd_gqa   = n_embd_head * n_head; // 1024
    const int64_t n_blk        = n_embd_gqa / 256;      // 4

    struct ggml_init_params iparams = { 32 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(iparams);
    ggml_backend_t backend = ggml_backend_cpu_init();

    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, n_tokens);
    std::vector<float> kvals(n_embd_head * n_head * n_tokens);
    for (int64_t t = 0; t < n_tokens; t++)
        for (int64_t h = 0; h < n_head; h++)
            for (int64_t e = 0; e < n_embd_head; e++)
                kvals[e + h * n_embd_head + t * n_embd_head * n_head] =
                    (float)((e - 128) * (h + 1)) / 100.0f + (float)t * 0.5f;

    ggml_tensor * k_flat = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);
    ggml_tensor * k_scale_new = ggml_pool_2d(ctx, k_flat, GGML_OP_POOL_MAX, 256, 1, 256, 1, 0.0f, 0.0f); // [4, n_tokens,1,1]

    const size_t blk_stride = ggml_row_size(k_scale_new->type, n_blk);
    ggml_tensor * k_scale_v    = ggml_view_3d(ctx, k_scale_new, n_blk, 1, n_tokens, blk_stride, blk_stride, 0);
    ggml_tensor * k_scale_rep  = ggml_repeat_4d(ctx, k_scale_v, n_blk, 256, n_tokens, 1); // [blk,rep,tok]
    ggml_tensor * k_scale_perm = ggml_permute(ctx, k_scale_rep, 1, 0, 2, 3);              // [rep,blk,tok]
    ggml_tensor * k_scale_cont = ggml_cont(ctx, k_scale_perm);
    ggml_tensor * k_scale_b    = ggml_reshape_2d(ctx, k_scale_cont, n_embd_gqa, n_tokens);

    ggml_tensor * k_norm = ggml_div(ctx, k_flat, k_scale_b);

    ggml_tensor * outputs[] = { k_norm, k_scale_new, k_scale_b };
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (auto * o : outputs) ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(k_cur, kvals.data(), 0, ggml_nbytes(k_cur));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> norm(n_embd_gqa * n_tokens);
    std::vector<float> scale(n_blk * n_tokens);
    std::vector<float> scaleb(n_embd_gqa * n_tokens);
    ggml_backend_tensor_get(k_norm, norm.data(), 0, ggml_nbytes(k_norm));
    ggml_backend_tensor_get(k_scale_new, scale.data(), 0, ggml_nbytes(k_scale_new));
    ggml_backend_tensor_get(k_scale_b, scaleb.data(), 0, ggml_nbytes(k_scale_b));

    printf("scale[blk + t*4]: ");
    for (int64_t t = 0; t < n_tokens; t++)
        for (int64_t blk = 0; blk < n_blk; blk++) printf("%.3f ", scale[blk + t * n_blk]);
    printf("\nscaleb[b + t*1024] b=0,255,256,511: ");
    for (int64_t b : {0LL, 255LL, 256LL, 511LL}) printf("%.3f ", scaleb[b + 0 * 1024]);
    printf("\n");

    int bad = 0;
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t blk = 0; blk < n_blk; blk++) {
            for (int64_t off = 0; off < 256; off++) {
                int64_t b = blk * 256 + off;
                float k  = kvals[off + blk * n_embd_head + t * n_embd_head * n_head];
                float sc = scale[blk + t * n_blk];
                float r  = norm[b + t * n_embd_gqa] * sc;
                if (std::fabs(r - k) > 1e-3f * std::fabs(k) + 1e-5f) {
                    if (bad < 8)
                        printf("MISMATCH t=%lld blk=%lld off=%lld k=%f sc=%f r=%f\n",
                               (long long)t, (long long)blk, (long long)off, k, sc, r);
                    bad++;
                }
            }
        }
    }
    printf("bad=%d (expect 0)\n", bad);

    ggml_backend_free(backend);
    ggml_free(ctx);
    return bad == 0 ? 0 : 1;
}
