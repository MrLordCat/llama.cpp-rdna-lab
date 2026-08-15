// Q4_K16 vec_dot unit test (research/q4-k16-quant, §3.3)
// Compares the scalar vec_dot against a dequantize + float64 reference dot.
//
// Build (from repo root):
//   g++ -std=c++17 -O2 -I ggml/include -I ggml/src -I ggml/src/ggml-cpu \
//       scripts/research/q4_k16_vecdot.cpp \
//       build-cpu/ggml/src/ggml-cpu/libggml-cpu.a build-cpu/ggml/src/ggml-base.a \
//       -o scripts/research/q4_k16_vecdot.exe
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"   // dequantize_row_q4_K16_*, quantize_row_q4_K16_*_ref
#include "ggml-cpu/quants.h" // vec_dot + quantize_row_q8_K
#include "ggml-cpu.h"      // ggml_cpu_init (fp16 lookup table)
#include "ggml-backend.h"  // ggml_backend_cpu_init

static double ref_dot(const float * xd, const float * y, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double) xd[i] * (double) y[i];
    return s;
}

static bool check_type(enum ggml_type type, const char * name, int n, const float * x_f32, const float * y_f32) {
    std::vector<char> x_q(ggml_row_size(type, n));
    std::vector<float> x_dq(n);
    std::vector<float> y_dq(n);
    std::vector<char> y_q(n/256 * sizeof(block_q8_K));

    switch (type) {
        case GGML_TYPE_Q4_K16_M: quantize_row_q4_K16_M_ref(x_f32, (block_q4_K16_M *) x_q.data(), n); dequantize_row_q4_K16_M((const block_q4_K16_M *) x_q.data(), x_dq.data(), n); break;
        case GGML_TYPE_Q4_K16:   quantize_row_q4_K16_ref  (x_f32, (block_q4_K16   *) x_q.data(), n); dequantize_row_q4_K16  ((const block_q4_K16   *) x_q.data(), x_dq.data(), n); break;
        case GGML_TYPE_Q4_K16_S: quantize_row_q4_K16_S_ref(x_f32, (block_q4_K16_S *) x_q.data(), n); dequantize_row_q4_K16_S((const block_q4_K16_S *) x_q.data(), x_dq.data(), n); break;
        default: return false;
    }

    quantize_row_q8_K(y_f32, y_q.data(), n);
    dequantize_row_q8_K((const block_q8_K *) y_q.data(), y_dq.data(), n);

    float s = 0.0f;
    switch (type) {
        case GGML_TYPE_Q4_K16_M: ggml_vec_dot_q4_K16_M_q8_1(n, &s, 0, x_q.data(), 0, y_q.data(), 0, 1); break;
        case GGML_TYPE_Q4_K16:   ggml_vec_dot_q4_K16_q8_1  (n, &s, 0, x_q.data(), 0, y_q.data(), 0, 1); break;
        case GGML_TYPE_Q4_K16_S: ggml_vec_dot_q4_K16_S_q8_1(n, &s, 0, x_q.data(), 0, y_q.data(), 0, 1); break;
        default: return false;
    }

    // the formula is exact: vec_dot must equal dot(x_dequantized, y_dequantized)
    // up to f32 summation order -> tolerance is tight
    const double ref = ref_dot(x_dq.data(), y_dq.data(), n);
    const double err = std::fabs((double) s - ref) / std::max(1.0, std::fabs(ref));

    const bool ok = err < 1e-4;
    printf("%-10s n=%6d  vec_dot=%.6f  ref=%.6f  rel_err=%.3e  %s\n",
            name, n, (double) s, ref, err, ok ? "OK" : "FAIL");
    return ok;
}

// graph-level check: MUL_MAT(x_q4_k16 [k, ncols], y_f32 [k, m]) via the CPU
// backend - exercises the full ggml path (type traits, vec_dot_type
// conversion of y to q8_K, vec_dot dispatch)
static bool check_graph(enum ggml_type type, const char * name, int k, int ncols, int m) {
    std::mt19937 rng(999);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<float> x_f32(k * ncols), y_f32(k * m);
    for (int i = 0; i < k * ncols; ++i) x_f32[i] = nd(rng);
    for (int i = 0; i < k * m;     ++i) y_f32[i] = ud(rng);

    std::vector<char> x_q(ncols * ggml_row_size(type, k));
    std::vector<float> x_dq(k * ncols);
    std::vector<float> y_dq(k * m);
    for (int c = 0; c < ncols; ++c) {
        switch (type) {
            case GGML_TYPE_Q4_K16_M: quantize_row_q4_K16_M_ref(&x_f32[c*k], (block_q4_K16_M *) &x_q[c*ggml_row_size(type, k)], k); break;
            case GGML_TYPE_Q4_K16:   quantize_row_q4_K16_ref  (&x_f32[c*k], (block_q4_K16   *) &x_q[c*ggml_row_size(type, k)], k); break;
            case GGML_TYPE_Q4_K16_S: quantize_row_q4_K16_S_ref(&x_f32[c*k], (block_q4_K16_S *) &x_q[c*ggml_row_size(type, k)], k); break;
            default: return false;
        }
    }
    switch (type) {
        case GGML_TYPE_Q4_K16_M: for (int c = 0; c < ncols; ++c) dequantize_row_q4_K16_M((const block_q4_K16_M *) &x_q[c*ggml_row_size(type, k)], &x_dq[c*k], k); break;
        case GGML_TYPE_Q4_K16:   for (int c = 0; c < ncols; ++c) dequantize_row_q4_K16  ((const block_q4_K16   *) &x_q[c*ggml_row_size(type, k)], &x_dq[c*k], k); break;
        case GGML_TYPE_Q4_K16_S: for (int c = 0; c < ncols; ++c) dequantize_row_q4_K16_S((const block_q4_K16_S *) &x_q[c*ggml_row_size(type, k)], &x_dq[c*k], k); break;
        default: return false;
    }
    // y goes through the q8_K conversion in MUL_MAT; replicate it for the ref
    for (int r = 0; r < m; ++r) {
        std::vector<char> yq(k/256 * sizeof(block_q8_K));
        quantize_row_q8_K(&y_f32[r*k], yq.data(), k);
        dequantize_row_q8_K((const block_q8_K *) yq.data(), &y_dq[r*k], k);
    }

    struct ggml_init_params params = { 1u << 22, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, type, k, ncols);
    struct ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    struct ggml_tensor * out = ggml_mul_mat(ctx, x, y);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == NULL) { printf("alloc failed\n"); return false; }
    ggml_backend_tensor_set(x, x_q.data(), 0, x_q.size());
    ggml_backend_tensor_set(y, y_f32.data(), 0, y_f32.size()*sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("graph compute failed\n"); return false; }

    std::vector<float> out_data(m * ncols);
    ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size()*sizeof(float));

    // reference: float64 dot of dequantized x and q8-dequantized y; out is [ncols, m] row-major
    bool ok = true;
    double worst = 0;
    for (int c = 0; c < ncols; ++c) {
        for (int r = 0; r < m; ++r) {
            double ref = 0;
            for (int i = 0; i < k; ++i) ref += (double) x_dq[c*k + i] * (double) y_dq[r*k + i];
            const double err = std::fabs((double) out_data[c + r*ncols] - ref) / std::max(1.0, std::fabs(ref));
            worst = std::max(worst, err);
            if (err > 1e-3) ok = false;
        }
    }
    printf("%-10s graph MUL_MAT k=%d cols=%d m=%d  worst_rel_err=%.3e  %s\n",
            name, k, ncols, m, worst, ok ? "OK" : "FAIL");
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// GET_ROWS path (model load): extract quantized rows, compare against the
// per-column quantized source
static bool check_get_rows(enum ggml_type type, const char * name, int k, int ncols) {
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x_f32(k * ncols);
    for (int i = 0; i < k * ncols; ++i) x_f32[i] = nd(rng);

    std::vector<char> x_q(ncols * ggml_row_size(type, k));
    std::vector<float> row_f32(k);
    std::vector<float> row_ref(k);
    for (int c = 0; c < ncols; ++c) {
        switch (type) {
            case GGML_TYPE_Q4_K16_M: quantize_row_q4_K16_M_ref(&x_f32[c*k], (block_q4_K16_M *) &x_q[c*ggml_row_size(type, k)], k); break;
            case GGML_TYPE_Q4_K16:   quantize_row_q4_K16_ref  (&x_f32[c*k], (block_q4_K16   *) &x_q[c*ggml_row_size(type, k)], k); break;
            case GGML_TYPE_Q4_K16_S: quantize_row_q4_K16_S_ref(&x_f32[c*k], (block_q4_K16_S *) &x_q[c*ggml_row_size(type, k)], k); break;
            default: return false;
        }
    }

    struct ggml_init_params params = { 1u << 22, NULL, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, type, k, ncols);
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * out = ggml_get_rows(ctx, x, idx);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == NULL) { printf("get_rows alloc failed\n"); return false; }
    ggml_backend_tensor_set(x, x_q.data(), 0, x_q.size());
    int32_t idx_val = 1;
    ggml_backend_tensor_set(idx, &idx_val, 0, sizeof(idx_val));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("get_rows compute failed\n"); return false; }
    ggml_backend_tensor_get(out, row_f32.data(), 0, row_f32.size()*sizeof(float));

    // get_rows_q dequantizes the row to f32; compare against dequantize of source row 1
    switch (type) {
        case GGML_TYPE_Q4_K16_M: dequantize_row_q4_K16_M((const block_q4_K16_M *) &x_q[ggml_row_size(type, k)], row_ref.data(), k); break;
        case GGML_TYPE_Q4_K16:   dequantize_row_q4_K16  ((const block_q4_K16   *) &x_q[ggml_row_size(type, k)], row_ref.data(), k); break;
        case GGML_TYPE_Q4_K16_S: dequantize_row_q4_K16_S((const block_q4_K16_S *) &x_q[ggml_row_size(type, k)], row_ref.data(), k); break;
        default: return false;
    }
    bool ok = true;
    double worst = 0;
    for (int i = 0; i < k; ++i) {
        const double err = std::fabs((double) row_f32[i] - (double) row_ref[i]);
        worst = std::max(worst, err);
        if (err > 1e-6f) ok = false;
    }
    printf("%-10s GET_ROWS k=%d  worst_abs_err=%.3e  %s\n", name, k, worst, ok ? "OK" : "FAIL");
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_cpu_init(); // init the fp16 lookup table used by the CPU vec_dot path

    const int nb = 32;                    // 32 super-blocks of 512
    const int n = nb * 512;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<float> x_f32(n), y_f32(n);
    for (int i = 0; i < n; ++i) {
        // per-16-sub-block magnitude pattern, as in real weights
        x_f32[i] = (0.05f + 0.5f * std::fabs((float) ((i/16) % 7 - 3))) * nd(rng);
        y_f32[i] = ud(rng);
    }

    bool ok = true;
    ok &= check_type(GGML_TYPE_Q4_K16_M, "q4_K16_M", n, x_f32.data(), y_f32.data());
    ok &= check_type(GGML_TYPE_Q4_K16,   "q4_K16",   n, x_f32.data(), y_f32.data());
    ok &= check_type(GGML_TYPE_Q4_K16_S, "q4_K16_S", n, x_f32.data(), y_f32.data());
    // also a wide row: n = 5120 (10 super-blocks, as ffn_gate rows)
    const int nw = 5120;
    std::vector<float> xw(nw), yw(nw);
    for (int i = 0; i < nw; ++i) {
        xw[i] = (0.05f + 0.5f * std::fabs((float) ((i/16) % 7 - 3))) * nd(rng);
        yw[i] = ud(rng);
    }
    ok &= check_type(GGML_TYPE_Q4_K16_M, "q4_K16_M(w)", nw, xw.data(), yw.data());

    // full graph path (MUL_MAT via CPU backend, y converted to q8_K)
    ok &= check_graph(GGML_TYPE_Q4_K16_M, "q4_K16_M", 5120, 2, 8);
    ok &= check_graph(GGML_TYPE_Q4_K16,   "q4_K16",   5120, 2, 8);
    ok &= check_graph(GGML_TYPE_Q4_K16_S, "q4_K16_S", 5120, 2, 8);
    ok &= check_graph(GGML_TYPE_Q4_K16_M, "q4_K16_M(gv)", 5120, 8, 16);

    // GET_ROWS (model load path)
    ok &= check_get_rows(GGML_TYPE_Q4_K16_M, "q4_K16_M", 5120, 4);
    ok &= check_get_rows(GGML_TYPE_Q4_K16,   "q4_K16",   5120, 4);
    ok &= check_get_rows(GGML_TYPE_Q4_K16_S, "q4_K16_S", 5120, 4);

    printf(ok ? "ALL OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
