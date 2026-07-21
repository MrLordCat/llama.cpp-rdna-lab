// Attention sparsity scout (P003 T5a gate).
//
// Measures how concentrated the post-softmax attention mass is across K/V
// positions for a real prompt, layer by layer. It exists to answer one
// question before any sparse-FlashAttention shader work: for a typical query,
// what fraction of the valid K/V positions carries 75/90/95/99% of the
// attention mass?
//
// Gate (T5a): a sparse-FA prototype is only worth building if >75% of the mass
// sits in <25% of the K/V blocks, i.e. the global frac75 mean is below 0.25.
//
// Flash Attention is forced off so the explicit `kq_soft_max` node is
// materialized in the graph; with FA on the softmax is fused and never exposed.
// Run CPU-only (build-cpu) to keep this off the GPU discovery/driver path.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Block-budget fractions for the cheap block-max selector recovery test.
// A real sparse-FA shader ranks K/V blocks by a cheap score (block max logit)
// and keeps only the top fraction; these are the budgets we evaluate.
static const double BUDGETS[]  = { 0.0625, 0.125, 0.25, 0.50 };
static const int    NBUD       = (int) (sizeof(BUDGETS) / sizeof(BUDGETS[0]));

struct layer_stats {
    int64_t rows      = 0;   // sampled (head, query) rows
    double  valid_sum = 0.0; // sum of valid (unmasked) key counts
    double  frac75    = 0.0; // sum of count_to_75 / valid
    double  frac90    = 0.0;
    double  frac95    = 0.0;
    double  frac99    = 0.0;
    double  bm_rec[NBUD] = { 0.0 }; // block-max top-k recovered mass per budget
    double  or_rec[NBUD] = { 0.0 }; // oracle (block-sum) top-k recovered mass per budget
    double  pq_rec[NBUD] = { 0.0 }; // per-query key-level (block=1) recovered mass per key budget
    int64_t tiles        = 0;       // sampled query tiles (gather test)
    double  tu_rec[NBUD] = { 0.0 }; // tile-union shared-key-set recovered mass per key budget
};

struct scout_data {
    std::vector<uint8_t>                    host;   // scratch for non-host tensors
    std::vector<float>                      row;    // scratch for one attention row
    std::vector<std::pair<float, double>>   blocks; // (block-max, block-sum) per row
    std::vector<double>                     gsum;   // per-tile summed prob per key
    std::vector<double>                     gsrt;   // sorted copy of gsum
    std::map<int, layer_stats>              per_layer;
    int query_stride = 1;
    int block_size   = 32;
    int tile         = 0;   // query-tile size for the gather test (0 = disabled)
};

static bool is_soft_max_node(const char * name) {
    return std::strncmp(name, "kq_soft_max", 11) == 0;
}

static bool scout_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * sd = (scout_data *) user_data;

    if (ask) {
        // only request data for the explicit attention softmax nodes
        return is_soft_max_node(t->name);
    }

    if (!is_soft_max_node(t->name) || t->type != GGML_TYPE_F32) {
        return true;
    }

    int il = -1;
    if (const char * dash = std::strrchr(t->name, '-')) {
        il = std::atoi(dash + 1);
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    uint8_t * data;
    if (is_host) {
        data = (uint8_t *) t->data;
    } else {
        const size_t n_bytes = ggml_nbytes(t);
        sd->host.resize(n_bytes);
        ggml_backend_tensor_get(t, sd->host.data(), 0, n_bytes);
        data = sd->host.data();
    }

    const int64_t n_kv   = t->ne[0];
    const int64_t n_q    = t->ne[1];
    const int64_t n_head = t->ne[2];
    const int     stride = sd->query_stride > 0 ? sd->query_stride : 1;

    sd->row.resize((size_t) n_kv);
    layer_stats & ls = sd->per_layer[il];

    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t q = 0; q < n_q; q += stride) {
            double  sum   = 0.0;
            int64_t valid = 0;
            for (int64_t k = 0; k < n_kv; ++k) {
                const float v = *(const float *) (data + h * t->nb[2] + q * t->nb[1] + k * t->nb[0]);
                sd->row[(size_t) k] = v;
                if (v > 0.0f) {
                    ++valid;
                    sum += v;
                }
            }
            if (valid == 0 || sum <= 0.0) {
                continue;
            }

            // --- cheap block-max selector recovery (uses the unsorted row) ---
            const int     B  = sd->block_size > 0 ? sd->block_size : 32;
            const int64_t nb = (n_kv + B - 1) / B;
            sd->blocks.clear();
            sd->blocks.reserve((size_t) nb);
            for (int64_t b = 0; b < nb; ++b) {
                const int64_t k0 = b * B;
                const int64_t k1 = std::min(k0 + (int64_t) B, n_kv);
                float  bmax = 0.0f;
                double bsum = 0.0;
                for (int64_t k = k0; k < k1; ++k) {
                    const float v = sd->row[(size_t) k];
                    if (v > bmax) { bmax = v; }
                    bsum += v;
                }
                if (bsum > 0.0) {
                    sd->blocks.emplace_back(bmax, bsum);
                }
            }
            const int64_t nvb = (int64_t) sd->blocks.size();
            for (int i = 0; i < NBUD; ++i) {
                int64_t keep = (int64_t) std::llround(BUDGETS[i] * (double) nvb);
                if (keep < 1)   { keep = 1; }
                if (keep > nvb) { keep = nvb; }

                std::partial_sort(sd->blocks.begin(), sd->blocks.begin() + keep, sd->blocks.end(),
                    [](const std::pair<float, double> & a, const std::pair<float, double> & b) { return a.first > b.first; });
                double rec_bm = 0.0;
                for (int64_t j = 0; j < keep; ++j) { rec_bm += sd->blocks[(size_t) j].second; }

                std::partial_sort(sd->blocks.begin(), sd->blocks.begin() + keep, sd->blocks.end(),
                    [](const std::pair<float, double> & a, const std::pair<float, double> & b) { return a.second > b.second; });
                double rec_or = 0.0;
                for (int64_t j = 0; j < keep; ++j) { rec_or += sd->blocks[(size_t) j].second; }

                ls.bm_rec[i] += rec_bm / sum;
                ls.or_rec[i] += rec_or / sum;
            }

            std::sort(sd->row.begin(), sd->row.begin() + (ptrdiff_t) n_kv, std::greater<float>());

            int64_t keepK[NBUD];
            for (int i = 0; i < NBUD; ++i) {
                int64_t kk = (int64_t) std::llround(BUDGETS[i] * (double) valid);
                if (kk < 1)     { kk = 1; }
                if (kk > valid) { kk = valid; }
                keepK[i] = kk;
            }

            double  acc = 0.0;
            int64_t c75 = 0, c90 = 0, c95 = 0, c99 = 0;
            bool    d75 = false, d90 = false, d95 = false, d99 = false;
            double  pqacc[NBUD] = { 0.0 };
            for (int64_t k = 0; k < n_kv; ++k) {
                acc += sd->row[(size_t) k];
                const double frac = acc / sum;
                if (!d75 && frac >= 0.75) { c75 = k + 1; d75 = true; }
                if (!d90 && frac >= 0.90) { c90 = k + 1; d90 = true; }
                if (!d95 && frac >= 0.95) { c95 = k + 1; d95 = true; }
                if (!d99 && frac >= 0.99) { c99 = k + 1; d99 = true; }
                for (int i = 0; i < NBUD; ++i) {
                    if (k + 1 == keepK[i]) { pqacc[i] = acc; }
                }
            }

            ls.rows      += 1;
            ls.valid_sum += (double) valid;
            ls.frac75    += (double) c75 / (double) valid;
            ls.frac90    += (double) c90 / (double) valid;
            ls.frac95    += (double) c95 / (double) valid;
            ls.frac99    += (double) c99 / (double) valid;
            for (int i = 0; i < NBUD; ++i) {
                ls.pq_rec[i] += pqacc[i] / sum;
            }
        }
    }

    // --- gather viability: shared key set across a tile of adjacent queries ---
    // A coopmat FA processes a whole query tile together, so it must gather ONE
    // shared K/V set for the tile. Rank keys by summed prob across the tile and
    // measure recovered mean mass at key budgets; compare to per-query pq_rec.
    if (sd->tile > 0) {
        const int T = sd->tile;
        sd->gsum.resize((size_t) n_kv);
        sd->gsrt.resize((size_t) n_kv);
        int tile_idx = 0;
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t qt = 0; qt < n_q; qt += T, ++tile_idx) {
                if ((tile_idx % stride) != 0) { continue; } // sample tiles at the per-query rate

                std::fill(sd->gsum.begin(), sd->gsum.begin() + (ptrdiff_t) n_kv, 0.0);
                int64_t vq = 0;
                const int64_t q1 = std::min(qt + (int64_t) T, n_q);
                for (int64_t q = qt; q < q1; ++q) {
                    double s = 0.0;
                    for (int64_t k = 0; k < n_kv; ++k) {
                        const float v = *(const float *) (data + h * t->nb[2] + q * t->nb[1] + k * t->nb[0]);
                        sd->gsum[(size_t) k] += v;
                        s += v;
                    }
                    if (s > 0.0) { ++vq; }
                }
                if (vq == 0) { continue; }

                int64_t nvk = 0;
                for (int64_t k = 0; k < n_kv; ++k) {
                    sd->gsrt[(size_t) k] = sd->gsum[(size_t) k];
                    if (sd->gsum[(size_t) k] > 0.0) { ++nvk; }
                }
                std::sort(sd->gsrt.begin(), sd->gsrt.begin() + (ptrdiff_t) n_kv, std::greater<double>());

                layer_stats & lst = sd->per_layer[il];
                for (int i = 0; i < NBUD; ++i) {
                    int64_t keep = (int64_t) std::llround(BUDGETS[i] * (double) nvk);
                    if (keep < 1)   { keep = 1; }
                    if (keep > nvk) { keep = nvk; }
                    double rec = 0.0;
                    for (int64_t j = 0; j < keep; ++j) { rec += sd->gsrt[(size_t) j]; }
                    lst.tu_rec[i] += rec / (double) vq;
                }
                lst.tiles += 1;
            }
        }
    }

    return true;
}

static void print_summary(const scout_data & sd, size_t n_prompt) {
    layer_stats total;
    for (const auto & kv : sd.per_layer) {
        const layer_stats & ls = kv.second;
        total.rows      += ls.rows;
        total.valid_sum += ls.valid_sum;
        total.frac75    += ls.frac75;
        total.frac90    += ls.frac90;
        total.frac95    += ls.frac95;
        total.frac99    += ls.frac99;
        total.tiles     += ls.tiles;
        for (int i = 0; i < NBUD; ++i) {
            total.bm_rec[i] += ls.bm_rec[i];
            total.or_rec[i] += ls.or_rec[i];
            total.pq_rec[i] += ls.pq_rec[i];
            total.tu_rec[i] += ls.tu_rec[i];
        }
    }

    std::fprintf(stdout, "ATTN_SPARSITY_HEADER n_prompt=%zu attn_layers=%zu query_stride=%d block_size=%d total_rows=%lld\n",
                 n_prompt, sd.per_layer.size(), sd.query_stride, sd.block_size, (long long) total.rows);

    if (total.rows == 0) {
        std::fprintf(stdout, "ATTN_SPARSITY_GLOBAL rows=0 note=no_kq_soft_max_nodes_captured "
                             "(is flash-attn actually off? is this a linear-attention-only model?)\n");
        std::fflush(stdout);
        return;
    }

    for (const auto & kv : sd.per_layer) {
        const int           il = kv.first;
        const layer_stats & ls = kv.second;
        if (ls.rows == 0) {
            continue;
        }
        std::fprintf(stdout,
            "ATTN_SPARSITY layer=%d rows=%lld valid_mean=%.1f frac75=%.4f frac90=%.4f frac95=%.4f frac99=%.4f "
            "bm06=%.4f bm12=%.4f bm25=%.4f bm50=%.4f\n",
            il, (long long) ls.rows, ls.valid_sum / ls.rows,
            ls.frac75 / ls.rows, ls.frac90 / ls.rows, ls.frac95 / ls.rows, ls.frac99 / ls.rows,
            ls.bm_rec[0] / ls.rows, ls.bm_rec[1] / ls.rows, ls.bm_rec[2] / ls.rows, ls.bm_rec[3] / ls.rows);
    }

    const double g75 = total.frac75 / total.rows;
    const double g90 = total.frac90 / total.rows;
    const double g95 = total.frac95 / total.rows;
    const double g99 = total.frac99 / total.rows;
    const double bm[NBUD] = {
        total.bm_rec[0] / total.rows, total.bm_rec[1] / total.rows,
        total.bm_rec[2] / total.rows, total.bm_rec[3] / total.rows,
    };
    const double or25 = total.or_rec[2] / total.rows; // oracle upper bound at the 25% budget

    std::fprintf(stdout,
        "ATTN_SPARSITY_GLOBAL rows=%lld valid_mean=%.1f frac75=%.4f frac90=%.4f frac95=%.4f frac99=%.4f "
        "bm06=%.4f bm12=%.4f bm25=%.4f bm50=%.4f or25=%.4f gate_75in25=%s gate_bm25_99=%s\n",
        (long long) total.rows, total.valid_sum / total.rows, g75, g90, g95, g99,
        bm[0], bm[1], bm[2], bm[3], or25,
        g75 < 0.25 ? "PASS" : "FAIL",
        bm[2] >= 0.99 ? "PASS" : "FAIL");

    // gather viability: per-query key-level (block=1) vs tile-union shared key set
    const double pq[NBUD] = {
        total.pq_rec[0] / total.rows, total.pq_rec[1] / total.rows,
        total.pq_rec[2] / total.rows, total.pq_rec[3] / total.rows,
    };
    if (sd.tile > 0 && total.tiles > 0) {
        const double tu[NBUD] = {
            total.tu_rec[0] / total.tiles, total.tu_rec[1] / total.tiles,
            total.tu_rec[2] / total.tiles, total.tu_rec[3] / total.tiles,
        };
        std::fprintf(stdout,
            "ATTN_GATHER_GLOBAL tile=%d tiles=%lld "
            "pq06=%.4f pq12=%.4f pq25=%.4f pq50=%.4f tu06=%.4f tu12=%.4f tu25=%.4f tu50=%.4f "
            "union_penalty25=%.4f gate_pq25_99=%s gate_tu25_99=%s\n",
            sd.tile, (long long) total.tiles,
            pq[0], pq[1], pq[2], pq[3], tu[0], tu[1], tu[2], tu[3],
            pq[2] - tu[2],
            pq[2] >= 0.99 ? "PASS" : "FAIL",
            tu[2] >= 0.99 ? "PASS" : "FAIL");
    } else {
        std::fprintf(stdout,
            "ATTN_GATHER_GLOBAL tile=0 pq06=%.4f pq12=%.4f pq25=%.4f pq50=%.4f gate_pq25_99=%s\n",
            pq[0], pq[1], pq[2], pq[3], pq[2] >= 0.99 ? "PASS" : "FAIL");
    }
    std::fflush(stdout);
}

static bool run_prefill(llama_context * ctx, const common_params & params, size_t & n_prompt_out) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens - provide a prompt with -p or -f\n", __func__);
        return false;
    }
    n_prompt_out = tokens.size();
    LOG_INF("%s: prefill of %zu tokens (flash-attn forced off)\n", __func__, tokens.size());

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("%s: llama_decode failed\n", __func__);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Flash Attention fuses the softmax; force it off so kq_soft_max is exposed.
    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LOG_WRN("%s: forcing flash-attn OFF so attention softmax is materialized\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    params.warmup = false;

    scout_data sd;
    if (const char * s = std::getenv("ATTN_SCOUT_QUERY_STRIDE")) {
        sd.query_stride = std::max(1, std::atoi(s));
    }
    if (const char * s = std::getenv("ATTN_SCOUT_BLOCK_SIZE")) {
        sd.block_size = std::max(1, std::atoi(s));
    }
    if (const char * s = std::getenv("ATTN_SCOUT_TILE")) {
        sd.tile = std::max(0, std::atoi(s));
    }

    params.cb_eval           = scout_cb_eval;
    params.cb_eval_user_data = &sd;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    LOG_INF("\n%s\n\n", common_params_get_system_info(params).c_str());

    size_t n_prompt = 0;
    const bool ok = run_prefill(ctx, params, n_prompt);
    if (ok) {
        print_summary(sd, n_prompt);
    }

    llama_backend_free();
    return ok ? 0 : 1;
}
