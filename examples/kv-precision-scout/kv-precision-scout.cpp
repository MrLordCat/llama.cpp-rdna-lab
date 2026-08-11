// D095 R6: diagnostic-only KV precision scout.
//
// Captures the post-RoPE Q/K and V tensors from the first full-attention
// layers of a real prompt. It compares raw E4M3, power-of-two block-scaled
// E4M3, q8_0 and f16 without changing the model graph or KV-cache format.
// Output is a small key=value stream consumed by
// scripts/research/d095_kv_precision_scout.py.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

enum class capture_kind {
    none,
    q,
    k,
    v,
};

struct layer_capture {
    int64_t d         = 0;
    int64_t kv_heads  = 0;
    int64_t q_heads   = 0;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> q_last;
};

struct scout_data {
    std::vector<uint8_t> host;
    std::map<int, layer_capture> layers;
    int  max_layer    = 11;
    int  key_stride   = 4;
    bool final_chunk  = false;
    int  errors       = 0;
    std::string label = "unknown";
};

static capture_kind parse_capture_name(const ggml_tensor * t, int & il) {
    const char * suffix = nullptr;
    capture_kind kind = capture_kind::none;

    if (std::strncmp(t->name, "Qcur-", 5) == 0) {
        suffix = t->name + 5;
        kind = capture_kind::q;
    } else if (std::strncmp(t->name, "Kcur-", 5) == 0) {
        suffix = t->name + 5;
        kind = capture_kind::k;
    } else if (std::strncmp(t->name, "Vcur-", 5) == 0) {
        suffix = t->name + 5;
        kind = capture_kind::v;
    } else {
        return capture_kind::none;
    }

    char * end = nullptr;
    const long parsed = std::strtol(suffix, &end, 10);
    if (end == suffix || *end != '\0' || parsed < 0) {
        return capture_kind::none;
    }

    // The projection and the post-RoPE tensor intentionally share Kcur/Vcur
    // names. Only the final tensor is three-dimensional [D, H, T].
    if (t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] <= 1) {
        return capture_kind::none;
    }

    il = (int) parsed;
    return kind;
}

static float read_scalar(const uint8_t * data, const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2) {
    const uint8_t * ptr = data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2];
    switch (t->type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *) ptr);
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*(const ggml_bf16_t *) ptr);
        default:
            return 0.0f;
    }
}

static bool supported_capture_type(enum ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static void capture_tensor(scout_data & sd, ggml_tensor * t, capture_kind kind, int il) {
    if (!supported_capture_type(t->type)) {
        LOG_ERR("kv scout: unsupported tensor type %s for %s\n", ggml_type_name(t->type), t->name);
        ++sd.errors;
        return;
    }

    const uint8_t * data = nullptr;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        const size_t n_bytes = ggml_nbytes(t);
        sd.host.resize(n_bytes);
        ggml_backend_tensor_get(t, sd.host.data(), 0, n_bytes);
        data = sd.host.data();
    }

    const int64_t d      = t->ne[0];
    const int64_t heads  = t->ne[1];
    const int64_t tokens = t->ne[2];
    layer_capture & lc = sd.layers[il];

    if (lc.d == 0) {
        lc.d = d;
    }
    if (lc.d != d) {
        LOG_ERR("kv scout: inconsistent head dimension at layer %d\n", il);
        ++sd.errors;
        return;
    }

    if (kind == capture_kind::q) {
        lc.q_heads = heads;
        lc.q_last.resize((size_t) heads*d);
        const int64_t token = tokens - 1;
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t i = 0; i < d; ++i) {
                lc.q_last[(size_t) h*d + i] = read_scalar(data, t, i, h, token);
            }
        }
        return;
    }

    if (lc.kv_heads == 0) {
        lc.kv_heads = heads;
    }
    if (lc.kv_heads != heads) {
        LOG_ERR("kv scout: inconsistent KV head count at layer %d\n", il);
        ++sd.errors;
        return;
    }

    std::vector<float> & dst = kind == capture_kind::k ? lc.k : lc.v;
    dst.reserve(dst.size() + (size_t) d*heads*tokens);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t i = 0; i < d; ++i) {
                dst.push_back(read_scalar(data, t, i, h, token));
            }
        }
    }
}

static bool scout_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * sd = (scout_data *) user_data;
    int il = -1;
    const capture_kind kind = parse_capture_name(t, il);
    const bool selected = kind != capture_kind::none && il <= sd->max_layer &&
        (kind != capture_kind::q || sd->final_chunk);

    if (ask) {
        return selected;
    }
    if (selected) {
        capture_tensor(*sd, t, kind, il);
    }
    return true;
}

static uint8_t encode_e4m3(float value) {
    uint32_t sign = value < 0.0f ? 0x80u : 0u;
    float f = std::fabs(value);
    if (f > 448.0f) {
        f = 448.0f;
    }

    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t exp32 = (bits >> 23u) & 0xffu;
    if (exp32 < 121u) {
        uint32_t man = (uint32_t) std::lround(f*512.0f);
        man = std::min(man, 7u);
        return (uint8_t) (sign | man);
    }

    uint32_t exp_field = exp32 - 120u;
    if (exp_field > 14u) {
        return (uint8_t) (sign | (14u << 3u) | 7u);
    }
    uint32_t man = ((bits & 0x7fffffu) + 0x80000u) >> 20u;
    exp_field += man >> 3u;
    man &= 7u;
    if (exp_field > 14u) {
        return (uint8_t) (sign | (14u << 3u) | 7u);
    }
    return (uint8_t) (sign | (exp_field << 3u) | man);
}

static float decode_e4m3(uint8_t value) {
    const uint32_t sign = value >> 7u;
    const uint32_t exp  = (value >> 3u) & 0x0fu;
    const uint32_t man  = value & 0x07u;
    float result = 0.0f;
    if (exp == 0u) {
        result = (float) man/512.0f;
    } else if (exp != 15u) {
        result = std::ldexp(1.0f + (float) man/8.0f, (int) exp - 7);
    }
    return sign != 0u ? -result : result;
}

enum class method_kind {
    raw_e4m3,
    block_e4m3,
    block_e4m3_linear,
    block_int8_p2,
    q8_0,
    f16,
};

struct method_spec {
    const char * name;
    method_kind kind;
    int block;
    double bytes_per_value;
};

static const method_spec METHODS[] = {
    { "raw_e4m3", method_kind::raw_e4m3,     0, 1.0 },
    { "bs_e4m3",  method_kind::block_e4m3, 16, 1.0 + 1.0/16.0 },
    { "bs_e4m3",  method_kind::block_e4m3, 32, 1.0 + 1.0/32.0 },
    { "bs_e4m3",  method_kind::block_e4m3, 64, 1.0 + 1.0/64.0 },
    { "gs_e4m3",  method_kind::block_e4m3_linear, 16, 1.0 + 2.0/16.0 },
    { "gs_e4m3",  method_kind::block_e4m3_linear, 32, 1.0 + 2.0/32.0 },
    { "gs_e4m3",  method_kind::block_e4m3_linear, 64, 1.0 + 2.0/64.0 },
    { "gs_e4m3",  method_kind::block_e4m3_linear, 256, 1.0 + 2.0/256.0 },
    { "bfp8_p2",   method_kind::block_int8_p2, 32, 1.0 + 1.0/32.0 },
    { "q8_0",      method_kind::q8_0,       32, 34.0/32.0 },
    { "f16",       method_kind::f16,         0, 2.0 },
};

struct quantized_values {
    std::vector<float> values;
    uint64_t saturation = 0;
    uint64_t zero       = 0;
    uint64_t subnormal  = 0;
};

static quantized_values reconstruct(const std::vector<float> & src, int64_t d, const method_spec & method) {
    quantized_values out;
    out.values.resize(src.size());

    if (method.kind == method_kind::raw_e4m3) {
        for (size_t i = 0; i < src.size(); ++i) {
            const uint8_t q = encode_e4m3(src[i]);
            out.values[i] = decode_e4m3(q);
            const uint8_t mag = q & 0x7fu;
            out.saturation += mag == 0x77u;
            out.zero       += mag == 0u;
            out.subnormal  += mag > 0u && mag < 8u;
        }
        return out;
    }

    if (method.kind == method_kind::f16) {
        for (size_t i = 0; i < src.size(); ++i) {
            const ggml_fp16_t q = ggml_fp32_to_fp16(src[i]);
            const uint16_t bits = (uint16_t) q;
            const uint16_t mag = bits & 0x7fffu;
            out.values[i] = ggml_fp16_to_fp32(q);
            out.saturation += mag == 0x7bffu;
            out.zero       += mag == 0u;
            out.subnormal  += mag > 0u && mag < 0x0400u;
        }
        return out;
    }

    const int block = method.block;
    for (size_t row = 0; row < src.size(); row += (size_t) d) {
        for (int64_t begin = 0; begin < d; begin += block) {
            const int64_t end = std::min<int64_t>(d, begin + block);
            float amax = 0.0f;
            for (int64_t i = begin; i < end; ++i) {
                amax = std::max(amax, std::fabs(src[row + (size_t) i]));
            }

            if (method.kind == method_kind::block_e4m3_linear) {
                const float scale_exact = amax/240.0f;
                const float scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale_exact));
                const float inv = scale != 0.0f ? 1.0f/scale : 0.0f;
                for (int64_t i = begin; i < end; ++i) {
                    const uint8_t q = encode_e4m3(src[row + (size_t) i]*inv);
                    const uint8_t mag = q & 0x7fu;
                    out.values[row + (size_t) i] = decode_e4m3(q)*scale;
                    out.saturation += mag == 0x77u;
                    out.zero       += mag == 0u;
                    out.subnormal  += mag > 0u && mag < 8u;
                }
                continue;
            }

            if (method.kind == method_kind::q8_0) {
                const float d_exact = amax/127.0f;
                const float inv = d_exact != 0.0f ? 1.0f/d_exact : 0.0f;
                const float d_stored = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d_exact));
                for (int64_t i = begin; i < end; ++i) {
                    int q = (int) std::lround(src[row + (size_t) i]*inv);
                    q = std::max(-127, std::min(127, q));
                    out.values[row + (size_t) i] = (float) q*d_stored;
                    out.saturation += std::abs(q) == 127;
                    out.zero       += q == 0;
                }
                continue;
            }

            if (method.kind == method_kind::block_int8_p2) {
                int scale_exp = 0;
                if (amax > 0.0f) {
                    scale_exp = (int) std::ceil(std::log2(amax/127.0f));
                    scale_exp = std::max(-126, std::min(127, scale_exp));
                }
                float scale = std::ldexp(1.0f, scale_exp);
                if (scale > 0.0f && amax/scale > 127.0f && scale_exp < 127) {
                    scale = std::ldexp(1.0f, ++scale_exp);
                }
                for (int64_t i = begin; i < end; ++i) {
                    int q = (int) std::lround(src[row + (size_t) i]/scale);
                    q = std::max(-127, std::min(127, q));
                    out.values[row + (size_t) i] = (float) q*scale;
                    out.saturation += std::abs(q) == 127;
                    out.zero       += q == 0;
                }
                continue;
            }

            int scale_exp = 0;
            if (amax > 0.0f) {
                scale_exp = (int) std::ceil(std::log2(amax/240.0f));
                scale_exp = std::max(-126, std::min(127, scale_exp));
            }
            float scale = std::ldexp(1.0f, scale_exp);
            if (scale > 0.0f && amax/scale > 240.0f && scale_exp < 127) {
                scale = std::ldexp(1.0f, ++scale_exp);
            }
            for (int64_t i = begin; i < end; ++i) {
                const uint8_t q = encode_e4m3(src[row + (size_t) i]/scale);
                const uint8_t mag = q & 0x7fu;
                out.values[row + (size_t) i] = decode_e4m3(q)*scale;
                out.saturation += mag == 0x77u;
                out.zero       += mag == 0u;
                out.subnormal  += mag > 0u && mag < 8u;
            }
        }
    }
    return out;
}

struct error_metrics {
    uint64_t n = 0;
    double mse = 0.0;
    double mae = 0.0;
    double cosine_error = 0.0;
    double max_abs_error = 0.0;
};

static error_metrics compare_vectors(const std::vector<float> & ref, const std::vector<float> & test) {
    error_metrics result;
    result.n = std::min(ref.size(), test.size());
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_test = 0.0;
    for (size_t i = 0; i < result.n; ++i) {
        const double a = ref[i];
        const double b = test[i];
        const double error = b - a;
        result.mse += error*error;
        result.mae += std::fabs(error);
        result.max_abs_error = std::max(result.max_abs_error, std::fabs(error));
        dot += a*b;
        norm_ref += a*a;
        norm_test += b*b;
    }
    if (result.n > 0) {
        result.mse /= (double) result.n;
        result.mae /= (double) result.n;
    }
    if (norm_ref > 0.0 && norm_test > 0.0) {
        result.cosine_error = 1.0 - dot/std::sqrt(norm_ref*norm_test);
    }
    return result;
}

static error_metrics compare_attention_logits(
        const layer_capture & lc,
        const std::vector<float> & k_test,
        int key_stride) {
    error_metrics result;
    if (lc.d <= 0 || lc.kv_heads <= 0 || lc.q_heads <= 0 || lc.q_last.empty() || lc.k.empty()) {
        return result;
    }

    const int64_t tokens = (int64_t) lc.k.size()/(lc.d*lc.kv_heads);
    const int64_t q_step = std::max<int64_t>(1, lc.q_heads/8);
    const int64_t k_step = std::max(1, key_stride);
    const double scale = 1.0/std::sqrt((double) lc.d);
    double dot_logits = 0.0;
    double norm_ref = 0.0;
    double norm_test = 0.0;

    for (int64_t qh = 0; qh < lc.q_heads; qh += q_step) {
        const int64_t kvh = qh*lc.kv_heads/lc.q_heads;
        const float * q = lc.q_last.data() + qh*lc.d;
        for (int64_t token = 0; token < tokens; token += k_step) {
            const size_t base = (size_t) (token*lc.kv_heads + kvh)*lc.d;
            double ref_logit = 0.0;
            double test_logit = 0.0;
            for (int64_t i = 0; i < lc.d; ++i) {
                ref_logit  += (double) q[i]*lc.k[base + (size_t) i];
                test_logit += (double) q[i]*k_test[base + (size_t) i];
            }
            ref_logit *= scale;
            test_logit *= scale;
            const double error = test_logit - ref_logit;
            result.mse += error*error;
            result.mae += std::fabs(error);
            result.max_abs_error = std::max(result.max_abs_error, std::fabs(error));
            dot_logits += ref_logit*test_logit;
            norm_ref += ref_logit*ref_logit;
            norm_test += test_logit*test_logit;
            ++result.n;
        }
    }

    if (result.n > 0) {
        result.mse /= (double) result.n;
        result.mae /= (double) result.n;
    }
    if (norm_ref > 0.0 && norm_test > 0.0) {
        result.cosine_error = 1.0 - dot_logits/std::sqrt(norm_ref*norm_test);
    }
    return result;
}

static void print_tensor_metrics(
        const scout_data & sd,
        int il,
        const char * tensor_name,
        const method_spec & method,
        const std::vector<float> & ref,
        const quantized_values & test) {
    const error_metrics metrics = compare_vectors(ref, test.values);
    const double denom = metrics.n > 0 ? (double) metrics.n : 1.0;
    std::printf(
        "KV_SCOUT_TENSOR task=%s layer=%d tensor=%s method=%s block=%d bpv=%.6f n=%llu "
        "mse=%.9e mae=%.9e cosine_error=%.9e max_abs_error=%.9e saturation_rate=%.9e "
        "zero_rate=%.9e subnormal_rate=%.9e\n",
        sd.label.c_str(), il, tensor_name, method.name, method.block, method.bytes_per_value,
        (unsigned long long) metrics.n, metrics.mse, metrics.mae, metrics.cosine_error,
        metrics.max_abs_error, test.saturation/denom, test.zero/denom, test.subnormal/denom);
}

static bool print_summary(const scout_data & sd, size_t prompt_tokens) {
    if (sd.layers.empty()) {
        LOG_ERR("kv scout: no Q/K/V tensors were captured\n");
        return false;
    }

    bool ok = sd.errors == 0;
    for (const auto & item : sd.layers) {
        const int il = item.first;
        const layer_capture & lc = item.second;
        const int64_t k_tokens = lc.d > 0 && lc.kv_heads > 0 ? (int64_t) lc.k.size()/(lc.d*lc.kv_heads) : 0;
        const int64_t v_tokens = lc.d > 0 && lc.kv_heads > 0 ? (int64_t) lc.v.size()/(lc.d*lc.kv_heads) : 0;
        std::printf(
            "KV_SCOUT_CAPTURE task=%s layer=%d d=%lld kv_heads=%lld q_heads=%lld "
            "prompt_tokens=%zu k_tokens=%lld v_tokens=%lld\n",
            sd.label.c_str(), il, (long long) lc.d, (long long) lc.kv_heads,
            (long long) lc.q_heads, prompt_tokens, (long long) k_tokens, (long long) v_tokens);

        if (k_tokens != (int64_t) prompt_tokens || v_tokens != (int64_t) prompt_tokens || lc.q_last.empty()) {
            LOG_ERR("kv scout: incomplete capture at layer %d\n", il);
            ok = false;
            continue;
        }

        for (const method_spec & method : METHODS) {
            const quantized_values k_test = reconstruct(lc.k, lc.d, method);
            const quantized_values v_test = reconstruct(lc.v, lc.d, method);
            print_tensor_metrics(sd, il, "K", method, lc.k, k_test);
            print_tensor_metrics(sd, il, "V", method, lc.v, v_test);

            const error_metrics logits = compare_attention_logits(lc, k_test.values, sd.key_stride);
            std::printf(
                "KV_SCOUT_LOGIT task=%s layer=%d method=%s block=%d bpv=%.6f samples=%llu "
                "mse=%.9e mae=%.9e cosine_error=%.9e max_abs_error=%.9e key_stride=%d\n",
                sd.label.c_str(), il, method.name, method.block, method.bytes_per_value,
                (unsigned long long) logits.n, logits.mse, logits.mae, logits.cosine_error,
                logits.max_abs_error, sd.key_stride);
        }
    }
    std::fflush(stdout);
    return ok;
}

static bool run_prefill(llama_context * ctx, const common_params & params, scout_data & sd, size_t & n_prompt_out) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens - provide a prompt with -p or -f\n", __func__);
        return false;
    }

    n_prompt_out = tokens.size();
    const size_t batch = (size_t) std::max(1, params.n_batch);
    LOG_INF("%s: capturing %zu tokens in batches of %zu through layer %d\n",
        __func__, tokens.size(), batch, sd.max_layer);
    for (size_t offset = 0; offset < tokens.size(); offset += batch) {
        const size_t count = std::min(batch, tokens.size() - offset);
        sd.final_chunk = offset + count == tokens.size();
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + offset, (int32_t) count))) {
            LOG_ERR("%s: llama_decode failed at token %zu\n", __func__, offset);
            return false;
        }
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
    params.warmup = false;

    scout_data sd;
    if (const char * value = std::getenv("KV_SCOUT_LABEL")) {
        sd.label = value;
    }
    if (const char * value = std::getenv("KV_SCOUT_MAX_LAYER")) {
        sd.max_layer = std::max(0, std::atoi(value));
    }
    if (const char * value = std::getenv("KV_SCOUT_KEY_STRIDE")) {
        sd.key_stride = std::max(1, std::atoi(value));
    }
    params.cb_eval = scout_cb_eval;
    params.cb_eval_user_data = &sd;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    size_t prompt_tokens = 0;
    const bool decoded = run_prefill(ctx, params, sd, prompt_tokens);
    const bool summarized = decoded && print_summary(sd, prompt_tokens);
    llama_backend_free();
    return summarized ? 0 : 1;
}