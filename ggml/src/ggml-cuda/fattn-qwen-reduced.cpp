#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct ggml_backend_cuda_context;

#if !defined(GGML_CUDA_NO_FA)
template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif // !defined(GGML_CUDA_NO_FA)

void ggml_cuda_set_device(int device);

enum qwen_reduced_fattn_kernel {
    QWEN_REDUCED_FATTN_NONE,
    QWEN_REDUCED_FATTN_VEC,
    QWEN_REDUCED_FATTN_WMMA_F16,
};

static constexpr int FATTN_REDUCED_KQ_STRIDE = 256;

static const char * ggml_cuda_reduced_fattn_kernel_name(const qwen_reduced_fattn_kernel kernel) {
    switch (kernel) {
        case QWEN_REDUCED_FATTN_NONE:     return "none";
        case QWEN_REDUCED_FATTN_VEC:      return "vec";
        case QWEN_REDUCED_FATTN_WMMA_F16: return "wmma_f16";
    }

    return "unknown";
}

static qwen_reduced_fattn_kernel ggml_cuda_reduced_fattn_forced_kernel() {
    static int cached = -1;
    if (cached == -1) {
        cached = QWEN_REDUCED_FATTN_NONE;
        const char * env = std::getenv("GGML_QWEN_FA_REDUCED_FORCE");
        if (env != nullptr) {
            if (std::strcmp(env, "vec") == 0) {
                cached = QWEN_REDUCED_FATTN_VEC;
            } else if (std::strcmp(env, "wmma") == 0 || std::strcmp(env, "wmma_f16") == 0) {
                cached = QWEN_REDUCED_FATTN_WMMA_F16;
            }
        }
    }

    return (qwen_reduced_fattn_kernel) cached;
}

static int ggml_cuda_reduced_ctx_device(const ggml_backend_cuda_context & ctx) {
    return *reinterpret_cast<const int *>(&ctx);
}

#if !defined(GGML_CUDA_NO_FA)
#define FATTN_REDUCED_VEC_CASE(D, type_K, type_V)                                                                \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }

#define FATTN_REDUCED_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_REDUCED_VEC_CASE( 64, type_K, type_V)       \
    FATTN_REDUCED_VEC_CASE(128, type_K, type_V)       \
    FATTN_REDUCED_VEC_CASE(256, type_K, type_V)

static void ggml_cuda_flash_attn_ext_vec_reduced(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    FATTN_REDUCED_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_REDUCED_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_REDUCED_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_REDUCED_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)

    GGML_ABORT("FlashAttention vec case is not available in GGML_HIP_QWEN_FA_REDUCED");
}

static bool ggml_cuda_reduced_fattn_type_supported(const ggml_tensor * K, const ggml_tensor * V) {
#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return false;
    }
#endif

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}
#endif // !defined(GGML_CUDA_NO_FA)

static qwen_reduced_fattn_kernel ggml_cuda_reduced_fattn_kernel(const ggml_tensor * dst) {
#ifdef GGML_CUDA_NO_FA
    GGML_UNUSED(dst);
    return QWEN_REDUCED_FATTN_NONE;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!(Q->ne[0] == 64 || Q->ne[0] == 128 || Q->ne[0] == 256)) {
        return QWEN_REDUCED_FATTN_NONE;
    }
    if (V->ne[0] != K->ne[0]) {
        return QWEN_REDUCED_FATTN_NONE;
    }
    if (mask && mask->ne[2] != 1) {
        return QWEN_REDUCED_FATTN_NONE;
    }
    if (!ggml_cuda_reduced_fattn_type_supported(K, V)) {
        return QWEN_REDUCED_FATTN_NONE;
    }
    if (K->ne[1] % FATTN_REDUCED_KQ_STRIDE != 0) {
        return QWEN_REDUCED_FATTN_NONE;
    }

    const qwen_reduced_fattn_kernel base_kernel = Q->ne[1] <= 2 ? QWEN_REDUCED_FATTN_VEC : QWEN_REDUCED_FATTN_WMMA_F16;
    const qwen_reduced_fattn_kernel forced_kernel = ggml_cuda_reduced_fattn_forced_kernel();
    const qwen_reduced_fattn_kernel selected_kernel = forced_kernel == QWEN_REDUCED_FATTN_NONE ? base_kernel : forced_kernel;

    if (std::getenv("GGML_TRACE_FATTN_SELECTED") != nullptr) {
        std::fprintf(
            stderr,
            "%s: reduced Q0=%lld Q1=%lld Q2=%lld Q3=%lld K0=%lld K1=%lld V0=%lld base=%s force=%s selected=%s\n",
            __func__,
            (long long) Q->ne[0],
            (long long) Q->ne[1],
            (long long) Q->ne[2],
            (long long) Q->ne[3],
            (long long) K->ne[0],
            (long long) K->ne[1],
            (long long) V->ne[0],
            ggml_cuda_reduced_fattn_kernel_name(base_kernel),
            ggml_cuda_reduced_fattn_kernel_name(forced_kernel),
            ggml_cuda_reduced_fattn_kernel_name(selected_kernel));
    }

    return selected_kernel;
#endif
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ggml_cuda_reduced_ctx_device(ctx));

#if defined(GGML_CUDA_NO_FA)
    GGML_UNUSED(dst);
    GGML_ABORT("FlashAttention is disabled by the active HIP experiment profile");
#else
    switch (ggml_cuda_reduced_fattn_kernel(dst)) {
        case QWEN_REDUCED_FATTN_VEC:
            ggml_cuda_flash_attn_ext_vec_reduced(ctx, dst);
            return;
        case QWEN_REDUCED_FATTN_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            return;
        case QWEN_REDUCED_FATTN_NONE:
            GGML_ABORT("FlashAttention shape is unsupported by GGML_HIP_QWEN_FA_REDUCED");
    }

    GGML_ABORT("fatal error");
#endif // defined(GGML_CUDA_NO_FA)
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
#if defined(GGML_CUDA_NO_FA)
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    GGML_UNUSED(device);
    return ggml_cuda_reduced_fattn_kernel(dst) != QWEN_REDUCED_FATTN_NONE;
#endif // defined(GGML_CUDA_NO_FA)
}