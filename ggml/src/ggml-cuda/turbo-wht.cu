#include "turbo-wht.cuh"

static __device__ __forceinline__ int turbo_wht_sign_device(int i) {
    uint32_t x = (uint32_t)i * 0x9E3779B9u + 0x85EBCA6Bu;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return (x & 1u) ? 1 : -1;
}

template <int direction>
static __global__ void turbo_wht_128_f32(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        int64_t                    n_groups,
        int64_t                    head_dim,
        int64_t                    groups_per_head) {
    const int64_t g = blockIdx.x;
    if (g >= n_groups) {
        return;
    }

    const int t = threadIdx.x;
    const int64_t head_idx    = g / groups_per_head;
    const int64_t group_in_hd = g % groups_per_head;
    const int64_t base        = head_idx * head_dim + group_in_hd * QK_TKV_0;

    __shared__ float x[QK_TKV_0];

    const float v = src[base + t];
    x[t] = direction == 0 ? v * (float) turbo_wht_sign_device(t) : v;
    __syncthreads();

#define WHT_STAGE(step) \
    if ((t % (2*(step))) < (step)) { \
        const float a = x[t]; \
        const float b = x[t + (step)]; \
        x[t]          = a + b; \
        x[t + (step)] = a - b; \
    } \
    __syncthreads();

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    WHT_STAGE(32)
    WHT_STAGE(64)

#undef WHT_STAGE

    constexpr float inv_sqrt128 = 0.08838834764831845f;
    float out = x[t] * inv_sqrt128;
    if (direction == 1) {
        out *= (float) turbo_wht_sign_device(t);
    }

    dst[base + t] = out;
}

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int direction  = ggml_get_op_params_i32(dst, 0);
    const int group_size = ggml_get_op_params_i32(dst, 1);

    GGML_ASSERT(direction == 0 || direction == 1);
    GGML_ASSERT(group_size == QK_TKV_0);
    GGML_ASSERT(src0->ne[0] % group_size == 0);

    const int64_t head_dim        = src0->ne[0];
    const int64_t n_heads         = ggml_nelements(src0) / head_dim;
    const int64_t groups_per_head = head_dim / group_size;
    const int64_t n_groups        = n_heads * groups_per_head;

    if (n_groups == 0) {
        return;
    }

    const float * src = (const float *) src0->data;
    float       * out = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();
    const dim3 blocks((unsigned int) n_groups);
    const dim3 threads(QK_TKV_0);

    if (direction == 0) {
        turbo_wht_128_f32<0><<<blocks, threads, 0, stream>>>(src, out, n_groups, head_dim, groups_per_head);
    } else {
        turbo_wht_128_f32<1><<<blocks, threads, 0, stream>>>(src, out, n_groups, head_dim, groups_per_head);
    }
}