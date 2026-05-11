#include "mmvq.cuh"

bool ggml_cuda_mmvq_dispatch_rest(ggml_type type_x, GGML_CUDA_MMVQ_TYPED_DISPATCH_ARGS) {
#define GGML_CUDA_MMVQ_DISPATCH_REST_CASE(type_name)                                                   \
    case type_name:                                                                                     \
        ggml_cuda_mmvq_dispatch_type_##type_name(                                                       \
            vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,                                     \
            stride_row_x, stride_col_y, stride_col_dst,                                                 \
            nchannels_x, nchannels_y, nchannels_dst,                                                    \
            stride_channel_x, stride_channel_y, stride_channel_dst,                                     \
            nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,             \
            ids_stride, stream);                                                                        \
        return true;

    switch (type_x) {
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_Q1_0)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_Q4_0)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_Q4_1)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_Q5_0)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_Q5_1)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_Q8_0)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_MXFP4)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_NVFP4)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_Q2_K)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_Q5_K)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ2_XXS)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ2_XS)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ2_S)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ3_XXS)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ1_S)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ1_M)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ4_NL)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ4_XS)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_IQ3_S)
        GGML_CUDA_MMVQ_DISPATCH_REST_CASE(GGML_TYPE_TQ3_0)
        default:
            return false;
    }

#undef GGML_CUDA_MMVQ_DISPATCH_REST_CASE
}
