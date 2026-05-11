#include "mmvq.cuh"

bool ggml_cuda_mmvq_dispatch_qwen_hot(ggml_type type_x, GGML_CUDA_MMVQ_TYPED_DISPATCH_ARGS) {
    switch (type_x) {
        case GGML_TYPE_Q3_K:
            ggml_cuda_mmvq_dispatch_type_GGML_TYPE_Q3_K(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                ids_stride, stream);
            return true;
        case GGML_TYPE_Q4_K:
            ggml_cuda_mmvq_dispatch_type_GGML_TYPE_Q4_K(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                ids_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            ggml_cuda_mmvq_dispatch_type_GGML_TYPE_Q6_K(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                ids_stride, stream);
            return true;
        default:
            return false;
    }
}
