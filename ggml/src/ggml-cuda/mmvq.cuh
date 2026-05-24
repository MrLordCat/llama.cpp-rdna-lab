#pragma once

#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// Internal typed dispatcher shared between MMVQ translation units.
void ggml_cuda_mmvq_switch_type(
    const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
    const int ncols_x, const int nrows_x, const int ncols_dst,
    const int stride_row_x, const int stride_col_y, const int stride_col_dst,
    const int nchannels_x, const int nchannels_y, const int nchannels_dst,
    const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
    const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
    const int ids_stride, const bool q3k_padded_storage, cudaStream_t stream);

#define GGML_CUDA_MMVQ_TYPED_DISPATCH_ARGS \
    const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst, \
    const int ncols_x, const int nrows_x, const int ncols_dst, \
    const int stride_row_x, const int stride_col_y, const int stride_col_dst, \
    const int nchannels_x, const int nchannels_y, const int nchannels_dst, \
    const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst, \
    const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst, \
    const int ids_stride, const bool q3k_padded_storage, cudaStream_t stream

#define GGML_CUDA_MMVQ_TYPE_LIST(X) \
    X(GGML_TYPE_Q1_0) \
    X(GGML_TYPE_Q4_0) \
    X(GGML_TYPE_Q4_1) \
    X(GGML_TYPE_Q5_0) \
    X(GGML_TYPE_Q5_1) \
    X(GGML_TYPE_Q8_0) \
    X(GGML_TYPE_MXFP4) \
    X(GGML_TYPE_NVFP4) \
    X(GGML_TYPE_Q2_K) \
    X(GGML_TYPE_Q3_K) \
    X(GGML_TYPE_Q4_K) \
    X(GGML_TYPE_Q5_K) \
    X(GGML_TYPE_Q6_K) \
    X(GGML_TYPE_IQ2_XXS) \
    X(GGML_TYPE_IQ2_XS) \
    X(GGML_TYPE_IQ2_S) \
    X(GGML_TYPE_IQ3_XXS) \
    X(GGML_TYPE_IQ1_S) \
    X(GGML_TYPE_IQ1_M) \
    X(GGML_TYPE_IQ4_NL) \
    X(GGML_TYPE_IQ4_XS) \
    X(GGML_TYPE_IQ3_S) \
    X(GGML_TYPE_TQ3_0)

#define GGML_CUDA_MMVQ_DECLARE_TYPED_DISPATCH(type_name) \
    void ggml_cuda_mmvq_dispatch_type_##type_name(GGML_CUDA_MMVQ_TYPED_DISPATCH_ARGS);

GGML_CUDA_MMVQ_TYPE_LIST(GGML_CUDA_MMVQ_DECLARE_TYPED_DISPATCH)

#undef GGML_CUDA_MMVQ_DECLARE_TYPED_DISPATCH

bool ggml_cuda_mmvq_dispatch_qwen_hot(ggml_type type_x, GGML_CUDA_MMVQ_TYPED_DISPATCH_ARGS);
bool ggml_cuda_mmvq_dispatch_rest(ggml_type type_x, GGML_CUDA_MMVQ_TYPED_DISPATCH_ARGS);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
