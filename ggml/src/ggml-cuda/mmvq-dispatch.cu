#include "mmvq.cuh"
#include "quantize.cuh"

#include <cstdlib>

static bool ggml_cuda_trace_mmvq_path_enabled() {
    static const bool enabled = std::getenv("GGML_TRACE_MMVQ_PATH") != nullptr;
    return enabled;
}

static size_t ggml_cuda_q3k_padded_storage_alloc_size_for_tensor(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(tensor->ne[0] % QK_K == 0);
    GGML_ASSERT(ggml_nelements(tensor) % QK_K == 0);

    size_t size = (ggml_nelements(tensor) / QK_K) * sizeof(block_q3_K_padded);

    if (tensor->ne[0] % MATRIX_ROW_PADDING != 0) {
        const int64_t pad_elems = MATRIX_ROW_PADDING - tensor->ne[0] % MATRIX_ROW_PADDING;
        GGML_ASSERT(pad_elems % QK_K == 0);
        size += (pad_elems / QK_K) * sizeof(block_q3_K_padded);
    }

    return size;
}

static bool ggml_cuda_mmvq_q3k_padded_storage_tensor(const ggml_tensor * tensor) {
    if (!(tensor->type == GGML_TYPE_Q3_K &&
            tensor->view_src == nullptr &&
            ggml_is_contiguous(tensor) &&
            tensor->ne[0] % QK_K == 0 &&
            ggml_nelements(tensor) % QK_K == 0 &&
            tensor->buffer != nullptr)) {
        return false;
    }

    return ggml_backend_buffer_get_alloc_size(tensor->buffer, tensor) ==
        ggml_cuda_q3k_padded_storage_alloc_size_for_tensor(tensor);
}

void ggml_cuda_mmvq_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const bool q3k_padded_storage, cudaStream_t stream) {
    const bool trace_path = ggml_cuda_trace_mmvq_path_enabled();
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

    if (ggml_cuda_mmvq_dispatch_qwen_hot(
            type_x, vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_x, nchannels_y, nchannels_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
            ids_stride, q3k_padded_storage, stream)) {
        if (trace_path) {
            GGML_LOG_INFO(
                "%s: type=%d/%s route=qwen-hot ncols_x=%d nrows_x=%d ncols_dst=%d ids=%d fusion=%d\n",
                __func__,
                (int) type_x,
                ggml_type_name(type_x),
                ncols_x,
                nrows_x,
                ncols_dst,
                ids != nullptr,
                has_fusion);
        }
        return;
    }

    if (ggml_cuda_mmvq_dispatch_rest(
            type_x, vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_x, nchannels_y, nchannels_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
            ids_stride, q3k_padded_storage, stream)) {
        if (trace_path) {
            GGML_LOG_INFO(
                "%s: type=%d/%s route=rest ncols_x=%d nrows_x=%d ncols_dst=%d ids=%d fusion=%d\n",
                __func__,
                (int) type_x,
                ggml_type_name(type_x),
                ncols_x,
                nrows_x,
                ncols_dst,
                ids != nullptr,
                has_fusion);
        }
        return;
    }

    GGML_ABORT("fatal error");
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;
    const bool q3k_padded_storage = ggml_cuda_mmvq_q3k_padded_storage_tensor(src0);
    if (q3k_padded_storage && fusion && fusion->gate) {
        GGML_ASSERT(ggml_cuda_mmvq_q3k_padded_storage_tensor(fusion->gate));
    }

    ggml_cuda_mmvq_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, q3k_padded_storage, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    ggml_cuda_mmvq_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, false, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
