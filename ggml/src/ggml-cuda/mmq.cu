#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <chrono>
#include <cstdlib>

static int ggml_rdna4_stream_k_min_ne11() {
    // Experimental override for RDNA4 stream-k threshold (default keeps existing behavior).
    int min_ne11 = 256;
    if (const char * env = std::getenv("GGML_MMQ_RDNA4_STREAM_K_MIN_NE11")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) {
            min_ne11 = parsed;
        }
    }
    return min_ne11;
}

static int ggml_rdna4_q4k_mmq_max_ne11() {
    int max_ne11 = 1024;
    if (const char * env = std::getenv("GGML_MMQ_RDNA4_Q4K_MAX_NE11")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) {
            max_ne11 = parsed;
        }
    }
    return max_ne11;
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

static bool ggml_cuda_q3k_padded_storage_enabled() {
    static bool result = [] {
        const char * env = std::getenv("GGML_CUDA_Q3K_PADDED_STORAGE");
#ifdef GGML_USE_HIP
        const bool enabled = env == nullptr ? true : std::atoi(env) != 0;
#else
        const bool enabled = env != nullptr && std::atoi(env) != 0;
#endif
        return enabled;
    }();

    return result;
}

static bool ggml_cuda_q3k_padded_storage_mmq_enabled() {
    static bool result = [] {
        const char * env = std::getenv("GGML_CUDA_Q3K_PADDED_STORAGE_MMQ");
#ifdef GGML_USE_HIP
        const bool enabled = env == nullptr ? true : std::atoi(env) != 0;
#else
        const bool enabled = env != nullptr && std::atoi(env) != 0;
#endif
        return enabled;
    }();

    return result;
}

static bool ggml_cuda_q3k_padded_storage_tensor(const ggml_tensor * tensor) {
    if (tensor->type == GGML_TYPE_Q3_K &&
            tensor->view_src != nullptr &&
            tensor->view_src->type == GGML_TYPE_Q3_K &&
            ggml_is_contiguous(tensor->view_src) &&
            tensor->view_src->ne[0] % QK_K == 0 &&
            ggml_nelements(tensor->view_src) % QK_K == 0 &&
            tensor->view_src->buffer != nullptr) {
        const size_t owner_raw_size = (ggml_nelements(tensor->view_src) / QK_K) * sizeof(block_q3_K);
        const size_t owner_alloc_size = ggml_backend_buffer_get_alloc_size(tensor->view_src->buffer, tensor->view_src);

        if (owner_alloc_size > owner_raw_size) {
            GGML_ABORT("Q3_K padded MMQ path does not support tensor views yet");
        }
    }

    if (!(ggml_cuda_q3k_padded_storage_enabled() &&
            tensor->type == GGML_TYPE_Q3_K &&
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

static bool ggml_cuda_q3k_padded_storage_mmq_tensor(const ggml_tensor * tensor) {
    return ggml_cuda_q3k_padded_storage_mmq_enabled() &&
        ggml_cuda_q3k_padded_storage_tensor(tensor);
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;
    const bool q3k_padded_storage = ggml_cuda_q3k_padded_storage_mmq_tensor(src0);
    const bool trace_src1_quant_timing = src0->type == GGML_TYPE_Q3_K &&
        std::getenv("GGML_TRACE_MMQ_SRC1_QUANT_TIMING") != nullptr;
    const bool trace_src1_quant_timing_sync = trace_src1_quant_timing &&
        std::getenv("GGML_TRACE_MMQ_SRC1_QUANT_TIMING_SYNC") != nullptr;
    const bool trace_src1_quant_timing_pre_sync = trace_src1_quant_timing_sync &&
        std::getenv("GGML_TRACE_MMQ_SRC1_QUANT_TIMING_PRE_SYNC") != nullptr;

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

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);
    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            double pre_sync_ms = 0.0;
            bool pre_sync_applied = false;
            int pre_sync_capture_active = 0;
            if (trace_src1_quant_timing_pre_sync) {
#ifdef GGML_USE_HIP
                hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
                CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
                pre_sync_capture_active = capture_status != hipStreamCaptureStatusNone;
#else
                cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
                CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
                pre_sync_capture_active = capture_status != cudaStreamCaptureStatusNone;
#endif
                if (!pre_sync_capture_active) {
                    const auto pre_sync_start = std::chrono::high_resolution_clock::now();
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    const auto pre_sync_end = std::chrono::high_resolution_clock::now();
                    pre_sync_ms = std::chrono::duration<double, std::milli>(pre_sync_end - pre_sync_start).count();
                    pre_sync_applied = true;
                }
            }
            const auto timing_start = trace_src1_quant_timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
            if (use_native_fp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
            if (trace_src1_quant_timing) {
                const auto timing_after_launch = std::chrono::high_resolution_clock::now();
                const double enqueue_ms = std::chrono::duration<double, std::milli>(timing_after_launch - timing_start).count();
                double sync_ms = 0.0;
                int capture_active = 0;
                bool sync_applied = false;
                if (trace_src1_quant_timing_sync) {
#ifdef GGML_USE_HIP
                    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
                    CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
                    capture_active = capture_status != hipStreamCaptureStatusNone;
#else
                    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
                    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
                    capture_active = capture_status != cudaStreamCaptureStatusNone;
#endif
                    if (!capture_active) {
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        const auto timing_after_sync = std::chrono::high_resolution_clock::now();
                        sync_ms = std::chrono::duration<double, std::milli>(timing_after_sync - timing_after_launch).count();
                        sync_applied = true;
                    }
                }
                GGML_LOG_INFO(
                    "GGML_TRACE_MMQ_SRC1_QUANT_TIMING: type=%d cc=%d dst=%s src0=%s src1=%s ne00=%lld ne01=%lld ne10=%lld ne11=%lld ne12=%lld ne13=%lld ne10_padded=%lld q3k_padded=%d sync_req=%d pre_sync_applied=%d sync_applied=%d pre_capture=%d capture=%d pre_sync_ms=%.3f enqueue_ms=%.3f sync_ms=%.3f total_ms=%.3f\n",
                    (int) src0->type,
                    cc,
                    dst->name,
                    src0->name,
                    src1->name,
                    (long long) ne00,
                    (long long) ne01,
                    (long long) ne10,
                    (long long) ne11,
                    (long long) ne12,
                    (long long) ne13,
                    (long long) ne10_padded,
                    q3k_padded_storage ? 1 : 0,
                    trace_src1_quant_timing_sync ? 1 : 0,
                    pre_sync_applied ? 1 : 0,
                    sync_applied ? 1 : 0,
                    pre_sync_capture_active,
                    capture_active,
                    pre_sync_ms,
                    enqueue_ms,
                    sync_ms,
                    enqueue_ms + sync_ms);
            }
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :  // block_fp4_mmq holds 256 values
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1, q3k_padded_storage};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_K == 8 * QK_MXFP4, "QK_K needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12, q3k_padded_storage};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc)
                            || (GGML_CUDA_CC_IS_RDNA4(cc) && ne11 >= ggml_rdna4_stream_k_min_ne11()))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols, ggml_cuda_q3k_padded_storage_mmq_tensor(src0)};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (std::getenv("GGML_CUDA_FORCE_MMQ_RUNTIME") != nullptr) {
        return true;
    }

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            if (n_experts >= 64) {
                return true;
            }

            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                    return ne11 <= 256;
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                    return ne11 <= ggml_rdna4_q4k_mmq_max_ne11();
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q6_K:
                    return ne11 <= 192;
                default:
                    return ne11 <= 128;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
