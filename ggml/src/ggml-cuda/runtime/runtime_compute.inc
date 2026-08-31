/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

struct cublas_force_compute_type {
    bool fp32 = false;
    bool fp16 = false;
};

static const cublas_force_compute_type & ggml_cuda_cublas_get_force_compute_type() {
    static const cublas_force_compute_type compute_type = [] {
        cublas_force_compute_type result;

        const bool ggml_cuda_force_cublas_compute_32f_env = getenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F") != nullptr;
        const bool ggml_cuda_force_cublas_compute_16f_env = getenv("GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F") != nullptr;

        GGML_ASSERT(ggml_cuda_force_cublas_compute_16f_env == false || ggml_cuda_force_cublas_compute_32f_env == false);

        if (ggml_cuda_force_cublas_compute_32f_env) {
            GGML_LOG_INFO("Detected GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F\n");
            result.fp32 = true;
        } else if (ggml_cuda_force_cublas_compute_16f_env) {
            GGML_LOG_INFO("Detected GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F\n");
            result.fp16 = true;
        }

        return result;
    }();

    return compute_type;
}

struct cublas_split_timing_config {
    bool enabled = false;
    bool detail = false;
    bool pre_sync = false;
    int64_t min_ncols = 256;
};

static const cublas_split_timing_config & ggml_cuda_cublas_get_split_timing_config() {
    static const cublas_split_timing_config config = [] {
        cublas_split_timing_config result;

        result.detail = getenv("GGML_TRACE_CUBLAS_SPLIT_DETAIL") != nullptr;
        result.enabled = getenv("GGML_TRACE_CUBLAS_SPLIT_TIMING") != nullptr || result.detail;
        result.pre_sync = getenv("GGML_TRACE_CUBLAS_SPLIT_TIMING_PRE_SYNC") != nullptr;

        if (const char * env = getenv("GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS")) {
            result.min_ncols = std::max<int64_t>(0, std::atoll(env));
        }

        if (result.enabled) {
            GGML_LOG_INFO(
                "Detected GGML_TRACE_CUBLAS_SPLIT_TIMING min_ncols=%lld detail=%d pre_sync=%d\n",
                (long long) result.min_ncols, result.detail ? 1 : 0, result.pre_sync ? 1 : 0);
        }

        return result;
    }();

    return config;
}

struct cublas_q3k_route_trace_config {
    bool enabled = false;
    int64_t min_ncols = 1024;
};

struct cublas_src1_reuse_trace_config {
    bool enabled = false;
    int64_t min_ncols = 1024;
};

struct q3k_hot_direct_route_config {
    bool enabled = false;
    int64_t min_ncols = 1024;
    int64_t match_row_diff = 0;
    int64_t match_ne00 = 0;
    int64_t match_ncols = 0;
};

struct q3k_cublas_pipeline_config {
    bool enabled = false;
    bool allow_graphs = false;
    int64_t min_ncols = 1024;
    int64_t min_rows = 4096;
    int64_t chunk_rows = 6144;
};

struct cublas_src1_f16_reuse_config {
    bool enabled = false;
    int64_t min_ncols = 1024;
};

static const cublas_q3k_route_trace_config & ggml_cuda_cublas_get_q3k_route_trace_config() {
    static const cublas_q3k_route_trace_config config = [] {
        cublas_q3k_route_trace_config result;

        result.enabled = getenv("GGML_TRACE_CUBLAS_Q3K_ROUTE") != nullptr;

        if (const char * env = getenv("GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS")) {
            result.min_ncols = std::max<int64_t>(0, std::atoll(env));
        }

        if (result.enabled) {
            GGML_LOG_INFO(
                "Detected GGML_TRACE_CUBLAS_Q3K_ROUTE min_ncols=%lld\n",
                (long long) result.min_ncols);
        }

        return result;
    }();

    return config;
}

static const cublas_src1_reuse_trace_config & ggml_cuda_cublas_get_src1_reuse_trace_config() {
    static const cublas_src1_reuse_trace_config config = [] {
        cublas_src1_reuse_trace_config result;

        result.enabled = getenv("GGML_TRACE_CUBLAS_SRC1_REUSE") != nullptr;

        if (const char * env = getenv("GGML_TRACE_CUBLAS_SRC1_REUSE_MIN_NCOLS")) {
            result.min_ncols = std::max<int64_t>(0, std::atoll(env));
        }

        if (result.enabled) {
            GGML_LOG_INFO(
                "Detected GGML_TRACE_CUBLAS_SRC1_REUSE min_ncols=%lld\n",
                (long long) result.min_ncols);
        }

        return result;
    }();

    return config;
}

static const q3k_hot_direct_route_config & ggml_cuda_get_q3k_hot_direct_route_config() {
    static const q3k_hot_direct_route_config config = [] {
        q3k_hot_direct_route_config result;

        result.enabled = getenv("GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE") != nullptr;

        if (const char * env = getenv("GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE_MIN_NCOLS")) {
            result.min_ncols = std::max<int64_t>(0, std::atoll(env));
        }

        if (const char * env = getenv("GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE_MATCH_ROW_DIFF")) {
            result.match_row_diff = std::max<int64_t>(0, std::atoll(env));
        }

        if (const char * env = getenv("GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE_MATCH_NE00")) {
            result.match_ne00 = std::max<int64_t>(0, std::atoll(env));
        }

        if (const char * env = getenv("GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE_MATCH_NCOLS")) {
            result.match_ncols = std::max<int64_t>(0, std::atoll(env));
        }

        if (result.enabled) {
            GGML_LOG_INFO(
                "Detected GGML_EXPERIMENTAL_Q3K_HOT_DIRECT_ROUTE min_ncols=%lld match_row_diff=%lld match_ne00=%lld match_ncols=%lld\n",
                (long long) result.min_ncols,
                (long long) result.match_row_diff,
                (long long) result.match_ne00,
                (long long) result.match_ncols);
        }

        return result;
    }();

    return config;
}

static const q3k_cublas_pipeline_config & ggml_cuda_get_q3k_cublas_pipeline_config() {
    static const q3k_cublas_pipeline_config config = [] {
        q3k_cublas_pipeline_config result;

        result.enabled = getenv("GGML_EXPERIMENTAL_Q3K_CUBLAS_PIPELINE") != nullptr;
        result.allow_graphs = getenv("GGML_EXPERIMENTAL_Q3K_CUBLAS_PIPELINE_ALLOW_GRAPHS") != nullptr;

        if (const char * env = getenv("GGML_EXPERIMENTAL_Q3K_CUBLAS_PIPELINE_MIN_NCOLS")) {
            result.min_ncols = std::max<int64_t>(0, std::atoll(env));
        }

        if (const char * env = getenv("GGML_EXPERIMENTAL_Q3K_CUBLAS_PIPELINE_MIN_ROWS")) {
            result.min_rows = std::max<int64_t>(0, std::atoll(env));
        }

        if (const char * env = getenv("GGML_EXPERIMENTAL_Q3K_CUBLAS_PIPELINE_CHUNK_ROWS")) {
            result.chunk_rows = std::max<int64_t>(1, std::atoll(env));
        }

        if (result.enabled) {
            GGML_LOG_INFO(
                "Detected GGML_EXPERIMENTAL_Q3K_CUBLAS_PIPELINE min_ncols=%lld min_rows=%lld chunk_rows=%lld allow_graphs=%d\n",
                (long long) result.min_ncols,
                (long long) result.min_rows,
                (long long) result.chunk_rows,
                result.allow_graphs ? 1 : 0);
        }

        return result;
    }();

    return config;
}

static const cublas_src1_f16_reuse_config & ggml_cuda_get_cublas_src1_f16_reuse_config() {
    static const cublas_src1_f16_reuse_config config = [] {
        cublas_src1_f16_reuse_config result;

        result.enabled = getenv("GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE") != nullptr;

        if (const char * env = getenv("GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE_MIN_NCOLS")) {
            result.min_ncols = std::max<int64_t>(0, std::atoll(env));
        }

        if (result.enabled) {
            GGML_LOG_INFO(
                "Detected GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE min_ncols=%lld\n",
                (long long) result.min_ncols);
        }

        return result;
    }();

    return config;
}

struct cublas_q3k_route_trace_stats {
    int64_t calls = 0;
    int64_t total_ncols = 0;
    int64_t total_convert_elems = 0;
    double total_convert_ms = 0.0;
    int64_t last_event_idx = -1;
    int64_t repeated_calls = 0;
    int64_t min_gap = std::numeric_limits<int64_t>::max();
    int64_t max_gap = 0;
    int64_t total_gap = 0;
};

struct cublas_src1_reuse_trace_stats {
    int64_t calls = 0;
    int64_t repeated_calls = 0;
    int64_t last_event_idx = -1;
    int64_t min_gap = std::numeric_limits<int64_t>::max();
    int64_t max_gap = 0;
    int64_t total_gap = 0;
    double total_src1_ms = 0.0;
    double repeated_src1_ms = 0.0;
    double tight_repeated_src1_ms = 0.0;
};

static std::string ggml_cuda_cublas_q3k_staging_key(
        const ggml_tensor * src0, const int dev,
        const int64_t row_low, const int64_t row_high, const int64_t ne00) {
    char key[256];
    snprintf(key, sizeof(key), "%p|%p|dev=%d|row=%lld:%lld|ne00=%lld",
        (const void *) src0, src0->data, dev,
        (long long) row_low, (long long) row_high, (long long) ne00);
    return std::string(key);
}

static std::string ggml_cuda_cublas_src1_reuse_key(
        const ggml_tensor * src1, const float * src1_ddf_i, const int dev,
        const int64_t ne10, const int64_t src1_ncols) {
    char key[256];
    snprintf(key, sizeof(key), "%p|%p|%p|dev=%d|ne10=%lld|ncols=%lld",
        (const void *) src1, src1->data, (const void *) src1_ddf_i, dev,
        (long long) ne10, (long long) src1_ncols);
    return std::string(key);
}

static void ggml_cuda_cublas_trace_q3k_route(
        const cublas_q3k_route_trace_config & config,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        const char * src0_dd_i, const int dev, const int cc,
        const int64_t row_low, const int64_t row_high,
        const int64_t ne00, const int64_t ne10, const int64_t src1_ncols,
        const double src0_ms, const double src0_convert_ms) {
    if (!config.enabled || src0->type != GGML_TYPE_Q3_K || src1_ncols < config.min_ncols) {
        return;
    }

    const int64_t row_diff = row_high - row_low;
    const int64_t convert_elems = row_diff * ne00;

    const std::string key = ggml_cuda_cublas_q3k_staging_key(src0, dev, row_low, row_high, ne00);

    static std::mutex mutex;
    static int64_t global_event_idx = 0;
    static std::map<std::string, cublas_q3k_route_trace_stats> stats_by_key;

    std::lock_guard<std::mutex> lock(mutex);
    const int64_t event_idx = global_event_idx++;
    cublas_q3k_route_trace_stats & stats = stats_by_key[key];
    int64_t gap_since_prev = -1;
    if (stats.last_event_idx >= 0) {
        gap_since_prev = event_idx - stats.last_event_idx;
        stats.repeated_calls += 1;
        stats.total_gap += gap_since_prev;
        stats.min_gap = std::min(stats.min_gap, gap_since_prev);
        stats.max_gap = std::max(stats.max_gap, gap_since_prev);
    }
    stats.last_event_idx = event_idx;
    stats.calls += 1;
    stats.total_ncols += src1_ncols;
    stats.total_convert_elems += convert_elems;
    stats.total_convert_ms += src0_convert_ms;

    const double avg_convert_ms = stats.calls > 0 ? stats.total_convert_ms / (double) stats.calls : 0.0;
    const double avg_gap = stats.repeated_calls > 0 ? (double) stats.total_gap / (double) stats.repeated_calls : 0.0;
    const int64_t min_gap = stats.repeated_calls > 0 ? stats.min_gap : -1;

    GGML_LOG_INFO(
        "GGML_TRACE_CUBLAS_Q3K_ROUTE: event_idx=%lld gap_since_prev=%lld dst=%s src0=%s src1=%s src0_tensor=%p src0_data=%p src0_dev_ptr=%p dev=%d cc=%d row_low=%lld row_high=%lld row_diff=%lld ne00=%lld ne10=%lld ncols=%lld convert_elems=%lld calls_for_key=%lld repeated_for_key=%lld min_gap_for_key=%lld avg_gap_for_key=%.2f max_gap_for_key=%lld total_ncols_for_key=%lld total_convert_elems_for_key=%lld src0_ms=%.3f src0_convert_ms=%.3f avg_convert_ms_for_key=%.3f\n",
        (long long) event_idx, (long long) gap_since_prev,
        dst->name, src0->name, src1->name,
        (const void *) src0, src0->data, (const void *) src0_dd_i, dev, cc,
        (long long) row_low, (long long) row_high, (long long) row_diff,
        (long long) ne00, (long long) ne10, (long long) src1_ncols,
        (long long) convert_elems, (long long) stats.calls, (long long) stats.repeated_calls,
        (long long) min_gap, avg_gap, (long long) stats.max_gap,
        (long long) stats.total_ncols, (long long) stats.total_convert_elems,
        src0_ms, src0_convert_ms, avg_convert_ms);
}

static void ggml_cuda_cublas_trace_src1_reuse(
        const cublas_src1_reuse_trace_config & config,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        const float * src1_ddf_i, const int dev, const int cc,
        const int64_t row_diff, const int64_t ne00, const int64_t ne10, const int64_t src1_ncols,
        const double src1_ms) {
    if (!config.enabled || src0->type != GGML_TYPE_Q3_K || src1_ncols < config.min_ncols || src1->type == GGML_TYPE_F16) {
        return;
    }

    const std::string key = ggml_cuda_cublas_src1_reuse_key(src1, src1_ddf_i, dev, ne10, src1_ncols);

    static std::mutex mutex;
    static int64_t global_event_idx = 0;
    static std::map<std::string, cublas_src1_reuse_trace_stats> stats_by_key;

    std::lock_guard<std::mutex> lock(mutex);
    const int64_t event_idx = global_event_idx++;
    cublas_src1_reuse_trace_stats & stats = stats_by_key[key];
    int64_t gap_since_prev = -1;
    if (stats.last_event_idx >= 0) {
        gap_since_prev = event_idx - stats.last_event_idx;
        stats.repeated_calls += 1;
        stats.repeated_src1_ms += src1_ms;
        if (gap_since_prev <= 2) {
            stats.tight_repeated_src1_ms += src1_ms;
        }
        stats.total_gap += gap_since_prev;
        stats.min_gap = std::min(stats.min_gap, gap_since_prev);
        stats.max_gap = std::max(stats.max_gap, gap_since_prev);
    }
    stats.last_event_idx = event_idx;
    stats.calls += 1;
    stats.total_src1_ms += src1_ms;

    const double avg_gap = stats.repeated_calls > 0 ? (double) stats.total_gap / (double) stats.repeated_calls : 0.0;
    const int64_t min_gap = stats.repeated_calls > 0 ? stats.min_gap : -1;

    GGML_LOG_INFO(
        "GGML_TRACE_CUBLAS_SRC1_REUSE: event_idx=%lld gap_since_prev=%lld dst=%s src0=%s src1=%s src1_tensor=%p src1_data=%p src1_dev_ptr=%p dev=%d cc=%d row_diff=%lld ne00=%lld ne10=%lld ncols=%lld calls_for_key=%lld repeated_for_key=%lld min_gap_for_key=%lld avg_gap_for_key=%.2f max_gap_for_key=%lld total_src1_ms_for_key=%.3f repeated_src1_ms_for_key=%.3f tight_repeated_src1_ms_for_key=%.3f src1_ms=%.3f\n",
        (long long) event_idx, (long long) gap_since_prev,
        dst->name, src0->name, src1->name,
        (const void *) src1, src1->data, (const void *) src1_ddf_i, dev, cc,
        (long long) row_diff, (long long) ne00, (long long) ne10, (long long) src1_ncols,
        (long long) stats.calls, (long long) stats.repeated_calls,
        (long long) min_gap, avg_gap, (long long) stats.max_gap,
        stats.total_src1_ms, stats.repeated_src1_ms, stats.tight_repeated_src1_ms, src1_ms);
}

static bool ggml_cuda_q3k_hot_direct_shape_match(
        const ggml_tensor * src0,
        const int64_t row_diff,
        const int64_t ne00,
        const int64_t src1_ncols,
        const q3k_hot_direct_route_config & config) {
    if (!config.enabled || src0->type != GGML_TYPE_Q3_K || src1_ncols < config.min_ncols) {
        return false;
    }

    const bool hot_row = row_diff == 17408 || row_diff == 10240 || row_diff == 6144 || row_diff == 5120;
    const bool hot_width = ne00 == 5120 || ne00 == 17408;
    const bool row_filter_ok = config.match_row_diff == 0 || row_diff == config.match_row_diff;
    const bool ne00_filter_ok = config.match_ne00 == 0 || ne00 == config.match_ne00;
    const bool ncols_filter_ok = config.match_ncols == 0 || src1_ncols == config.match_ncols;
    return hot_row && hot_width && row_filter_ok && ne00_filter_ok && ncols_filter_ok;
}

static void ggml_cuda_trace_q3k_hot_direct_route_match(
        const q3k_hot_direct_route_config & config,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        const int dev,
        const int cc,
        const int64_t row_low,
        const int64_t row_high,
        const int64_t ne00,
        const int64_t ne10,
        const int64_t src1_ncols) {
    const int64_t row_diff = row_high - row_low;
    if (!ggml_cuda_q3k_hot_direct_shape_match(src0, row_diff, ne00, src1_ncols, config)) {
        return;
    }

    GGML_LOG_INFO(
        "GGML_TRACE_Q3K_HOT_DIRECT_ROUTE_MATCH: dst=%s src0=%s src1=%s dev=%d cc=%d row_low=%lld row_high=%lld row_diff=%lld ne00=%lld ne10=%lld ncols=%lld\n",
        dst->name, src0->name, src1->name, dev, cc,
        (long long) row_low, (long long) row_high, (long long) row_diff,
        (long long) ne00, (long long) ne10, (long long) src1_ncols);
}

static double ggml_cuda_cublas_split_timing_sync_elapsed(
        cudaStream_t stream,
        const std::chrono::high_resolution_clock::time_point & start,
        bool & sync_applied,
        int & capture_active) {
    sync_applied = false;
    capture_active = 0;

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
        sync_applied = true;
    }

    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void ggml_cuda_q3k_cublas_pipeline_sync(
        ggml_backend_cuda_context & ctx,
        const char * src0_dd_i, const half * src1_ptr, float * dst_dd_i,
        const int id, const int64_t row_diff, const int64_t ne00, const int64_t ne10,
        const int64_t src1_ncols, const int64_t ldc, const bool use_q3k_padded_storage,
        const int64_t configured_chunk_rows, cudaStream_t main_stream) {
    constexpr int buffers = 2;

    const int64_t chunk_rows = std::max<int64_t>(1, std::min<int64_t>(configured_chunk_rows, row_diff));
    const size_t chunk_elems = (size_t) chunk_rows * (size_t) ne00;

    ggml_cuda_pool_alloc<half> chunk0(ctx.pool(id));
    ggml_cuda_pool_alloc<half> chunk1(ctx.pool(id));
    chunk0.alloc(chunk_elems);
    chunk1.alloc(chunk_elems);
    half * chunks[buffers] = { chunk0.get(), chunk1.get() };

    cudaStream_t dequant_streams[buffers] = {
        ctx.stream(id, 4),
        ctx.stream(id, 5),
    };
    cudaStream_t gemm_streams[buffers] = {
        ctx.stream(id, 6),
        ctx.stream(id, 7),
    };
    cudaEvent_t src1_ready_event = nullptr;
    cudaEvent_t ready_events[buffers] = { nullptr, nullptr };
    CUDA_CHECK(cudaEventCreateWithFlags(&src1_ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(src1_ready_event, main_stream));
    for (int i = 0; i < buffers; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ready_events[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamWaitEvent(gemm_streams[i], src1_ready_event, 0));
        CUBLAS_CHECK(cublasSetStream(ctx.q3k_pipeline_cublas_handle(id, i), gemm_streams[i]));
    }

    const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(GGML_TYPE_Q3_K);
    GGML_ASSERT(to_fp16_cuda != nullptr);
    GGML_ASSERT(ne00 % QK_K == 0);
    const size_t blocks_per_row = (size_t) ne00 / QK_K;
    const size_t q3_block_size = use_q3k_padded_storage ? sizeof(block_q3_K_padded) : ggml_type_size(GGML_TYPE_Q3_K);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int64_t row_start = 0, chunk = 0; row_start < row_diff; row_start += chunk_rows, ++chunk) {
        const int64_t rows = std::min<int64_t>(chunk_rows, row_diff - row_start);
        const int buf = chunk % buffers;
        if (chunk >= buffers) {
            CUDA_CHECK(cudaStreamSynchronize(gemm_streams[buf]));
        }

        const char * src0_chunk = src0_dd_i + (size_t) row_start * blocks_per_row * q3_block_size;
        const int64_t ne = rows * ne00;
        if (use_q3k_padded_storage) {
            ggml_cuda_q3_K_padded_to_fp16(src0_chunk, chunks[buf], ne, dequant_streams[buf]);
        } else {
            to_fp16_cuda(src0_chunk, chunks[buf], ne, dequant_streams[buf]);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ready_events[buf], dequant_streams[buf]));
        CUDA_CHECK(cudaStreamWaitEvent(gemm_streams[buf], ready_events[buf], 0));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.q3k_pipeline_cublas_handle(id, buf), CUBLAS_OP_T, CUBLAS_OP_N,
                    rows, src1_ncols, ne10,
                    &alpha, chunks[buf], CUDA_R_16F, ne00,
                            src1_ptr,    CUDA_R_16F, ne10,
                    &beta,  dst_dd_i + row_start, CUDA_R_32F, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    for (int i = 0; i < buffers; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(gemm_streams[i]));
        CUDA_CHECK(cudaEventDestroy(ready_events[i]));
    }
    CUDA_CHECK(cudaEventDestroy(src1_ready_event));
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int cc = ggml_cuda_info().devices[id].cc;
    const auto & split_timing_config = ggml_cuda_cublas_get_split_timing_config();
    const auto & q3k_route_trace_config = ggml_cuda_cublas_get_q3k_route_trace_config();
    const auto & src1_reuse_trace_config = ggml_cuda_cublas_get_src1_reuse_trace_config();
    const auto & q3k_hot_direct_route_config = ggml_cuda_get_q3k_hot_direct_route_config();
    const bool trace_split_timing = split_timing_config.enabled && src1_ncols >= split_timing_config.min_ncols;
    int trace_capture_active = 0;
    auto trace_stage_start = trace_split_timing ?
        std::chrono::high_resolution_clock::now() :
        std::chrono::high_resolution_clock::time_point{};
    auto trace_stage_ms = [&](const bool stage_enabled) -> double {
        if (!trace_split_timing || !stage_enabled) {
            return 0.0;
        }

        bool sync_applied = false;
        int capture_active = 0;
        const double elapsed_ms = ggml_cuda_cublas_split_timing_sync_elapsed(stream, trace_stage_start, sync_applied, capture_active);
        GGML_UNUSED(sync_applied);
        trace_capture_active |= capture_active;
        trace_stage_start = std::chrono::high_resolution_clock::now();
        return elapsed_ms;
    };

    if (trace_split_timing && split_timing_config.pre_sync) {
        bool pre_sync_applied = false;
        int pre_sync_capture_active = 0;
        const auto pre_sync_start = std::chrono::high_resolution_clock::now();
        (void) ggml_cuda_cublas_split_timing_sync_elapsed(stream, pre_sync_start, pre_sync_applied, pre_sync_capture_active);
        GGML_UNUSED(pre_sync_applied);
        trace_capture_active |= pre_sync_capture_active;
        trace_stage_start = std::chrono::high_resolution_clock::now();
    }

    const bool supports_bf16 = GGML_CUDA_CC_IS_NVIDIA(cc) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);

    // G09 (2026-08-30): RDNA4 has no f32 MFMA, so the pure f32 GEMM path
    // (GDN projections and rotation K/V in this lane) runs an order of
    // magnitude slower than the f16-input route (measured 0.59 ms vs
    // 0.058 ms at 48x1024x5120 on gfx1201). Send f32 GEMMs through the
    // f16 conversion + f32 accumulate path, which is byte-identical on the
    // locked lane and gives prompt +5.35% (r3). Default ON for RDNA4;
    // GGML_ROCM_F32_GEMM_F16=0 restores the pure f32 path.
    const bool f32_gemm_f16 = [&]() {
        if (src0->type != GGML_TYPE_F32 || !GGML_CUDA_CC_IS_RDNA4(cc)) {
            return false;
        }
        if (const char * env = std::getenv("GGML_ROCM_F32_GEMM_F16")) {
            return std::atoi(env) != 0;
        }
        return true;
    }();

    const bool use_fp16 =
        src0->type != GGML_TYPE_NVFP4 &&
        (src0->type == GGML_TYPE_F16 || (src0->type == GGML_TYPE_F32 && f32_gemm_f16) || ggml_is_quantized(src0->type)) &&
        ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] &&
        dst->op_params[0] == GGML_PREC_DEFAULT;

    if (supports_bf16 && src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
            GGML_ASSERT(to_bf16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1_ddf_i, src1_as_bf16.get(), ne, stream);
        }
        const double src1_ms = trace_stage_ms(src1->type != GGML_TYPE_BF16);
        const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16 * src0_ptr = (const nv_bfloat16 *)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32  = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                                 src1_ptr,       CUDA_R_16BF, ne10,
                    &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        const double gemm_ms = trace_stage_ms(true);

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff*src1_ncols, stream);
        const double dst_ms = trace_stage_ms(true);
        if (trace_split_timing) {
            GGML_LOG_INFO(
                "GGML_TRACE_CUBLAS_SPLIT_TIMING: path=bf16 compute=32f dst=%s src0_type=%s src1_type=%s dev=%d cc=%d row_diff=%lld ne00=%lld ne10=%lld ncols=%lld ldc=%lld src0_ms=0.000 src1_ms=%.3f gemm_ms=%.3f dst_ms=%.3f sum_ms=%.3f capture=%d\n",
                dst->name, ggml_type_name(src0->type), ggml_type_name(src1->type), id, cc,
                (long long) row_diff, (long long) ne00, (long long) ne10, (long long) src1_ncols, (long long) ldc,
                src1_ms, gemm_ms, dst_ms, src1_ms + gemm_ms + dst_ms, trace_capture_active);
        }
    } else if (fast_fp16_hardware_available(cc) && use_fp16) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        ggml_cuda_pool_alloc<block_q3_K_padded> src0_as_q3k_padded(ctx.pool(id));
        const bool use_q3k_padded_storage =
            src0->type == GGML_TYPE_Q3_K && ggml_cuda_q3k_padded_storage_tensor_physical(src0);
        const bool use_q3k_padded_dequant_probe =
            src0->type == GGML_TYPE_Q3_K && !use_q3k_padded_storage && ggml_cuda_q3k_padded_dequant_probe_enabled();
        const bool use_q3k_padded_dequant = use_q3k_padded_storage || use_q3k_padded_dequant_probe;
        const auto & force_compute_type = ggml_cuda_cublas_get_force_compute_type();
        const auto & q3k_cublas_pipeline_config = ggml_cuda_get_q3k_cublas_pipeline_config();
        const auto & src1_f16_reuse_config = ggml_cuda_get_cublas_src1_f16_reuse_config();
        const uint64_t src1_f16_reuse_epoch = ++ctx.cublas_src1_f16_reuse_epoch[id];
        const bool use_compute32 = !force_compute_type.fp16 && (GGML_CUDA_CC_IS_CDNA(cc)
                                        || GGML_CUDA_CC_IS_RDNA4(cc)
                                        || cc == GGML_CUDA_CC_VOLTA
                                        || force_compute_type.fp32);
        bool q3k_cublas_pipeline_graph_blocked = false;
#ifdef USE_CUDA_GRAPH
        if (q3k_cublas_pipeline_config.enabled && !q3k_cublas_pipeline_config.allow_graphs) {
#ifdef GGML_USE_HIP
            hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
            CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
            q3k_cublas_pipeline_graph_blocked = ctx.any_cuda_graph_enabled() || capture_status != hipStreamCaptureStatusNone;
#else
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
            q3k_cublas_pipeline_graph_blocked = ctx.any_cuda_graph_enabled() || capture_status != cudaStreamCaptureStatusNone;
#endif // GGML_USE_HIP
        }
#endif // USE_CUDA_GRAPH
        const bool use_q3k_cublas_pipeline =
            q3k_cublas_pipeline_config.enabled &&
            !q3k_cublas_pipeline_graph_blocked &&
            ctx.curr_stream_no == 0 &&
            use_compute32 &&
            src0->type == GGML_TYPE_Q3_K &&
            src0->op == GGML_OP_NONE &&
            !use_q3k_padded_dequant_probe &&
            src1_ncols >= q3k_cublas_pipeline_config.min_ncols &&
            row_diff >= q3k_cublas_pipeline_config.min_rows &&
            ne00 % QK_K == 0;
        double src0_alloc_ms = 0.0;
        double src0_convert_ms = 0.0;
        if (src0->type != GGML_TYPE_F16 && !use_q3k_cublas_pipeline) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            GGML_ASSERT(!use_q3k_padded_dequant || ne % QK_K == 0);
            if (split_timing_config.detail) {
                const auto src0_alloc_start = std::chrono::high_resolution_clock::now();
                src0_as_f16.alloc(ne);
                if (use_q3k_padded_dequant_probe) {
                    src0_as_q3k_padded.alloc(ne / QK_K);
                }
                const auto src0_alloc_end = std::chrono::high_resolution_clock::now();
                src0_alloc_ms = std::chrono::duration<double, std::milli>(src0_alloc_end - src0_alloc_start).count();
                trace_stage_start = std::chrono::high_resolution_clock::now();
            } else {
                src0_as_f16.alloc(ne);
                if (use_q3k_padded_dequant_probe) {
                    src0_as_q3k_padded.alloc(ne / QK_K);
                }
            }
            if (use_q3k_padded_dequant_probe) {
                ggml_cuda_q3_K_pack_to_padded(src0_dd_i, src0_as_q3k_padded.get(), ne, stream);
                ggml_cuda_q3_K_padded_to_fp16(src0_as_q3k_padded.get(), src0_as_f16.get(), ne, stream);
            } else if (use_q3k_padded_storage) {
                ggml_cuda_q3_K_padded_to_fp16(src0_dd_i, src0_as_f16.get(), ne, stream);
            } else {
                to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
            }
            if (split_timing_config.detail) {
                src0_convert_ms = trace_stage_ms(true);
            }
        }
        const double src0_ms = use_q3k_cublas_pipeline ? 0.0 :
            (split_timing_config.detail ? src0_alloc_ms + src0_convert_ms : trace_stage_ms(src0->type != GGML_TYPE_F16));
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();
        ggml_cuda_cublas_trace_q3k_route(
            q3k_route_trace_config, src0, src1, dst, src0_dd_i, id, cc,
            row_low, row_high, ne00, ne10, src1_ncols, src0_ms, src0_convert_ms);
        ggml_cuda_trace_q3k_hot_direct_route_match(
            q3k_hot_direct_route_config, src0, src1, dst, id, cc,
            row_low, row_high, ne00, ne10, src1_ncols);

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        double src1_alloc_ms = 0.0;
        double src1_convert_ms = 0.0;
        const bool can_reuse_src1_f16 =
            src1_f16_reuse_config.enabled &&
            ctx.curr_stream_no == 0 &&
            src1->type != GGML_TYPE_F16 &&
            src1_ncols >= src1_f16_reuse_config.min_ncols;
        const bool reuse_src1_f16 =
            can_reuse_src1_f16 &&
            ctx.cublas_src1_f16_reuse_valid[id] &&
            ctx.cublas_src1_f16_reuse_key_epoch[id] + 1 == src1_f16_reuse_epoch &&
            ctx.cublas_src1_f16_reuse_tensor[id] == src1 &&
            ctx.cublas_src1_f16_reuse_tensor_data[id] == src1->data &&
            ctx.cublas_src1_f16_reuse_src_ptr[id] == src1_ddf_i &&
            ctx.cublas_src1_f16_reuse_ne10[id] == ne10 &&
            ctx.cublas_src1_f16_reuse_ncols[id] == src1_ncols &&
            ctx.cublas_src1_f16_reuse_stream_no[id] == ctx.curr_stream_no;
        if (src1->type != GGML_TYPE_F16) {
            size_t ne = src1_ncols*ne10;
            if (reuse_src1_f16) {
                ctx.cublas_src1_f16_reuse_key_epoch[id] = src1_f16_reuse_epoch;
                if (trace_split_timing) {
                    trace_stage_start = std::chrono::high_resolution_clock::now();
                }
            } else {
                const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
                GGML_ASSERT(to_fp16_cuda != nullptr);
                half * src1_f16_dst = nullptr;
                if (split_timing_config.detail) {
                    const auto src1_alloc_start = std::chrono::high_resolution_clock::now();
                    src1_f16_dst = can_reuse_src1_f16 ? ctx.cublas_src1_f16_reuse_buffer(id, ne) : src1_as_f16.alloc(ne);
                    const auto src1_alloc_end = std::chrono::high_resolution_clock::now();
                    src1_alloc_ms = std::chrono::duration<double, std::milli>(src1_alloc_end - src1_alloc_start).count();
                    trace_stage_start = std::chrono::high_resolution_clock::now();
                } else {
                    src1_f16_dst = can_reuse_src1_f16 ? ctx.cublas_src1_f16_reuse_buffer(id, ne) : src1_as_f16.alloc(ne);
                }
                to_fp16_cuda(src1_ddf_i, src1_f16_dst, ne, stream);
                if (can_reuse_src1_f16) {
                    ctx.cublas_src1_f16_reuse_tensor[id] = src1;
                    ctx.cublas_src1_f16_reuse_tensor_data[id] = src1->data;
                    ctx.cublas_src1_f16_reuse_src_ptr[id] = src1_ddf_i;
                    ctx.cublas_src1_f16_reuse_ne10[id] = ne10;
                    ctx.cublas_src1_f16_reuse_ncols[id] = src1_ncols;
                    ctx.cublas_src1_f16_reuse_stream_no[id] = ctx.curr_stream_no;
                    ctx.cublas_src1_f16_reuse_key_epoch[id] = src1_f16_reuse_epoch;
                    ctx.cublas_src1_f16_reuse_valid[id] = true;
                }
                if (split_timing_config.detail) {
                    src1_convert_ms = trace_stage_ms(true);
                }
            }
        }
        const double src1_ms = split_timing_config.detail ? src1_alloc_ms + src1_convert_ms : trace_stage_ms(src1->type != GGML_TYPE_F16 && !reuse_src1_f16);
        ggml_cuda_cublas_trace_src1_reuse(
            src1_reuse_trace_config, src0, src1, dst, src1_ddf_i, id, cc,
            row_diff, ne00, ne10, src1_ncols, src1_ms);
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i :
            (can_reuse_src1_f16 ? ctx.cublas_src1_f16_reuse_buffers[id] : src1_as_f16.get());

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

        if (use_q3k_cublas_pipeline)
        {
            ggml_cuda_q3k_cublas_pipeline_sync(
                ctx, src0_dd_i, src1_ptr, dst_dd_i, id, row_diff, ne00, ne10,
                src1_ncols, ldc, use_q3k_padded_storage, q3k_cublas_pipeline_config.chunk_rows, stream);
            const double pipeline_ms = trace_stage_ms(true);
            if (trace_split_timing) {
                GGML_LOG_INFO(
                    "GGML_TRACE_CUBLAS_SPLIT_TIMING: path=q3k_pipeline_sync compute=32f dst=%s src0_type=%s src1_type=%s dev=%d cc=%d row_diff=%lld ne00=%lld ne10=%lld ncols=%lld ldc=%lld src0_ms=0.000 src1_ms=%.3f gemm_ms=%.3f dst_ms=0.000 sum_ms=%.3f pipe_chunk=%lld capture=%d\n",
                    dst->name, ggml_type_name(src0->type), ggml_type_name(src1->type), id, cc,
                    (long long) row_diff, (long long) ne00, (long long) ne10, (long long) src1_ncols, (long long) ldc,
                    src1_ms, pipeline_ms, src1_ms + pipeline_ms,
                    (long long) q3k_cublas_pipeline_config.chunk_rows, trace_capture_active);
            }
        }
        else if (use_compute32)
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha, src0_ptr,  CUDA_R_16F, ne00,
                                src1_ptr,  CUDA_R_16F, ne10,
                        &beta,   dst_dd_i, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            const double gemm_ms = trace_stage_ms(true);
            if (trace_split_timing) {
                if (split_timing_config.detail) {
                    GGML_LOG_INFO(
                        "GGML_TRACE_CUBLAS_SPLIT_TIMING: path=fp16 compute=32f dst=%s src0_type=%s src1_type=%s dev=%d cc=%d row_diff=%lld ne00=%lld ne10=%lld ncols=%lld ldc=%lld src0_ms=%.3f src1_ms=%.3f gemm_ms=%.3f dst_ms=0.000 sum_ms=%.3f src0_alloc_ms=%.3f src0_convert_ms=%.3f src1_alloc_ms=%.3f src1_convert_ms=%.3f capture=%d\n",
                        dst->name, ggml_type_name(src0->type), ggml_type_name(src1->type), id, cc,
                        (long long) row_diff, (long long) ne00, (long long) ne10, (long long) src1_ncols, (long long) ldc,
                        src0_ms, src1_ms, gemm_ms, src0_ms + src1_ms + gemm_ms,
                        src0_alloc_ms, src0_convert_ms, src1_alloc_ms, src1_convert_ms, trace_capture_active);
                } else {
                    GGML_LOG_INFO(
                        "GGML_TRACE_CUBLAS_SPLIT_TIMING: path=fp16 compute=32f dst=%s src0_type=%s src1_type=%s dev=%d cc=%d row_diff=%lld ne00=%lld ne10=%lld ncols=%lld ldc=%lld src0_ms=%.3f src1_ms=%.3f gemm_ms=%.3f dst_ms=0.000 sum_ms=%.3f capture=%d\n",
                        dst->name, ggml_type_name(src0->type), ggml_type_name(src1->type), id, cc,
                        (long long) row_diff, (long long) ne00, (long long) ne10, (long long) src1_ncols, (long long) ldc,
                        src0_ms, src1_ms, gemm_ms, src0_ms + src1_ms + gemm_ms, trace_capture_active);
                }
            }
        } else {
            ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

            const half alpha_f16 = 1.0f;
            const half beta_f16 = 0.0f;

            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f16, src0_ptr,      CUDA_R_16F, ne00,
                                    src1_ptr,      CUDA_R_16F, ne10,
                        &beta_f16,  dst_f16.get(), CUDA_R_16F, ldc,
                        CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            const double gemm_ms = trace_stage_ms(true);

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
            to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
            const double dst_ms = trace_stage_ms(true);
            if (trace_split_timing) {
                if (split_timing_config.detail) {
                    GGML_LOG_INFO(
                        "GGML_TRACE_CUBLAS_SPLIT_TIMING: path=fp16 compute=16f dst=%s src0_type=%s src1_type=%s dev=%d cc=%d row_diff=%lld ne00=%lld ne10=%lld ncols=%lld ldc=%lld src0_ms=%.3f src1_ms=%.3f gemm_ms=%.3f dst_ms=%.3f sum_ms=%.3f src0_alloc_ms=%.3f src0_convert_ms=%.3f src1_alloc_ms=%.3f src1_convert_ms=%.3f capture=%d\n",
                        dst->name, ggml_type_name(src0->type), ggml_type_name(src1->type), id, cc,
                        (long long) row_diff, (long long) ne00, (long long) ne10, (long long) src1_ncols, (long long) ldc,
                        src0_ms, src1_ms, gemm_ms, dst_ms, src0_ms + src1_ms + gemm_ms + dst_ms,
                        src0_alloc_ms, src0_convert_ms, src1_alloc_ms, src1_convert_ms, trace_capture_active);
                } else {
                    GGML_LOG_INFO(
                        "GGML_TRACE_CUBLAS_SPLIT_TIMING: path=fp16 compute=16f dst=%s src0_type=%s src1_type=%s dev=%d cc=%d row_diff=%lld ne00=%lld ne10=%lld ncols=%lld ldc=%lld src0_ms=%.3f src1_ms=%.3f gemm_ms=%.3f dst_ms=%.3f sum_ms=%.3f capture=%d\n",
                        dst->name, ggml_type_name(src0->type), ggml_type_name(src1->type), id, cc,
                        (long long) row_diff, (long long) ne00, (long long) ne10, (long long) src1_ncols, (long long) ldc,
                        src0_ms, src1_ms, gemm_ms, dst_ms, src0_ms + src1_ms + gemm_ms + dst_ms, trace_capture_active);
                }
            }
        }
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        const double src0_ms = trace_stage_ms(src0->type != GGML_TYPE_F32);
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        const double src1_ms = trace_stage_ms(src1->type != GGML_TYPE_F32);

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
        const double gemm_ms = trace_stage_ms(true);
        if (trace_split_timing) {
            GGML_LOG_INFO(
                "GGML_TRACE_CUBLAS_SPLIT_TIMING: path=f32 compute=32f dst=%s src0_type=%s src1_type=%s dev=%d cc=%d row_diff=%lld ne00=%lld ne10=%lld ncols=%lld ldc=%lld src0_ms=%.3f src1_ms=%.3f gemm_ms=%.3f dst_ms=0.000 sum_ms=%.3f capture=%d\n",
                dst->name, ggml_type_name(src0->type), ggml_type_name(src1->type), id, cc,
                (long long) row_diff, (long long) ne00, (long long) ne10, (long long) src1_ncols, (long long) ldc,
                src0_ms, src1_ms, gemm_ms, src0_ms + src1_ms + gemm_ms, trace_capture_active);
        }
    }

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

    if (dstDevice != srcDevice && !ggml_cuda_peer_copy_enabled()) {
        const size_t host_size = width * height;
        void * host = malloc(host_size);
        if (host == nullptr) {
            return cudaErrorMemoryAllocation;
        }

        ggml_cuda_set_device(srcDevice);
        cudaError_t err = cudaDeviceSynchronize();
        if (err == cudaSuccess) {
            err = cudaMemcpy2DAsync(host, width, src, spitch, width, height, cudaMemcpyDeviceToHost, cudaStreamPerThread);
        }
        if (err == cudaSuccess) {
            err = cudaStreamSynchronize(cudaStreamPerThread);
        }
        if (err == cudaSuccess) {
            ggml_cuda_set_device(dstDevice);
            err = cudaMemcpy2DAsync(dst, dpitch, host, width, width, height, cudaMemcpyHostToDevice, cudaStreamPerThread);
        }
        if (err == cudaSuccess) {
            err = cudaStreamSynchronize(cudaStreamPerThread);
        }

        free(host);
        return err;
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

static cudaError_t ggml_cuda_MemcpyPeerAsync(
    void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream) {

    if (dstDevice == srcDevice || ggml_cuda_peer_copy_enabled()) {
        return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
    }

    void * host = malloc(count);
    if (host == nullptr) {
        return cudaErrorMemoryAllocation;
    }

    ggml_cuda_set_device(srcDevice);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        err = cudaMemcpy(host, src, count, cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess) {
        ggml_cuda_set_device(dstDevice);
        err = cudaMemcpyAsync(dst, host, count, cudaMemcpyHostToDevice, cudaStreamPerThread);
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(cudaStreamPerThread);
    }

    free(host);
    return err;
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int64_t i02_divisor = ne12 / ne02;
    const int64_t i03_divisor = ne13 / ne03;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));
    GGML_ASSERT(!(split && ne03 < ne13));

    ggml_tensor_extra_gpu * src0_extra = split ? (ggml_tensor_extra_gpu *) src0->extra : nullptr;


    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(tensor_split);

            if (id != 0) {
                dev[id].row_low  = ne01*tensor_split[id];
                if (dev[id].row_low < ne01) {
                    dev[id].row_low -= dev[id].row_low % rounding;
                }
            }

            if (id != ggml_backend_cuda_get_device_count() - 1) {
                dev[id].row_high  = ne01*tensor_split[id + 1];
                if (dev[id].row_high < ne01) {
                    dev[id].row_high -= dev[id].row_high % rounding;
                }
            }
        }
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char *) src0_extra->data_device[id] : (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer.
            // This buffer needs to be cleared entirely because multiple regions will function as padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            const size_t nbytes_data    = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low)*ne00);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && src1_is_contiguous) {
                quantize_src1(
                    dev[id].src1_ddf, nullptr, dev[id].src1_ddq, src0->type, ne10,
                    nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
                    src1_padded_col_size, ne11, ne12, ne13, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low)*ne1 : ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaEventRecord(src0_extra->events[ctx.device][0], ctx.stream()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            // wait for main GPU data if necessary
            if (split && (id != ctx.device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[ctx.device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                const size_t nbytes_src0_matrix = ne01*ne00*src0_ts / src0_bs;
                char  *  src0_dd_i =  dev[id].src0_dd + ((i03/i03_divisor)*ne02 + (i02/i02_divisor)) * nbytes_src0_matrix;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(ggml_cuda_MemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(ggml_cuda_MemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous) {
                    quantize_src1(
                        src1_ddf_i, nullptr, src1_ddq_i, src0->type, ne10, ne10, ne11*ne10, ne12*ne11*ne10,
                        src1_padded_col_size, src1_ncols, 1, 1, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i03 % i03_divisor == 0 && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                        src0_dd_i, src0, i03/i03_divisor, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[id].row_low;
                        CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(
                            dhf_dst_i, ctx.device, ne0*sizeof(float), dst_dd_i, id, row_diff*sizeof(float), row_diff*sizeof(float), src1_ncols, stream));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_backend_cuda_get_device_count() > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

        ggml_cuda_set_device(ctx.device);
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (dev[id].row_low == dev[id].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), src0_extra->events[id][is], 0));
            }
        }
    }
}

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type src0_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<src0_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft));
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float * dst_ddf = (float *) dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == ts_src1);
    int64_t s11 = nb11 / ts_src1;
    int64_t s12 = nb12 / ts_src1;
    int64_t s13 = nb13 / ts_src1;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // Handle src0
    src0_ptr = (const cuda_t *) src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_alloc.alloc(ne_src1);

        const auto convert_func = traits::get_nc_converter(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        src1_ptr = src1_alloc.get();
        s11 = ne10;
        s12 = ne11*s11;
        s13 = ne12*s12;

        is_src1_cont_2 = true;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const auto & force_compute_type = ggml_cuda_cublas_get_force_compute_type();

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    static constexpr bool is_src0_type_f16 = src0_type == GGML_TYPE_F16;

    // bf16 and fp32 are already being computed in fp32 (ensure it using static_assert),
    // so checking necessity of forced fp32 only for fp16 src0_type
    static_assert(is_src0_type_f16 || traits::compute_type == CUBLAS_COMPUTE_32F);

    const bool need_compute_32f = is_src0_type_f16 && !force_compute_type.fp16 && (GGML_CUDA_CC_IS_CDNA(cc)
                                                                                  || GGML_CUDA_CC_IS_RDNA4(cc)
                                                                                  || cc == GGML_CUDA_CC_VOLTA
                                                                                  || force_compute_type.fp32);

    if (dst->op_params[0] == GGML_PREC_DEFAULT && !need_compute_32f) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_t = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    } else {
        dst_t = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
        const int64_t smb = ne12 == 1 ? s13       : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11,       smb,     // strideB
                beta,     dst_t, cu_data_type,   ne0,       ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        size_t src1_stride_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        dim3 block_dims(threads_x, threads_y);

        dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                (src1->type == src0_type) ? nb12 : s12*src1_stride_size,
                (src1->type == src0_type) ? nb13 : s13*src1_stride_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("Unsupported type");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (glu->src[0] != ffn_gate_bias || glu->src[1] != ffn_up_bias) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up || ffn_up_bias->src[1] == ffn_up;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate || ffn_gate_bias->src[1] == ffn_gate;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up || ffn_gate_bias->src[0] != ffn_gate) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    } else {
        if (glu->src[0] != ffn_gate && glu->src[1] != ffn_up) {
            return false;
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (ffn_up->src[2] && (ffn_up->src[2] != ffn_gate->src[2])) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 3> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    const bool split = ggml_backend_buft_is_cuda_split(ffn_up->src[0]->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(ffn_gate->src[0]->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(src1->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(src1->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    bool use_mul_mat_vec_f = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_f     = !ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    // D094: accept pre-quantized Q8_1 src1 (experiment-2)
    const bool src1_f32_or_q8_1 = src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_Q8_1;
    bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1_f32_or_q8_1 && dst->type == GGML_TYPE_F32;

    bool any_gpus_with_slow_fp16 = false;

    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            const int cc            = ggml_cuda_info().devices[id].cc;
            const int warp_size     = ggml_cuda_info().devices[id].warp_size;
            use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
            use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
            use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
            use_mul_mat_vec_q       = use_mul_mat_vec_q         && ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1]);
            any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
        }
    } else {
        const int cc            = ggml_cuda_info().devices[ctx.device].cc;
        const int warp_size     = ggml_cuda_info().devices[ctx.device].warp_size;
        use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
        use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
        use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
        use_mul_mat_vec_q       = use_mul_mat_vec_q         && ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1]);
        any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
    }

    const auto & q3k_hot_direct_route_config = ggml_cuda_get_q3k_hot_direct_route_config();
    const bool force_hot_q3k_direct_route = !split &&
        ggml_cuda_q3k_hot_direct_shape_match(src0, src0->ne[1], src0->ne[0], src1->ne[1], q3k_hot_direct_route_config);

    if (force_hot_q3k_direct_route) {
        use_mul_mat_q = true;
    }

    if (src0->type == GGML_TYPE_Q3_K && ggml_cuda_q3k_padded_dequant_probe_enabled()) {
        use_mul_mat_vec_q = false;
        use_mul_mat_q = false;
    }

    if (src0->type == GGML_TYPE_Q3_K && ggml_cuda_q3k_padded_storage_tensor_physical(src0) &&
            !ggml_cuda_q3k_padded_storage_mmq_enabled()) {
        use_mul_mat_q = false;
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    //TODO update for generic tensor parallelism
    const int cc                 = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_batched_cublas_f16  = src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
    bool use_batched_cublas_bf16 = src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
    bool use_batched_cublas_f32  = src0->type == GGML_TYPE_F32;

    const bool trace_mul_mat_route = std::getenv("GGML_TRACE_CUDA_MUL_MAT_ROUTE") != nullptr;
    const auto log_mul_mat_route = [&](const char * route) {
        if (!trace_mul_mat_route) {
            return;
        }

        GGML_LOG_INFO(
            "GGML_TRACE_CUDA_MUL_MAT_ROUTE: route=%s dst=%s src0=%s src1=%s src0_type=%s src1_type=%s dst_type=%s split=%d bad_padding=%d use_vec_f=%d use_mat_f=%d use_vec_q=%d use_mat_q=%d use_batched_cublas_f16=%d use_batched_cublas_bf16=%d use_batched_cublas_f32=%d src0_ne=(%lld,%lld,%lld,%lld) src1_ne=(%lld,%lld,%lld,%lld) dst_ne=(%lld,%lld,%lld,%lld) src0_nb=(%zu,%zu,%zu,%zu) src1_nb=(%zu,%zu,%zu,%zu) dst_nb=(%zu,%zu,%zu,%zu) src0_mod128=%llu src1_mod128=%llu dst_mod128=%llu\n",
            route,
            dst->name,
            src0->name,
            src1->name,
            ggml_type_name(src0->type),
            ggml_type_name(src1->type),
            ggml_type_name(dst->type),
            split ? 1 : 0,
            bad_padding_clear ? 1 : 0,
            use_mul_mat_vec_f ? 1 : 0,
            use_mul_mat_f ? 1 : 0,
            use_mul_mat_vec_q ? 1 : 0,
            use_mul_mat_q ? 1 : 0,
            use_batched_cublas_f16 ? 1 : 0,
            use_batched_cublas_bf16 ? 1 : 0,
            use_batched_cublas_f32 ? 1 : 0,
            (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
            (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
            (long long) dst->ne[0],  (long long) dst->ne[1],  (long long) dst->ne[2],  (long long) dst->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3],
            (unsigned long long) ((uintptr_t) src0->data & 127),
            (unsigned long long) ((uintptr_t) src1->data & 127),
            (unsigned long long) ((uintptr_t) dst->data  & 127));
    };

    if (!split && use_mul_mat_vec_f) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        log_mul_mat_route("mul_mat_vec_f_direct");
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_f) {
        log_mul_mat_route("mul_mat_f_direct");
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_vec_q) {
        log_mul_mat_route("mul_mat_vec_q_direct");
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_q) {
        log_mul_mat_route(force_hot_q3k_direct_route ? "mul_mat_q_direct_hotshape" : "mul_mat_q_direct");
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && (use_batched_cublas_f16 || use_batched_cublas_bf16 || use_batched_cublas_f32)
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        log_mul_mat_route("batched_cublas");
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_mul_mat_vec_f) {
        log_mul_mat_route("mul_mat_vec_f_backend");
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_f, nullptr);
    } else if (use_mul_mat_vec_q) {
        log_mul_mat_route("mul_mat_vec_q_backend");
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        log_mul_mat_route("mul_mat_q_backend");
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        log_mul_mat_route("cublas_backend");
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft) && "mul_mat_id does not support split buffers");

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool trace_mul_mat_route = std::getenv("GGML_TRACE_CUDA_MUL_MAT_ROUTE") != nullptr
        || std::getenv("GGML_TRACE_MUL_MAT_ID_ROUTE") != nullptr;
    const auto log_mul_mat_id_route = [&](const char * route) {
        if (!trace_mul_mat_route) {
            return;
        }

        GGML_LOG_INFO(
            "GGML_TRACE_CUDA_MUL_MAT_ID_ROUTE: route=%s dst=%s src0=%s src1=%s src0_type=%s src1_type=%s dst_type=%s cc=%d ne2=%lld ne02=%lld ne12=%lld ids_ne0=%lld src0_ne=(%lld,%lld,%lld,%lld) src1_ne=(%lld,%lld,%lld,%lld) dst_ne=(%lld,%lld,%lld,%lld)\n",
            route,
            dst->name,
            src0->name,
            src1->name,
            ggml_type_name(src0->type),
            ggml_type_name(src1->type),
            ggml_type_name(dst->type),
            cc,
            (long long) ne2,
            (long long) ne02,
            (long long) ne12,
            (long long) ids->ne[0],
            (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
            (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
            (long long) dst->ne[0],  (long long) dst->ne[1],  (long long) dst->ne[2],  (long long) dst->ne[3]);
    };

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    log_mul_mat_id_route("mul_mat_vec_q_direct");
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    log_mul_mat_id_route("mul_mat_vec_f_direct");
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            log_mul_mat_id_route("mul_mat_q_direct");
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            log_mul_mat_id_route("mul_mat_f_direct");
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    // TODO: add asserts to verify this. should work with CUDA, HIP, etc.
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    log_mul_mat_id_route("fallback_sorted_rows");

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

static double ggml_cuda_trace_stream_sync(cudaStream_t stream, bool & sync_applied, int & capture_active) {
    sync_applied = false;
    capture_active = 0;

#ifdef GGML_USE_HIP
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
    capture_active = capture_status != hipStreamCaptureStatusNone;
#else
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    capture_active = capture_status != cudaStreamCaptureStatusNone;
#endif

    if (capture_active) {
        return 0.0;
    }

    const auto sync_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto sync_end = std::chrono::high_resolution_clock::now();
    sync_applied = true;
    return std::chrono::duration<double, std::milli>(sync_end - sync_start).count();
}

static unsigned long long ggml_cuda_trace_tensor_ptr_mod(const ggml_tensor * tensor, uintptr_t mod) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return 0;
    }

    return (unsigned long long) ((uintptr_t) tensor->data & (mod - 1));
}

static const char * ggml_cuda_trace_tensor_type_name(const ggml_tensor * tensor) {
    return tensor == nullptr ? "none" : ggml_type_name(tensor->type);
}

static const char * ggml_cuda_trace_tensor_buffer_name(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "none";
    }

    const ggml_tensor * owner = tensor->view_src != nullptr ? tensor->view_src : tensor;
    return owner->buffer == nullptr ? "none" : ggml_backend_buffer_name(owner->buffer);
}

static long long ggml_cuda_trace_tensor_buffer_offset(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return -1;
    }

    const ggml_tensor * owner = tensor->view_src != nullptr ? tensor->view_src : tensor;
    if (owner->buffer == nullptr) {
        return -1;
    }

    const char * base = (const char *) ggml_backend_buffer_get_base(owner->buffer);
    const char * data = (const char *) tensor->data;
    return (long long) (data - base);
}

static unsigned long long ggml_cuda_trace_tensor_nbytes(const ggml_tensor * tensor) {
    return tensor == nullptr ? 0 : (unsigned long long) ggml_nbytes(tensor);
}
