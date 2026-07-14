ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
        for (int j = 0; j < 2; ++j) {
            if (q3k_pipeline_cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(q3k_pipeline_cublas_handles[i][j]));
            }
        }
        if (cublas_src1_f16_reuse_buffers[i] != nullptr) {
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaFree(cublas_src1_f16_reuse_buffers[i]));
        }
        if (!cublas_src1_f16_reuse_retired_buffers[i].empty()) {
            ggml_cuda_set_device(i);
            for (half * ptr : cublas_src1_f16_reuse_retired_buffers[i]) {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
    }
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft);
static size_t ggml_cuda_q3k_padded_storage_alloc_size(const ggml_tensor * tensor);

static bool ggml_cuda_q3k_padded_dequant_probe_enabled() {
    static bool result = [] {
        const bool enabled = getenv("GGML_CUDA_Q3K_PADDED_DEQUANT_PROBE") != nullptr;
        if (enabled) {
            GGML_LOG_INFO("Detected GGML_CUDA_Q3K_PADDED_DEQUANT_PROBE=1\n");
        }
        return enabled;
    }();

    return result;
}

static bool ggml_cuda_q3k_padded_storage_enabled() {
    static bool result = [] {
        const char * env = getenv("GGML_CUDA_Q3K_PADDED_STORAGE");
#ifdef GGML_USE_HIP
        const bool enabled = env == nullptr ? true : std::atoi(env) != 0;
#else
        const bool enabled = env != nullptr && std::atoi(env) != 0;
#endif

        if (env != nullptr) {
            GGML_LOG_INFO("Detected GGML_CUDA_Q3K_PADDED_STORAGE=%d\n", enabled ? 1 : 0);
#ifdef GGML_USE_HIP
        } else if (enabled) {
            GGML_LOG_INFO("Using default GGML_CUDA_Q3K_PADDED_STORAGE=1 on HIP\n");
#endif
        }

        return enabled;
    }();

    return result;
}

static bool ggml_cuda_q3k_padded_storage_mmq_enabled() {
    static bool result = [] {
        const char * env = getenv("GGML_CUDA_Q3K_PADDED_STORAGE_MMQ");
#ifdef GGML_USE_HIP
        const bool enabled = env == nullptr ? true : std::atoi(env) != 0;
#else
        const bool enabled = env != nullptr && std::atoi(env) != 0;
#endif

        if (env != nullptr) {
            GGML_LOG_INFO("Detected GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=%d\n", enabled ? 1 : 0);
#ifdef GGML_USE_HIP
        } else if (enabled) {
            GGML_LOG_INFO("Using default GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1 on HIP\n");
#endif
        }

        return enabled;
    }();

    return result;
}

static bool ggml_cuda_q3k_padded_storage_tensor(const ggml_tensor * tensor) {
    if (ggml_cuda_q3k_padded_storage_enabled() &&
            tensor->type == GGML_TYPE_Q3_K &&
            tensor->view_src != nullptr &&
            tensor->view_src->type == GGML_TYPE_Q3_K &&
            ggml_is_contiguous(tensor->view_src) &&
            tensor->view_src->ne[0] % QK_K == 0 &&
            ggml_nelements(tensor->view_src) % QK_K == 0 &&
            tensor->view_src->buffer != nullptr &&
            !ggml_backend_buft_is_cuda_split(tensor->view_src->buffer->buft)) {
        const size_t owner_raw_size = (ggml_nelements(tensor->view_src) / QK_K) * sizeof(block_q3_K);
        const size_t owner_alloc_size = ggml_backend_buffer_get_alloc_size(tensor->view_src->buffer, tensor->view_src);

        if (owner_alloc_size > owner_raw_size) {
            GGML_ABORT("Q3_K padded storage does not support tensor views yet");
        }
    }

    if (!(ggml_cuda_q3k_padded_storage_enabled() &&
        tensor->type == GGML_TYPE_Q3_K &&
        tensor->view_src == nullptr &&
        ggml_is_contiguous(tensor) &&
        tensor->ne[0] % QK_K == 0 &&
        ggml_nelements(tensor) % QK_K == 0)) {
        return false;
    }

    return tensor->buffer == nullptr || !ggml_backend_buft_is_cuda_split(tensor->buffer->buft);
}

static bool ggml_cuda_q3k_padded_storage_tensor_physical(const ggml_tensor * tensor) {
    if (!ggml_cuda_q3k_padded_storage_tensor(tensor) || tensor->buffer == nullptr) {
        return false;
    }

    return ggml_backend_buffer_get_alloc_size(tensor->buffer, tensor) ==
        ggml_cuda_q3k_padded_storage_alloc_size(tensor);
}

static size_t ggml_cuda_q3k_padded_storage_nblocks(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(tensor->ne[0] % QK_K == 0);
    GGML_ASSERT(ggml_nelements(tensor) % QK_K == 0);
    return ggml_nelements(tensor) / QK_K;
}

static size_t ggml_cuda_q3k_padded_storage_raw_size(const ggml_tensor * tensor) {
    return ggml_cuda_q3k_padded_storage_nblocks(tensor) * sizeof(block_q3_K);
}

static size_t ggml_cuda_q3k_padded_storage_alloc_size(const ggml_tensor * tensor) {
    size_t size = ggml_cuda_q3k_padded_storage_nblocks(tensor) * sizeof(block_q3_K_padded);

    if (tensor->ne[0] % MATRIX_ROW_PADDING != 0) {
        const int64_t pad_elems = MATRIX_ROW_PADDING - tensor->ne[0] % MATRIX_ROW_PADDING;
        GGML_ASSERT(pad_elems % QK_K == 0);
        size += (pad_elems / QK_K) * sizeof(block_q3_K_padded);
    }

    return size;
}

static size_t ggml_cuda_q3k_padded_storage_offset_from_raw(const size_t raw_offset) {
    return (raw_offset / sizeof(block_q3_K)) * sizeof(block_q3_K_padded);
}

static size_t ggml_cuda_q3k_padded_storage_nblocks_from_raw_slice(
        const ggml_tensor * tensor,
        const size_t offset,
        const size_t size) {
    const size_t raw_size = ggml_cuda_q3k_padded_storage_raw_size(tensor);
    GGML_ASSERT(offset <= raw_size);
    GGML_ASSERT(size <= raw_size - offset);
    if (size == 0) {
        return 0;
    }

    const size_t raw_block_size = sizeof(block_q3_K);
    const size_t raw_block_offset = offset % raw_block_size;
    return (raw_block_offset + size + raw_block_size - 1) / raw_block_size;
}

static void ggml_cuda_q3k_copy_host_to_padded_storage(
        ggml_tensor * tensor,
        const void * data,
        const size_t offset,
        const size_t size,
        cudaStream_t stream) {
    const size_t nblocks = ggml_cuda_q3k_padded_storage_nblocks_from_raw_slice(tensor, offset, size);
    if (nblocks == 0) {
        return;
    }

    const size_t raw_block_size   = sizeof(block_q3_K);
    const size_t raw_block_offset = offset % raw_block_size;
    const size_t padded_offset    = ggml_cuda_q3k_padded_storage_offset_from_raw(offset);
    const bool partial_edge = raw_block_offset != 0 || size % raw_block_size != 0;

    std::vector<block_q3_K_padded> packed(nblocks);
    if (partial_edge) {
        CUDA_CHECK(cudaMemcpyAsync(packed.data(), (const char *) tensor->data + padded_offset,
            packed.size() * sizeof(block_q3_K_padded), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const uint8_t * src_bytes = (const uint8_t *) data;
    size_t copied = 0;
    size_t remaining = size;
    size_t dst_offset = raw_block_offset;

    for (size_t i = 0; i < nblocks; ++i) {
        size_t chunk = raw_block_size - dst_offset;
        if (chunk > remaining) {
            chunk = remaining;
        }

        std::memcpy((uint8_t *) &packed[i] + dst_offset, src_bytes + copied, chunk);
        packed[i].pad[0] = 0;
        packed[i].pad[1] = 0;

        copied += chunk;
        remaining -= chunk;
        dst_offset = 0;
    }

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + padded_offset, packed.data(),
        packed.size() * sizeof(block_q3_K_padded), cudaMemcpyHostToDevice, stream));
}

static void ggml_cuda_q3k_copy_padded_storage_to_host(
        const ggml_tensor * tensor,
        void * data,
        const size_t offset,
        const size_t size,
        cudaStream_t stream) {
    const size_t nblocks = ggml_cuda_q3k_padded_storage_nblocks_from_raw_slice(tensor, offset, size);
    if (nblocks == 0) {
        return;
    }

    const size_t raw_block_size   = sizeof(block_q3_K);
    const size_t raw_block_offset = offset % raw_block_size;
    const size_t padded_offset    = ggml_cuda_q3k_padded_storage_offset_from_raw(offset);

    std::vector<block_q3_K_padded> packed(nblocks);
    CUDA_CHECK(cudaMemcpyAsync(packed.data(), (const char *) tensor->data + padded_offset,
        packed.size() * sizeof(block_q3_K_padded), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint8_t * dst_bytes = (uint8_t *) data;
    size_t copied = 0;
    size_t remaining = size;
    size_t src_offset = raw_block_offset;

    for (size_t i = 0; i < nblocks; ++i) {
        size_t chunk = raw_block_size - src_offset;
        if (chunk > remaining) {
            chunk = remaining;
        }

        std::memcpy(dst_bytes + copied, (const uint8_t *) &packed[i] + src_offset, chunk);

        copied += chunk;
        remaining -= chunk;
        src_offset = 0;
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_q3k_padded_storage_tensor_physical(tensor)) {
        ggml_cuda_q3k_copy_host_to_padded_storage(tensor, data, offset, size, cudaStreamPerThread);
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_q3k_padded_storage_tensor_physical(tensor)) {
        ggml_cuda_q3k_copy_padded_storage_to_host(tensor, data, offset, size, cudaStreamPerThread);
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_q3k_padded_storage_tensor_physical(tensor)) {
        for (size_t i = 0; i < n_copies; ++i) {
            const size_t raw_offset = offset + i * stride_tensor;
            ggml_cuda_q3k_copy_host_to_padded_storage(tensor, (const char *) data + i * stride_data,
                raw_offset, size, cudaStreamPerThread);
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_q3k_padded_storage_tensor_physical(tensor)) {
        for (size_t i = 0; i < n_copies; ++i) {
            const size_t raw_offset = offset + i * stride_tensor;
            ggml_cuda_q3k_copy_padded_storage_to_host(tensor, (char *) data + i * stride_data,
                raw_offset, size, cudaStreamPerThread);
        }
        return;
    }

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_cuda_q3k_padded_storage_tensor_physical(src) || ggml_cuda_q3k_padded_storage_tensor_physical(dst)) {
        return false;
    }

    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            if (!ggml_cuda_peer_copy_enabled()) {
                const size_t nbytes = ggml_nbytes(src);
                static thread_local std::vector<uint8_t> host_stage;
                try {
                    if (host_stage.size() < nbytes) {
                        host_stage.resize(nbytes);
                    }
                } catch (...) {
                    return false;
                }

                ggml_cuda_set_device(src_ctx->device);
                CUDA_CHECK(cudaMemcpy(host_stage.data(), src->data, nbytes, cudaMemcpyDeviceToHost));
                ggml_cuda_set_device(dst_ctx->device);
                CUDA_CHECK(cudaMemcpyAsync(dst->data, host_stage.data(), nbytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
            } else {
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
            }
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (ggml_cuda_q3k_padded_storage_tensor(tensor)) {
        return ggml_cuda_q3k_padded_storage_alloc_size(tensor);
    }

    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

static int64_t get_row_rounding(const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split) {
    int64_t row_rounding = 0;
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            continue;
        }

        const int cc = ggml_cuda_info().devices[id].cc;
        row_rounding = std::max(row_rounding, (int64_t)get_mmq_y_host(cc));
    }
    return row_rounding;
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;

    if (id == ggml_backend_cuda_get_device_count() - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_cuda_split_buffer_type_context {
    int main_device;
    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    std::string name;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
                for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
                    if (extra->events[id][is] != nullptr) {
                        CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
                    }
                }
                if (extra->data_device[id] != nullptr) {
                    CUDA_CHECK(cudaFree(extra->data_device[id]));
                }
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};


static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_cuda_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if cudaMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_cuda_set_device(id);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, size, id));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }

        extra->data_device[id] = buf;

        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
            CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

static const char * ggml_backend_cuda_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

#ifdef GGML_USE_NCCL
struct ggml_backend_cuda_comm_context {
    std::vector<ggml_backend_t> backends;
    std::vector<ncclComm_t> comms;

    ~ggml_backend_cuda_comm_context() {
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
    }
};
#endif // GGML_USE_NCCL

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
#ifdef GGML_USE_NCCL
    if (comm_ctx_v == nullptr) {
        return;
    }
    ggml_backend_cuda_comm_context * comm_ctx = (ggml_backend_cuda_comm_context *) comm_ctx_v;
    delete comm_ctx;
#else
    GGML_UNUSED(comm_ctx_v);
#endif // GGML_USE_NCCL
}

static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
#ifdef GGML_USE_NCCL
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }
    ggml_backend_cuda_comm_context * ret = new ggml_backend_cuda_comm_context;
    std::vector<int> dev_ids;
    ret->backends.reserve(n_backends);
    dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->backends.push_back(backends[i]);
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;
        dev_ids.push_back(cuda_ctx->device);
    }

    ret->comms.resize(n_backends);
    NCCL_CHECK(ncclCommInitAll(ret->comms.data(), n_backends, dev_ids.data()));
    return ret;
#else
    // If NCCL is installed it is used by default for optimal performance.
    // However, NVIDIA does not distribute NCCL with CUDA so users may be unwittingly missing this package.
    // RCCL is disabled by default, users are explicitly opting in.
    // Therefore print no warning for RCCL.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    static bool warning_printed = false;
    if (!warning_printed) {
        GGML_LOG_WARN("%s: NVIDIA Collective Communications Library (NCCL) is unavailable, multi GPU performance will be suboptimal\n", __func__);
        warning_printed = true;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(backends, n_backends);
    return nullptr;
#endif // GGML_USE_NCCL
}

static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
#ifdef GGML_USE_NCCL
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    GGML_ASSERT(comm_ctx_v != nullptr);
    ggml_backend_cuda_comm_context * comm_ctx = (ggml_backend_cuda_comm_context *) comm_ctx_v;
    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());

        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
#else
    GGML_UNUSED_VARS(comm_ctx_v, tensors);
    return false;
#endif // GGML_USE_NCCL
}

ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<std::pair<int, std::array<float, GGML_CUDA_MAX_DEVICES>>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_CUDA_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_cuda_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find({main_device, tensor_split_arr});
    if (it != buft_map.end()) {
        return &it->second;
    }
    auto * ctx = new ggml_backend_cuda_split_buffer_type_context{
        main_device,
        tensor_split_arr,
        GGML_CUDA_NAME + std::to_string(main_device) + "_Split",
    };

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), main_device),
        /* .context = */ ctx,
    };

    auto result = buft_map.emplace(std::make_pair(main_device, tensor_split_arr), buft);
    return &result.first->second;
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

