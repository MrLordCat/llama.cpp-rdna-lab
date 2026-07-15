////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (ggml_cuda_q3k_padded_storage_tensor_physical(tensor)) {
        ggml_cuda_q3k_copy_host_to_padded_storage(tensor, data, offset, size, cuda_ctx->stream());
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (ggml_cuda_q3k_padded_storage_tensor_physical(tensor)) {
        ggml_cuda_q3k_copy_padded_storage_to_host(tensor, data, offset, size, cuda_ctx->stream());
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (ggml_cuda_q3k_padded_storage_tensor_physical(tensor)) {
        for (size_t i = 0; i < n_copies; ++i) {
            const size_t raw_offset = offset + i * stride_tensor;
            ggml_cuda_q3k_copy_host_to_padded_storage(tensor, (const char *) data + i * stride_data,
                raw_offset, size, cuda_ctx->stream());
        }
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        return;
    }

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (ggml_cuda_q3k_padded_storage_tensor_physical(tensor)) {
        for (size_t i = 0; i < n_copies; ++i) {
            const size_t raw_offset = offset + i * stride_tensor;
            ggml_cuda_q3k_copy_padded_storage_to_host(tensor, (char *) data + i * stride_data,
                raw_offset, size, cuda_ctx->stream());
        }
        return;
    }

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    if (ggml_cuda_q3k_padded_storage_tensor_physical(src) || ggml_cuda_q3k_padded_storage_tensor_physical(dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            if (!ggml_cuda_peer_copy_enabled()) {
#if defined(GGML_USE_HIP) && defined(_WIN32)
                static const bool async_host_stage = [] {
                    const bool enabled = ggml_cuda_env_opt_in_enabled("GGML_ROCM_ASYNC_CROSS_DEVICE_STAGE");
                    if (enabled) {
                        GGML_LOG_INFO("ROCm async cross-device host staging enabled\n");
                    }
                    return enabled;
                }();
                static const bool trace_async_host_stage =
                    getenv("GGML_TRACE_ROCM_ASYNC_CROSS_DEVICE_STAGE") != nullptr;
                if (!async_host_stage) {
                    return false;
                }

                const size_t nbytes = ggml_nbytes(dst);
                auto & slot = cuda_ctx_src->async_host_stage_slots[
                    cuda_ctx_src->async_host_stage_cursor++ % ggml_backend_cuda_context::ASYNC_HOST_STAGE_SLOTS];

                if (slot.pending) {
                    ggml_cuda_set_device(slot.destination_device);
                    CUDA_CHECK(cudaEventSynchronize(slot.destination_done));
                    slot.pending = false;
                }

                const bool devices_changed = slot.source_device != cuda_ctx_src->device ||
                    slot.destination_device != cuda_ctx_dst->device;
                if (devices_changed) {
                    if (slot.source_ready != nullptr) {
                        ggml_cuda_set_device(slot.source_device);
                        CUDA_CHECK(cudaEventDestroy(slot.source_ready));
                        slot.source_ready = nullptr;
                    }
                    if (slot.destination_done != nullptr) {
                        ggml_cuda_set_device(slot.destination_device);
                        CUDA_CHECK(cudaEventDestroy(slot.destination_done));
                        slot.destination_done = nullptr;
                    }
                    slot.source_device = cuda_ctx_src->device;
                    slot.destination_device = cuda_ctx_dst->device;
                }

                if (slot.capacity < nbytes) {
                    if (slot.host != nullptr) {
                        CUDA_CHECK(cudaFreeHost(slot.host));
                        slot.host = nullptr;
                        slot.capacity = 0;
                    }
                    slot.host = ggml_cuda_host_malloc(nbytes);
                    if (slot.host == nullptr) {
                        return false;
                    }
                    slot.capacity = nbytes;
                }

                ggml_cuda_set_device(cuda_ctx_src->device);
                if (slot.source_ready == nullptr) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&slot.source_ready, cudaEventDisableTiming));
                }
                CUDA_CHECK(cudaMemcpyAsync(slot.host, src->data, nbytes, cudaMemcpyDeviceToHost, cuda_ctx_src->stream()));
                CUDA_CHECK(cudaEventRecord(slot.source_ready, cuda_ctx_src->stream()));

                ggml_cuda_set_device(cuda_ctx_dst->device);
                if (slot.destination_done == nullptr) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&slot.destination_done, cudaEventDisableTiming));
                }
                cudaError_t async_status = cudaStreamWaitEvent(cuda_ctx_dst->stream(), slot.source_ready, 0);
                if (async_status == cudaSuccess) {
                    async_status = cudaMemcpyAsync(
                        dst->data, slot.host, nbytes, cudaMemcpyHostToDevice, cuda_ctx_dst->stream());
                }
                if (async_status != cudaSuccess) {
                    GGML_LOG_WARN(
                        "ROCm async cross-device host staging unavailable for %d -> %d: %s; using synchronous fallback\n",
                        cuda_ctx_src->device, cuda_ctx_dst->device, cudaGetErrorString(async_status));
                    (void) cudaGetLastError();
                    ggml_cuda_set_device(cuda_ctx_src->device);
                    CUDA_CHECK(cudaEventSynchronize(slot.source_ready));
                    return false;
                }
                CUDA_CHECK(cudaEventRecord(slot.destination_done, cuda_ctx_dst->stream()));
                slot.pending = true;

                if (trace_async_host_stage) {
                    GGML_LOG_INFO(
                        "GGML_TRACE_ROCM_ASYNC_CROSS_DEVICE_STAGE: src=%d dst=%d tensor=%s bytes=%zu slot=%zu\n",
                        cuda_ctx_src->device, cuda_ctx_dst->device, src->name, nbytes,
                        (cuda_ctx_src->async_host_stage_cursor - 1) % ggml_backend_cuda_context::ASYNC_HOST_STAGE_SLOTS);
                }
                return true;
#else
                return false;
#endif
            }
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}
