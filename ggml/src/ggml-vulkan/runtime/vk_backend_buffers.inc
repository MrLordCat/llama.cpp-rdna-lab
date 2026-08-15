// GGML Vulkan buffer and buffer-type backend interfaces.
// backend interface

#define UNUSED GGML_UNUSED

// device backend

static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_vk_buffer_type_name;
}

static void ggml_backend_vk_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    ggml_vk_destroy_buffer(ctx->dev_buffer);
    delete ctx;
}

static void * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer) {
    return vk_ptr_base;

    UNUSED(buffer);
}

static enum ggml_status ggml_backend_vk_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << buffer << " (" << buffer->context << "), " << tensor << ")");
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_vk_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    if (size == 0) {
        return;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    size_t device_offset = offset / ggml_type_size(tensor->type) * ggml_vk_device_type_size(tensor->type)
        + offset % ggml_type_size(tensor->type);
    size_t device_size = size / ggml_type_size(tensor->type) * ggml_vk_device_type_size(tensor->type)
        + size % ggml_type_size(tensor->type);
    ggml_vk_buffer_memset(buf, vk_tensor_view_offset(tensor) + device_offset, val32, device_size);
}

static void ggml_backend_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    if (size == 0) {
        return;
    }

    if (ggml_nbytes(tensor) != ggml_vk_device_size(tensor)) {
        ggml_vk_buffer_write_padded(buf, vk_tensor_view_offset(tensor), offset, data, size, tensor->type);
    } else {
        ggml_vk_buffer_write(buf, vk_tensor_view_offset(tensor) + offset, data, size);
    }
}

static void ggml_backend_vk_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_data << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    if (size == 0) {
        return;
    }

    if (ggml_nbytes(tensor) != ggml_vk_device_size(tensor)) {
        ggml_vk_buffer_write_2d_padded(buf, vk_tensor_view_offset(tensor), offset, data, stride_data, stride_tensor, size, n_copies, tensor->type);
    } else {
        ggml_vk_buffer_write_2d(buf, vk_tensor_view_offset(tensor) + offset, data, stride_data, stride_tensor, size, n_copies);
    }
}

static void ggml_backend_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    if (size == 0) {
        return;
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    if (ggml_nbytes(tensor) != ggml_vk_device_size(tensor)) {
        ggml_vk_buffer_read_padded(buf, vk_tensor_view_offset(tensor), offset, data, size, tensor->type);
    } else {
        ggml_vk_buffer_read(buf, vk_tensor_view_offset(tensor) + offset, data, size);
    }
}

static void ggml_backend_vk_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_data << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    if (size == 0) {
        return;
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    if (ggml_nbytes(tensor) != ggml_vk_device_size(tensor)) {
        ggml_vk_buffer_read_2d_padded(buf, vk_tensor_view_offset(tensor), offset, data, stride_tensor, stride_data, size, n_copies, tensor->type);
    } else {
        ggml_vk_buffer_read_2d(buf, vk_tensor_view_offset(tensor) + offset, data, stride_tensor, stride_data, size, n_copies);
    }
}

static bool ggml_backend_vk_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_nbytes(src) == 0) {
        return true;
    }

    if (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;
        ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;

        vk_buffer src_buf = src_buf_ctx->dev_buffer;
        vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

        ggml_vk_buffer_copy(dst_buf, vk_tensor_view_offset(dst), src_buf, vk_tensor_view_offset(src), ggml_vk_device_size(src));

        return true;
    }
    return false;

    UNUSED(buffer);
}

static void ggml_backend_vk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    ggml_vk_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
};

// vk buffer type
static const char * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;

    vk_buffer dev_buffer = nullptr;
    try {
        dev_buffer = ggml_vk_create_buffer_device(ctx->device, size);
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: buffer type alloc failed: name=" << ctx->name << " size=" << size << std::endl;
        return nullptr;
    }

    ggml_backend_vk_buffer_context * bufctx = new ggml_backend_vk_buffer_context(ctx->device, std::move(dev_buffer), ctx->name);

    return ggml_backend_buffer_init(buft, ggml_backend_vk_buffer_interface, bufctx, size);
}

static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->properties.limits.minStorageBufferOffsetAlignment;
}

static size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->suballocation_block_size;
}

static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_vk_device_size(tensor);

    UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_num << ")");

    vk_device dev = ggml_vk_get_device(dev_num);

    return &dev->buffer_type;
}

// host buffer type

static const char * ggml_backend_vk_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_VK_NAME "_Host";

    UNUSED(buft);
}

static const char * ggml_backend_vk_host_direct_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_VK_NAME "_Host_Direct";

    UNUSED(buft);
}

static const char * ggml_backend_vk_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_VK_NAME "_Host";

    UNUSED(buffer);
}

static void ggml_backend_vk_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(vk_instance.devices[0], buffer->context);
}

static ggml_backend_buffer_t ggml_backend_vk_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    void * ptr = nullptr;
    try {
        ptr = ggml_vk_host_malloc(vk_instance.devices[0], size);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory (%s)\n", e.what());
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_vk_host_buffer_free_buffer;

    return buffer;

    UNUSED(buft);
}

static size_t ggml_backend_vk_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return vk_instance.devices[0]->properties.limits.minMemoryMapAlignment;

    UNUSED(buft);
}

static size_t ggml_backend_vk_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return vk_instance.devices[0]->suballocation_block_size;

    UNUSED(buft);
}

// Should be changed to return device-specific host buffer type
// but that probably requires changes in llama.cpp
ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_vk_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_vk_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_vk_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_vk_host_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_vk_host_buffer_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ nullptr,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    return &ggml_backend_vk_buffer_type_host;
}

ggml_backend_buffer_type_t ggml_backend_vk_host_direct_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_vk_buffer_type_host_direct = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_vk_host_direct_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_vk_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_vk_host_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_vk_host_buffer_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ nullptr,
    };

    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    return &ggml_backend_vk_buffer_type_host_direct;
}


// backend

static const char * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return ctx->name.c_str();
}

static void ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    delete ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_vk_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return &ctx->device->buffer_type;
}

static void ggml_backend_vk_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    if (size == 0) {
        return;
    }

    if (ggml_nbytes(tensor) != ggml_vk_device_size(tensor)) {
        ggml_vk_synchronize(ctx);
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        ggml_backend_vk_buffer_set_tensor_2d(buffer, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context cpy_ctx;

    if (ctx->device->async_use_transfer_queue) {
        if (ctx->transfer_ctx.expired()) {
            cpy_ctx = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);
            ctx->transfer_ctx = cpy_ctx;
            ggml_vk_ctx_begin(ctx->device, cpy_ctx);
        } else {
            cpy_ctx = ctx->transfer_ctx.lock();
        }
    } else {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    auto dst_offset = vk_tensor_view_offset(tensor) + offset;

    bool ret = ggml_vk_buffer_write_2d_async(cpy_ctx, buf, dst_offset, data, stride_data, stride_tensor, size, n_copies);

    if (!ret) {
        const size_t staging_size = size * n_copies;
        ggml_vk_ensure_sync_staging_buffer(ctx, staging_size);
        ggml_vk_sync_buffers(nullptr, cpy_ctx);

        std::vector<vk::BufferCopy> slices(1);
        if (size == stride_tensor) {
            slices[0].srcOffset = 0;
            slices[0].dstOffset = dst_offset;
            slices[0].size = staging_size;
        } else {
            slices.resize(n_copies);
            for (size_t i = 0; i < n_copies; i++) {
                slices[i].srcOffset = i * size;
                slices[i].dstOffset = dst_offset + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->buffer->buf.copyBuffer(ctx->sync_staging->buffer, buf->buffer, slices);

        if (size == stride_data) {
            deferred_memcpy(ctx->sync_staging->ptr, data, staging_size, &cpy_ctx->in_memcpys);
        } else {
            for (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (const uint8_t *)data + i * stride_data, size, &cpy_ctx->in_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

static void ggml_backend_vk_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_set_tensor_2d_async(backend, tensor, data, offset, size, 1, size, size);
}

static void ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    if (size == 0) {
        return;
    }

    if (ggml_nbytes(tensor) != ggml_vk_device_size(tensor)) {
        ggml_vk_synchronize(ctx);
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        ggml_backend_vk_buffer_get_tensor_2d(buffer, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_buffer buf = buf_ctx->dev_buffer;

    auto src_offset = vk_tensor_view_offset(tensor) + offset;
    bool ret = ggml_vk_buffer_read_2d_async(compute_ctx, buf, src_offset, data, stride_tensor, stride_data, size, n_copies);

    if (!ret) {
        const size_t staging_size = size * n_copies;
        ggml_vk_ensure_sync_staging_buffer(ctx, staging_size);
        ggml_vk_sync_buffers(nullptr, compute_ctx);

        std::vector<vk::BufferCopy> slices(1);
        if (size == stride_tensor) {
            slices[0].srcOffset = src_offset;
            slices[0].dstOffset = 0;
            slices[0].size = staging_size;
        } else {
            slices.resize(n_copies);
            for (size_t i = 0; i < n_copies; i++) {
                slices[i].srcOffset = src_offset + i * stride_tensor;
                slices[i].dstOffset = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->buffer->buf.copyBuffer(buf->buffer, ctx->sync_staging->buffer, slices);

        if (size == stride_data) {
            deferred_memcpy(data, ctx->sync_staging->ptr, staging_size, &compute_ctx->out_memcpys);
        } else {
            for (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)data + i * stride_data, (const uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->out_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

static void ggml_backend_vk_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, data, offset, size, 1, size, size);
}

static bool ggml_backend_vk_get_tensor_batch(
        ggml_backend_t backend,
        const ggml_backend_tensor_get_entry * entries,
        size_t n_entries) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_batch(" << n_entries << ")");

    if (n_entries == 0) {
        return true;
    }

    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) backend->context;
    size_t staging_size = 0;
    const bool trace_checkpoint = getenv("LLAMA_CHECKPOINT_TIMING") != nullptr;

    // Validate the whole batch before recording any command so false remains a
    // side-effect-free request for the generic fallback.
    for (size_t i = 0; i < n_entries; ++i) {
        const ggml_tensor * tensor = entries[i].tensor;
        if (entries[i].size == 0) {
            continue;
        }
        if (tensor->buffer->buft != ggml_backend_vk_get_default_buffer_type(backend) ||
            ggml_nbytes(tensor) != ggml_vk_device_size(tensor) ||
            (entries[i].offset % 4) != 0 || (entries[i].size % 4) != 0) {
            if (trace_checkpoint) {
                GGML_LOG_INFO("checkpoint Vulkan batch fallback: tensor=%s type=%s nbytes=%zu device_size=%zu offset=%zu size=%zu default_buft=%d\n",
                        tensor->name, ggml_type_name(tensor->type), ggml_nbytes(tensor), ggml_vk_device_size(tensor),
                        entries[i].offset, entries[i].size,
                        tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend));
            }
            return false;
        }

        ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *) tensor->buffer->context;
        if (buf_ctx->dev_buffer->device != ctx->device) {
            if (trace_checkpoint) {
                GGML_LOG_INFO("checkpoint Vulkan batch fallback: tensor=%s belongs to another device\n", tensor->name);
            }
            return false;
        }

        staging_size = (staging_size + 3) & ~size_t(3);
        if (entries[i].size > SIZE_MAX - staging_size) {
            return false;
        }
        staging_size += entries[i].size;
    }

    if (staging_size == 0) {
        return true;
    }

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    ggml_vk_ensure_sync_staging_buffer(ctx, staging_size);
    ggml_vk_sync_buffers(nullptr, compute_ctx);

    size_t staging_offset = 0;
    for (size_t i = 0; i < n_entries; ++i) {
        const ggml_backend_tensor_get_entry & entry = entries[i];
        if (entry.size == 0) {
            continue;
        }

        staging_offset = (staging_offset + 3) & ~size_t(3);
        ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *) entry.tensor->buffer->context;
        vk::BufferCopy slice(
                vk_tensor_view_offset(entry.tensor) + entry.offset,
                staging_offset,
                entry.size);
        compute_ctx->s->buffer->buf.copyBuffer(
                buf_ctx->dev_buffer->buffer,
                ctx->sync_staging->buffer,
                { slice });
        deferred_memcpy(
                entry.data,
                (const uint8_t *) ctx->sync_staging->ptr + staging_offset,
                entry.size,
                &compute_ctx->out_memcpys);
        staging_offset += entry.size;
    }

    ggml_vk_synchronize(ctx);
    if (trace_checkpoint) {
        GGML_LOG_INFO("checkpoint Vulkan batch: tensors=%zu bytes=%zu\n", n_entries, staging_size);
    }
    return true;
}

static bool ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_vk_device_size(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    if (ggml_nbytes(src) == 0) {
        return true;
    }

    if (dst->buffer->buft != ggml_backend_vk_get_default_buffer_type(backend_dst)) {
        return false;
    }

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

    if (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;

        // Async copy only works within the same device
        if (src_buf_ctx->dev_buffer->device != dst_buf->device) {
            return false;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_buffer_copy_async(compute_ctx, dst_buf, vk_tensor_view_offset(dst),
                                   src_buf_ctx->dev_buffer, vk_tensor_view_offset(src),
                                   ggml_vk_device_size(src));
        return true;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (ggml_vk_device_size(src) != ggml_nbytes(src)) {
            return false;
        }

        vk_buffer pinned_buf = nullptr;
        size_t pinned_offset = 0;
        ggml_vk_host_get(ctx->device, src->data, pinned_buf, pinned_offset);
        if (pinned_buf == nullptr) {
            return false;
        }

        vk_context cpy_ctx;
        if (ctx->device->async_use_transfer_queue) {
            if (ctx->transfer_ctx.expired()) {
                cpy_ctx = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);
                ctx->transfer_ctx = cpy_ctx;
                ggml_vk_ctx_begin(ctx->device, cpy_ctx);
            } else {
                cpy_ctx = ctx->transfer_ctx.lock();
            }
        } else {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        return ggml_vk_buffer_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_view_offset(dst),
                                          src->data, ggml_nbytes(src));
    }

    GGML_UNUSED(backend_src);
    return false;
}
