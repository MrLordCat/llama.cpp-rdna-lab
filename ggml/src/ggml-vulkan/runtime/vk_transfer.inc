// Pipeline lookup plus host/device staging, transfer, copy, and memset helpers.
static vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            break;
        default:
            return nullptr;
    }

    return ctx->device->pipeline_dequant[type];
}

static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    VK_LOG_DEBUG("ggml_vk_get_mul_mat_mat_pipeline(" << ggml_type_name(src0_type) << ", " << ggml_type_name(src1_type) << ", " << prec << ")");
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
        return ctx->device->pipeline_matmul_f32;
    }
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
        return ctx->device->pipeline_matmul_f32_f16;
    }
    if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16) {
        return ctx->device->pipeline_matmul_bf16;
    }
    if (prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support)) {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_f16_f32.f16acc;
        }
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
            return ctx->device->pipeline_matmul_f16.f16acc;
        }
    } else {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_f16_f32.f32acc;
        }
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
            return ctx->device->pipeline_matmul_f16.f32acc;
        }
    }

    // MMQ
    if (src1_type == GGML_TYPE_Q8_1) {
        vk_matmul_pipeline pipelines = ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0_type].f32acc;

        if (pipelines->is_empty()) {
            return nullptr;
        }

        return pipelines;
    }

    if (src1_type != GGML_TYPE_F32 && !ctx->device->coopmat2) {
        return nullptr;
    }

    switch (src0_type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            break;
        default:
            return nullptr;
    }

    if (ctx->device->coopmat2) {
        assert(src1_type == GGML_TYPE_F16);
        return prec == GGML_PREC_DEFAULT ? ctx->device->pipeline_dequant_mul_mat_mat_f16[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat_f16[src0_type].f32acc;
    }
    if (ctx->device->coopmat_support) {
        return (ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT) ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;
    }
    return (ctx->device->fp16 && prec == GGML_PREC_DEFAULT) ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;
}

static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t num_cols, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(num_cols >= 1 && num_cols <= mul_mat_vec_max_cols);

    if (b_type == GGML_TYPE_Q8_1) {
        switch (a_type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
                break;
            default:
                return nullptr;
        }
    }

    switch (a_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            break;
        default:
            return nullptr;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
    if ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        if (a_type == GGML_TYPE_Q6_K) {
            if (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_SIZE_LARGE;
            }
        } else {
            if (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_SIZE_LARGE;
            }
        }
    }

    if (b_type == GGML_TYPE_Q8_1) {
        if (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
        }
        return ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][num_cols-1];
    }

    return b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][num_cols-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][num_cols-1];
}

static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_id_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    VK_LOG_DEBUG("ggml_vk_get_mul_mat_mat_id_pipeline()");
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
        return ctx->device->pipeline_matmul_id_f32;
    }
    if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16) {
        return ctx->device->pipeline_matmul_id_bf16;
    }
    if (prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support)) {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_id_f16_f32.f16acc;
        }
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
            return ctx->device->pipeline_matmul_id_f16.f16acc;
        }
    } else {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_id_f16_f32.f32acc;
        }
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
            return ctx->device->pipeline_matmul_id_f16.f32acc;
        }
    }

    // MMQ
    if (src1_type == GGML_TYPE_Q8_1) {
        vk_matmul_pipeline pipelines = ctx->device->pipeline_dequant_mul_mat_mat_id_q8_1[src0_type].f32acc;

        if (pipelines->is_empty()) {
            return nullptr;
        }

        return pipelines;
    }

    GGML_ASSERT(src1_type == GGML_TYPE_F32 || (ctx->device->coopmat2 && src1_type == GGML_TYPE_F16));

    switch (src0_type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            break;
        default:
            return nullptr;
    }

    vk_matmul_pipeline2& mmp = ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type];
    // XXX TODO 'prec' is not actually allowed in mul_mat_id.
    bool prefer_fp16acc = ctx->device->fp16 /*&& prec == GGML_PREC_DEFAULT*/;
    bool support_fp16acc = !mmp.f16acc->is_empty();
    bool support_fp32acc = !mmp.f32acc->is_empty();

    if (support_fp16acc && (prefer_fp16acc || !support_fp32acc)) {
        return mmp.f16acc;
    } else {
        GGML_ASSERT(support_fp32acc);
        return mmp.f32acc;
    }
}

static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    if (b_type == GGML_TYPE_Q8_1) {
        switch (a_type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
                break;
            default:
                return nullptr;
        }
    }

    switch (a_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            break;
        default:
            return nullptr;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
    if ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        if (a_type == GGML_TYPE_Q6_K) {
            if (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_SIZE_LARGE;
            }
        } else {
            if (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_SIZE_LARGE;
            }
        }
    }

    if (b_type == GGML_TYPE_Q8_1) {
        if (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
        }
        return ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    return ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

static void * ggml_vk_host_malloc(vk_device& device, size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");
    vk_buffer buf = ggml_vk_create_buffer(device, size,
        {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});

    if(!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n",
            size/1024.0/1024.0);
        device->device.freeMemory(buf->device_memory);
        device->device.destroyBuffer(buf->buffer);
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    device->pinned_memory.push_back(std::make_tuple(buf->ptr, size, buf));

    return buf->ptr;
}

static void ggml_vk_host_free(vk_device& device, void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    vk_buffer buf;
    size_t index;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            index = i;
            break;
        }
    }
    if (buf == nullptr) {
        fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
        return;
    }

    ggml_vk_destroy_buffer(buf);

    device->pinned_memory.erase(device->pinned_memory.begin() + index);
}

static void ggml_vk_host_get(const vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    buf = nullptr;
    buf_offset = 0;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            buf_offset = ((const uint8_t *)ptr) - addr;
            break;
        }
    }
}

static vk_subbuffer ggml_vk_tensor_subbuffer(
    const ggml_backend_vk_context * ctx, const ggml_tensor * tensor, bool allow_misalign = false) {

    vk_buffer buffer = nullptr;
    size_t offset = 0;
    if (ctx->device->uma || ggml_backend_buffer_is_host(tensor->buffer)) {
        ggml_vk_host_get(ctx->device, tensor->data, buffer, offset);
    }
    if (!buffer) {
        GGML_ASSERT(!ggml_backend_buffer_is_host(tensor->buffer));
        auto buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;
        buffer = buf_ctx->dev_buffer;
        offset = vk_tensor_view_offset(tensor);
    }
    GGML_ASSERT(buffer != nullptr);

    size_t size = ggml_vk_device_size(tensor);

    size_t misalign_bytes = offset & (ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_bytes == 0);
    offset &= ~misalign_bytes;
    size += misalign_bytes;

    return vk_subbuffer{buffer, offset, size};
}

// Get a command buffer from pool. Create a new one if no reusable buffer is available
static vk_command_buffer* ggml_vk_get_or_create_cmd_buffer(vk_device& device, vk_command_pool& pool) {
    for (auto& cmd_buffer : pool.cmd_buffers) {
        if (!cmd_buffer.in_use) {
            cmd_buffer.use_counter++;
            cmd_buffer.in_use = true;
            return &cmd_buffer;
        }
    }
    return ggml_vk_create_cmd_buffer(device, pool);
}

static vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, bool one_time = true) {
    vk_submission s;
    s.buffer = ggml_vk_get_or_create_cmd_buffer(device, p);
    if (one_time) {
        s.buffer->buf.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    } else {
        s.buffer->buf.begin({ vk::CommandBufferUsageFlags{} });
    }

    return s;
}

template <typename T> size_t push_constant_size(const T &t) {
    static_assert(std::is_class<T>::value, "T must be a struct/class");
    GGML_UNUSED(t);
    return sizeof(T);
}
template <typename T> size_t push_constant_size(const std::vector<T> &t) {
    GGML_UNUSED(t);
    return sizeof(T) * t.size();
}
template <typename T, uint32_t N> size_t push_constant_size(const std::array<T, N> &t) {
    GGML_UNUSED(t);
    return sizeof(T) * N;
}

template <typename T> const T *push_constant_data(const T &t) {
    static_assert(std::is_class<T>::value, "T must be a struct/class");
    return &t;
}
template <typename T> const T *push_constant_data(const std::vector<T> &t) {
    return t.data();
}
template <typename T, uint32_t N> const T *push_constant_data(const std::array<T, N> &t) {
    return t.data();
}

template <typename T>
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos, const T &push_constants, std::array<uint32_t, 3> elements) {
    const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    for (auto& buffer : descriptor_buffer_infos) {
        std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.maxComputeWorkGroupCount[0] &&
                wg1 <= ctx->device->properties.limits.maxComputeWorkGroupCount[1] &&
                wg2 <= ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
    GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
    GGML_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size());
    GGML_ASSERT(pipeline->push_constant_size == push_constant_size(push_constants));

    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];
    vk::WriteDescriptorSet write_descriptor_set{ descriptor_set, 0, 0, pipeline->parameter_count, vk::DescriptorType::eStorageBuffer, nullptr, descriptor_buffer_infos.begin() };
    ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});

    subctx->s->buffer->buf.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0, push_constant_size(push_constants), push_constant_data(push_constants));
    subctx->s->buffer->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer->buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_set },
                                {});
    subctx->s->buffer->buf.dispatch(wg0, wg1, wg2);
}

static void ggml_vk_end_submission(vk_submission& s, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores) {
    s.buffer->buf.end();

    s.wait_semaphores = std::move(wait_semaphores);
    s.signal_semaphores = std::move(signal_semaphores);
}

static void ggml_vk_ctx_end(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    if (ctx->s == nullptr) {
        return;
    }

    ctx->s->buffer->buf.end();
    ctx->s = nullptr;
}

static void ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    if (subctx->s != nullptr) {
        ggml_vk_ctx_end(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].data();
}

static vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context result;
    if (!ctx->compute_ctx.expired()) {
        result = ctx->compute_ctx.lock();
    } else {
        result = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = result;
        ggml_vk_ctx_begin(ctx->device, result);
    }

    if (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_last_submitted < ctx->transfer_semaphore.value) {
        result->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_last_submitted = ctx->transfer_semaphore.value;
    }

    return result;
}

// Submit any pending transfer queue work and signal the transfer semaphore.
// The next compute context created via ggml_vk_get_compute_ctx will wait on this semaphore.
// Returns true if work was submitted.
static bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    if (!ctx->device->async_use_transfer_queue || ctx->transfer_ctx.expired()) {
        return false;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.lock();
    ggml_vk_ctx_end(cpy_ctx);

    for (auto& cpy : cpy_ctx->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    return true;
}

static size_t ggml_vk_align_size(size_t width, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << width << ", " << align << ")");
    return CEIL_DIV(width, align) * align;
}

static void deferred_memcpy(void * dst, const void * src, size_t size, std::vector<vk_staging_memcpy>* memcpys = nullptr) {
    if (memcpys == nullptr) {
        memcpy(dst, src, size);
    } else {
        memcpys->emplace_back(dst, src, size);
    }
}

static void deferred_memset(void * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets = nullptr) {
    if (memsets == nullptr) {
        memset(dst, val, size);
    } else {
        memsets->emplace_back(dst, val, size);
    }
}

static void ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    if (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static void ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    if (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Staging buffer required
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    if (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer((VkBuffer)staging_buffer->buffer, (VkBuffer)dst->buffer, slices);

    if (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (const uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    return true;
}

static bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    return ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

static void ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        for (size_t i = 0; i < height; i++) {
            memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (const uint8_t *) src + i * spitch, width);
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue.cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        bool ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        for (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_buffer_write_2d waitForFences");
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

static void ggml_vk_buffer_write(vk_buffer& dst, size_t offset, const void * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

// view_offset is expected to be in device bytes, all other values in host/ggml bytes
static void ggml_vk_buffer_write_2d_padded(vk_buffer& dst, size_t view_offset, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_padded(" << width << ", " << height << ")");

    size_t host_block_size = ggml_type_size(type);
    size_t device_block_size = ggml_vk_device_type_size(type);

    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        for (size_t i = 0; i < height; i++) {
            size_t row_pos = 0;
            while (row_pos < width) {
                size_t host_bytes_write_pos = i * dpitch + offset + row_pos;
                size_t block = host_bytes_write_pos / host_block_size;
                size_t block_pos = host_bytes_write_pos % host_block_size;
                size_t nbytes = std::min(width - row_pos, host_block_size - block_pos);
                memcpy((uint8_t *) dst->ptr + view_offset + block * device_block_size + block_pos,
                       (const uint8_t *) src + i * spitch + row_pos,
                       nbytes);
                row_pos += nbytes;
            }
        }
    } else {
        size_t dev_offset = offset / host_block_size * device_block_size + offset % host_block_size;
        GGML_ASSERT(height == 1 || dpitch % host_block_size == 0);
        size_t dev_stride = dpitch / host_block_size * device_block_size;

        size_t dev_width = width / host_block_size * device_block_size + width % host_block_size;
        uint8_t *hostbuf = (uint8_t *)ggml_vk_host_malloc(dst->device, height * dev_width + 2 * device_block_size);
        size_t view_start_block_pos = offset % host_block_size;

        for (size_t i = 0; i < height; i++) {
            size_t row_pos = 0;
            size_t start_block = (i * dpitch + offset) / host_block_size;
            while (row_pos < width) {
                size_t host_bytes_write_pos = i * dpitch + offset + row_pos;
                size_t block = host_bytes_write_pos / host_block_size;
                size_t block_pos = host_bytes_write_pos % host_block_size;
                size_t nbytes = std::min(width - row_pos, host_block_size - block_pos);
                memcpy((uint8_t *) hostbuf + i * dev_width + (block - start_block) * device_block_size + block_pos - view_start_block_pos,
                       (const uint8_t *) src + i * spitch + row_pos,
                       nbytes);
                row_pos += nbytes;
            }
        }
        ggml_vk_buffer_write_2d(dst, view_offset + dev_offset, hostbuf, dev_width, dev_stride, dev_width, height);
        ggml_vk_host_free(dst->device, hostbuf);
    }
}

static void ggml_vk_buffer_write_padded(vk_buffer& dst, size_t view_offset, size_t offset, const void * src, size_t size, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_padded(" << size << ")");
    ggml_vk_buffer_write_2d_padded(dst, view_offset, offset, src, size, size, size, 1, type);
}

static bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Fall back to staging buffer
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    if (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } else {
        staging_slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    if (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    return true;
}

static bool ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t size, bool sync_staging = false) {
    return ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

static void ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        for (size_t i = 0; i < height; i++) {
            memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) src->ptr + offset + i * spitch, width);
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue.cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        bool ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_read_2d waitForFences");
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

static void ggml_vk_buffer_read(vk_buffer& src, size_t offset, void * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

// view_offset is expected to be in device bytes, all other values in host/ggml bytes
static void ggml_vk_buffer_read_2d_padded(vk_buffer& src, size_t view_offset, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_padded(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");

    size_t host_block_size = ggml_type_size(type);
    size_t device_block_size = ggml_vk_device_type_size(type);

    if (src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        for (size_t i = 0; i < height; i++) {
            size_t row_pos = 0;
            while (row_pos < width) {
                size_t host_bytes_read_pos = i * spitch + offset + row_pos;
                size_t block = host_bytes_read_pos / host_block_size;
                size_t block_pos = host_bytes_read_pos % host_block_size;
                size_t nbytes = std::min(width - row_pos, host_block_size - block_pos);
                memcpy((uint8_t *) dst + i * dpitch + row_pos,
                       (const uint8_t *) src->ptr + view_offset + block * device_block_size + block_pos,
                       nbytes);
                row_pos += nbytes;
            }
        }
    } else {
        size_t dev_offset = offset / host_block_size * device_block_size + offset % host_block_size;
        GGML_ASSERT(height == 1 || spitch % host_block_size == 0);
        size_t dev_stride = spitch / host_block_size * device_block_size;
        size_t view_start_block_pos = offset % host_block_size;

        size_t dev_width = width / host_block_size * device_block_size + width % host_block_size;
        uint8_t *hostbuf = (uint8_t *)ggml_vk_host_malloc(src->device, height * dev_width + 2 * device_block_size);
        ggml_vk_buffer_read_2d(src, view_offset + dev_offset, hostbuf, dev_stride, dev_width, dev_width, height);

        for (size_t i = 0; i < height; i++) {
            size_t row_pos = 0;
            size_t start_block = (i * spitch + offset) / host_block_size;
            while (row_pos < width) {
                size_t host_bytes_read_pos = i * spitch + offset + row_pos;
                size_t block = host_bytes_read_pos / host_block_size;
                size_t block_pos = host_bytes_read_pos % host_block_size;
                size_t nbytes = std::min(width - row_pos, host_block_size - block_pos);
                memcpy((uint8_t *) dst + i * dpitch + row_pos,
                       (const uint8_t *) hostbuf + i * dev_width + (block - start_block) * device_block_size + block_pos - view_start_block_pos,
                       nbytes);
                row_pos += nbytes;
            }
        }
        ggml_vk_host_free(src->device, hostbuf);
    }
}

static void ggml_vk_buffer_read_padded(vk_buffer& src, size_t view_offset, size_t offset, void * dst, size_t size, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_padded(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d_padded(src, view_offset, offset, dst, size, size, size, 1, type);
}

static void ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

static void ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue.cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_copy waitForFences");
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } else {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

static void ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        return;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

static void ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + offset, c, size);
        return;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue.cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_memset waitForFences");
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}
