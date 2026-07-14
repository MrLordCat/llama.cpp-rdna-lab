
// Public backend API, multi-device communication, device metadata, and registration.
static ggml_guid_t ggml_backend_vk_guid() {
    static ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    return &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_num) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_num << ")");

    ggml_backend_vk_context * ctx = new ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_num);

    ggml_backend_t vk_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_num),
        /* .context = */ ctx,
    };

    if (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = nullptr;
    }

    return vk_backend;
}

bool ggml_backend_is_vk(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

struct ggml_backend_vk_comm_context {
    std::vector<ggml_backend_t> backends;
    std::vector<void *> host_buffers;
    std::vector<vk_buffer> device_buffers;
    size_t host_buffer_size = 0;
    size_t device_buffer_size = 0;
    std::mutex mutex;
    size_t timing_size = 0;
    size_t timing_count = 0;
    double timing_read_ms = 0.0;
    double timing_reduce_ms = 0.0;
    double timing_write_ms = 0.0;
    ~ggml_backend_vk_comm_context() {
        for (size_t i = 0; i < host_buffers.size(); ++i) {
            ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) backends[i]->context;
            if (host_buffers[i] != nullptr) {
                ggml_vk_host_free(ctx->device, host_buffers[i]);
            }
            ggml_vk_destroy_buffer(device_buffers[i]);
        }
    }
};

static void ggml_backend_vk_comm_free(void * comm_ctx_v) {
    delete (ggml_backend_vk_comm_context *) comm_ctx_v;
}

static void * ggml_backend_vk_comm_init(ggml_backend_t * backends, size_t n_backends) {
    // Keep this experimental until the host-mediated path has passed correctness
    // and performance gates on multiple Vulkan drivers.
    if (std::getenv("GGML_VK_TENSOR_ALLREDUCE") == nullptr || n_backends < 2) {
        return nullptr;
    }

    std::unique_ptr<ggml_backend_vk_comm_context> comm(new ggml_backend_vk_comm_context);
    comm->backends.reserve(n_backends);
    comm->host_buffers.resize(n_backends, nullptr);
    comm->device_buffers.resize(n_backends);
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_vk(backends[i])) {
            return nullptr;
        }
        comm->backends.push_back(backends[i]);
    }

    GGML_LOG_INFO("ggml_vulkan: enabled host-mediated tensor all-reduce for %zu devices\n", n_backends);
    return comm.release();
}

static bool ggml_backend_vk_comm_ensure_buffers(ggml_backend_vk_comm_context * comm, size_t transfer_size, bool use_bf16) {
    if (comm->host_buffer_size < transfer_size) {
        for (size_t i = 0; i < comm->host_buffers.size(); ++i) {
            ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) comm->backends[i]->context;
            ggml_vk_host_free(ctx->device, comm->host_buffers[i]);
            comm->host_buffers[i] = ggml_vk_host_malloc(ctx->device, transfer_size);
            if (comm->host_buffers[i] == nullptr) {
                comm->host_buffer_size = 0;
                return false;
            }
        }
        comm->host_buffer_size = transfer_size;
    }

    if (use_bf16 && comm->device_buffer_size < transfer_size) {
        for (size_t i = 0; i < comm->device_buffers.size(); ++i) {
            ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) comm->backends[i]->context;
            ggml_vk_destroy_buffer(comm->device_buffers[i]);
            comm->device_buffers[i] = ggml_vk_create_buffer_check(
                ctx->device, transfer_size, vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
        comm->device_buffer_size = transfer_size;
    }
    return true;
}

#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
static void ggml_backend_vk_reduce_bf16_2_avx2(uint16_t * a, uint16_t * b, size_t n) {
    const __m256i round_bias = _mm256_set1_epi32(0x7fff);
    const __m256i one = _mm256_set1_epi32(1);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i a32 = _mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) (a + i))), 16);
        const __m256i b32 = _mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) (b + i))), 16);
        const __m256 sum = _mm256_add_ps(_mm256_castsi256_ps(a32), _mm256_castsi256_ps(b32));
        __m256i bits = _mm256_castps_si256(sum);
        const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), one);
        bits = _mm256_srli_epi32(_mm256_add_epi32(bits, _mm256_add_epi32(round_bias, lsb)), 16);
        const __m128i packed = _mm_packus_epi32(
            _mm256_castsi256_si128(bits), _mm256_extracti128_si256(bits, 1));
        _mm_storeu_si128((__m128i *) (a + i), packed);
        _mm_storeu_si128((__m128i *) (b + i), packed);
    }
    for (; i < n; ++i) {
        const ggml_bf16_t sum = GGML_FP32_TO_BF16(
            GGML_BF16_TO_FP32(((ggml_bf16_t *) a)[i]) +
            GGML_BF16_TO_FP32(((ggml_bf16_t *) b)[i]));
        ((ggml_bf16_t *) a)[i] = sum;
        ((ggml_bf16_t *) b)[i] = sum;
    }
}
#endif

static bool ggml_backend_vk_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx_v != nullptr);
    ggml_backend_vk_comm_context * comm = (ggml_backend_vk_comm_context *) comm_ctx_v;
    std::lock_guard<std::mutex> guard(comm->mutex);

    const size_t n_backends = comm->backends.size();
    const size_t size = ggml_nbytes(tensors[0]);
    if (size == 0) {
        return true;
    }
    if (tensors[0]->type != GGML_TYPE_F32 || !ggml_is_contiguously_allocated(tensors[0])) {
        return false;
    }
    for (size_t i = 1; i < n_backends; ++i) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 ||
            ggml_nbytes(tensors[i]) != size || !ggml_is_contiguously_allocated(tensors[i])) {
            return false;
        }
    }
    const bool use_bf16 = std::getenv("GGML_VK_TENSOR_ALLREDUCE_BF16") != nullptr && size >= 1024 * 1024;
    const size_t transfer_size = use_bf16 ? size / 2 : size;
    if (!ggml_backend_vk_comm_ensure_buffers(comm, transfer_size, use_bf16)) {
        return false;
    }

    using clock = std::chrono::steady_clock;
    const bool trace_timing = std::getenv("GGML_VK_COMM_TIMING") != nullptr;
    const auto read_start = trace_timing ? clock::now() : clock::time_point{};

    // Queue both readbacks before waiting. The per-device waits run concurrently,
    // allowing the two DMA engines and the pending compute streams to overlap.
    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            if (use_bf16) {
                ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) comm->backends[i]->context;
                ggml_backend_vk_buffer_context * tensor_buf_ctx =
                    (ggml_backend_vk_buffer_context *) tensors[i]->buffer->context;
                const size_t tensor_offset = vk_tensor_view_offset(tensors[i]);
                ggml_pipeline_request_descriptor_sets(
                    ctx, ctx->device->pipeline_contig_cpy_f32_bf16, 1);
                vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_cpy_to_contiguous(
                    ctx, compute_ctx, ctx->device->pipeline_contig_cpy_f32_bf16, tensors[i],
                    { tensor_buf_ctx->dev_buffer, tensor_offset, size },
                    { comm->device_buffers[i], 0, transfer_size });
                ggml_vk_buffer_read_async(
                    compute_ctx, comm->device_buffers[i], 0, comm->host_buffers[i], transfer_size);
            } else {
                ggml_backend_vk_get_tensor_async(comm->backends[i], tensors[i], comm->host_buffers[i], 0, size);
            }
        } else {
            std::memset(comm->host_buffers[i], 0, transfer_size);
        }
    }

    std::vector<std::thread> waiters;
    waiters.reserve(n_backends);
    for (size_t i = 0; i < n_backends; ++i) {
        waiters.emplace_back([&, i]() { ggml_backend_synchronize(comm->backends[i]); });
    }
    for (std::thread & waiter : waiters) {
        waiter.join();
    }

    const auto reduce_start = trace_timing ? clock::now() : clock::time_point{};

    const size_t n = size / sizeof(float);
    if (use_bf16) {
        ggml_bf16_t * reduced = (ggml_bf16_t *) comm->host_buffers[0];
        if (n_backends == 2) {
            ggml_bf16_t * other = (ggml_bf16_t *) comm->host_buffers[1];
#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
            if (__builtin_cpu_supports("avx2")) {
                ggml_backend_vk_reduce_bf16_2_avx2(
                    (uint16_t *) reduced, (uint16_t *) other, n);
            } else
#endif
            {
            for (size_t k = 0; k < n; ++k) {
                const ggml_bf16_t sum = GGML_FP32_TO_BF16(
                    GGML_BF16_TO_FP32(reduced[k]) + GGML_BF16_TO_FP32(other[k]));
                reduced[k] = sum;
                other[k] = sum;
            }
            }
        } else {
            for (size_t j = 1; j < n_backends; ++j) {
                const ggml_bf16_t * src = (const ggml_bf16_t *) comm->host_buffers[j];
                for (size_t k = 0; k < n; ++k) {
                    reduced[k] = GGML_FP32_TO_BF16(
                        GGML_BF16_TO_FP32(reduced[k]) + GGML_BF16_TO_FP32(src[k]));
                }
            }
            for (size_t j = 1; j < n_backends; ++j) {
                std::memcpy(comm->host_buffers[j], reduced, transfer_size);
            }
        }
    } else {
        float * reduced = (float *) comm->host_buffers[0];
        if (n_backends == 2) {
            float * other = (float *) comm->host_buffers[1];
            for (size_t k = 0; k < n; ++k) {
                const float sum = reduced[k] + other[k];
                reduced[k] = sum;
                other[k] = sum;
            }
        } else {
            for (size_t j = 1; j < n_backends; ++j) {
                const float * src = (const float *) comm->host_buffers[j];
                for (size_t k = 0; k < n; ++k) {
                    reduced[k] += src[k];
                }
            }
            for (size_t j = 1; j < n_backends; ++j) {
                std::memcpy(comm->host_buffers[j], reduced, transfer_size);
            }
        }
    }
    // Each host buffer is pinned by its matching Vulkan device. Queue both
    // uploads before waiting so the two transfer paths can progress in parallel.
    for (size_t i = 0; i < n_backends; ++i) {
        if (use_bf16) {
            ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) comm->backends[i]->context;
            ggml_backend_vk_buffer_context * tensor_buf_ctx =
                (ggml_backend_vk_buffer_context *) tensors[i]->buffer->context;
            const size_t tensor_offset = vk_tensor_view_offset(tensors[i]);
            ggml_pipeline_request_descriptor_sets(
                ctx, ctx->device->pipeline_contig_cpy_bf16_f32, 1);
            vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
            ggml_vk_buffer_write_async(
                compute_ctx, comm->device_buffers[i], 0, comm->host_buffers[i], transfer_size);

            ggml_tensor bf16_desc = *tensors[i];
            bf16_desc.type = GGML_TYPE_BF16;
            bf16_desc.nb[0] = sizeof(ggml_bf16_t);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                bf16_desc.nb[d] = bf16_desc.nb[d - 1] * bf16_desc.ne[d - 1];
            }
            ggml_vk_cpy_to_contiguous(
                ctx, compute_ctx, ctx->device->pipeline_contig_cpy_bf16_f32, &bf16_desc,
                { comm->device_buffers[i], 0, transfer_size },
                { tensor_buf_ctx->dev_buffer, tensor_offset, size });
        } else {
            ggml_backend_vk_set_tensor_async(comm->backends[i], tensors[i], comm->host_buffers[i], 0, size);
        }
    }

    const auto write_start = trace_timing ? clock::now() : clock::time_point{};

    waiters.clear();
    for (size_t i = 0; i < n_backends; ++i) {
        waiters.emplace_back([&, i]() { ggml_backend_synchronize(comm->backends[i]); });
    }
    for (std::thread & waiter : waiters) {
        waiter.join();
    }

    if (trace_timing) {
        const auto done = clock::now();
        if (comm->timing_size != size) {
            comm->timing_size = size;
            comm->timing_count = 0;
            comm->timing_read_ms = 0.0;
            comm->timing_reduce_ms = 0.0;
            comm->timing_write_ms = 0.0;
        }
        comm->timing_count++;
        comm->timing_read_ms += std::chrono::duration<double, std::milli>(reduce_start - read_start).count();
        comm->timing_reduce_ms += std::chrono::duration<double, std::milli>(write_start - reduce_start).count();
        comm->timing_write_ms += std::chrono::duration<double, std::milli>(done - write_start).count();
        if (comm->timing_count % 127 == 0) {
            GGML_LOG_INFO("ggml_vulkan: all-reduce timing: bytes=%zu transfer_bytes=%zu mode=%s count=%zu read_ms=%.3f reduce_ms=%.3f write_ms=%.3f\n",
                size, transfer_size, use_bf16 ? "bf16" : "f32", comm->timing_count,
                comm->timing_read_ms / comm->timing_count,
                comm->timing_reduce_ms / comm->timing_count,
                comm->timing_write_ms / comm->timing_count);
        }
    }
    return true;
}

int ggml_backend_vk_get_device_count() {
    return ggml_vk_get_device_count();
}

void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size) {
    GGML_ASSERT(device < (int) vk_instance.device_indices.size());
    int dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_size);
}

void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total) {
    GGML_ASSERT(device < (int) vk_instance.device_indices.size());
    GGML_ASSERT(device < (int) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetprops;
    vk::PhysicalDeviceMemoryProperties2 memprops = {};
    const bool membudget_supported = vk_instance.device_supports_membudget[device];
    const bool is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    if (membudget_supported) {
        memprops.pNext = &budgetprops;
    }
    vkdev.getMemoryProperties2(&memprops);

    *total = 0;
    *free = 0;

    for (uint32_t i = 0; i < memprops.memoryProperties.memoryHeapCount; ++i) {
        const vk::MemoryHeap & heap = memprops.memoryProperties.memoryHeaps[i];

        if (is_integrated_gpu || (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
            *total += heap.size;

            if (membudget_supported && i < budgetprops.heapUsage.size()) {
                *free += budgetprops.heapBudget[i] - budgetprops.heapUsage[i];
            } else {
                *free += heap.size;
            }
        }
    }
}

static vk::PhysicalDeviceType ggml_backend_vk_get_device_type(int device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    return props.properties.deviceType;
}

static std::string ggml_backend_vk_get_device_pci_id(int device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    bool ext_support = false;

    for (const auto& properties : ext_props) {
        if (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = true;
            break;
        }
    }

    if (!ext_support) {
        return "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.pNext = &pci_bus_info;

    device.getProperties2(&props);

    const uint32_t pci_domain = pci_bus_info.pciDomain;
    const uint32_t pci_bus = pci_bus_info.pciBus;
    const uint32_t pci_device = pci_bus_info.pciDevice;
    const uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    char pci_bus_id[16] = {};
    snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    return std::string(pci_bus_id);
}

//////////////////////////

struct ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
    bool is_integrated_gpu;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_vk_device_get_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_memory(ctx->device, free, total);
}

static ggml_backend_buffer_type_t ggml_backend_vk_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ggml_backend_vk_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_vk_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return ggml_backend_vk_host_buffer_type();
}

static enum ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    return ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ggml_backend_vk_init(ctx->device);
}

static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    const vk_device& device = ggml_vk_get_device(ctx->device);

    const bool uses_bda = (op->op == GGML_OP_IM2COL || op->op == GGML_OP_IM2COL_3D) &&
                          device->shader_int64 && device->buffer_device_address;

    auto const & tensor_size_supported = [&](size_t tensor_size) {
        if (tensor_size > device->max_buffer_size) {
            return false;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        if (!uses_bda && !device->shader_64b_indexing) {
            if (tensor_size > device->properties.limits.maxStorageBufferRange) {
                return false;
            }
        }
        return true;
    };
    // reject any tensors larger than the max buffer size
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && !tensor_size_supported(ggml_vk_device_size(op->src[i]))) {
            return false;
        }
    }
    if (!tensor_size_supported(ggml_vk_device_size(op))) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_SGN:
                    return ggml_is_contiguous(op->src[0]) &&
                           (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                if (op->op == GGML_OP_MUL_MAT_ID) {
                    if (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        return false;
                    }
                }
                switch (src0_type) {
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
                        return false;
                }
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                if (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    return false;
                }
                if (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    return false;
                }

                return true;
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                bool coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                if ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    return false;
                }
                if (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    return false;
                }
                // mismatching K/V type is currently supported for coopmat2 only.
                if (op->src[1]->type != op->src[2]->type && !coopmat2) {
                    static std::once_flag mixed_kv_warn_once;
                    std::call_once(mixed_kv_warn_once, []() {
                        GGML_LOG_WARN("ggml_vulkan: mixed K/V Flash Attention requires coopmat2; this Vulkan device will fall back to a non-Vulkan backend. Use matching cache types on this device to avoid large graph splits.\n");
                    });
                    return false;
                }
                auto fa_kv_ok = [coopmat2](ggml_type t) {
                    switch (t) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_0:
                        return true;
                    case GGML_TYPE_Q1_0:
                        return coopmat2;
                    default:
                        return false;
                    }
                };
                if (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    return false;
                }
                if (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    return false;
                }
                return true;
            }
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
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
                    case GGML_TYPE_I32:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_SET_ROWS:
            {
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != nullptr ? op->src[1]->type : src0_type;

                if (src0_type == GGML_TYPE_F32) {
                    switch (src1_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        break;
                    }
                }
                if (src1_type == GGML_TYPE_F32) {
                    switch (src0_type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        break;
                    }
                }

                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }

                if (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    return true;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                if (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_vk_device_type_size(src0_type) % 2) == 0) {
                    return true;
                }
                return false;
            }
        case GGML_OP_REPEAT:
            return ggml_type_size(op->type) == sizeof(float) && ggml_type_size(op->src[0]->type) == sizeof(float);
        case GGML_OP_REPEAT_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ROPE:
            return ggml_is_contiguous_rows(op) && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ROPE_BACK:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]) &&
                   op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_ADD_ID:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LOG:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        case GGML_OP_ARGSORT:
            {
                if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    return false;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                if (device->vulkan_memory_model) {
                    return true;
                } else {
                    return op->ne[0] <= (1 << device->max_workgroup_size_log2);
                }
            }
        case GGML_OP_TOP_K:
            {
                if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    return false;
                }
                // We could potentially support larger, using argsort to sort the
                // whole thing. Not clear if this is needed.
                uint32_t min_pipeline = (uint32_t)log2f(float(op->ne[0])) + 1;
                if (min_pipeline >= num_topk_pipelines ||
                    !device->pipeline_topk_f32[min_pipeline]) {
                    return false;
                }
            }
            return true;
        case GGML_OP_UPSCALE:
            if (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                if ((op->op_params[0] & 0xFF) != GGML_SCALE_MODE_BILINEAR) {
                    return false;
                }
            }
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ACC:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_SET:
            return op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        case GGML_OP_CONCAT:
            return ggml_type_size(op->src[0]->type) == ggml_type_size(GGML_TYPE_F32);
        case GGML_OP_ADD1:
            return (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_FILL:
            return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        case GGML_OP_SCALE:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
        case GGML_OP_ROLL:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_DIAG_MASK_INF:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SOFT_MAX:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        case GGML_OP_SOFT_MAX_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_CUMSUM:
            {
                if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
                    return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
                }
                return false;
            }
        case GGML_OP_SOLVE_TRI:
            {
                if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }
                const uint32_t N = op->src[0]->ne[0];
                const uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                if (K > 1u << device->max_workgroup_size_log2) {
                    return false;
                }
                const uint32_t batch_N = device->properties.limits.maxComputeSharedMemorySize / ((N + K) * sizeof(float));

                if (batch_N == 0) {
                    return false;
                }
                return true;
            }
        case GGML_OP_ARGMAX:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_COUNT_EQUAL:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_IM2COL_3D:
            return op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true; // all inputs are contiguous, see ggml.c
        case GGML_OP_GATED_DELTA_NET:
            {
                const uint32_t S_v = op->src[2]->ne[0];
                if (S_v != 32 && S_v != 64 && S_v != 128) {
                    return false;
                }
                for (int i = 0; i < 6; i++) {
                    if (op->src[i] == nullptr || op->src[i]->type != GGML_TYPE_F32) {
                        return false;
                    }
                }
                return op->type == GGML_TYPE_F32;
            }
        case GGML_OP_SSM_SCAN:
            {
                for (int i = 0; i < 6; i++) {
                    if (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        return false;
                    }
                }
                if (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    return false;
                }
                if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    return false;
                }

                const uint32_t d_state = op->src[0]->ne[0];
                const uint32_t head_dim = op->src[0]->ne[1];

                bool is_mamba2 = (op->src[3] && op->src[3]->nb[1] == sizeof(float));
                if (!is_mamba2) {
                    return false;
                }

                if ((d_state != 128 && d_state != 256) || head_dim % 16 != 0) {
                    return false;
                }

                size_t shmem_size = d_state * sizeof(float);

                if (shmem_size > device->properties.limits.maxComputeSharedMemorySize) {
                    return false;
                }

                if (!device->subgroup_basic) {
                    return false;
                }

                return true;
            }
        case GGML_OP_SSM_CONV:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_1D:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                // Channel-contiguous format is not supported yet.
                return ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        default:
            return false;
    }

    UNUSED(dev);
}

static bool ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name == ggml_backend_vk_host_direct_buffer_type_name) {
        return true;
    }
    if (buft->iface.get_name != ggml_backend_vk_buffer_type_name) {
        return false;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_buffer_type_context * buft_ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return buft_ctx->device->idx == ctx->device;
}

static int64_t ggml_vk_get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    return ggml_vk_get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = new vk_event;
    if (!vkev) {
        return nullptr;
    }

    // No events initially, they get created on demand
    vkev->has_event = false;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

static void ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    for (auto& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    for (auto& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    if (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    delete vkev;
    delete event;
}

static void ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    if (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize");

        // Reset and move submitted events
        for (auto& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        if (vkev->cmd_buffer) {
            // Only flag for reuse if it hasn't been reused already
            if (vkev->cmd_buffer_use_counter == vkev->cmd_buffer->use_counter) {
                vkev->cmd_buffer->in_use = false;
                vkev->cmd_buffer->buf.reset();
            }
            vkev->cmd_buffer = nullptr;
        }
    }
}

static vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, void * ptr, size_t size) {
    if (!device->external_memory_host) {
        return {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    if (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }
    if (size & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }

    const vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    try {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    return buf;
}

static ggml_backend_buffer_t ggml_backend_vk_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_UNUSED(max_tensor_size);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_buffer buf = ggml_vk_buffer_from_host_ptr(device, ptr, size);

    if (!buf) {
        return {};
    }

    ggml_backend_vk_buffer_context * bufctx = new ggml_backend_vk_buffer_context(device, std::move(buf), device->name);

    ggml_backend_buffer_t ret = ggml_backend_buffer_init(ggml_backend_vk_device_get_buffer_type(dev), ggml_backend_vk_buffer_interface, bufctx, size);

    return ret;
}

static const struct ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_memory,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

static const char * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_VK_NAME;
}

static size_t ggml_backend_vk_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return ggml_backend_vk_get_device_count();
}

static ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    static std::vector<ggml_backend_dev_t> devices;

    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            for (int i = 0; i < ggml_backend_vk_get_device_count(); i++) {
                ggml_backend_vk_device_context * ctx = new ggml_backend_vk_device_context;
                char desc[256];
                ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_size = min_batch_size;
                devices.push_back(new ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = true;
        }
    }

    GGML_ASSERT(device < devices.size());
    return devices[device];
}

static void * ggml_backend_vk_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) ggml_backend_vk_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) ggml_backend_vk_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *) ggml_backend_vk_comm_allreduce_tensor;
    }
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_count,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ ggml_backend_vk_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ nullptr,
    };
    try {
        ggml_vk_instance_init();
        return &reg;
    } catch (const vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        return nullptr;
    } catch (const std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        return nullptr;
    } catch (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        return nullptr;
    }
}
