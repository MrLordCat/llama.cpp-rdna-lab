// Pipeline compilation, command pools, synchronization primitives, and memory allocation.
static VkDeviceSize ggml_vk_get_max_buffer_range(const ggml_backend_vk_context * ctx, const vk_buffer &buf, const VkDeviceSize offset) {
    const VkDeviceSize range = std::min(VkDeviceSize{buf->size - offset},
                                        VkDeviceSize{ctx->device->properties.limits.maxStorageBufferRange});
    return range;
}

// Wait for ctx->fence to be signaled.
static void ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    if (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, true, UINT64_MAX), "almost_ready_fence");
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = false;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::Result result;
    while ((result = ctx->device->device.getFenceStatus(ctx->fence)) != vk::Result::eSuccess) {
        if (result != vk::Result::eNotReady) {
            fprintf(stderr, "ggml_vulkan: error %s at %s:%d\n", to_string(result).c_str(), __FILE__, __LINE__);
            exit(1);
        }
        for (uint32_t i = 0; i < 100; ++i) {
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
        }
    }
    ctx->device->device.resetFences({ ctx->fence });
}

// variables to track number of compiles in progress
static uint32_t compile_count = 0;
static std::mutex compile_count_mutex;
static std::condition_variable compile_count_cond;

static void ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_size, const void* spv_data, const std::string entrypoint,
                                         uint32_t parameter_count, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         bool disable_robustness, bool require_full_subgroups, uint32_t required_subgroup_size) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_count <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_size << ")");
    static const bool trace_pipeline_create = std::getenv("GGML_VK_PIPELINE_CREATE_TRACE") != nullptr;
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(parameter_count <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    if (trace_pipeline_create) {
        std::ostringstream ss;
        ss << "ggml_vulkan: create pipeline begin: device=" << device->name
           << " name=" << pipeline->name
           << " entry=" << entrypoint
           << " params=" << parameter_count
           << " wg_denoms=(" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << ")"
           << " disable_robustness=" << disable_robustness
           << " full_subgroups=" << require_full_subgroups
           << " required_subgroup_size=" << required_subgroup_size
           << " spv_size=" << spv_size
           << " spec=[";
        for (size_t i = 0; i < specialization_constants.size(); ++i) {
            if (i > 0) {
                ss << ",";
            }
            ss << specialization_constants[i];
        }
        ss << "]";
        std::cerr << ss.str() << std::endl;
    }

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t *>(spv_data));

    // Patch SPIR-V to enable RTE rounding for FP16, avoiding the need for
    // separate shader variants compiled with -DRTE16.
    std::vector<uint32_t> spv;
    if (device->float_controls_rte_fp16) {
        const uint32_t* spv_words = reinterpret_cast<const uint32_t *>(spv_data);
        size_t word_count = spv_size / sizeof(uint32_t);
        spv.assign(spv_words, spv_words + word_count);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_insert_pos = pos;
        size_t ext_insert_pos = pos;
        size_t exec_insert_pos = pos;
        uint32_t entry_point_id = 0;

        while (pos < spv.size()) {
            uint32_t opcode = spv[pos] & spv::OpCodeMask;
            uint32_t len    = spv[pos] >> spv::WordCountShift;
            if (len == 0) break;

            if (opcode == spv::OpCapability) {
                cap_insert_pos = pos + len;
                ext_insert_pos = pos + len;
            } else if (opcode == spv::OpExtension) {
                ext_insert_pos = pos + len;
            } else if (opcode == spv::OpEntryPoint) {
                entry_point_id = spv[pos + 2];
                exec_insert_pos = pos + len;
            } else if (opcode == spv::OpExecutionMode || opcode == spv::OpExecutionModeId) {
                exec_insert_pos = pos + len;
            } else if (entry_point_id != 0) {
                break;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        // OpExecutionMode %entrypoint RoundingModeRTE 16
        uint32_t exec_mode[] = { (4u << spv::WordCountShift) | spv::OpExecutionMode, entry_point_id, spv::ExecutionModeRoundingModeRTE, 16 };
        spv.insert(spv.begin() + exec_insert_pos, std::begin(exec_mode), std::end(exec_mode));

        // OpExtension "SPV_KHR_float_controls"
        const char ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_words = CEIL_DIV(sizeof(ext_str), sizeof(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_words, 0);
        extension[0] = (uint32_t)((1 + ext_str_words) << spv::WordCountShift) | spv::OpExtension;
        memcpy(&extension[1], ext_str, sizeof(ext_str));
        spv.insert(spv.begin() + ext_insert_pos, extension.begin(), extension.end());

        // OpCapability RoundingModeRTE
        uint32_t capability[] = { (2u << spv::WordCountShift) | spv::OpCapability, spv::CapabilityRoundingModeRTE };
        spv.insert(spv.begin() + cap_insert_pos, std::begin(capability), std::end(capability));

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spv.size() * sizeof(uint32_t), spv.data());
    }

    pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline->push_constant_size
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);

    std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

    for (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].offset = i * sizeof(uint32_t);
        specialization_entries[i].size = sizeof(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.data(),
        specialization_constants.size() * sizeof(uint32_t),
        specialization_constants.data()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    if (device->subgroup_require_full_support && require_full_subgroups) {
        pipeline_shader_stage_create_flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::ShaderStageFlagBits::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info;
    pipeline_shader_stage_required_subgroup_size_create_info.requiredSubgroupSize = required_subgroup_size;
    if (device->subgroup_size_control && required_subgroup_size > 0) {
        GGML_ASSERT(device->subgroup_min_size <= required_subgroup_size && required_subgroup_size <= device->subgroup_max_size);
        pipeline_shader_create_info.setPNext(&pipeline_shader_stage_required_subgroup_size_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::PipelineCreateFlagBits::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    if (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        rci.uniformBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        compute_pipeline_create_info.setPNext(&rci);
    }

#if defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    if (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64BitIndexingEXT;
        if (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.setPNext(compute_pipeline_create_info.pNext);
        compute_pipeline_create_info.setPNext(&pipelineFlags2CreateInfo);
    }
#endif

    if (trace_pipeline_create) {
        std::cerr << "ggml_vulkan: create compute pipeline: device=" << device->name
                  << " name=" << pipeline->name << std::endl;
    }

    try {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Compute pipeline creation failed for " << pipeline->name << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
    pipeline->compiled = true;

    if (trace_pipeline_create) {
        std::cerr << "ggml_vulkan: create pipeline done: device=" << device->name
                  << " name=" << pipeline->name << std::endl;
    }

    if (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    if (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        auto statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        bool print_stats = !vk_pipeline_stats_filter.empty() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        if (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::endl;
        }

        for (auto & s : statistics) {
            if (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.data() << ": ";
                switch (s.format) {
                    case vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        break;
                    case vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        break;
                    case vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        break;
                    case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        break;
                }
                std::cerr << std::endl;
            }
            // "Register Count" is reported by NVIDIA drivers.
            if (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->register_count = (uint32_t)s.value.u64;
            }
        }
    }

    device->all_pipelines.push_back(pipeline);

    {
        std::lock_guard<std::mutex> guard(compile_count_mutex);
        assert(compile_count > 0);
        compile_count--;
    }
    compile_count_cond.notify_all();
}

static void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_set_requirements += n;
    if (!pipeline->compiled) {
        pipeline->needed = true;
        ggml_vk_load_shaders(ctx->device);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

static void ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    if (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
        // Enough descriptors are available
        return;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_set_requirements});
    uint32_t to_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_SIZE;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;

    while (to_alloc > 0) {
        const uint32_t alloc_count = std::min(pool_remaining, to_alloc);
        to_alloc -= alloc_count;
        pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE;

        if (pool_idx >= ctx->descriptor_pools.size()) {
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, MAX_PARAMETER_COUNT * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, descriptor_pool_size);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = device->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx], alloc_count, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
}

static vk_command_buffer* ggml_vk_create_cmd_buffer(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        p.pool,
        vk::CommandBufferLevel::ePrimary,
        1);
    const std::vector<vk::CommandBuffer> cmd_buffers = device->device.allocateCommandBuffers(command_buffer_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, true });
    return &p.cmd_buffers[p.cmd_buffers.size()-1];
}

static void ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    if (ctx->seqs.empty()) {
        if (fence) {
            std::lock_guard<std::mutex> guard(queue_mutex);
            ctx->p->q->queue.submit({}, fence);
        }
        return;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    int idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    for (const auto& sequence : ctx->seqs) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    for (const auto& sequence : ctx->seqs) {
        for (const auto& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            for (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            for (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].data(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].data(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].pNext = nullptr;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].data(),
                stage_flags[idx].data(),
                1,
                &submission.buffer->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].data(),
            };
            si.setPNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    std::lock_guard<std::mutex> guard(queue_mutex);
    ctx->p->q->queue.submit(submit_infos, fence);

    ctx->seqs.clear();
}

static uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, const vk::QueueFlags& required, const vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_num_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    const uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    for (uint32_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            return i;
        }
    }

    // Fall back to only required
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to reusing compute queue
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to ignoring min_num_queries
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    if (compute_index >= 0) {
        return compute_index;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::endl;

    for(auto &q_family : queue_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.queueCount) << " flags: " + to_string(q_family.queueFlags) << std::endl;
    }
    abort();
}

static void ggml_vk_create_queue(vk_device& device, vk_queue& q, uint32_t queue_family_index, uint32_t queue_index, vk::PipelineStageFlags&& stage_flags, bool transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    q.queue_family_index = queue_family_index;
    q.transfer_only = transfer_only;

    q.cmd_pool.init(device, &q);

    q.queue = device->device.getQueue(queue_family_index, queue_index);

    q.stage_flags = stage_flags;
}

static vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << result << ")");
    ctx->gc.contexts.emplace_back(result);
    result->p = &p;
    return result;
}

static vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << result << ")");
    result->p = &p;
    return result;
}

static vk_semaphore * ggml_vk_create_binary_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eBinary, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    return &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

static vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    if (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    return &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

static vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    if (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    return ctx->gc.events[ctx->event_idx++];
}

static void ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    for (auto& cmd_buffer : p.cmd_buffers) {
        cmd_buffer.in_use = false;
    }
}

static void ggml_vk_queue_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    static constexpr uint32_t cleanup_frequency = 10;

    if (device->compute_queue.cmd_pool.buffers_in_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_queue.cmd_pool);
    }
    if (device->transfer_queue.cmd_pool.buffers_in_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_queue.cmd_pool);
    }
}

static std::vector<uint32_t> ggml_vk_find_memory_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    return indices;
}

static float ggml_vk_memory_priority_for_allocation(size_t size) {
    const char * threshold_env = getenv("GGML_VK_MEMORY_PRIORITY_LARGE_THRESHOLD_MB");
    if (threshold_env == nullptr) {
        return 1.0f;
    }

    const long threshold_mb = strtol(threshold_env, nullptr, 10);
    if (threshold_mb <= 0) {
        return 1.0f;
    }

    const uint64_t threshold = (uint64_t) threshold_mb * 1024ull * 1024ull;
    if (size >= threshold) {
        return 1.0f;
    }

    const char * small_priority_env = getenv("GGML_VK_MEMORY_PRIORITY_SMALL");
    float small_priority = small_priority_env != nullptr ? (float) atof(small_priority_env) : 0.25f;
    small_priority = std::max(0.0f, std::min(1.0f, small_priority));
    return small_priority;
}

static vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, const std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       void *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    if (size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (size == 0) {
        buf->size = 0;
        return buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    if (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::ExternalMemoryBufferCreateInfo external_memory_bci;
    if (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    const float memory_priority = ggml_vk_memory_priority_for_allocation(size);
    const vk::MemoryPriorityAllocateInfoEXT mem_priority_info { memory_priority };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    if (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    if (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        try {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } catch (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            return {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        for (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            if (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }
            if (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            if ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                break;
            }
        }
        if (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            return {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        try {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } catch (const vk::SystemError& e) {
        }
    } else {
        for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            const auto & req_flags = *it;

            const std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            if (memory_type_indices.empty()) {
                continue;
            }
            buf->memory_property_flags = req_flags;

            bool done = false;

            for (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                try {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    done = true;
                    break;
                } catch (const vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    if (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        throw e;
                    }
                }
            }

            if (done) {
                break;
            }
        }
    }

    if (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        throw vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    if (import_ptr) {
        buf->ptr = import_ptr;
    } else {
        if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    if (device->buffer_device_address) {
        const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    return buf;
}

static vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0)) {
    try {
        return ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

static vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    try {
        if (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } else if (device->uma) {
            // Fall back to host memory type
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } else if (device->disable_host_visible_vidmem) {
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } else {
            // use rebar if available, otherwise fallback to device only visible memory
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    return buf;
}

static void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }

    if (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

static vk_subbuffer ggml_vk_subbuffer(const ggml_backend_vk_context* ctx, const vk_buffer& buf, size_t offset = 0) {
    return { buf, offset, ggml_vk_get_max_buffer_range(ctx, buf, offset) };
}

static void ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    const bool transfer_queue = subctx->p->q->transfer_only;

    if (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;
    }

    subctx->s->buffer->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) },
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) }
        } },
        {},
        {}
    );
}

static void ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->buffer->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

static void ggml_vk_set_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->buffer->buf.setEvent(
        event,
        ctx->p->q->stage_flags
    );
}

static void ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    if (events.empty()) {
        return;
    }

    ctx->s->buffer->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

struct vk_fa_tuning_params {
    FaCodePath path;
    uint32_t workgroup_size;
    uint32_t subgroup_size;
    uint32_t block_rows;
    uint32_t block_cols;
    uint32_t d_split;
    uint32_t row_split;
    bool shmem_staging;
    bool disable_subgroups;
    uint32_t limit_occupancy_shmem;

    void print() const {
        std::cerr << "path=" << path << " workgroup_size=" << workgroup_size << " subgroup_size=" << subgroup_size <<
                     " block_rows=" << block_rows << " block_cols=" << block_cols << " d_split=" << d_split <<
                     " row_split=" << row_split << " shmem_staging=" << shmem_staging << " disable_subgroups=" << disable_subgroups <<
                     " limit_occupancy_shmem=" << limit_occupancy_shmem << std::endl;
    }
};

static bool ggml_vk_flash_attn_scalar_shmem_support(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type kv_type);
static bool ggml_vk_flash_attn_coopmat_shmem_support(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc);

static vk_fa_tuning_params get_fa_tuning_params_scalar(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type kv_type, bool f32acc) {

    vk_fa_tuning_params result{};
    result.path = FA_SCALAR;

    if (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        result.subgroup_size = 32;
        result.disable_subgroups = true;
    } else if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        result.subgroup_size = n_rows < 4 ? 32 : device->subgroup_size;
    } else {
        result.subgroup_size = device->subgroup_size;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t row_split_max_hsk = 64;
    if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        row_split_max_hsk = n_rows <= 8 ? 64 : 128;
    }
    result.row_split = (n_rows < 4 || hsk <= row_split_max_hsk) ? 1 : 4;

    if (result.subgroup_size > 32 && (n_rows < 4 || hsk < (result.row_split == 1 ? 128 : 64))) {
        result.workgroup_size = result.subgroup_size * 2;
    } else {
        result.workgroup_size = result.subgroup_size * 4;
    }

    const uint32_t D = hsk | hsv;

    const bool reduce_block_rows = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    if (n_rows == 1) {
        result.block_rows = 1;
        result.block_cols = 64;
    } else {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        if (result.row_split == 1) {
            result.block_rows = n_rows == 2 ? 2 : ((n_rows <= 4 || reduce_block_rows) ? 4 : 8);
        } else {
            result.block_rows = n_rows <= 4 ? 4 : ((n_rows <= 8 || reduce_block_rows) ? 8 : 16);
        }

        result.block_cols = (D & 8) ? 64 : 32;
    }

    // Local fork (D094 cycle-4): scalar/MMQ FA geometry knobs for RDNA4 tuning.
    if (const char * env = getenv("GGML_VK_FA_SCALAR_BC")) {
        result.block_cols = (uint32_t) std::max(1, std::atoi(env));
    }
    if (const char * env = getenv("GGML_VK_FA_SCALAR_RS")) {
        result.row_split = (uint32_t) std::max(1, std::atoi(env));
    }

    const uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    result.d_split = std::min(std::min(result.subgroup_size, 8u), D_lsb / 4);

    if (const char * env = getenv("GGML_VK_FA_SCALAR_DSPLIT")) {
        result.d_split = (uint32_t) std::max(1, std::atoi(env));
    }

    result.shmem_staging = ((device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) || kv_type == GGML_TYPE_F8_E4M3) ? 1 : 0;

    // Local fork (D096 D2): single-row (decode) f8 FA reads K/V directly -
    // no LDS staging. At n_rows==1 the K/V block is never reused, so staging
    // is a pure LDS round-trip; the shader dequantizes f8 in registers
    // instead (dequantize4, SHMEM_STAGING==0 path). Measured 2026-08-12:
    // +3.8% decode at 12K lane, parity at 49K lane. GGML_VK_FA_F8_DIRECT
    // forces direct for all n_rows; GGML_VK_FA_SCALAR_STAGING forces staging.
    if (kv_type == GGML_TYPE_F8_E4M3 && (n_rows == 1 || getenv("GGML_VK_FA_F8_DIRECT"))) {
        result.shmem_staging = 0;
    }

    // Local fork (D094 cycle-4): scalar/MMQ FA: stage K/V tiles through LDS
    // (avoids per-element on-the-fly dequantize4 in the hot loop).
    if (getenv("GGML_VK_FA_SCALAR_STAGING")) {
        result.shmem_staging = 1;
    }

    // Local fork (D094 cycle-4): scalar/MMQ FA overrides (applied last so they
    // win over the auto-tuning above). LDS overflow would fail pipeline
    // creation, which is caught and reverted per run.
    if (const char * env = getenv("GGML_VK_FA_SCALAR_BR")) {
        result.block_rows = (uint32_t) std::max(1, std::atoi(env));
    }
    if (const char * env = getenv("GGML_VK_FA_SCALAR_SUBGROUP")) {
        result.subgroup_size = (uint32_t) std::max(1, std::atoi(env));
        result.workgroup_size = result.subgroup_size * 4;
    }

    if (!reduce_block_rows && !ggml_vk_flash_attn_scalar_shmem_support(device, result, hsk, hsv, f32acc, kv_type)) {
        result.block_rows /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
        if (device->architecture != AMD_GCN && n_rows >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            result.limit_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } else if (device->architecture == AMD_GCN && n_rows <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            result.limit_occupancy_shmem = (n_rows < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    return result;
}

static vk_fa_tuning_params get_fa_tuning_params_coopmat1(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type kv_type, bool f32acc) {
    GGML_UNUSED(n_rows);
    GGML_UNUSED(n_kv);
    GGML_UNUSED(kv_type);
    GGML_UNUSED(f32acc);

    vk_fa_tuning_params result{};
    result.path = FA_COOPMAT1;

    const uint32_t D = hsk | hsv;

    const uint32_t coopmat_block_rows = 16;
    const uint32_t coopmat_block_cols = 16;

    const uint32_t num_subgroups = 4;

    result.block_rows = coopmat_block_rows;
    result.block_cols = coopmat_block_cols * num_subgroups;
    result.row_split = num_subgroups;
    result.subgroup_size = device->subgroup_size;
    result.workgroup_size = num_subgroups * result.subgroup_size;

    const uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    result.d_split = std::min(std::min(result.subgroup_size, 8u), D_lsb / 4);

    // D096 P1: the canonical DATA_A_F8_E4M3 cm1 shader already supports
    // tile-local K/V dequantization when SHMEM_STAGING=0. Keep the historical
    // full-tile staging default, but allow the direct-f8 experiment to avoid
    // exceeding RDNA4's 32 KiB LDS limit without maintaining a second FA kernel.
    const bool f8_direct = kv_type == GGML_TYPE_F8_E4M3 && getenv("GGML_VK_FA_F8_DIRECT") != nullptr;
    result.shmem_staging = ((device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ||
                            (kv_type == GGML_TYPE_F8_E4M3 && !f8_direct)) ? 1 : 0;

    // Diagnostic knobs for coopmat1 FA geometry experiments (local fork).
    // The shmem gate in ggml_vk_flash_attn() still applies, so unsupported
    // combinations fall back to scalar FA instead of producing broken shaders.
    if (const char * env = getenv("GGML_VK_FA_BLOCK_ROWS")) {
        result.block_rows = std::max(coopmat_block_rows, (uint32_t) std::atoi(env));
    }
    if (const char * env = getenv("GGML_VK_FA_BLOCK_COLS")) {
        result.block_cols = std::max(coopmat_block_cols, (uint32_t) std::atoi(env));
    }
    if (const char * env = getenv("GGML_VK_FA_ROW_SPLIT")) {
        const uint32_t row_split = std::max(1u, (uint32_t) std::atoi(env));
        // rows_per_thread and the shmem layout both derive from row_split, so
        // only accept power-of-two splits that evenly divide block_rows.
        // row_split > 4 (512-thread workgroups) crashed on the RX 9070 XT
        // Windows driver during prefill and is intentionally capped here.
        if (row_split <= 4 && row_split <= result.block_rows && (result.block_rows % row_split) == 0) {
            result.row_split = row_split;
            result.workgroup_size = row_split * result.subgroup_size;
        }
    }

    return result;
}

static vk_fa_tuning_params get_fa_tuning_params_coopmat2(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    GGML_UNUSED(n_kv);
    GGML_UNUSED(f32acc);

    vk_fa_tuning_params result{};
    result.path = FA_COOPMAT2;

    const uint32_t D = hsk | hsv;

    const bool small_rows = n_rows < 32;

    if (small_rows) {
        result.block_rows = 32;
        result.block_cols = 32;
    } else if (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        result.block_rows = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        result.block_cols = 32;
    } else {
        result.block_rows = 64;
        result.block_cols = 64;
    }

    result.subgroup_size = device->subgroup_size;
    result.workgroup_size = (small_rows && (D % 32) == 0) ? 256 : 128;

    return result;
}

static vk_fa_tuning_params get_fa_tuning_params(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    // Mixed K/V is only implemented on the coopmat2 (flash_attn_cm2) path; never use scalar/cm1.
    if (k_type != v_type) {
        GGML_ASSERT(device->coopmat2);
        return get_fa_tuning_params_coopmat2(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    }

    // D096 P5: same fused-Q/PV fp8 transforms as P4, but V is read raw f8
    // (fp8 P*V, f32 accumulator) and there is no f16 preconvert pass.
    if (k_type == GGML_TYPE_F8_E4M3 && v_type == k_type && hsk == 256 && f32acc && n_rows > 8 &&
        device->coopmat1_fa_support && device->coopmat_support_16x16x16_f32acc &&
        getenv("GGML_VK_FA_F8_P5") != nullptr) {
        vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
        if (params.block_rows == 16) {
            params.path = FA_F8_P5;
            params.shmem_staging = 0;
            return params;
        }
    }

    // D096 P4: same fused-Q transform as P3, but V comes from a dense f16
    // preconvert buffer so the canonical PV path needs no tile-local dequant.
    if (k_type == GGML_TYPE_F8_E4M3 && v_type == k_type && hsk == 256 && f32acc && n_rows > 8 &&
        device->coopmat1_fa_support && device->coopmat_support_16x16x16_f32acc &&
        getenv("GGML_VK_FA_F8_P5") == nullptr && getenv("GGML_VK_FA_F8_P4") != nullptr) {
        vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
        if (params.block_rows == 16) {
            params.path = FA_F8_P4;
            params.shmem_staging = 0;
            return params;
        }
    }

    // D096 P3: the same generated S-only fp8 transform as P2, but Q is
    // quantized into shared E4M3 inside the canonical FA workgroup. The fixed
    // 4096-byte Q tile ABI is deliberately shape-gated to Br=16, HSK=256.
    if (k_type == GGML_TYPE_F8_E4M3 && v_type == k_type && hsk == 256 && f32acc && n_rows > 8 &&
        device->coopmat1_fa_support && device->coopmat_support_16x16x16_f32acc &&
        getenv("GGML_VK_FA_F8_P5") == nullptr && getenv("GGML_VK_FA_F8_P4") == nullptr && getenv("GGML_VK_FA_F8_P3") != nullptr) {
        vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
        if (params.block_rows == 16) {
            params.path = FA_F8_P3;
            params.shmem_staging = 0;
            return params;
        }
    }

    // D096 P2: generated canonical cm1 with only K*Q->S transformed to fp8
    // cooperative matrices. Restrict to ordinary prefill; short GQA requests
    // keep the canonical route and are not part of the tile-scale ABI.
    if (k_type == GGML_TYPE_F8_E4M3 && v_type == k_type && f32acc && n_rows > 8 &&
        device->coopmat1_fa_support && device->coopmat_support_16x16x16_f32acc &&
        getenv("GGML_VK_FA_F8_P3") == nullptr && getenv("GGML_VK_FA_F8_P2") != nullptr) {
        vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
        params.path = FA_F8_P2;
        params.shmem_staging = 0;
        return params;
    }

    // D096-A2 (local fork): fp8 native FA kernel - hand-written SPIR-V assembly
    // (fp8 coopmat reads K/V directly from the f8 KV cache; no f16 preconvert,
    // no shmem K staging). f32acc only; opt-in via GGML_VK_FA_F8_NATIVE.
    if (k_type == GGML_TYPE_F8_E4M3 && v_type == k_type && f32acc && n_rows > 1 &&
        device->coopmat1_fa_support && device->coopmat_support_16x16x16_f32acc &&
        getenv("GGML_VK_FA_F8_NATIVE") != nullptr) {
        vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
        params.path = FA_F8_NATIVE;
        params.shmem_staging = 0; // native kernel reads K directly from the f8 KV cache
        return params;
    }

    // D096 D3 (probe): native fp8 coopmat on single-row decode (n_rows == 1)
    // under GGML_VK_FA_F8_NATIVE_DECODE; measured against scalar on 49K lane.
    if (k_type == GGML_TYPE_F8_E4M3 && v_type == k_type && f32acc && n_rows == 1 &&
        device->coopmat1_fa_support && device->coopmat_support_16x16x16_f32acc &&
        getenv("GGML_VK_FA_F8_NATIVE") != nullptr && getenv("GGML_VK_FA_F8_NATIVE_DECODE") != nullptr) {
        vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
        params.path = FA_F8_NATIVE;
        params.shmem_staging = 0; // native kernel reads K directly from the f8 KV cache
        return params;
    }

    FaCodePath path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    // Local fork (D094 cycle-4): A/B the scalar FA path (flash_attn.comp with
    // MMQ int8 dot products) against coopmat1 on RDNA4. Matmul uses the same
    // MMQ-style pipeline at 55-65 TFLOPS while coopmat FA sits at ~16.
    if (getenv("GGML_VK_FA_FORCE_SCALAR")) {
        path = FA_SCALAR;
    }

    if (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    if (path == FA_COOPMAT1) {
        bool shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        const vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
        bool shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc);

        if (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    if (n_rows == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    // Q1_0 K/V is only implemented on coopmat2 (flash_attn_cm2); there is no scalar FA shader for it.
    if ((k_type == GGML_TYPE_Q1_0 || v_type == GGML_TYPE_Q1_0) && device->coopmat2) {
        path = FA_COOPMAT2;
    }

    switch (path) {
    case FA_SCALAR:
        return get_fa_tuning_params_scalar(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
    case FA_COOPMAT1:
        return get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, f32acc);
    case FA_COOPMAT2:
        return get_fa_tuning_params_coopmat2(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    default:
        throw std::runtime_error("unsupported FaCodePath");
    }
}

static vk_fa_pipeline_state get_fa_pipeline_state(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc,
                                                  bool use_mask, bool use_mask_opt, bool use_logit_softcap, ggml_type k_type, ggml_type v_type) {
    const bool old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_mask_opt      ? 1 : 0) |
                     (use_mask          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0);

    // Local fork (D094 cycle-4): diagnostic - skip causal-mask work on the
    // scalar FA path (incorrect attention; timing diagnostics only).
    if (params.path == FA_SCALAR && getenv("GGML_VK_FA_SCALAR_NOMASK")) {
        flags &= ~3u;
    }

    const uint32_t subgroup_size = params.disable_subgroups ? 0 : params.subgroup_size;

    return vk_fa_pipeline_state{hsk, hsv, params.block_rows, params.block_cols, params.d_split, params.row_split, params.shmem_staging, params.path, params.workgroup_size, subgroup_size, aligned, f32acc, flags, params.limit_occupancy_shmem, k_type, v_type};
}

static std::vector<uint32_t> get_fa_spec_constants(const vk_fa_pipeline_state& state) {
    const auto fa_block_bytes = [](ggml_type t) -> uint32_t {
        // decodeBufF32 uses a block of vec4s for a better memory access pattern.
        return t == GGML_TYPE_F32 ? 16u : (uint32_t) ggml_vk_device_type_size(t);
    };
    return {
        /* 0 WorkGroupSize   */ state.workgroup_size,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_split,
        /* 7 row_split       */ state.row_split,
        /* 8 SubGroupSize    */ state.subgroup_size,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.limit_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_block_bytes(state.k_type),
        /*15 FaBlockBytesV   */ fa_block_bytes(state.v_type),
    };
}

static bool ggml_vk_matmul_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    uint32_t lut_size = 0;
    switch (src0_type) {
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ1_M:
        lut_size = 2*2048 + 4*2048;
        break;
    case GGML_TYPE_IQ2_XXS:
        lut_size = 8*256;
        break;
    case GGML_TYPE_IQ2_XS:
        lut_size = 8*512;
        break;
    case GGML_TYPE_IQ2_S:
        lut_size = 8*1024;
        break;
    case GGML_TYPE_IQ3_XXS:
        lut_size = 4*256;
        break;
    case GGML_TYPE_IQ3_S:
        lut_size = 4*512;
        break;
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_MXFP4:
        lut_size = 4*16;
        break;
    case GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_size = 4*16 + 128u * (uint32_t)sizeof(float);
        break;
    default:
        break;
    }

    // Needs to be kept up to date on shader changes
    const uint32_t bank_conflict_offset = device->coopmat_support ? 8 : 1;
    const uint32_t type_size = device->fp16 ? sizeof(ggml_fp16_t) : sizeof(float);
    const uint32_t warps = warptile[0] / warptile[10];

    const uint32_t load_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_offset) * type_size;
    const uint32_t mmid_row_ids = mul_mat_id ? (warptile[2] * 2 * sizeof(uint16_t)) : 0;
    const uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * sizeof(float) : 0;
    const uint32_t ballots_sh = mul_mat_id ? (warps * 4 * sizeof(uint32_t)) : 0;

    const uint32_t total_size = load_bufs + mmid_row_ids + coopmat_stage + lut_size + ballots_sh;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    return supported;
}

static bool ggml_vk_matmul_warptile_layout_valid(const std::vector<uint32_t>& warptile) {
    GGML_ASSERT(warptile.size() >= 11);
    const uint32_t bm      = warptile[1];
    const uint32_t bn      = warptile[2];
    const uint32_t wm      = warptile[4];
    const uint32_t wn      = warptile[5];
    const uint32_t tm      = warptile[7];
    const uint32_t tn      = warptile[8];
    const uint32_t warp    = warptile[10];

    if (bm == 0 || bn == 0 || wm == 0 || wn == 0 || tm == 0 || tn == 0 || warp == 0) {
        return false;
    }
    if (bm % wm != 0 || bn % wn != 0 || wm % tm != 0 || wn % tn != 0) {
        return false;
    }

    const uint32_t storestride = warp / tm;
    if (storestride == 0 || warp % tm != 0 || storestride > tn || tn % storestride != 0) {
        return false;
    }
    return true;
}

static bool ggml_vk_matmul_prepare_variant_warptile(const vk_device& device, std::vector<uint32_t>& warptile) {
    GGML_ASSERT(warptile.size() >= 11);
    if (!ggml_vk_matmul_warptile_layout_valid(warptile)) {
        return false;
    }

    const uint64_t block_size = uint64_t(warptile[10]) * (warptile[1] / warptile[4]) * (warptile[2] / warptile[5]);
    if (block_size > device->properties.limits.maxComputeWorkGroupInvocations) {
        return false;
    }

    if (warptile[0] != block_size) {
        VK_LOG_DEBUG("ggml_vulkan: retile MMQ workgroup size " << warptile[0] << " -> " << block_size);
        warptile[0] = uint32_t(block_size);
    }

    return true;
}

struct GpuPipelineConfig {
    // GPU architecture identifier.
    // Example: vk_device_architecture::AMD_GCN
    vk_device_architecture arch;

    // Mapping of pipeline names to their specific subgroup sizes.
    // Example: {"soft_max_f32", 64}
    std::unordered_map<std::string, uint32_t> pipelines;

    // Default subgroup size for this GPU.
    // Defaults to 0 if not explicitly provided.
    uint32_t default_subgroup_size = 0;
};

// Pipeline configuration for RDNA1 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

// Pipeline configuration for RDNA2 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

static constexpr uint32_t RDNA_DEFAULT_SUBGROUP_SIZE = 32;

// Define configurations for different GPUs.
static std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
};

static uint32_t get_subgroup_size(const std::string &pipeline_name, const vk_device_architecture &arch) {
    for (const auto &config : gpu_pipeline_configs) {
        if (config.arch == arch) {
            auto pipIt = config.pipelines.find(pipeline_name);
            if (pipIt != config.pipelines.end()) {
                return pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
            for (const auto &entry : sorted_pipelines) {
                if (pipeline_name.find(entry.first) != std::string::npos) {
                    return entry.second;
                }
            }
            return config.default_subgroup_size;
        }
    }
    return 0; // If no matching configuration is found
}
