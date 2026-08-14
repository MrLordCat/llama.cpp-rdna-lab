// Graph recording, command submission boundaries, preallocation, and cleanup.
static void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#if defined(GGML_VULKAN_RUN_TESTS)
    const std::vector<size_t> vals {
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
        8, 8, 8,
        100, 46, 576,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
    };
    const size_t num_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q4_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q4_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q4_0, true);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q8_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q8_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q8_0, true);

    abort();

    for (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2);
        std::cerr << '\n' << std::endl;

        if (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::endl;
        }

        if (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::endl;
        }
    }

    GGML_ABORT("fatal error");
#endif

    if (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = true;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    if (ctx->prealloc_x == nullptr || (ctx->prealloc_size_x > 0 && ctx->prealloc_x->size < ctx->prealloc_size_x)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_size_x << ")");
        // Resize buffer
        if (ctx->prealloc_x != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_x);
    }
    if (ctx->prealloc_y == nullptr || (ctx->prealloc_size_y > 0 && ctx->prealloc_y->size < ctx->prealloc_size_y)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_size_y << ")");
        // Resize buffer
        if (ctx->prealloc_y != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_y);
        ctx->prealloc_y_last_tensor_used = nullptr;
    }
    if (ctx->prealloc_split_k == nullptr || (ctx->prealloc_size_split_k > 0 && ctx->prealloc_split_k->size < ctx->prealloc_size_split_k)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_size_split_k << ")");
        // Resize buffer
        if (ctx->prealloc_split_k != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_split_k);
        }
        ctx->prealloc_split_k = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_split_k);
    }
    if (ctx->prealloc_fa_k16 == nullptr || (ctx->prealloc_size_fa_k16 > 0 && ctx->prealloc_fa_k16->size < ctx->prealloc_size_fa_k16)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(fa_k16_size: " << ctx->prealloc_size_fa_k16 << ")");
        if (ctx->prealloc_fa_k16 != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_fa_k16);
        }
        ctx->prealloc_fa_k16 = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_fa_k16);
    }
    if (ctx->prealloc_fa_v16 == nullptr || (ctx->prealloc_size_fa_v16 > 0 && ctx->prealloc_fa_v16->size < ctx->prealloc_size_fa_v16)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(fa_v16_size: " << ctx->prealloc_size_fa_v16 << ")");
        if (ctx->prealloc_fa_v16 != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_fa_v16);
        }
        ctx->prealloc_fa_v16 = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_fa_v16);
    }
    if (ctx->prealloc_fa_mask16 == nullptr || (ctx->prealloc_size_fa_mask16 > 0 && ctx->prealloc_fa_mask16->size < ctx->prealloc_size_fa_mask16)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(fa_mask16_size: " << ctx->prealloc_size_fa_mask16 << ")");
        if (ctx->prealloc_fa_mask16 != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_fa_mask16);
        }
        ctx->prealloc_fa_mask16 = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_fa_mask16);
    }
    if (ctx->prealloc_add_rms_partials == nullptr || (ctx->prealloc_size_add_rms_partials > 0 && ctx->prealloc_add_rms_partials->size < ctx->prealloc_size_add_rms_partials)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_add_rms_partials << ")");
        // Resize buffer
        if (ctx->prealloc_add_rms_partials != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
        }
        ctx->prealloc_add_rms_partials = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_add_rms_partials);
    }
}

static void ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, int tensor_idx, bool almost_ready);

// Returns true if node has enqueued work into the queue, false otherwise
// If submit is true the current all operations queued so far are being submitted to Vulkan to overlap cmdlist creation and GPU execution.
static bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int node_idx, ggml_tensor *node_begin, int node_idx_begin, bool last_node, bool almost_ready, bool submit){
    ggml_tensor * node = cgraph->nodes[node_idx];
    if (ggml_is_empty(node) || ggml_op_is_empty(node->op) || !node->buffer) {
        return false;
    }
    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << node << ", " << ggml_op_name(node->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * src2 = node->src[2];
    ggml_tensor * src3 = node->src[3];

    if (node->op == GGML_OP_ADD) {
        int next_node_idx = node_idx + 1 + ctx->num_additional_fused_ops;
        if (next_node_idx < cgraph->n_nodes &&
            cgraph->nodes[next_node_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[next_node_idx]->src[0] == cgraph->nodes[next_node_idx - 1] &&
            ggml_nrows(cgraph->nodes[next_node_idx]) == 1 &&
            ctx->device->add_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_size(ctx, cgraph->nodes[node_idx]);
            ctx->do_add_rms_partials_offset_calculation = true;
            if (ctx->prealloc_size_add_rms_partials_offset + size <= ctx->prealloc_size_add_rms_partials) {
                ctx->do_add_rms_partials = true;
            }
        }
    }

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    {
        // This logic detects dependencies between modes in the graph and calls ggml_vk_sync_buffers
        // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
        // there is no auxiliary memory use, it shouldn't be necessary to call ggml_vk_sync_buffers
        // outside of this logic. When a node uses one of the prealloc buffers for something like
        // dequantization or split_k, additional synchronization is needed between those passes.
        bool need_sync = false;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        auto const &overlaps_unsynced = [&](const ggml_tensor *node, const std::vector<const ggml_tensor *> &unsynced_nodes) -> bool {
            if (unsynced_nodes.size() == 0) {
                return false;
            }
            auto tensor_storage = [&](const ggml_tensor * tensor, vk_buffer & buffer, size_t & base, size_t & size) {
                buffer = nullptr;
                base = 0;
                size = ggml_vk_device_size(tensor);

                if (ggml_backend_buffer_is_host(tensor->buffer)) {
                    ggml_vk_host_get(ctx->device, tensor->data, buffer, base);
                    return;
                }

                ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;
                buffer = buf_ctx->dev_buffer;
                base = vk_tensor_view_offset(tensor);
            };

            size_t n_base = 0;
            size_t n_size = 0;
            vk_buffer a_buf = nullptr;
            tensor_storage(node, a_buf, n_base, n_size);
            if (!a_buf) {
                return true;
            }
            for (auto &other : unsynced_nodes) {
                size_t o_base = 0;
                size_t o_size = 0;
                vk_buffer o_buf = nullptr;
                tensor_storage(other, o_buf, o_base, o_size);
                if (!o_buf) {
                    return true;
                }
                if (a_buf == o_buf) {
                    if ((o_base <= n_base && n_base < o_base + o_size) ||
                        (n_base <= o_base && o_base < n_base + n_size)) {
                        return true;
                    }
                }
            }
            return false;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1 && !need_sync; ++i) {
            const ggml_tensor *cur_node = cgraph->nodes[node_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            if (ctx->fused_ops_write_mask & (1 << i)) {
                if (overlaps_unsynced(cur_node, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_node, ctx->unsynced_nodes_written)) {
                    need_sync = true;
                    break;
                }
            }
            for (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                if (!cur_node->src[j]) {
                    continue;
                }
                if (overlaps_unsynced(cur_node->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = true;
                    break;
                }
            }
        }

        if (need_sync) {
            if (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::endl;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            if (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_node_idx[ctx->query_idx] = node_idx;
                compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
            const ggml_tensor *cur_node = cgraph->nodes[node_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            if (ctx->fused_ops_write_mask & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_node);
            }
            for (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                if (!cur_node->src[j]) {
                    continue;
                }
                ctx->unsynced_nodes_read.push_back(cur_node->src[j]);
            }
        }
    }
    if (vk_enable_sync_logger) {
        for (int i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
            auto *n = cgraph->nodes[node_idx + i];
            std::cerr << node_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            if (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            if (n->op == GGML_OP_ROPE) {
                const int mode = ((const int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::endl;
        }
    }

    switch (node->op) {
    case GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_REPEAT_BACK:
        ggml_vk_repeat_back(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ACC:
    case GGML_OP_SET:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_GET_ROWS:
        ggml_vk_get_rows(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ADD:
        if (ctx->num_additional_fused_ops) {
            ggml_vk_multi_add(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_add(ctx, compute_ctx, src0, src1, node);
        }
        break;
    case GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_MUL:
        ggml_vk_mul(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ADD_ID:
        ggml_vk_add_id(ctx, compute_ctx, src0, src1, src2, node);

        break;
    case GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, node);

        break;
    case GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, node);

        break;
    case GGML_OP_SCALE:
        ggml_vk_scale(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SET_ROWS:
        ggml_vk_set_rows(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_SILU_BACK:
        ggml_vk_silu_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_GROUP_NORM:
        ggml_vk_group_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, node_idx, (float *)node->op_params);
        break;
    case GGML_OP_RMS_NORM_BACK:
        ggml_vk_rms_norm_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_UNARY:
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
            break;
        }

        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
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
            ggml_vk_unary(ctx, compute_ctx, src0, node);
            break;
        case GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, node);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(node)) {
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, node);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_DIAG_MASK_INF:
        ggml_vk_diag_mask_inf(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SOFT_MAX:
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, node);
        }

        break;
    case GGML_OP_SOFT_MAX_BACK:
        ggml_vk_soft_max_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, node_idx, false);

        break;
    case GGML_OP_ROPE_BACK:
        ggml_vk_rope(ctx, compute_ctx, cgraph, node_idx, true);

        break;
    case GGML_OP_ARGSORT:
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_argsort(ctx, compute_ctx, src0, node);
        }

        break;
    case GGML_OP_TOP_K:
        ggml_vk_topk(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SUM:
        ggml_vk_sum(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SUM_ROWS:
        ggml_vk_sum_rows(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_MEAN:
        ggml_vk_mean(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_COUNT_EQUAL:
        ggml_vk_count_equal(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_IM2COL:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_IM2COL_3D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, node_idx);

        break;
    case GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, node_idx);

        break;

    case GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, node->src[4], node);

        break;

    case GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, node);

        break;

    case GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, node);

        break;

    case GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_SCAN:
        ggml_vk_ssm_scan(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, node);

        break;

    case GGML_OP_OPT_STEP_ADAMW:
        ggml_vk_opt_step_adamw(ctx, compute_ctx, node);

        break;

    case GGML_OP_OPT_STEP_SGD:
        ggml_vk_opt_step_sgd(ctx, compute_ctx, src0, src1, src2, node);

        break;
    default:
        return false;
    }

    ctx->tensor_ctxs[node_idx] = compute_ctx;

#if defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    last_node = true;
#endif

    if (submit || last_node) {
        ggml_vk_ctx_end(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        if (last_node) {
            compute_ctx->exit_tensor_idx = node_idx_begin;
        }
        else {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, node_begin, node_idx_begin, almost_ready);
    }
    return true;
}

static void ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, int tensor_idx, bool almost_ready = false) {
    GGML_UNUSED(cgraph);
    GGML_UNUSED(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();

    // Only run if ctx hasn't been submitted yet
    if (!subctx->seqs.empty()) {
#ifdef GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#endif

        // Do staging buffer copies
        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        for (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        if (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = true;
        } else {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = true;

#ifdef GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#endif
    }

    if (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        subctx->in_memcpys.clear();
        subctx->out_memcpys.clear();
        subctx->memsets.clear();
    }
}

// Clean up after graph processing is done
static void ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_last_pipeline_used = {};

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;
    ctx->prealloc_fa_k16_need_sync = ctx->prealloc_fa_v16_need_sync = ctx->prealloc_fa_mask16_need_sync = false;
    ctx->prealloc_size_fa_k16 = 0;
    ctx->prealloc_size_fa_v16 = 0;
    ctx->prealloc_size_fa_mask16 = 0;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    if (ctx->device->async_use_transfer_queue) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    for (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    for (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
}

// Clean up on backend free
static void ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_buffer(ctx->prealloc_x);
    ggml_vk_destroy_buffer(ctx->prealloc_fa_k16);
    ggml_vk_destroy_buffer(ctx->prealloc_fa_v16);
    ggml_vk_destroy_buffer(ctx->prealloc_fa_mask16);
    ggml_vk_destroy_buffer(ctx->prealloc_y);
    ggml_vk_destroy_buffer(ctx->prealloc_split_k);
    ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
    ggml_vk_destroy_buffer(ctx->sync_staging);

    ctx->prealloc_y_last_pipeline_used = nullptr;

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    for (auto& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    if (ctx->device->async_use_transfer_queue) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    if (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(true);
    }
}

static int ggml_vk_get_device_count() {
    ggml_vk_instance_init();

    return vk_instance.device_indices.size();
}

static void ggml_vk_get_device_description(int device, char * description, size_t description_size) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_size, "%s", props.deviceName.data());
}
