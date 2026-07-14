// Backend synchronization, graph execution, fusion, optimization, and events.
static void ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    bool do_transfer = !ctx->compute_ctx.expired();

    if (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = true;
    }

    vk_context compute_ctx;
    vk_command_buffer* cmd_buf = nullptr;
    if (do_transfer) {
        compute_ctx = ctx->compute_ctx.lock();
        if (compute_ctx->s) {
            cmd_buf = compute_ctx->s->buffer;
        }

        ggml_vk_ctx_end(compute_ctx);

        for (auto& cpy : compute_ctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ggml_vk_submit(compute_ctx, {});
        ctx->submit_pending = true;
    }

    if (ctx->submit_pending) {
        if (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_last_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, nullptr,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_queue.stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, nullptr,
                0, nullptr,
            };
            si.setPNext(&tl_info);
            std::lock_guard<std::mutex> guard(queue_mutex);
            ctx->device->compute_queue.queue.submit({ si }, ctx->fence);
            ctx->transfer_semaphore_last_submitted = ctx->transfer_semaphore.value;
        } else {
            std::lock_guard<std::mutex> guard(queue_mutex);
            ctx->device->compute_queue.queue.submit({}, ctx->fence);
        }
        ggml_vk_wait_for_fence(ctx);
        ctx->submit_pending = false;
        if (cmd_buf) {
            cmd_buf->in_use = false;
            cmd_buf->buf.reset();
        }
    }

    if (do_transfer) {
        for (auto& cpy : compute_ctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

static void ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);
}

static bool ggml_vk_is_empty(ggml_tensor * node) {
    return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
}

static bool ggml_vk_is_dense_ffn_glu_candidate(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor ** up, const ggml_tensor ** gate, const ggml_tensor ** glu) {
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, { node_idx + 2 })) {
        return false;
    }

    const ggml_tensor * mm0 = cgraph->nodes[node_idx];
    const ggml_tensor * mm1 = cgraph->nodes[node_idx + 1];
    const ggml_tensor * cur_glu = cgraph->nodes[node_idx + 2];

    if (cur_glu->src[0] == mm0 && cur_glu->src[1] == mm1) {
        *gate = mm0;
        *up = mm1;
    } else if (cur_glu->src[0] == mm1 && cur_glu->src[1] == mm0) {
        *gate = mm1;
        *up = mm0;
    } else {
        return false;
    }

    if ((*up)->src[0]->type != (*gate)->src[0]->type ||
        !ggml_are_same_shape((*up)->src[0], (*gate)->src[0]) ||
        !ggml_are_same_stride((*up)->src[0], (*gate)->src[0])) {
        return false;
    }

    if ((*up)->src[1] != (*gate)->src[1]) {
        return false;
    }

    if ((*up)->src[2] != (*gate)->src[2]) {
        return false;
    }

    const ggml_glu_op glu_op = ggml_get_glu_op(cur_glu);
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU && glu_op != GGML_GLU_OP_SWIGLU_OAI) {
        return false;
    }

    if (ggml_get_op_params_i32(cur_glu, 1) != 0) {
        return false;
    }

    *glu = cur_glu;
    return true;
}

static bool ggml_vk_is_dense_ffn_block_candidate(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor ** up, const ggml_tensor ** gate, const ggml_tensor ** glu, const ggml_tensor ** down) {
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT }, { node_idx + 3 })) {
        return false;
    }

    if (!ggml_vk_is_dense_ffn_glu_candidate(cgraph, node_idx, up, gate, glu)) {
        return false;
    }

    const ggml_tensor * cur_down = cgraph->nodes[node_idx + 3];
    if (cur_down->src[1] != *glu) {
        return false;
    }

    if (cur_down->src[0]->type != (*up)->src[0]->type) {
        return false;
    }

    *down = cur_down;
    return true;
}

static bool ggml_vk_is_dense_ffn_view_block_candidate(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * up, const ggml_tensor * glu, const ggml_tensor ** down, int * view_count, bool * view_hazard, bool * fuse_reject) {
    int block_node_idxs[8];
    ggml_op block_ops[8];
    int block_count = 0;

    block_node_idxs[block_count] = node_idx;
    block_ops[block_count++] = GGML_OP_MUL_MAT;
    block_node_idxs[block_count] = node_idx + 1;
    block_ops[block_count++] = GGML_OP_MUL_MAT;
    block_node_idxs[block_count] = node_idx + 2;
    block_ops[block_count++] = GGML_OP_GLU;

    const ggml_tensor * activation = glu;
    int next_node_idx = node_idx + 3;
    *view_count = 0;
    *view_hazard = false;
    *fuse_reject = false;

    while (next_node_idx < cgraph->n_nodes && cgraph->nodes[next_node_idx]->op == GGML_OP_VIEW && cgraph->nodes[next_node_idx]->src[0] == activation && *view_count < 4) {
        const ggml_tensor * view = cgraph->nodes[next_node_idx];
        if (view->type != activation->type || ggml_nelements(view) != ggml_nelements(activation) || view->view_offs != 0) {
            *view_hazard = true;
            return false;
        }

        block_node_idxs[block_count] = next_node_idx;
        block_ops[block_count++] = GGML_OP_VIEW;
        activation = view;
        (*view_count)++;
        next_node_idx++;
    }

    if (*view_count == 0 || next_node_idx >= cgraph->n_nodes || cgraph->nodes[next_node_idx]->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const ggml_tensor * cur_down = cgraph->nodes[next_node_idx];
    if (cur_down->src[1] != activation || cur_down->src[0]->type != up->src[0]->type) {
        return false;
    }

    block_node_idxs[block_count] = next_node_idx;
    block_ops[block_count++] = GGML_OP_MUL_MAT;
    const int output_idx = next_node_idx;
    if (!ggml_can_fuse_subgraph_ext(cgraph, block_node_idxs, block_count, block_ops, &output_idx, 1)) {
        *fuse_reject = true;
        return false;
    }

    *down = cur_down;
    return true;
}

static int ggml_vk_find_dense_ffn_down_node(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * up, const ggml_tensor * glu, int search_limit) {
    const int search_end = std::min(cgraph->n_nodes, node_idx + search_limit);
    for (int search_idx = node_idx + 3; search_idx < search_end; ++search_idx) {
        const ggml_tensor * candidate = cgraph->nodes[search_idx];
        if (candidate->op != GGML_OP_MUL_MAT) {
            continue;
        }
        if (candidate->src[1] != glu) {
            continue;
        }
        if (candidate->src[0]->type != up->src[0]->type) {
            continue;
        }
        return search_idx;
    }

    return -1;
}

static bool ggml_vk_can_fuse_dense_ffn_nonadjacent_block(const struct ggml_cgraph * cgraph, int node_idx, int down_idx) {
    int block_node_idxs[4] = { node_idx, node_idx + 1, node_idx + 2, down_idx };
    ggml_op block_ops[4] = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT };
    return ggml_can_fuse_subgraph_ext(cgraph, block_node_idxs, 4, block_ops, &down_idx, 1);
}

static void ggml_vk_trace_dense_ffn_glu_candidates(const struct ggml_cgraph * cgraph) {
    int candidates = 0;
    int block_candidates = 0;
    int view_block_candidates = 0;
    int scan_block_candidates = 0;
    int scan_block_misses = 0;
    int scan_block_fuse_reject = 0;
    int view_block_hazard = 0;
    int view_block_fuse_reject = 0;
    int view_block_struct_reject = 0;
    int block_missing_next = 0;
    int block_next_not_mul_mat = 0;
    int block_src_mismatch = 0;
    int block_type_mismatch = 0;
    int block_fuse_reject = 0;
    int prefill_candidates = 0;
    int q3_candidates = 0;
    std::map<std::string, int> shapes;
    std::map<std::string, int> block_shapes;
    std::map<std::string, int> view_block_shapes;
    std::map<std::string, int> scan_block_shapes;
    std::map<std::string, int> block_next_ops;
    std::map<std::string, int> block_next_view_sources;

    for (int node_index = 0; node_index + 2 < cgraph->n_nodes; ++node_index) {
        const ggml_tensor * up = nullptr;
        const ggml_tensor * gate = nullptr;
        const ggml_tensor * glu = nullptr;
        if (!ggml_vk_is_dense_ffn_glu_candidate(cgraph, node_index, &up, &gate, &glu)) {
            continue;
        }

        candidates++;
        if (up->ne[1] > 1) {
            prefill_candidates++;
        }
        if (up->src[0]->type == GGML_TYPE_Q3_K) {
            q3_candidates++;
        }

        std::ostringstream key;
        key << ggml_type_name(up->src[0]->type)
            << " glu=" << ggml_glu_op_name(ggml_get_glu_op(glu))
            << " m=" << up->ne[0]
            << " n=" << up->ne[1]
            << " k=" << up->src[1]->ne[0];
        shapes[key.str()]++;

        const int scan_down_idx = ggml_vk_find_dense_ffn_down_node(cgraph, node_index, up, glu, 32);
        if (scan_down_idx >= 0) {
            const ggml_tensor * scan_down = cgraph->nodes[scan_down_idx];
            if (ggml_vk_can_fuse_dense_ffn_nonadjacent_block(cgraph, node_index, scan_down_idx)) {
                scan_block_candidates++;

                std::ostringstream scan_key;
                scan_key << ggml_type_name(up->src[0]->type)
                         << " glu=" << ggml_glu_op_name(ggml_get_glu_op(glu))
                         << " gap=" << (scan_down_idx - node_index)
                         << " gate_up_m=" << up->ne[0]
                         << " gate_up_n=" << up->ne[1]
                         << " gate_up_k=" << up->src[1]->ne[0]
                         << " down_m=" << scan_down->ne[0]
                         << " down_n=" << scan_down->ne[1]
                         << " down_k=" << scan_down->src[1]->ne[0];
                scan_block_shapes[scan_key.str()]++;
            } else {
                scan_block_fuse_reject++;
            }
        } else {
            scan_block_misses++;
        }

        const ggml_tensor * down = nullptr;
        if (node_index + 3 >= cgraph->n_nodes) {
            block_missing_next++;
        } else {
            down = cgraph->nodes[node_index + 3];
            if (down->op != GGML_OP_MUL_MAT) {
                block_next_not_mul_mat++;
                block_next_ops[ggml_op_name(down->op)]++;
                if (down->op == GGML_OP_VIEW) {
                    std::ostringstream view_src_key;
                    const ggml_tensor * view_src = down->src[0];
                    view_src_key << "src=" << (view_src != nullptr ? ggml_op_name(view_src->op) : "null")
                                 << " src_is_glu=" << (view_src == glu ? 1 : 0)
                                 << " view_ne=" << down->ne[0] << "x" << down->ne[1]
                                 << " glu_ne=" << glu->ne[0] << "x" << glu->ne[1];
                    block_next_view_sources[view_src_key.str()]++;
                }

                int view_count = 0;
                bool view_hazard = false;
                bool fuse_reject = false;
                const ggml_tensor * view_down = nullptr;
                if (ggml_vk_is_dense_ffn_view_block_candidate(cgraph, node_index, up, glu, &view_down, &view_count, &view_hazard, &fuse_reject)) {
                    view_block_candidates++;

                    std::ostringstream view_block_key;
                    view_block_key << ggml_type_name(up->src[0]->type)
                                   << " glu=" << ggml_glu_op_name(ggml_get_glu_op(glu))
                                   << " views=" << view_count
                                   << " gate_up_m=" << up->ne[0]
                                   << " gate_up_n=" << up->ne[1]
                                   << " gate_up_k=" << up->src[1]->ne[0]
                                   << " down_m=" << view_down->ne[0]
                                   << " down_n=" << view_down->ne[1]
                                   << " down_k=" << view_down->src[1]->ne[0];
                    view_block_shapes[view_block_key.str()]++;
                } else if (view_hazard) {
                    view_block_hazard++;
                } else if (fuse_reject) {
                    view_block_fuse_reject++;
                } else {
                    view_block_struct_reject++;
                }
            } else if (down->src[1] != glu) {
                block_src_mismatch++;
            } else if (down->src[0]->type != up->src[0]->type) {
                block_type_mismatch++;
            } else if (!ggml_can_fuse_subgraph(cgraph, node_index, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT }, { node_index + 3 })) {
                block_fuse_reject++;
            } else if (ggml_vk_is_dense_ffn_block_candidate(cgraph, node_index, &up, &gate, &glu, &down)) {
                block_candidates++;

                std::ostringstream block_key;
                block_key << ggml_type_name(up->src[0]->type)
                          << " glu=" << ggml_glu_op_name(ggml_get_glu_op(glu))
                          << " gate_up_m=" << up->ne[0]
                          << " gate_up_n=" << up->ne[1]
                          << " gate_up_k=" << up->src[1]->ne[0]
                          << " down_m=" << down->ne[0]
                          << " down_n=" << down->ne[1]
                          << " down_k=" << down->src[1]->ne[0];
                block_shapes[block_key.str()]++;
            }
        }
    }

    if (candidates == 0) {
        return;
    }

    std::cerr << "ggml_vulkan: ffn_route_trace candidates=" << candidates
              << " blocks=" << block_candidates
              << " view_blocks=" << view_block_candidates
              << " scan_blocks=" << scan_block_candidates
              << " prefill=" << prefill_candidates
              << " q3=" << q3_candidates << std::endl;
    for (const auto & item : shapes) {
        std::cerr << "ggml_vulkan: ffn_route_trace " << item.first
                  << " count=" << item.second << std::endl;
    }
    for (const auto & item : block_shapes) {
        std::cerr << "ggml_vulkan: ffn_block_trace " << item.first
                  << " count=" << item.second << std::endl;
    }
    for (const auto & item : view_block_shapes) {
        std::cerr << "ggml_vulkan: ffn_view_block_trace " << item.first
                  << " count=" << item.second << std::endl;
    }
    for (const auto & item : scan_block_shapes) {
        std::cerr << "ggml_vulkan: ffn_scan_block_trace " << item.first
                  << " count=" << item.second << std::endl;
    }
    if (scan_block_misses != 0 || scan_block_fuse_reject != 0) {
        std::cerr << "ggml_vulkan: ffn_scan_block_trace rejects"
                  << " missing_down=" << scan_block_misses
                  << " fuse_reject=" << scan_block_fuse_reject << std::endl;
    }
    if (view_block_hazard != 0 || view_block_fuse_reject != 0 || view_block_struct_reject != 0) {
        std::cerr << "ggml_vulkan: ffn_view_block_trace rejects"
                  << " view_hazard=" << view_block_hazard
                  << " fuse_reject=" << view_block_fuse_reject
                  << " struct_reject=" << view_block_struct_reject << std::endl;
    }
    if (block_missing_next != 0 || block_next_not_mul_mat != 0 || block_src_mismatch != 0 || block_type_mismatch != 0 || block_fuse_reject != 0) {
        std::cerr << "ggml_vulkan: ffn_block_trace rejects"
                  << " missing_next=" << block_missing_next
                  << " next_not_mul_mat=" << block_next_not_mul_mat
                  << " src_mismatch=" << block_src_mismatch
                  << " type_mismatch=" << block_type_mismatch
                  << " fuse_reject=" << block_fuse_reject << std::endl;
        for (const auto & item : block_next_ops) {
            std::cerr << "ggml_vulkan: ffn_block_trace reject_next_op " << item.first
                      << " count=" << item.second << std::endl;
        }
        for (const auto & item : block_next_view_sources) {
            std::cerr << "ggml_vulkan: ffn_block_trace reject_next_view " << item.first
                      << " count=" << item.second << std::endl;
        }
    }
}

static bool ggml_vk_can_fuse(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }
        // rms_norm shader assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
    }
    auto const &mm_add_ok = [&](const ggml_tensor *mul, const ggml_tensor *add) {
        const ggml_tensor *bias = add->src[0] == mul ? add->src[1] : add->src[0];

        // mat-vec only
        if (ggml_nrows(mul) != 1) {
            return false;
        }
        // shaders assume the types match
        if (mul->type != bias->type) {
            return false;
        }
        // shaders reuse the D shape for bias
        if (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, bias) != 0) {
            return false;
        }
        return true;
    };

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        // additional constraints specific to this fusion
        const ggml_tensor *mul = cgraph->nodes[node_idx];
        const ggml_tensor *add = cgraph->nodes[node_idx + 1];

        if (!mm_add_ok(mul, add)) {
            return false;
        }
        if (ops.size() == 3) {
            if (ops.begin()[2] != GGML_OP_ADD) {
                return false;
            }
            if (!mm_add_ok(add, cgraph->nodes[node_idx + 2])) {
                return false;
            }
        }
    }

    auto const &mmid_mul_ok = [&](const ggml_tensor *mmid, const ggml_tensor *mul) {
        const ggml_tensor *scale = mul->src[1];

        if (mmid != mul->src[0]) {
            return false;
        }
        // mat-vec only
        if (!ggml_vk_use_mul_mat_vec_id(cgraph, node_idx)) {
            return false;
        }
        // shaders assume the types match
        if (mmid->type != scale->type) {
            return false;
        }
        // shaders assume the bias is contiguous
        if (!ggml_is_contiguous(scale)) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, scale) != 0) {
            return false;
        }
        // shader only indexes by expert index
        if (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            return false;
        }
        return true;
    };

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_ADD_ID) {
        // additional constraints specific to this fusion
        const ggml_tensor *mul = cgraph->nodes[node_idx];
        const ggml_tensor *add = cgraph->nodes[node_idx + 1];
        const ggml_tensor *bias = add->src[1];

        if (mul != add->src[0]) {
            return false;
        }
        // mat-vec only
        if (!ggml_vk_use_mul_mat_vec_id(cgraph, node_idx)) {
            return false;
        }
        // shaders assume the types match
        if (mul->type != bias->type) {
            return false;
        }
        // shaders assume the bias is contiguous
        if (!ggml_is_contiguous(bias)) {
            return false;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        if (mul->src[2] != add->src[2]) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, bias) != 0) {
            return false;
        }

        if (ops.size() == 3) {
            if (ops.begin()[2] != GGML_OP_MUL) {
                return false;
            }
            const ggml_tensor *mul = cgraph->nodes[node_idx + 2];
            return mmid_mul_ok(add, mul);
        }
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        const ggml_tensor *mmid = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx + 1];

        if (!mmid_mul_ok(mmid, mul)) {
            return false;
        }
    }

    return true;
}

static bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                      int node_idx, topk_moe_mode mode) {

    const ggml_tensor * softmax;
    const ggml_tensor * weights;
    const ggml_tensor * get_rows;
    const ggml_tensor * argsort;

    switch (mode) {
    case TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[node_idx + 0];
        weights = cgraph->nodes[node_idx + 9];
        get_rows = cgraph->nodes[node_idx + 4];
        argsort = cgraph->nodes[node_idx + 2];
        break;
    case TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[node_idx + 0]; // really sigmoid
        weights = cgraph->nodes[node_idx + 10];
        get_rows = cgraph->nodes[node_idx + 5];
        argsort = cgraph->nodes[node_idx + 3];
        if (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        // bias is expected to be 1D
        if (ggml_nrows(cgraph->nodes[node_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[node_idx + 2]->src[1])) {
            return false;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        if (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            return false;
        }
        break;
    case TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[node_idx + 0];
        weights = cgraph->nodes[node_idx + 4];
        get_rows = cgraph->nodes[node_idx + 4];
        argsort = cgraph->nodes[node_idx + 2];
        break;
    case TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[node_idx + 4];
        weights = cgraph->nodes[node_idx + 5];
        get_rows = cgraph->nodes[node_idx + 2];
        argsort = cgraph->nodes[node_idx + 0];
        break;
    default:
        return false;
    }

    ggml_tensor * probs = get_rows->src[0];
    if (probs->op != GGML_OP_RESHAPE) {
        return false;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    if (probs != selection_probs && mode != TOPK_MOE_SIGMOID_NORM_BIAS) {
        return false;
    }

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        return false;
    }

    if (softmax->op == GGML_OP_SOFT_MAX) {
        const float * op_params = (const float *)softmax->op_params;

        float scale = op_params[0];
        float max_bias = op_params[1];

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    }

    const int n_expert = softmax->ne[0];
    if (n_expert > (1 << (num_topk_moe_pipelines-1))) {
        return false;
    }

    if (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_full_support ||
        ctx->device->disable_fusion) {
        return false;
    }

    return true;
}

static bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                           int node_idx) {
    GGML_UNUSED(ctx);
    const ggml_tensor *rope = cgraph->nodes[node_idx + 0];
    const ggml_tensor *view = cgraph->nodes[node_idx + 1];
    const ggml_tensor *set_rows = cgraph->nodes[node_idx + 2];

    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/mrope shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_MROPE) {
        return false;
    }

    return true;
}

// Check whether the tensors overlap in memory.
// Fusions can potentially overwrite src tensors in ways that are not prevented
// by ggml-alloc. If the fusion src is being applied in a way that's elementwise
// with the destination, then it's OK for them to overlap if they are exactly equal.
static bool ggml_vk_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b, bool elementwise) {
    ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)a->buffer->context;
    vk_buffer a_buf = a_buf_ctx->dev_buffer;
    ggml_backend_vk_buffer_context * b_buf_ctx = (ggml_backend_vk_buffer_context *)b->buffer->context;
    vk_buffer b_buf = b_buf_ctx->dev_buffer;
    if (a_buf == b_buf) {
        auto a_base = vk_tensor_view_offset(a);
        auto a_size = ggml_vk_device_size(a);
        auto b_base = vk_tensor_view_offset(b);
        auto b_size = ggml_vk_device_size(b);

        if (elementwise && a_base == b_base && a_size == b_size) {
            return false;
        }

        if ((b_base <= a_base && a_base < b_base + b_size) ||
            (a_base <= b_base && b_base < a_base + a_size)) {
            return true;
        }
    }
    return false;
}

static bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                               int node_idx) {
    GGML_UNUSED(ctx);
    const ggml_tensor *rms = cgraph->nodes[node_idx + 0];
    const ggml_tensor *mul = cgraph->nodes[node_idx + 1];
    const ggml_tensor *rope = cgraph->nodes[node_idx + 2];

    const int mode = ((const int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    if (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        return false;
    }

    // only norm/neox are handled in the shader
    if (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_NORMAL) {
        return false;
    }

    // shared memory size for passing data from mul->rope
    if (mul->ne[0] > 1024) {
        return false;
    }

    // conditions for pipeline creation
    if (sizeof(vk_op_rms_norm_mul_rope_push_constants) > ctx->device->properties.limits.maxPushConstantsSize) {
        return false;
    }

    return true;
}

static uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {

    const ggml_tensor *first_node = cgraph->nodes[node_idx];
    if (first_node->op != GGML_OP_ADD) {
        return 0;
    }

    if (!ctx->device->multi_add) {
        return 0;
    }

    int32_t num_adds = 1;
    while (node_idx + num_adds < cgraph->n_nodes &&
           cgraph->nodes[node_idx + num_adds]->op == GGML_OP_ADD &&
           num_adds < MAX_FUSED_ADDS) {
        num_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    for (int32_t i = 0; i < num_adds; ++i) {
        const ggml_tensor *next_node = cgraph->nodes[node_idx + i];
        if (!ggml_are_same_shape(first_node, next_node->src[0]) ||
            !ggml_are_same_shape(first_node, next_node->src[1]) ||
            next_node->type != GGML_TYPE_F32 ||
            next_node->src[0]->type != GGML_TYPE_F32 ||
            next_node->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_bytes(ctx, next_node) ||
            get_misalign_bytes(ctx, next_node->src[0]) ||
            get_misalign_bytes(ctx, next_node->src[1])) {
            num_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    for (int32_t i = 0; i < num_adds; ++i) {
        adds[i] = GGML_OP_ADD;
    }

    // decrease num_adds if they can't all be fused
    while (num_adds > 1 && !ggml_can_fuse(cgraph, node_idx, adds, num_adds)) {
        num_adds--;
    }

    // a single add is not "fused", so just return zero
    if (num_adds == 1) {
        return 0;
    }
    return num_adds;
}

static int32_t find_first_set(uint32_t x) {
    int32_t ret = 0;
    if (!x) {
        return -1;
    }
    while (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    return ret;
}

static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    static const bool trace_dense_ffn = getenv("GGML_VK_FFN_ROUTE_TRACE") != nullptr;

    if (vk_instance.debug_utils_support) {
        vk::DebugUtilsLabelEXT dul = {};
        dul.pLabelName = "ggml_backend_vk_graph_compute";
        dul.color = std::array<float,4>{1.0f, 1.0f, 1.0f, 1.0f};
        vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT(ctx->device->compute_queue.queue, reinterpret_cast<VkDebugUtilsLabelEXT*>(&dul));
    }

    ctx->prealloc_size_add_rms_partials_offset = 0;
    ctx->do_add_rms_partials = false;
    ctx->do_add_rms_partials_offset_calculation = false;

    if (trace_dense_ffn) {
        ggml_vk_trace_dense_ffn_glu_candidates(cgraph);
    }

    int last_node = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    while (last_node > 0 && (ggml_vk_is_empty(cgraph->nodes[last_node]) || ((cgraph->nodes[last_node]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        last_node -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    bool first_node_in_batch = true; // true if next node will be first node in a batch
    int submit_node_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    if (vk_perf_logger_enabled) {
        // Scheduler reserve can leave an open context before the first traced graph.
        // Timestamp queries are synchronous and cannot safely reset/reuse their pool
        // until that work has been submitted and completed.
        if (!ctx->compute_ctx.expired()) {
            ggml_vk_synchronize(ctx);
        }

        // allocate/resize the query pool
        if (ctx->num_queries < cgraph->n_nodes + 1) {
            if (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.queryCount = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->num_queries = query_create_info.queryCount;
            ctx->query_fusion_names.resize(ctx->num_queries);
            ctx->query_fusion_node_count.resize(ctx->num_queries);
            ctx->query_nodes.resize(ctx->num_queries);
            ctx->query_node_idx.resize(ctx->num_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_names.begin(), ctx->query_fusion_names.end(), nullptr);
        std::fill(ctx->query_fusion_node_count.begin(), ctx->query_fusion_node_count.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), nullptr);
        std::fill(ctx->query_node_idx.begin(), ctx->query_node_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;

    if (ctx->prealloc_size_add_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, nullptr);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_buffer_memset_async(compute_ctx, ctx->prealloc_add_rms_partials, 0, 0, ctx->prealloc_size_add_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated to overlap command-buffer generation
    // with GPU execution. FLOPs track prompt, decode, and attention work more closely
    // than the static size of a weight tensor.
    uint32_t submitted_nodes = 0;
    uint32_t submit_count = 0;
    uint64_t batch_flops = 0;
    uint64_t total_flops = 0;
    uint64_t flops_per_submit = std::min(uint64_t(200'000'000'000), ctx->last_total_flops / 40u);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (first_node_in_batch) {
            submit_node_idx = i;
        }

        const uint64_t node_flops = ggml_vk_get_node_flops(cgraph->nodes[i]);
        batch_flops += node_flops;
        total_flops += node_flops;

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        bool op_srcs_fused_elementwise[12];

        ctx->fused_topk_moe_mode = TOPK_MOE_COUNT;
        ctx->fused_topk_moe_scale = false;
        const char *fusion_string {};
        if (!ctx->device->disable_fusion) {
            uint32_t num_adds = ggml_vk_fuse_multi_add(ctx, cgraph, i);
            if (num_adds) {
                ctx->num_additional_fused_ops = num_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, true);
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD })) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { i + 4 }) &&
                       ggml_check_edges(cgraph, i, rms_norm_mul_rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i + 2)) {
                ctx->num_additional_fused_ops = 4;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = false;
                op_srcs_fused_elementwise[2] = false;
                op_srcs_fused_elementwise[3] = false;
                op_srcs_fused_elementwise[4] = false;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE })&&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = true;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { i + 2 }) &&
                       ggml_check_edges(cgraph, i, rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = false;
                op_srcs_fused_elementwise[2] = false;
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_norm_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->num_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                ctx->fused_topk_moe_mode = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_sigmoid_norm_bias_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->num_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 4;
                ctx->fused_topk_moe_mode = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->num_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                ctx->fused_topk_moe_mode = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_late_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->num_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 1;
                ctx->fused_topk_moe_mode = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            }
            if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                if (ggml_can_fuse_subgraph(cgraph, i + ctx->num_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->num_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->num_additional_fused_ops, { GGML_OP_GET_ROWS, GGML_OP_SCALE }, { i + ctx->num_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = true;
                    ctx->num_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->num_additional_fused_ops] = false;
                }
            }
        }
        GGML_ASSERT(ctx->num_additional_fused_ops < (int)(sizeof(op_srcs_fused_elementwise) / sizeof(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_mask |= 1 << ctx->num_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        if (ctx->num_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t bits = ctx->fused_ops_write_mask & ~(1 << ctx->num_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->num_additional_fused_ops];
            if (bits) {
                int output_idx = find_first_set(bits);
                GGML_ASSERT(bits == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            bool need_disable = false;

            // topk_moe often overwrites the source, but for a given row all the src values are
            // loaded before anything is stored. If there's only one row, this is safe, so treat
            // this as a special case.
            bool is_topk_moe_single_row = ctx->fused_topk_moe_mode != TOPK_MOE_COUNT &&
                                          ggml_nrows(cgraph->nodes[i]->src[0]) == 1;

            if (!is_topk_moe_single_row) {
                for (int j = 0; j < 2; ++j) {
                    ggml_tensor *dst = output_nodes[j];
                    if (!dst) {
                        continue;
                    }
                    // Loop over all srcs of all nodes in the fusion. If the src overlaps
                    // the destination and the src is not an intermediate node that's being
                    // elided, then disable fusion.
                    for (int k = 0; k <= ctx->num_additional_fused_ops; ++k) {
                        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                            ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                            if (!src || src->op == GGML_OP_NONE) {
                                continue;
                            }
                            if (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                                bool found = false;
                                for (int n = 0; n < k; ++n) {
                                    if (cgraph->nodes[i + n] == src) {
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found) {
                                    need_disable = true;
                                }
                            }
                        }
                    }
                }
            }
            if (need_disable) {
                ctx->num_additional_fused_ops = 0;
                ctx->fused_ops_write_mask = 1;
                ctx->fused_topk_moe_mode = TOPK_MOE_COUNT;
                ctx->fused_topk_moe_scale = false;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        bool almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        bool submit = (submitted_nodes >= ctx->device->max_nodes_per_submit) ||
                      (flops_per_submit != 0 && batch_flops >= flops_per_submit) ||
                      (i + ctx->num_additional_fused_ops >= last_node) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        bool enqueued = ggml_vk_build_graph(ctx, cgraph, i, cgraph->nodes[submit_node_idx], submit_node_idx, i + ctx->num_additional_fused_ops >= last_node, almost_ready, submit);

        if (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            if (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_names[ctx->query_idx] = fusion_string;
                compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } else {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_names[i] = fusion_string;
                ctx->query_fusion_node_count[i] = ctx->num_additional_fused_ops;
            }
        }

        if (enqueued) {
            ++submitted_nodes;

#ifndef GGML_VULKAN_CHECK_RESULTS
            if (first_node_in_batch) {
                first_node_in_batch = false;
            }
#endif
        }

        if (submit && enqueued) {
            first_node_in_batch = true;
            submitted_nodes = 0;
            batch_flops = 0;
            if (submit_count < 3) {
                flops_per_submit *= 2;
            }
            submit_count++;
        }
        i += ctx->num_additional_fused_ops;
        ctx->num_additional_fused_ops = 0;
        ctx->fused_ops_write_mask = 0;
    }

    ctx->last_total_flops = total_flops;

    if (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.lock();
        ggml_vk_ctx_end(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, true, UINT64_MAX), "GGML_VULKAN_PERF waitForFences");
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait), "get timestamp results");
        if (!vk_perf_logger_concurrent) {
            // Log each op separately
            for (int i = 1; i < ctx->query_idx; i++) {
                auto node = ctx->query_nodes[i];
                auto name = ctx->query_fusion_names[i];
                ctx->perf_logger->log_timing(node, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } else {
            // Log each group of nodes
            int prev_node_idx = 0;
            for (int i = 1; i < ctx->query_idx; i++) {
                auto cur_node_idx = ctx->query_node_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<const char *> names;
                for (int node_idx = prev_node_idx; node_idx < cur_node_idx; ++node_idx) {
                    if (ggml_op_is_empty(cgraph->nodes[node_idx]->op)) {
                        continue;
                    }
                    nodes.push_back(cgraph->nodes[node_idx]);
                    names.push_back(ctx->query_fusion_names[node_idx]);
                    node_idx += ctx->query_fusion_node_count[node_idx];
                }
                prev_node_idx = cur_node_idx;
                ctx->perf_logger->log_timing(nodes, names, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    if (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    return GGML_STATUS_SUCCESS;

    UNUSED(backend);
}

// Sort the graph for improved parallelism.
static void ggml_vk_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * graph)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    if (ctx->device->disable_graph_optimize) {
        return;
    }

    auto const &is_empty = [](ggml_tensor * node) -> bool {
        return node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    auto const &is_src_of = [](const ggml_tensor *dst, const ggml_tensor *src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor *dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor *src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    std::vector<ggml_tensor *> new_order;
    std::vector<bool> used(graph->n_nodes, false);
    std::set<ggml_tensor *> used_node_set;

    int first_unused = 0;
    while (first_unused < graph->n_nodes) {
        std::vector<int> current_set;

        // Check for fusion patterns and avoid reordering them
        auto const &match_pattern = [&](const std::initializer_list<ggml_op> &pattern, int start) -> bool {
            if (start + (int)pattern.size() <= graph->n_nodes) {
                bool is_pattern = true;
                for (size_t j = 0; j < pattern.size(); ++j) {
                    if (graph->nodes[start + j]->op != pattern.begin()[j] || used[start + j]) {
                        is_pattern = false;
                    }
                }
                return is_pattern;
            }
            return false;
        };

        auto const &keep_pattern = [&](const std::initializer_list<ggml_op> &pattern) -> bool {
            if (match_pattern(pattern, first_unused)) {
                for (size_t j = 0; j < pattern.size(); ++j) {
                    new_order.push_back(graph->nodes[first_unused + j]);
                    used_node_set.insert(graph->nodes[first_unused + j]);
                    used[first_unused + j] = true;
                }
                while (first_unused < graph->n_nodes && used[first_unused]) {
                    first_unused++;
                }
                return true;
            }
            return false;
        };

        if (keep_pattern(topk_moe_early_softmax_norm)) {
            continue;
        }
        if (keep_pattern(topk_moe_sigmoid_norm_bias)) {
            continue;
        }
        if (keep_pattern(topk_moe_early_softmax)) {
            continue;
        }
        if (keep_pattern(topk_moe_late_softmax)) {
            continue;
        }

        // First, grab the next unused node.
        current_set.push_back(first_unused);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        const int NUM_TO_CHECK = 20;
        for (int j = first_unused+1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
            if (used[j]) {
                continue;
            }
            if (is_empty(graph->nodes[j])) {
                continue;
            }
            // Don't pull forward nodes from fusion patterns
            if (match_pattern(topk_moe_early_softmax_norm, j) ||
                match_pattern(topk_moe_sigmoid_norm_bias, j) ||
                match_pattern(topk_moe_early_softmax, j) ||
                match_pattern(topk_moe_late_softmax, j)) {
                continue;
            }
            bool ok = true;
            for (int c = first_unused; c < j; ++c) {
                if (!used[c] &&
                    is_src_of(graph->nodes[j], graph->nodes[c]) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_ADD_ID) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_ADD && graph->nodes[j]->op == GGML_OP_ADD)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                current_set.push_back(j);

                int rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == nullptr || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_set.push_back(rope_idx);
                            used[rope_idx] = true;
                            break;
                        }
                    }
                }
                // Look for ROPE + VIEW + SET_ROWS and make them consecutive
                if (graph->nodes[rope_idx]->op == GGML_OP_ROPE) {
                    int view_idx = -1;
                    int set_rows_idx = -1;
                    for (int k = rope_idx+1; k < std::min(rope_idx + 10, graph->n_nodes); ++k) {
                        if (view_idx == -1 &&
                            graph->nodes[k]->op == GGML_OP_VIEW &&
                            graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            continue;
                        }
                        if (view_idx != -1 &&
                            set_rows_idx == -1 &&
                            graph->nodes[k]->op == GGML_OP_SET_ROWS &&
                            graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            set_rows_idx = k;
                            break;
                        }
                    }
                    if (set_rows_idx != -1) {
                        current_set.push_back(view_idx);
                        current_set.push_back(set_rows_idx);
                        used[view_idx] = true;
                        used[set_rows_idx] = true;
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_node_set.find(graph->nodes[k]->src[1]) != used_node_set.end())) {
                            current_set.push_back(k);
                            used[k] = true;
                            break;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_ADD &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_node_set.find(graph->nodes[k]->src[1]) != used_node_set.end())) {
                            current_set.push_back(k);
                            used[k] = true;
                            break;
                        }
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        if (graph->nodes[current_set.back()]->op != GGML_OP_ADD) {
            for (int j = first_unused+1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
                if (used[j]) {
                    continue;
                }
                if (!is_empty(graph->nodes[j])) {
                    continue;
                }
                bool ok = true;
                for (int c = first_unused; c < j; ++c) {
                    bool c_in_current_set = std::find(current_set.begin(), current_set.end(), c) != current_set.end();
                    // skip views whose srcs haven't been processed.
                    if (!used[c] &&
                        is_src_of(graph->nodes[j], graph->nodes[c]) &&
                        !c_in_current_set) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    current_set.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        for (auto c : current_set) {
            new_order.push_back(graph->nodes[c]);
            used_node_set.insert(graph->nodes[c]);
            used[c] = true;
        }
        while (first_unused < graph->n_nodes && used[first_unused]) {
            first_unused++;
        }
    }
    // Replace the graph with the new order.
    for (int i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_order[i];
    }
}

static void ggml_backend_vk_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    auto* cmd_buf = compute_ctx->s->buffer; // retrieve pointer before it gets reset

    if (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    if (vkev->events_free.empty()) {
        vkev->event = ctx->device->device.createEvent({});
    } else {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = true;

    ggml_vk_set_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_end(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = true;
    vkev->cmd_buffer = cmd_buf;
    vkev->cmd_buffer_use_counter = cmd_buf->use_counter;
    ctx->compute_ctx.reset();
}

static void ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    if (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });
    }
}

// TODO: enable async and synchronize
static ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_vk_graph_compute,
    /* .event_record            = */ ggml_backend_vk_event_record,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_graph_optimize,
    /* .tensor_get_batch        = */ ggml_backend_vk_get_tensor_batch,
};
