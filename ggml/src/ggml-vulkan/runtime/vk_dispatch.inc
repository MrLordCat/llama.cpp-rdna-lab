// Matmul/attention route selection and generic operation-to-pipeline dispatch.
static uint32_t ggml_vk_guess_split_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, bool disable_split_k, ggml_type src0_type, const vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_split_k << ", " << ggml_type_name(src0_type) << ")");

    if (disable_split_k || getenv("GGML_VK_MM_DISABLE_SPLIT_K")) {
        if (getenv("GGML_VK_MM_TRACE_SPLIT")) { fprintf(stderr, "MMSPLIT: m=%u n=%u k=%u split_k=1 (disabled)\n", m, n, k); }
        return 1;
    }

    uint32_t split_k = 1;
    if (ctx->device->shader_core_count != 0 && m >= pipeline->wg_denoms[0] && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        if (k >= 2048) {
            if (m_tiles * n_tiles <= ctx->device->shader_core_count / 2) {
                split_k = ctx->device->shader_core_count / (m_tiles * n_tiles);
            } else if (m_tiles * n_tiles <= ctx->device->shader_core_count * 2 / 3) {
                split_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            split_k = std::min(split_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            while (true) {
                if (split_k == 1) {
                    break;
                }
                uint32_t k_split = CEIL_DIV(k, split_k);
                k_split = ROUNDUP_POW2(k_split, 256);
                if (k_split * (split_k - 1) < k) {
                    break;
                }
                split_k--;
            }
        }
    }

    static const int ffn_down_split_k_override = []() -> int {
        const char * env = getenv("GGML_VK_Q3K_FFN_DOWN_SPLIT_K");
        if (env == nullptr) {
            return -1;
        }
        return atoi(env);
    }();
    if (src0_type == GGML_TYPE_Q3_K && m == 5120 && k == 17408 && n >= 128) {
        uint32_t target_split_k = 3;
        if (ffn_down_split_k_override >= 0) {
            if (ffn_down_split_k_override < 2) {
            return split_k;
            }
            target_split_k = std::min<uint32_t>((uint32_t) ffn_down_split_k_override, 8u);
        }

        split_k = std::max(split_k, target_split_k);

        while (split_k > 1) {
            uint32_t k_split = CEIL_DIV(k, split_k);
            k_split = ROUNDUP_POW2(k_split, 256);
            if (k_split * (split_k - 1) < k) {
                break;
            }
            split_k--;
        }
    }

    static const int low_tile_split_k_override = []() -> int {
        const char * env = getenv("GGML_VK_QK_LOW_TILE_SPLIT_K");
        if (env == nullptr) {
            return -1;
        }
        return atoi(env);
    }();
    const bool q3_low_tile_candidate =
        src0_type == GGML_TYPE_Q3_K && n >= 128 &&
        ((m == 10240 && k == 5120) ||
         (m ==  6144 && k == 5120) ||
         (m ==  5120 && k == 6144));
    const bool q4_low_tile_candidate =
        src0_type == GGML_TYPE_Q4_K && n >= 128 &&
        m == 5120 && k == 6144;
    int low_tile_split_k = low_tile_split_k_override;
    if (low_tile_split_k < 0 && q3_low_tile_candidate &&
        ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
        ctx->device->driver_id == vk::DriverId::eAmdProprietary &&
        getenv("GGML_VK_DISABLE_QK_LOW_TILE_DEFAULT") == nullptr) {
        low_tile_split_k = n >= 512 ? 2 : 3;
    }
    if (low_tile_split_k >= 2 && (q3_low_tile_candidate || q4_low_tile_candidate)) {
        split_k = std::max(split_k, std::min<uint32_t>((uint32_t) low_tile_split_k, 8u));

        while (split_k > 1) {
            uint32_t k_split = CEIL_DIV(k, split_k);
            k_split = ROUNDUP_POW2(k_split, 256);
            if (k_split * (split_k - 1) < k) {
                break;
            }
            split_k--;
        }
    }

    return split_k;
}

static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, ggml_type src0_type, ggml_type src1_type) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_pipeline(" << m << ", " << n << ", " << aligned << ", " << ggml_type_name(src0_type) << ", " << ggml_type_name(src1_type) << ")");

    if (ctx->device->coopmat2) {
        const uint32_t shader_core_count = ctx->device->shader_core_count;
        const uint32_t tiles_l = CEIL_DIV(m, mmp->a_l->wg_denoms[0]) * CEIL_DIV(n, mmp->a_l->wg_denoms[1]);
        const uint32_t tiles_m = CEIL_DIV(m, mmp->a_m->wg_denoms[0]) * CEIL_DIV(n, mmp->a_m->wg_denoms[1]);

        // Use large shader when the N dimension is greater than the medium shader's tile size
        uint32_t crossover_large = mmp->m->wg_denoms[1];

        // Prefer large over medium if either:
        // - medium or large tiles would overfill the GPU
        // - large tiles with a split_k==3 fits in the GPU and medium tiles with split_k==2 does not
        //   (medium with split_k==2 is probably better if it fits - more workgroups running and less split_k overhead)
        bool prefer_large = tiles_m > shader_core_count || tiles_l > shader_core_count ||
                            // split_k==3 with large tiles likely better than medium tiles with no split_k.
                            (tiles_l <= shader_core_count / 3 && tiles_m > shader_core_count / 2);

        if ((ctx->device->mul_mat_l[src0_type] && (n > crossover_large && prefer_large)) || (!ctx->device->mul_mat_m[src0_type] && !ctx->device->mul_mat_s[src0_type])) {
            return aligned ? mmp->a_l : mmp->l;
        }
        // Use medium shader when the N dimension is greater than the small shader's tile size
        uint32_t crossover_medium = mmp->s->wg_denoms[1];
        if ((ctx->device->mul_mat_m[src0_type] && (n > crossover_medium)) || !ctx->device->mul_mat_s[src0_type]) {
            return aligned ? mmp->a_m : mmp->m;
        }
        return aligned ? mmp->a_s : mmp->s;
    }

    if ((ctx->device->mul_mat_s[src0_type] && (m <= 32 || n <= 32)) || (!ctx->device->mul_mat_m[src0_type] && !ctx->device->mul_mat_l[src0_type])) {
        return aligned ? mmp->a_s : mmp->s;
    }
    if ((ctx->device->mul_mat_m[src0_type] && (m <= 64 || n <= 64)) || !ctx->device->mul_mat_l[src0_type]) {
        return aligned ? mmp->a_m : mmp->m;
    }
    return aligned ? mmp->a_l : mmp->l;

    GGML_UNUSED(src1_type);
}

static uint32_t ggml_vk_guess_matmul_pipeline_align(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, ggml_type src0_type, ggml_type src1_type) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_pipeline_align(" << m << ", " << n << ", " << ggml_type_name(src0_type) << ", " << ggml_type_name(src1_type) << ")");
    return ggml_vk_guess_matmul_pipeline(ctx, mmp, m, n, true, src0_type, src1_type)->align;
}

static void ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& split_k_buffer,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size << "), split_k: (" << (split_k_buffer.buffer != nullptr ? split_k_buffer.buffer->buffer : VK_NULL_HANDLE) << ", " << split_k_buffer.offset << ", " << split_k_buffer.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << split_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
        if (getenv("GGML_VK_MM_TRACE_SPLIT")) { fprintf(stderr, "MMSPLIT: m=%u n=%u k=%u split_k=%u q8=%d\n", (unsigned)m, (unsigned)n, (unsigned)k, split_k, split_k_buffer.buffer != nullptr); }
    if (split_k == 1) {
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.maxComputeWorkGroupCount[2]));

        uint32_t base_work_group_z = 0;
        while (base_work_group_z < batch) {
            uint32_t groups_z = std::min(batch - base_work_group_z, ctx->device->properties.limits.maxComputeWorkGroupCount[2]);

            const vk_mat_mat_push_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, base_work_group_z, batch, k, ne02, ne12, broadcast2, broadcast3, padded_n };
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d }, pc, { m, n, groups_z });
            base_work_group_z += groups_z;
        }
        return;
    }

    if (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_split = CEIL_DIV(k, split_k);
    k_split = ROUNDUP_POW2(k_split, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.maxComputeWorkGroupCount[2]));

    uint32_t base_work_group_z = 0;
    while (base_work_group_z < batch) {
        uint32_t groups_z = std::min(batch - base_work_group_z, ctx->device->properties.limits.maxComputeWorkGroupCount[2]);

        const vk_mat_mat_push_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, base_work_group_z, batch, k_split, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, split_k_buffer }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * split_k, n, groups_z });
        base_work_group_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    const std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), split_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_split_k_reduce, { split_k_buffer, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_split_k_need_sync = true;
}

static vk_pipeline ggml_vk_guess_matmul_id_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, ggml_type src0_type) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_id_pipeline(" << m << ", " << n << ", " << aligned << ", " << ggml_type_name(src0_type) << ")");

    if (ctx->device->coopmat2) {
        // Use large shader when the N dimension is greater than the medium shader's tile size
        uint32_t crossover_large = mmp->m->wg_denoms[1];
        if ((ctx->device->mul_mat_id_l[src0_type] && (n > crossover_large)) || (!ctx->device->mul_mat_id_m[src0_type] && !ctx->device->mul_mat_id_s[src0_type])) {
            return aligned ? mmp->a_l : mmp->l;
        }
        // Use medium shader when the N dimension is greater than the small shader's tile size
        uint32_t crossover_medium = mmp->s->wg_denoms[1];
        if ((ctx->device->mul_mat_id_m[src0_type] && (n > crossover_medium)) || !ctx->device->mul_mat_id_s[src0_type]) {
            return aligned ? mmp->a_m : mmp->m;
        }
        return aligned ? mmp->a_s : mmp->s;
    }

    if ((ctx->device->mul_mat_id_s[src0_type] && (m <= 32 || n <= 32)) || (!ctx->device->mul_mat_id_m[src0_type] && !ctx->device->mul_mat_id_l[src0_type])) {
        return aligned ? mmp->a_s : mmp->s;
    }
    if ((ctx->device->mul_mat_id_m[src0_type] && (m <= 64 || n <= 64)) || !ctx->device->mul_mat_id_l[src0_type]) {
        return aligned ? mmp->a_m : mmp->m;
    }
    return aligned ? mmp->a_l : mmp->l;
}

static uint32_t ggml_vk_guess_matmul_id_pipeline_align(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, ggml_type src0_type) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_pipeline_align(" << m << ", " << n << ", " << ggml_type_name(src0_type) << ")");
    return ggml_vk_guess_matmul_id_pipeline(ctx, mmp, m, n, true, src0_type)->align;
}

static void ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, const vk_subbuffer & expert_count_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t padded_n) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size << "), ids: (" << ids.buffer->buffer << ", " << ids.offset << ", " << ids.size << "), expert_count: (" << expert_count_buf.buffer->buffer << ", " << expert_count_buf.offset << ", " << expert_count_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    const vk_mat_mat_id_push_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, padded_n };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_count_buf }, pc, { m, nei1, n_as });
}

static bool ggml_vk_dim01_contiguous(const ggml_tensor * tensor) {
    return
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_size(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

static vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src, const ggml_tensor * dst, ggml_type to) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    bool contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    bool transpose = dst && src->nb[1] == ggml_type_size(to) && ggml_are_same_shape(dst, src);

    if (transpose && src->type == to) {
        if (ggml_vk_device_type_size(to) == 4) {
            return ctx->device->pipeline_cpy_transpose_32;
        } else if (ggml_vk_device_type_size(to) == 2) {
            return ctx->device->pipeline_cpy_transpose_16;
        }
    }

    if (src->type == GGML_TYPE_F32 && to == GGML_TYPE_F32) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f32_f32;
        } else {
            return ctx->device->pipeline_cpy_f32_f32;
        }
    }
    if (src->type == GGML_TYPE_F32 && to == GGML_TYPE_F16) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f32_f16;
        } else {
            return ctx->device->pipeline_cpy_f32_f16;
        }
    }
    if (src->type == GGML_TYPE_F16 && to == GGML_TYPE_F16) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f16_f16;
        } else {
            return ctx->device->pipeline_cpy_f16_f16;
        }
    }
    if (src->type == GGML_TYPE_F16 && to == GGML_TYPE_F32) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f16_f32;
        } else {
            return ctx->device->pipeline_cpy_f16_f32;
        }
    }
    if (src->type == GGML_TYPE_F32 && to == GGML_TYPE_BF16) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f32_bf16;
        } else {
            return ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    if (src->type == GGML_TYPE_BF16 && to == GGML_TYPE_F32) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_bf16_f32;
        } else {
            return ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    if (src->type == GGML_TYPE_F32 && to == GGML_TYPE_I32) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f32_i32;
        } else {
            return ctx->device->pipeline_cpy_f32_i32;
        }
    }
    if (src->type == GGML_TYPE_I32 && to == GGML_TYPE_F32) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_i32_f32;
        } else {
            return ctx->device->pipeline_cpy_i32_f32;
        }
    }
    if (src->type == GGML_TYPE_F32) {
        switch (to) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_F8_E4M3:
            return ctx->device->pipeline_cpy_f32_quant[to];
        default:
            break;
        }
    }

    if (to == GGML_TYPE_F32) {
        switch (src->type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_F8_E4M3:
            return ctx->device->pipeline_cpy_quant_f32[src->type];
        default:
            break;
        }
    }

    if (src->type == to) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(to) || ggml_vk_device_type_size(src->type) == 2 || ggml_vk_device_type_size(src->type) == 4);
        if ((ggml_vk_device_type_size(src->type) % 4) == 0) {
            if (contig) {
                return ctx->device->pipeline_contig_cpy_f32_f32;
            } else {
                return ctx->device->pipeline_cpy_f32_f32;
            }
        } else {
            if (contig) {
                return ctx->device->pipeline_contig_cpy_f16_f16;
            } else {
                return ctx->device->pipeline_cpy_f16_f16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(to) << std::endl;
    GGML_ABORT("fatal error");
}

static void ggml_vk_cpy_to_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, const ggml_tensor * tensor, const vk_subbuffer & in, const vk_subbuffer & out) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ")");

    const uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    if (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } else if (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } else {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_push_constants pc = vk_op_unary_push_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

// Copy/convert tensor into a caller-defined dense layout. Destination strides
// are in output elements, not bytes.
static void ggml_vk_cpy_to_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, const ggml_tensor * tensor,
        const vk_subbuffer & in, const vk_subbuffer & out,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ")");

    const uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    if (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } else if (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } else {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_push_constants pc = vk_op_unary_push_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

static vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    switch(type) {
        case GGML_TYPE_Q8_1:
            return ctx->device->pipeline_quantize_q8_1_x4;
        default:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::endl;
            GGML_ABORT("fatal error");
    }
}

static void ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, const vk_subbuffer & in, const vk_subbuffer & out, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    const uint32_t num_blocks = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    const uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.maxComputeWorkGroupCount[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    const uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    const vk_quantize_q8_1_push_constants pc = {
        ne,
        num_blocks,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

static vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_UNUSED(ctx);
#if defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    while (*ptr) {
        if ((*ptr)->is_64b_indexing) {
            return *ptr;
        }
        ptr = &(*ptr)->next;
    }
#endif
    return pipeline;
}

static void ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool disable_split_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t ne21 = dst->ne[1];
    const uint32_t stride_d = dst->nb[1] / ggml_type_size(dst->type);
    const uint32_t stride_batch_d = stride_d*ne21;

    const uint64_t r2 = ne12 / ne02;
    const uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;

    vk_buffer d_Qx = nullptr;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
                              !ggml_vk_dim01_contiguous(src1);

    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;

    const bool y_contiguous = ggml_is_contiguous(src1);
    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && y_contiguous && !y_non_contig && (ne11 * ne10) % 4 == 0;
    const bool quantize_y_initial = quantize_y;
    const vk_matmul_pipeline q8_candidate_pipeline = quantize_y ? ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0->type].f32acc : nullptr;
    const bool q8_candidate_empty = q8_candidate_pipeline == nullptr || q8_candidate_pipeline->is_empty();

    // Check for mmq first
    vk_matmul_pipeline mmp = quantize_y ? ggml_vk_get_mul_mat_mat_pipeline(ctx, src0->type, GGML_TYPE_Q8_1, (ggml_prec)dst->op_params[0]) : nullptr;
    const bool q8_mmp_found = mmp != nullptr;

    if (mmp == nullptr) {
        // Fall back to f16 dequant mul mat
        mmp = ggml_vk_get_mul_mat_mat_pipeline(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0]);
        quantize_y = false;
    }

    // D104 R1 (probe): force the generic predequant route for Q6_K
    // (dequant_q6_k -> f16 staging -> f16 coopmat matmul). Measurement-only.
    const bool q6k_predequant_probe =
        std::getenv("GGML_VK_PREDEQUANT_Q6K") != nullptr && src0->type == GGML_TYPE_Q6_K;
    const bool qx_needs_dequant = mmp == nullptr || x_non_contig || q6k_predequant_probe;
    // D094: pre-quantized Q8_1 src1 is consumed directly by MMQ kernels.
    const bool qy_needs_dequant = !quantize_y && src1->type != GGML_TYPE_Q8_1 && ((src1->type != f16_type && !y_f32_kernel) || y_non_contig);

    if (qx_needs_dequant) {
        // Fall back to dequant + f16 mulmat
        mmp = ggml_vk_get_mul_mat_mat_pipeline(ctx, f16_type, y_f32_kernel ? GGML_TYPE_F32 : f16_type, (ggml_prec)dst->op_params[0]);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint32_t kpad = quantize_y ? 0 : ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ctx, mmp, ne01, ne11, qx_needs_dequant ? f16_type : src0->type, quantize_y ? GGML_TYPE_Q8_1 : (y_f32_kernel ? GGML_TYPE_F32 : src1->type)));
    const bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, ne11, aligned, qx_needs_dequant ? f16_type : src0->type, quantize_y ? GGML_TYPE_Q8_1 : (y_f32_kernel ? GGML_TYPE_F32 : src1->type));

    static const bool q3_k_quad_dequant = []() -> bool {
        if (std::getenv("GGML_VK_DISABLE_Q3K_QUAD_DEQUANT") != nullptr) {
            return false;
        }
        if (const char * env = std::getenv("GGML_VK_Q3K_QUAD_DEQUANT")) {
            return atoi(env) != 0;
        }
        return true;
    }();
    const bool q3_k_quad_shape =
        ne11 >= 128 &&
        ((ne01 == 17408 && ne10 ==  5120) ||
         (ne01 ==  5120 && ne10 == 17408) ||
         (ne01 ==  8704 && ne10 ==  5120) ||
         (ne01 ==  5120 && ne10 ==  8704) ||
         (ne01 ==  5120 && ne10 ==  6144) ||
         (ne01 ==  6144 && ne10 ==  5120) ||
         (ne01 ==  5120 && ne10 ==  3072) ||
         (ne01 ==  3072 && ne10 ==  5120) ||
         (ne01 == 12288 && ne10 ==  5120));
    if (q3_k_quad_dequant && src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_F32 &&
        !quantize_y && !qx_needs_dequant && !qy_needs_dequant && aligned && q3_k_quad_shape) {
        vk_matmul_pipeline q3_quad_mmp = nullptr;
        const ggml_prec prec = (ggml_prec) dst->op_params[0];
        if (ctx->device->coopmat_support) {
            q3_quad_mmp = (ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT) ?
                ctx->device->pipeline_dequant_mul_mat_mat_q3_k_quad.f16acc :
                ctx->device->pipeline_dequant_mul_mat_mat_q3_k_quad.f32acc;
        } else if (ctx->device->fp16 && prec == GGML_PREC_DEFAULT) {
            q3_quad_mmp = ctx->device->pipeline_dequant_mul_mat_mat_q3_k_quad.f16acc;
        } else {
            q3_quad_mmp = ctx->device->pipeline_dequant_mul_mat_mat_q3_k_quad.f32acc;
        }

        if (q3_quad_mmp != nullptr && !q3_quad_mmp->is_empty()) {
            pipeline = ggml_vk_guess_matmul_pipeline(ctx, q3_quad_mmp, ne01, ne11, aligned, src0->type, src1->type);
        }

    }
    if (ggml_vk_device_size(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    static const bool route_trace = std::getenv("GGML_VK_MATMUL_ROUTE_TRACE") != nullptr;
    if (route_trace && pipeline) {
        static std::mutex route_trace_mutex;
        static std::set<std::string> route_trace_seen;

        std::ostringstream key;
        key << pipeline->name << "|src0=" << ggml_type_name(src0->type)
            << "|src1=" << ggml_type_name(src1->type)
            << "|m=" << ne01 << "|n=" << ne11 << "|k=" << ne10
            << "|aligned=" << aligned
            << "|integer_dot=" << ctx->device->integer_dot_product
            << "|y_contiguous=" << y_contiguous
            << "|quantize_y_initial=" << quantize_y_initial
            << "|q8_candidate_empty=" << q8_candidate_empty
            << "|q8_mmp_found=" << q8_mmp_found
            << "|quantize_y=" << quantize_y
            << "|x_non_contig=" << x_non_contig
            << "|y_non_contig=" << y_non_contig
            << "|qx_dequant=" << qx_needs_dequant
            << "|qy_dequant=" << qy_needs_dequant;

        std::lock_guard<std::mutex> guard(route_trace_mutex);
        if (route_trace_seen.insert(key.str()).second) {
            std::cerr << "ggml_vulkan: matmul route: " << key.str() << std::endl;
        }
    }

    const uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, src0->type, pipeline);
    if (getenv("GGML_VK_MM_TRACE_SPLIT")) { fprintf(stderr, "MMSPLIT: m=%lld n=%lld k=%lld split_k=%u disable=%d\n", (long long) ne01, (long long) ne11, (long long) ne10, split_k, disable_split_k ? 1 : 0); }
    if (route_trace && split_k > 1 && pipeline) {
        static std::mutex split_trace_mutex;
        static std::set<std::string> split_trace_seen;

        std::ostringstream key;
        key << pipeline->name
            << "|src0=" << ggml_type_name(src0->type)
            << "|src1=" << ggml_type_name(src1->type)
            << "|m=" << ne01 << "|n=" << ne11 << "|k=" << ne10
            << "|split_k=" << split_k;

        std::lock_guard<std::mutex> guard(split_trace_mutex);
        if (split_trace_seen.insert(key.str()).second) {
            std::cerr << "ggml_vulkan: matmul split-k route: " << key.str() << std::endl;
        }
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    const uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    const uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    const uint64_t d_ne = ggml_nelements(dst);

    const uint64_t qx_sz = ggml_vk_device_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);
    const uint64_t qy_sz = ggml_vk_device_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = !qx_needs_dequant ? qx_sz : sizeof(ggml_fp16_t) * x_ne;
    const uint64_t y_sz = quantize_y ? (ggml_vk_align_size(y_ne, 128) * ggml_vk_device_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) : (y_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne);
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    vk_pipeline to_q8_1 = nullptr;

    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type);
    } else {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, f16_type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    if (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        const uint64_t split_k_size = split_k > 1 ? d_sz * split_k : 0;
        if (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (split_k > 1 && split_k_size > ctx->device->properties.limits.maxStorageBufferRange)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (qx_needs_dequant && ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (split_k > 1 && ctx->prealloc_size_split_k < split_k_size) {
            ctx->prealloc_size_split_k = split_k_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        if (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        if (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        if (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        if (split_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, 1);
        }
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    const uint64_t d_buf_offset = vk_tensor_view_offset(dst);
    GGML_ASSERT(d_D != nullptr);
    GGML_ASSERT(d_D->size >= d_buf_offset + d_sz);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_buffer;
        qx_buf_offset = vk_tensor_view_offset(src0);
        GGML_ASSERT(d_Qx != nullptr);
    }
    if (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_view_offset(src1);
        GGML_ASSERT(d_Qy != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } else if (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    if (x_non_contig || qx_needs_dequant) {
        if (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    if (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_offset), ggml_vk_subbuffer(ctx, d_X, 0));
    } else if (qx_needs_dequant) {
        const std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    if (y_non_contig) {
        if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
        }
    }
    if (quantize_y) {
        if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
        }
    }

    uint32_t stride_batch_x = ne00*ne01;
    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_sz }, { d_Y, y_buf_offset, y_sz },
        ggml_vk_subbuffer(ctx, d_D, d_buf_offset), { ctx->prealloc_split_k, 0, d_sz * split_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        split_k, ne12*ne13, ne02, ne12, r2, r3, padded_n
    );  // NOLINT

    if (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = true;
    }
    if (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = true;
    }
}

// Device tuning
static bool ggml_vk_should_use_mmvq(const vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    if (device->mmvq_mode == 1) {
        return true;
    } else if (device->mmvq_mode == -1) {
        return false;
    }

    // MMVQ is generally good for batches
    if (n > 1) {
        return true;
    }

    // Quantization overhead is not worth it for small k
    switch (device->vendor_id) {
    case VK_VENDOR_ID_NVIDIA:
        if (src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            return true;
        }

        if (k <= 4096) {
            return false;
        }

        switch (src0_type) {
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q8_0:
            return device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        default:
            return true;
        }
    case VK_VENDOR_ID_AMD:
        if (k < 2048) {
            return false;
        }

        switch (src0_type) {
        case GGML_TYPE_Q8_0:
            return device->architecture == vk_device_architecture::AMD_GCN;
        default:
            return true;
        }
    case VK_VENDOR_ID_INTEL:
        if (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance is worse than fp16, see
            // https://github.com/ggml-org/llama.cpp/issues/17628
            return false;
        }

        if (k < 2048) {
            return false;
        }

        switch (src0_type) {
        // From tests on A770 Linux, may need more tuning
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_1:
            return false;
        default:
            return true;
        }
    default:
        return true;
    }

    GGML_UNUSED(m);
}

static void ggml_vk_mul_mat_vec_q_f16(
        ggml_backend_vk_context * ctx,
        vk_context & subctx,
        const struct ggml_cgraph * cgraph,
        int node_idx,
        uint32_t row_offset = 0,
        uint32_t row_count = 0) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11_full = src1->ne[1];
    const uint64_t ne11 = row_count > 0 ? row_count : ne11_full;
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    const uint64_t r2 = ne12 / ne02;
    const uint64_t r3 = ne13 / ne03;

    GGML_ASSERT(row_offset + ne11 <= ne11_full);

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    bool batch_n = ne11 > 1;

    const bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;
    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : nullptr;
    vk_pipeline to_q8_1 = nullptr;

    if (dmmv == nullptr) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, src1->type, ne11, ne20, ne00);
        quantize_y = false;
    }

    if (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    if (ggml_vk_device_size(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    const bool qx_needs_dequant = x_non_contig;
    // D094: pre-quantized Q8_1 src1 is consumed directly by MMVQ kernels.
    const bool qy_needs_dequant = !quantize_y && src1->type != GGML_TYPE_Q8_1 && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    const uint64_t x_ne = ggml_nelements(src0);
    const uint64_t y_ne = ne10 * ne11 * ne12 * ne13;

    const uint64_t qx_sz = ggml_vk_align_size(ggml_vk_device_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t x_sz = x_non_contig ? ggml_vk_align_size(ggml_vk_device_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) : qx_sz;
    const uint64_t y_sz = quantize_y ? (ggml_vk_align_size(y_ne, 128) * ggml_vk_device_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne);

    {
        if (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.maxStorageBufferRange)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (qx_needs_dequant && ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        if (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        if (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        if (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[node_idx + ctx->num_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    if (row_offset > 0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

        const uint64_t y_offset = row_offset * src1->nb[1];
        const uint64_t d_offset = row_offset * dst->nb[1];
        const uint64_t alignment = ctx->device->properties.limits.minStorageBufferOffsetAlignment;

        GGML_ASSERT(y_offset <= d_Qy.size && d_offset <= d_D.size);
        GGML_ASSERT(y_offset % alignment == 0 && d_offset % alignment == 0);

        d_Qy.offset += y_offset;
        d_Qy.size -= y_offset;
        d_D.offset += d_offset;
        d_D.size -= d_offset;
    }

    if (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } else {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } else {
        d_Y = d_Qy;
    }

    if (x_non_contig) {
        if (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_size(ggml_vk_device_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, d_Qx, d_X);
    }
    if (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_vk_device_type_size(src1->type) * y_ne);
        if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
        }
    }
    if (quantize_y) {
        if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : ne00*ne01;
    uint32_t stride_batch_y = batch_n ? ne10 : (ne10*ne11);
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    if (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    if (ctx->num_additional_fused_ops > 0) {
        const ggml_tensor * add = cgraph->nodes[node_idx + 1];
        const ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    if (ctx->num_additional_fused_ops == 2) {
        const ggml_tensor * add = cgraph->nodes[node_idx + 2];
        const ggml_tensor * bias = add->src[0] == cgraph->nodes[node_idx + 1] ? add->src[1] : add->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.maxComputeWorkGroupCount[1]));

    uint32_t base_work_group_y = 0;
    while (base_work_group_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - base_work_group_y, ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
        const vk_mat_vec_push_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, base_work_group_y,
            (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
                                  {
                                    d_X,
                                    d_Y,
                                    d_D,
                                    d_F0,
                                    d_F1,
                                  },
                                  pc, { groups_x, groups_y, groups_z });
        base_work_group_y += groups_y;
    }

    if (x_non_contig) {
        ctx->prealloc_x_need_sync = true;
    }
    if (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = true;
    }
}

static void ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_ratio = (uint32_t)ne12 / (uint32_t)ne02;
    if (gqa_ratio > 8 || gqa_ratio == 0 || ne12 != ne02 * gqa_ratio) {
        gqa_ratio = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1];

    if (ggml_vk_device_size(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[node_idx + ctx->num_additional_fused_ops], true);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, true);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    if (ctx->num_additional_fused_ops > 0) {
        const ggml_tensor * add = cgraph->nodes[node_idx + 1];
        const ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    if (ctx->num_additional_fused_ops > 1) {
        const ggml_tensor * bias = cgraph->nodes[node_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_push_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, nullptr, nullptr, cgraph->nodes[node_idx + ctx->num_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    if (gqa_ratio > 1) {
        workgroups_z /= gqa_ratio;
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { 1, (uint32_t)ne01, workgroups_z });
}

static void ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t nb01 = src0->nb[1];
    const uint64_t nb02 = src0->nb[2];

    const uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    const uint32_t nb03 = (uint32_t)(src0->nb[3] / sizeof(ggml_fp16_t));
    const uint32_t nb13 = (uint32_t)(src1->nb[3] / sizeof(float));
    const uint32_t nb23 = (uint32_t)(dst->nb[3] / sizeof(float));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    const uint32_t row_stride_x = nb01 / sizeof(ggml_fp16_t);
    const uint32_t channel_stride_x = nb02 / sizeof(ggml_fp16_t);
    const uint32_t channel_stride_y = nb12 / sizeof(float);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    if (ggml_vk_device_size(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[node_idx + ctx->num_additional_fused_ops], true);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, true);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    if (ctx->num_additional_fused_ops > 0) {
        const ggml_tensor * add = cgraph->nodes[node_idx + 1];
        const ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    if (ctx->num_additional_fused_ops > 1) {
        const ggml_tensor * bias = cgraph->nodes[node_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_push_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        row_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, nullptr, nullptr, cgraph->nodes[node_idx + ctx->num_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

static void ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    const size_t nbytes = ggml_vk_device_size(src0);
    const bool needs_split = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.maxStorageBufferRange;
    const bool batched_mul_mat_vec =
        dst->ne[1] > 1 && dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1;
    bool avoid_batched_mul_mat_vec = false;
    bool split_batched_mul_mat_vec = false;
#if defined(_WIN32)
    // AMDVLK can crash while compiling the F32xF32 batched mat-vec shader with
    // COLS >= 5. MTP n_max=4 hits this shape because verify uses draft+target.
    avoid_batched_mul_mat_vec =
        batched_mul_mat_vec &&
        dst->ne[1] >= 5 &&
        src0->type == GGML_TYPE_F32 &&
        src1->type == GGML_TYPE_F32 &&
        ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
        std::getenv("GGML_VK_ALLOW_AMD_F32_BATCHED_MUL_MAT_VEC_5_PLUS") == nullptr;

    const uint64_t alignment = ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    split_batched_mul_mat_vec =
        avoid_batched_mul_mat_vec &&
        ctx->num_additional_fused_ops == 0 &&
        (4 * src1->nb[1]) % alignment == 0 &&
        (4 * dst->nb[1]) % alignment == 0;
#endif

    if (needs_split) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        const uint32_t M_split = ctx->device->properties.limits.maxStorageBufferRange / (2 * src0->nb[1]);
        uint32_t m_offset = 0;
        while (m_offset < dst->ne[0]) {
            const uint32_t cur_M_size = std::min(M_split, (uint32_t)(dst->ne[0] - m_offset));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_offset * dst->nb[0];
            src02.view_offs += m_offset * src0->nb[1];
            dst2.ne[0] = cur_M_size;
            src02.ne[1] = cur_M_size;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, true);

            m_offset += cur_M_size;
        }
    } else if (src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.maxComputeWorkGroupCount[1] &&
        src1->ne[2] <= ctx->device->properties.limits.maxComputeWorkGroupCount[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, node_idx);
    } else if (src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.maxComputeWorkGroupCount[0] &&
               src0->ne[1] <= ctx->device->properties.limits.maxComputeWorkGroupCount[1] &&
               src1->ne[2] <= ctx->device->properties.limits.maxComputeWorkGroupCount[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, node_idx);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size
    // when ne12 and ne13 are one.
    } else if (split_batched_mul_mat_vec) {
        // Keep each specialization below the Windows AMD compiler-crash boundary.
        for (uint32_t row = 0; row < dst->ne[1]; row += 4) {
            const uint32_t rows = std::min<uint32_t>(4, dst->ne[1] - row);
            ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, node_idx, row, rows);
        }
    } else if ((dst->ne[1] == 1 || (batched_mul_mat_vec && !avoid_batched_mul_mat_vec)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, node_idx);
    } else {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, false);
    }
}

static void ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t nei0 = ids->ne[0];
    const uint64_t nei1 = ids->ne[1];

    const uint32_t nbi0 = ids->nb[0];
    const uint32_t nbi1 = ids->nb[1];
    const uint32_t nbi2 = ids->nb[2];

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    const uint64_t n_as = ne02;

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;
    ggml_backend_vk_buffer_context * ids_buf_ctx = (ggml_backend_vk_buffer_context *)ids->buffer->context;

    vk_buffer d_Qx = nullptr;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;
    vk_buffer d_ids = nullptr;
    size_t ids_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;
    bool ids_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        ggml_vk_host_get(ctx->device, ids->data, d_ids, ids_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
        ids_uma = d_ids != nullptr;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
                              !ggml_vk_dim01_contiguous(src1);

    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;

    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0;

    // Check for mmq first
    vk_matmul_pipeline mmp = quantize_y ? ggml_vk_get_mul_mat_mat_id_pipeline(ctx, src0->type, GGML_TYPE_Q8_1, (ggml_prec)dst->op_params[0]) : nullptr;

    if (mmp == nullptr) {
        // Fall back to f16 dequant mul mat
        mmp = ggml_vk_get_mul_mat_mat_id_pipeline(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0]);
        quantize_y = false;
    }

    const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
    // D094: pre-quantized Q8_1 src1 is consumed directly by MMQ kernels.
    const bool qy_needs_dequant = !quantize_y && src1->type != GGML_TYPE_Q8_1 && ((src1->type != f16_type && !y_f32_kernel) || y_non_contig);

    if (qx_needs_dequant) {
        // Fall back to dequant + f16 mulmat
        mmp = ggml_vk_get_mul_mat_mat_id_pipeline(ctx, f16_type, y_f32_kernel ? GGML_TYPE_F32 : f16_type, (ggml_prec)dst->op_params[0]);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint32_t kpad = quantize_y ? 0 : ggml_vk_align_size(ne10, ggml_vk_guess_matmul_id_pipeline_align(ctx, mmp, ne01, nei1, qx_needs_dequant ? f16_type : src0->type));
    const bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_id_pipeline(ctx, mmp, ne01, nei1, aligned, qx_needs_dequant ? f16_type : src0->type);

    if (ggml_vk_device_size(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) :ne11;
    const uint64_t x_ne = ggml_nelements(src0);
    const uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    const uint64_t d_ne = ggml_nelements(dst);

    const uint64_t qx_sz = ggml_vk_device_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);
    const uint64_t qy_sz = ggml_vk_device_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = !qx_needs_dequant ? qx_sz : sizeof(ggml_fp16_t) * x_ne;
    const uint64_t y_sz = quantize_y ? (ggml_vk_align_size(y_ne, 128) * ggml_vk_device_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) : (y_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne);
    const uint64_t ids_sz = nbi2;
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    vk_pipeline to_q8_1 = nullptr;

    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type);
    } else {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, f16_type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    if (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline count_experts = ctx->device->pipeline_count_experts;

    uint32_t expert_count_size = sizeof(uint32_t) * n_as;

    {
        if (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.maxStorageBufferRange)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (qx_needs_dequant && ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (ctx->prealloc_size_split_k < expert_count_size) {
            ctx->prealloc_size_split_k = expert_count_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        if (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        if (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        if (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, count_experts, 1);
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    const uint64_t d_buf_offset = vk_tensor_view_offset(dst);
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_buffer;
        qx_buf_offset = vk_tensor_view_offset(src0);
        GGML_ASSERT(d_Qx != nullptr);
    }
    if (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_view_offset(src1);
        GGML_ASSERT(d_Qy != nullptr);
    }
    if (!ids_uma) {
        d_ids = ids_buf_ctx->dev_buffer;
        ids_buf_offset = vk_tensor_view_offset(ids);
        GGML_ASSERT(d_ids != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } else if (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    if (x_non_contig || qx_needs_dequant) {
        if (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    // Count how many times each expert is used
    vk_subbuffer expert_count_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_split_k, 0);
    if (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        const std::vector<uint32_t> pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_size(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_size(ids->type)),
                                           (uint32_t)(get_misalign_bytes(ctx, ids) / ggml_vk_device_type_size(ids->type)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, count_experts,
            { vk_subbuffer{ d_ids, ids_buf_offset, ids_sz }, expert_count_buf }, pc, { (uint32_t)n_as, 1, 1});
    }

    if (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_offset), ggml_vk_subbuffer(ctx, d_X, 0));
    } else if (qx_needs_dequant) {
        const std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    if (y_non_contig) {
        if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
        }
    }
    if (quantize_y) {
        if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    uint32_t stride_batch_x = ne00*ne01;
    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_sz }, { d_Y, y_buf_offset, y_sz },
        { d_D, d_buf_offset, d_sz }, { d_ids, ids_buf_offset, ids_sz }, expert_count_buf,
        ne01, ne21, ne10, ne10, ne10, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_size(ids->type), ne11, padded_n
    );  // NOLINT

    if (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = true;
    }
    if (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = true;
    }
    ctx->prealloc_split_k_need_sync = true;
}

static void ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * ids = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    const uint64_t nei0 = ids->ne[0];
    const uint64_t nei1 = ids->ne[1];
    const uint32_t nbi1 = (uint32_t)(ids->nb[1] / sizeof(int));

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    const bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;
    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : nullptr;
    vk_pipeline to_q8_1 = nullptr;

    if (dmmv == nullptr) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = false;
    }

    if (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    const bool qx_needs_dequant = x_non_contig;
    // D094: pre-quantized Q8_1 src1 is consumed directly by MMVQ kernels.
    const bool qy_needs_dequant = !quantize_y && src1->type != GGML_TYPE_Q8_1 && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    if (ggml_vk_device_size(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    const uint64_t x_ne = ggml_nelements(src0);
    const uint64_t y_ne = ggml_nelements(src1);

    const uint64_t qx_sz = ggml_vk_align_size(ggml_vk_device_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t x_sz = x_non_contig ? ggml_vk_align_size(ggml_vk_device_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) : qx_sz;
    const uint64_t y_sz = quantize_y ? (ggml_vk_align_size(y_ne, 128) * ggml_vk_device_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne);

    {
        if (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.maxStorageBufferRange)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (qx_needs_dequant && ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        if (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        if (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        if (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[node_idx + ctx->num_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    if (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } else {
        d_X = d_Qx;
    }
    if (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } else {
        d_Y = d_Qy;
    }

    if (x_non_contig) {
        if (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    if (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_size(ggml_vk_device_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, d_Qx, d_X);
    }
    if (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_vk_device_type_size(src1->type) * y_ne);
        if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
        }
    }
    if (quantize_y) {
        if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
        }
    }

    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_size(src1->type);
    }

    const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    if (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    if (ctx->num_additional_fused_ops > 0) {
        const ggml_tensor * bias = cgraph->nodes[node_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        if (cgraph->nodes[node_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } else {
            GGML_ASSERT(cgraph->nodes[node_idx + 1]->op == GGML_OP_ADD_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    if (ctx->num_additional_fused_ops > 1) {
        const ggml_tensor * scale = cgraph->nodes[node_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    for (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        const vk_mat_vec_id_push_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            (uint32_t)(ne00 * ne01), stride_batch_y, (uint32_t)(ne20 * ne21),
            fusion_flags,
            (uint32_t)nei0, (uint32_t)ne11, expert_i1, nbi1
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
            {
                d_X,
                d_Y,
                d_D,
                d_F0,
                d_F1,
                d_ids,
            },
            pc, { groups_x, (uint32_t)nei0, groups_z });
    }

    if (x_non_contig) {
        ctx->prealloc_x_need_sync = true;
    }
    if (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = true;
    }
}

static bool ggml_vk_use_mul_mat_vec_id(const struct ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src2 = dst->src[2];
    return (src2->ne[1] <= 8) && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

static void ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    if (ggml_vk_use_mul_mat_vec_id(cgraph, node_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, node_idx);
    } else {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

static bool ggml_vk_flash_attn_scalar_shmem_support(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type kv_type) {
    GGML_UNUSED(f32acc);
    // Needs to be kept up to date on shader changes
    const uint32_t wg_size = params.workgroup_size;
    const uint32_t Br = params.block_rows;
    const uint32_t Bc = params.block_cols;

    const uint32_t float_type_size = device->fp16 ? sizeof(ggml_fp16_t) : sizeof(float);

    const bool mmq = device->integer_dot_product && device->subgroup_clustered &&
                     (kv_type == GGML_TYPE_Q4_0 || kv_type == GGML_TYPE_Q4_1 ||
                      kv_type == GGML_TYPE_Q5_0 || kv_type == GGML_TYPE_Q5_1 ||
                      kv_type == GGML_TYPE_Q8_0 || kv_type == GGML_TYPE_IQ4_NL);

    // tmpsh is overestimated slightly
    const uint32_t tmpsh = wg_size * sizeof(float);
    const uint32_t tmpshv4 = wg_size * 4 * float_type_size;

    const uint32_t masksh = Bc * (Br + 1) * float_type_size;

    uint32_t Qf, kvsh, kblocksh_size;
    if (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        const uint32_t block_b_size = 8 * sizeof(int32_t) + 2 * float_type_size;
        Qf = Br * (hsk / 32) * block_b_size;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_size : 4 * float_type_size;

        // block_a_cache size depends on quant type
        uint32_t block_a_size;
        switch (kv_type) {
            case GGML_TYPE_Q4_0:  block_a_size = 4 * sizeof(uint32_t) + float_type_size; break;
            case GGML_TYPE_Q4_1:  block_a_size = 4 * sizeof(uint32_t) + 2 * float_type_size; break;
            case GGML_TYPE_Q5_0:  block_a_size = 4 * sizeof(uint32_t) + sizeof(uint32_t) + float_type_size; break;
            case GGML_TYPE_Q5_1:  block_a_size = 4 * sizeof(uint32_t) + sizeof(uint32_t) + 2 * float_type_size; break;
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_IQ4_NL: block_a_size = 8 * sizeof(int32_t) + float_type_size; break;
            default: block_a_size = 0; break;
        }
        kblocksh_size = params.shmem_staging ? Bc * (hsk / 32) * block_a_size : block_a_size;
    } else {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_size;

        const uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_size : 4 * float_type_size;

        kblocksh_size = 0;
    }

    const uint32_t total_size = tmpsh + tmpshv4 + masksh + Qf + kvsh + kblocksh_size;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << total_size << ", supported=" << supported);

    return supported;
}

static bool ggml_vk_flash_attn_coopmat_shmem_support(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc) {
    // Needs to be kept up to date on shader changes
    const uint32_t Br = params.block_rows;
    const uint32_t Bc = params.block_cols;

    const uint32_t MatBr = 16, MatBc = 16;

    const uint32_t row_split = Bc / MatBc;

    const uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    const uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    const uint32_t acctype = f32acc ? 4 : 2;
    const uint32_t f16vec4 = 8;

    const uint32_t tmpsh = (Bc / MatBc) * sizeof(float);

    const uint32_t qstride = hsk_pad / 4 + 2;
    const uint32_t Qf = Br * qstride * f16vec4;

    const uint32_t psh_stride = Br / 4 + 2;
    const uint32_t Psh = Bc * psh_stride * f16vec4;

    const uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    const uint32_t sfsh = Bc * sfshstride * acctype;

    const uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    const uint32_t vsh_stride = MatBc / 4 * row_split;
    const uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    const uint32_t osh_stride = params.row_split * MatBr / 4;
    const uint32_t pvsh = MatBc * osh_stride * f16vec4;

    const uint32_t slope = Br * acctype;

    const uint32_t total_size = tmpsh + Qf + Psh + sfsh + ksh + pvsh + slope;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << total_size << ", supported=" << supported);

    return supported;
}

static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask, const ggml_tensor * sinks, const ggml_tensor * k_scale, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    if (sinks) {
        std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] << ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] << ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "))");

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const uint32_t nem0 = mask ? mask->ne[0] : 0;
    const uint32_t nem1 = mask ? mask->ne[1] : 0;
    const uint32_t nem2 = mask ? mask->ne[2] : 0;
    const uint32_t nem3 = mask ? mask->ne[3] : 0;

    const uint32_t HSK = nek0;
    const uint32_t HSV = nev0;
    uint32_t N = neq1;
    const uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    assert(dst->type == GGML_TYPE_F32);
    assert(q->type == GGML_TYPE_F32);
    uint32_t gqa_ratio = 1;
    uint32_t qk_ratio = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    bool f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32;

    // Diagnostic knob: force f16 accumulator FA (local fork). On RDNA4 the
    // f16acc coopmat path has roughly double the throughput of f32acc and the
    // P/sfsh shared-memory staging is halved. f32acc=0 enables f16acc.
    if (const char * env = getenv("GGML_VK_FA_F32ACC")) {
        f32acc = std::atoi(env) != 0;
    }

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k->type, v->type, f32acc);
    const uint32_t max_gqa = std::min(tuning_params.block_rows, 32u);

    if (N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa &&
        qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_ratio = qk_ratio;
        N = gqa_ratio;
        workgroups_y /= gqa_ratio;
    }

    // D094 cycle-6 (local fork): ROCm-style pre-dequant for q8_0 / f8_e4m3 KV on prefill.
    // K/V are converted to f16 in a separate pass outside the FA kernel, so the
    // coopmat1 kernel runs the pure-f16 path (no per-j dequant staging, which was
    // ~32% of FA time on q8_0). Decode (N == 1) keeps the quantized KV untouched
    // (smaller KV reads are faster there). Same idea as ggml-cuda's f16_extra_data path.
    // f8_e4m3 added D095-D3: without it the forced shmem_staging=1 pushes coopmat1
    // shmem over the RDNA4 32 KiB limit and prefill falls back to scalar FA (-36% ptps).
    // D096-A2 (local fork): fp8 native FA kernel - the kernel reads the f8 KV
    // cache directly via fp8 coopmat (no f16 preconvert staging). Opt-in via
    // GGML_VK_FA_F8_NATIVE; f32acc only (see get_fa_tuning_params).
    const bool fa_f8_native = (k->type == GGML_TYPE_F8_E4M3) && v->type == k->type && neq1 > 1 && f32acc &&
                              getenv("GGML_VK_FA_F8_NATIVE") != nullptr;

    // D096 P1: route raw f8 K/V through the generated DATA_A_F8_E4M3 cm1
    // shader during prefill. The env is also consumed by FA tuning, where it
    // can make the direct cm1 layout fit for GQA decode instead of falling back
    // to scalar; it is therefore a broad route probe, not an isolated scalar
    // staging switch.
    const bool fa_f8_direct = !fa_f8_native &&
                              (k->type == GGML_TYPE_F8_E4M3) && v->type == k->type && neq1 > 1 &&
                              k_scale == nullptr &&
                              getenv("GGML_VK_FA_F8_DIRECT") != nullptr;

    // D096 D4.3 (local fork): preconvert quantized prefill K/V to f16 before
    // cooperative FA. Decode stays on the compact scalar route. A 2026-08-13
    // audit found that the earlier 13.4 tps "direct f8" result had inherited
    // GGML_VK_FA_F8_NATIVE_DECODE and was actually the diagnostic native
    // cooperative route. Clean scalar f8 measured 25.45 tps at 49K versus
    // 27.18 for f16; forced native f8 reproduced 13.28 tps.
    const bool fa_preconvert = !fa_f8_native && !fa_f8_direct &&
                               (k->type == GGML_TYPE_Q8_0 || k->type == GGML_TYPE_F8_E4M3) &&
                               v->type == k->type && neq1 > 1 &&
                               getenv("GGML_VK_FA_NO_PRECONVERT") == nullptr;
    const ggml_type k_type_eff = fa_preconvert ? GGML_TYPE_F16 : k->type;
    const ggml_type v_type_eff = fa_preconvert ? GGML_TYPE_F16 : v->type;

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    if (tuning_params.path != FA_COOPMAT2) {
        GGML_ASSERT(k->type == v->type || (k->type == GGML_TYPE_F8_E4M3 && v->type == GGML_TYPE_F16));
    }

    const uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_size(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_size(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_size(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    if (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    if (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    size_t fa_k16_sz = 0;
    size_t fa_v16_sz = 0;
    size_t nbk2_eff = nbk2;
    size_t nbk3_eff = nbk3;
    size_t nbv2_eff = nbv2;
    size_t nbv3_eff = nbv3;
    if (fa_preconvert) {
        fa_k16_sz = (size_t) nek0 * nek1 * nek2 * nek3 * sizeof(ggml_fp16_t);
        fa_v16_sz = (size_t) nev0 * nev1 * nev2 * nev3 * sizeof(ggml_fp16_t);
        if (fa_k16_sz > ctx->device->properties.limits.maxStorageBufferRange ||
            fa_v16_sz > ctx->device->properties.limits.maxStorageBufferRange) {
            GGML_ABORT("Requested FA pre-dequant buffer size is too large");
        }
        if (ctx->prealloc_size_fa_k16 < fa_k16_sz) {
            if (ctx->prealloc_fa_k16_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
                ctx->prealloc_fa_k16_need_sync = false;
            }
            ctx->prealloc_size_fa_k16 = fa_k16_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (ctx->prealloc_size_fa_v16 < fa_v16_sz) {
            if (ctx->prealloc_fa_v16_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
                ctx->prealloc_fa_v16_need_sync = false;
            }
            ctx->prealloc_size_fa_v16 = fa_v16_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        k_stride = nek0; // f16 elements per row (dense f16 copy)
        v_stride = nev0;
        nbk2_eff = (size_t) k_stride * sizeof(ggml_fp16_t) * nek1;
        nbk3_eff = nbk2_eff * nek2;
        nbv2_eff = (size_t) v_stride * sizeof(ggml_fp16_t) * nev1;
        nbv3_eff = nbv2_eff * nev2;
    }

    const uint32_t alignment = tuning_params.block_cols;
    bool aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    if (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = false;
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    bool use_mask_opt = mask && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.block_cols * 16;
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   mask != nullptr, use_mask_opt, logit_softcap != 0, k_type_eff, v_type_eff);

    vk_pipeline pipeline = nullptr;

    {
        std::lock_guard<std::recursive_mutex> guard(ctx->device->mutex);
        // Use the effective type: with fa_preconvert the q8_0 KV is staged as f16,
        // so the pure-f16 coopmat pipeline must be selected (FaTypeK spec constant).
        auto &pipelines = ctx->device->pipeline_flash_attn_f32_f16[k_type_eff];
        auto it = pipelines.find(fa_pipeline_state);
        if (it != pipelines.end()) {
            pipeline = it->second;
        } else {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    assert(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t split_kv = KV;
    uint32_t split_k = 1;

    // Intel Alchemist prefers more workgroups
    const uint32_t shader_core_count_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    const uint32_t shader_core_count = ctx->device->shader_core_count ? ctx->device->shader_core_count * shader_core_count_multiplier : 16;

    const uint32_t Br = fa_pipeline_state.Br;
    const uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    const uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    if (gqa_ratio > 1 && workgroups_x <= Br) {
        split_k = shader_core_count * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } else if (gqa_ratio <= 1) {
        uint32_t total_wgs_no_split = Tr * workgroups_y * workgroups_z;
        if (total_wgs_no_split < shader_core_count * 2) {
            split_k = shader_core_count * 2 / total_wgs_no_split;
        }
    }

    // Diagnostic knob: force additional KV splitting for long-prefill FA
    // experiments (local fork). Larger split_k trades a small reduce pass for
    // shorter per-workgroup KV loops and more independent workgroups.
    if (const char * env = getenv("GGML_VK_FA_FORCE_SPLIT_K")) {
        split_k = std::max(split_k, (uint32_t) std::max(1, std::atoi(env)));
    }
    // D094 c8 diagnostic: disable split_k entirely (isolate reduce-path issues).
    if (getenv("GGML_VK_FA_DISABLE_SPLIT_K")) {
        split_k = 1;
    }
    // D096-A2: the fp8 native kernel does not yet write split-local O/L/M with
    // the split index offset; force split_k=1 (single-KV-chunk dispatch).
    if (fa_pipeline_state.path == FA_F8_NATIVE) {
        split_k = 1;
    }

    if (split_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        split_kv = ROUNDUP_POW2(std::max(1u, KV / split_k), alignment);
        split_k = CEIL_DIV(KV, split_kv);
    }
    static const bool route_trace = std::getenv("GGML_VK_FA_ROUTE_TRACE") != nullptr;
    if (route_trace && pipeline) {
        static std::mutex route_trace_mutex;
        static std::set<std::string> route_trace_seen;

        std::ostringstream key;
        key << pipeline->name
            << "|path=" << ggml_vk_fa_path_name(fa_pipeline_state.path)
            << "|k=" << ggml_type_name(k->type)
            << "|v=" << ggml_type_name(v->type)
            << "|q=" << ggml_type_name(q->type)
            << "|HSK=" << HSK
            << "|HSV=" << HSV
            << "|N=" << N
            << "|KV=" << KV
            << "|Br=" << Br
            << "|Bc=" << Bc
            << "|D_split=" << fa_pipeline_state.D_split
            << "|row_split=" << fa_pipeline_state.row_split
            << "|shmem_staging=" << fa_pipeline_state.shmem_staging
            << "|workgroup_size=" << fa_pipeline_state.workgroup_size
            << "|subgroup_size=" << fa_pipeline_state.subgroup_size
            << "|aligned=" << fa_pipeline_state.aligned
            << "|f32acc=" << fa_pipeline_state.f32acc
            << "|flags=" << fa_pipeline_state.flags
            << "|limit_occupancy_shmem=" << fa_pipeline_state.limit_occupancy_shmem
            << "|use_mask_opt=" << use_mask_opt
            << "|gqa_ratio=" << gqa_ratio
            << "|preconvert=" << fa_preconvert
            << "|f8_direct=" << fa_f8_direct
            << "|split_k=" << split_k
            << "|split_kv=" << split_kv
            << "|Tr=" << Tr
            << "|wg_x=" << workgroups_x
            << "|wg_y=" << workgroups_y
            << "|wg_z=" << workgroups_z
            << "|shader_cores=" << shader_core_count;

        std::lock_guard<std::mutex> guard(route_trace_mutex);
        if (route_trace_seen.insert(key.str()).second) {
            std::cerr << "ggml_vulkan: flash_attn route: " << key.str() << std::endl;
        }
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    const uint64_t split_k_size = split_k > 1 ? (HSV * ne1 * sizeof(float) + ne1 * sizeof(float) * 2) * split_k * ne2 * ne3 : 0;
    if (split_k_size > ctx->device->properties.limits.maxStorageBufferRange) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    if (ctx->prealloc_size_split_k < split_k_size) {
        ctx->prealloc_size_split_k = split_k_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    const uint32_t mask_opt_num_dwords = CEIL_DIV(nem0, 16 * Bc);
    const uint64_t mask_opt_size = sizeof(uint32_t) * mask_opt_num_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_mask_opt = nullptr;
    if (use_mask_opt) {
        std::lock_guard<std::recursive_mutex> guard(ctx->device->mutex);
        auto &pipelines = ctx->device->pipeline_fa_mask_opt;
        auto it = pipelines.find({Br, Bc});
        if (it != pipelines.end()) {
            pipeline_fa_mask_opt = it->second;
        } else {
            pipelines[{Br, Bc}] = pipeline_fa_mask_opt = std::make_shared<vk_pipeline_struct>();
        }
        assert(pipeline_fa_mask_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_mask_opt, 1);

        if (ctx->prealloc_size_y < mask_opt_size) {
            ctx->prealloc_size_y = mask_opt_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    const uint32_t n_head_kv   = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head_kv));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer mask_buf = mask ? ggml_vk_tensor_subbuffer(ctx, mask) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer mask_opt_buf = use_mask_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    // D131 R9: windowed f16 K block scales (one per 256 values), aligned with
    // the K view this call reads. Nullptr keeps binding 7 unused (flag 0).
    vk_subbuffer k_scale_buf = k_scale ? ggml_vk_tensor_subbuffer(ctx, k_scale) : q_buf;

    if (fa_preconvert) {
        // q8_0 / f8_e4m3 -> f16 conversion passes (dequant shaders, f16 output)
        const ggml_type kv_type = k->type;
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_dequant[kv_type], 2);
        const uint32_t k_nelem = (uint32_t)(nek0 * nek1 * nek2 * nek3);
        const uint32_t v_nelem = (uint32_t)(nev0 * nev1 * nev2 * nev3);
        const vk_subbuffer k16_buf{ ctx->prealloc_fa_k16, 0, fa_k16_sz };
        const vk_subbuffer v16_buf{ ctx->prealloc_fa_v16, 0, fa_v16_sz };
        // D131 R9: K dequant folds the block scale into the f16 copy (pc[2]=1);
        // V has no scale (pc[2]=0). M=n_kv, K=n_head_kv drive the permuted-K
        // scale indexing inside the f8 dequant shader.
        const std::vector<uint32_t> pc_k = { (uint32_t)nek1, (uint32_t)nek2, k_scale != nullptr ? 1u : 0u, 0, k_nelem };
        if (kv_type == GGML_TYPE_F8_E4M3) {
            ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_dequant[kv_type],
                                      { k_buf, k16_buf, k_scale_buf }, pc_k,
                                      { CEIL_DIV(k_nelem, 4096), 1, 1 });
        } else {
            ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_dequant[kv_type],
                                      { k_buf, k16_buf }, pc_k,
                                      { CEIL_DIV(k_nelem, 4096), 1, 1 });
        }
        const std::vector<uint32_t> pc_v = { 1, v_nelem, 0, 0, v_nelem };
        if (kv_type == GGML_TYPE_F8_E4M3) {
            ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_dequant[kv_type],
                                      { v_buf, v16_buf, q_buf }, pc_v,
                                      { CEIL_DIV(v_nelem, 4096), 1, 1 });
        } else {
            ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_dequant[kv_type],
                                      { v_buf, v16_buf }, pc_v,
                                      { CEIL_DIV(v_nelem, 4096), 1, 1 });
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k16_buf;
        v_buf = v16_buf;
    }

    // D096-C: the native fp8 FA kernel reads the mask as f16, but llama's
    // attn_inp_kq_mask is F32 (0.0/-inf). f16-views of f32 data read -inf as 0
    // (low 16 bits of 0xFF800000) so the mask silently does nothing. Convert the
    // mask F32->F16 into prealloc_fa_mask16 for the native path.
    if (fa_pipeline_state.path == FA_F8_NATIVE && mask && mask->type == GGML_TYPE_F32) {
        std::cerr << "ggml_vulkan: fa_native mask F32->F16 convert: nelem=" << (nem0 * nem1 * nem2 * nem3) << std::endl;
        const uint32_t mask_nelem = (uint32_t)(nem0 * nem1 * nem2 * nem3);
        const size_t mask16_sz = (size_t) mask_nelem * sizeof(ggml_fp16_t);
        if (ctx->prealloc_size_fa_mask16 < mask16_sz) {
            if (ctx->prealloc_fa_mask16_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
                ctx->prealloc_fa_mask16_need_sync = false;
            }
            ctx->prealloc_size_fa_mask16 = mask16_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_dequant[GGML_TYPE_F32], 2);
        const vk_subbuffer mask16_buf{ ctx->prealloc_fa_mask16, 0, mask16_sz };
        const std::vector<uint32_t> pc_m = { 1, mask_nelem, 0, 0, mask_nelem };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_dequant[GGML_TYPE_F32],
                                  { mask_buf, mask16_buf }, pc_m,
                                  { CEIL_DIV(mask_nelem, 4096), 1, 1 });
        ggml_vk_sync_buffers(ctx, subctx);
        mask_buf = mask16_buf;
        ctx->prealloc_fa_mask16_need_sync = true;
    } else if (fa_pipeline_state.path == FA_F8_NATIVE && mask) {
        std::cerr << "ggml_vulkan: fa_native mask type=" << ggml_type_name(mask->type) << " (no conversion)" << std::endl;
    }

    // D096-E: host-side KV/Q/mask dump (no kernel writes). On the first call
    // with a non-empty V cache (real prefill; warmup/checkpoint-init runs have
    // an all-zero cache) read K/V/Q/mask to files for offline NaN analysis.
    // v130: HOSTDUMP AND the kernel arm both fire on the first NQ>4 call with
    // a non-empty V cache, so hkv_*.bin and fa_kdump.bin describe the SAME
    // dispatch and can be cross-checked (expected S = Q*K in Python).
    static bool fa8_host_dumped = false;
    static bool fa_f8_dbg_armed = false;   // moved up from the dump-armed block
    static bool fa_f8_dbg_write = false;   // moved up from the dump-armed block
    static bool fa_f8_dbg_done = false;    // one arm/read cycle per session
    if (fa_pipeline_state.path == FA_F8_NATIVE && !fa8_host_dumped &&
        getenv("GGML_VK_FA8_HOSTDUMP") && neq1 > 4 && v && k) {
        const uint32_t v_bytes = (uint32_t)(nev0 * nev1 * nev2 * nev3);
        const uint32_t k_bytes = (uint32_t)(nek0 * nek1 * nek2 * nek3);
        std::vector<uint8_t> khead(std::min<size_t>(k_bytes, 4096));
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_buffer_read(k_buf.buffer, k_buf.offset, khead.data(), khead.size());
        bool any = false;
        for (uint8_t b : khead) { if (b != 0) { any = true; break; } }
        if (any) {
            std::vector<uint8_t> vfull(v_bytes), kfull(k_bytes);
            ggml_vk_buffer_read(v_buf.buffer, v_buf.offset, vfull.data(), v_bytes);
            ggml_vk_buffer_read(k_buf.buffer, k_buf.offset, kfull.data(), k_bytes);
            const uint32_t q_bytes = (uint32_t)(q->ne[0] * q->ne[1] * q->ne[2] * q->ne[3] * 4);
            std::vector<uint8_t> qfull(q_bytes);
            ggml_vk_buffer_read(q_buf.buffer, q_buf.offset, qfull.data(), q_bytes);
            FILE * f = fopen("build_logs/agent-workload/hkv_k.bin", "wb");
            if (f) { fwrite(kfull.data(), 1, kfull.size(), f); fclose(f); }
            f = fopen("build_logs/agent-workload/hkv_v.bin", "wb");
            if (f) { fwrite(vfull.data(), 1, vfull.size(), f); fclose(f); }
            f = fopen("build_logs/agent-workload/hkv_q.bin", "wb");
            if (f) { fwrite(qfull.data(), 1, qfull.size(), f); fclose(f); }
            if (mask) {
                const uint32_t m_bytes = (uint32_t)(mask->ne[0] * mask->ne[1] * mask->ne[2] * mask->ne[3] * ggml_element_size(mask));
                std::vector<uint8_t> mfull(m_bytes);
                ggml_vk_buffer_read(mask_buf.buffer, mask_buf.offset, mfull.data(), m_bytes);
                f = fopen("build_logs/agent-workload/hkv_mask.bin", "wb");
                if (f) { fwrite(mfull.data(), 1, mfull.size(), f); fclose(f); }
            }
            std::cerr << "ggml_vulkan: fa_native HOSTDUMP saved (NQ=" << neq1 << " K=" << k_bytes
                      << " V=" << v_bytes << " Q=" << q_bytes << ")" << std::endl;
            fa8_host_dumped = true;
            // arm the kernel dump on this same call: the kernel writes the
            // debug buffer during this dispatch (pc bit 25); READ picks it up
            // on the next native call.
            GGML_ASSERT(ggml_vk_fa8_dbg_buf_ref() && "fa_native dump buffer must be initialized in load_shaders");
            fa_f8_dbg_write = true;
            fa_f8_dbg_armed = true;
            std::cerr << "ggml_vulkan: fa_native kernel dump ARMED (NQ=" << neq1 << " N=" << N << " KV=" << KV
                      << " gqa=" << gqa_ratio << " HSK=" << HSK << " host-dump call)" << std::endl;
        }
    }

// D096-C diagnostic v2 (kernel-side): the kernel writes S (after mask add),
// the applied raw mask, M and L to a debug buffer AT EXECUTION time when
// pc bit 25 of mask_n_head_log2 is set. Armed together with the HOSTDUMP
// (same call - first NQ>4 with a non-empty V cache); one-shot write.
    if (fa_pipeline_state.path == FA_F8_NATIVE && getenv("GGML_VK_FA_F8_DUMP") && !fa_f8_dbg_done) {
        if (!fa_f8_dbg_armed) {
            if (neq1 > 4) {
                GGML_ASSERT(ggml_vk_fa8_dbg_buf_ref() && "fa_native dump buffer must be initialized in load_shaders");
                fa_f8_dbg_armed = true;
                fa_f8_dbg_write = true;  // single write: only the arm call sets bit 25
                std::cerr << "ggml_vulkan: fa_native kernel dump ARMED (NQ=" << neq1 << " N=" << N << " KV=" << KV << " gqa=" << gqa_ratio << " HSK=" << HSK << ")" << std::endl;
            }
        } else {
            ggml_vk_sync_buffers(ctx, subctx);
            std::vector<float> dbg(512 * 1024 / 4);
            ggml_vk_buffer_read(ggml_vk_fa8_dbg_buf_ref(), 0, dbg.data(), dbg.size() * sizeof(float));
            if (dbg[0] == 0.0f) {
                std::cerr << "ggml_vulkan: fa_native kernel dump SKIP (empty cache, call NQ=" << neq1 << " N=" << N << " KV=" << KV << " gqa=" << gqa_ratio << ")" << std::endl;
                // keep armed: wait for the first call with real (non-zero) data
            } else {
                FILE * f = fopen("build_logs/agent-workload/fa_kdump.bin", "wb");
                if (f) {
                    fwrite(dbg.data(), sizeof(float), dbg.size(), f);
                    fclose(f);
                    std::cerr << "ggml_vulkan: fa_native kernel dump READ (NQ=" << neq1 << " N=" << N << " KV=" << KV << " gqa=" << gqa_ratio << " dbg0=" << dbg[0] << ")" << std::endl;
                } else {
                    std::cerr << "ggml_vulkan: fa_native kernel dump FAILED to open file" << std::endl;
                }
                // D096 v101: dump dst on the same call as the kernel dump, so
                // store markers and the output buffer come from one dispatch.
                const size_t out_nelem2 = (size_t) ne0 * ne1 * ne2 * ne3;
                std::vector<float> out2(out_nelem2);
                ggml_vk_buffer_read(dst_buf.buffer, dst_buf.offset, out2.data(), out2.size() * sizeof(float));
                FILE * f2 = fopen("build_logs/agent-workload/fa_out_samecall.bin", "wb");
                if (f2) {
                    fwrite(out2.data(), sizeof(float), out2.size(), f2);
                    fclose(f2);
                    std::cerr << "ggml_vulkan: FA OUTDUMP samecall NQ=" << neq1 << " nelem=" << out_nelem2 << std::endl;
                }
                fa_f8_dbg_armed = false;
                fa_f8_dbg_done = true;
            }
        }
    }
    uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | n_head_log2;
    // D131 R9: bit 26 tells the scalar shader to apply binding 7 (K scales)
    if (k_scale != nullptr && tuning_params.path == FA_SCALAR && !fa_preconvert) {
        mask_n_head_log2 |= (1u << 26);
    }
    if (fa_pipeline_state.path == FA_F8_NATIVE && fa_f8_dbg_write) {
        mask_n_head_log2 |= (1u << 25);
        fa_f8_dbg_write = false;  // one-shot: only the arm call writes the debug buffer
    }

    if (use_mask_opt)
    {
        const vk_op_flash_attn_mask_opt_push_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(mask->nb[1] / sizeof(ggml_fp16_t)),
            (uint32_t)(mask->nb[2] / sizeof(ggml_fp16_t)),
            (uint32_t)(mask->nb[3] / sizeof(ggml_fp16_t)),
            mask_opt_num_dwords,
            mask_opt_num_dwords * CEIL_DIV(nem1, Br),
            mask_opt_num_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_mask_opt,
                                  { mask_buf, mask_opt_buf }, opt_pc,
                                  { mask_opt_num_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    // D131 R9: the scale multiplies the finished score inside the main kernel
    // (before the partial sums land in split_k_buffer), so split_k needs no
    // changes in the reduce pass.
    const vk_flash_attn_push_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, (uint32_t)nbk2_eff, (uint32_t)nbk3_eff,
                                              v_stride, (uint32_t)nbv2_eff, (uint32_t)nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              mask_n_head_log2, m0, m1,
                                              gqa_ratio, split_kv, split_k };

    if (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_split_k_reduce, 1);

        if (ctx->prealloc_split_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        if (gqa_ratio > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = split_k * workgroups_x;
        } else {
            dispatch_x = Tr * split_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer split_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_split_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                  {q_buf, k_buf, v_buf, mask_buf, sinks_buf, split_k_buf, mask_opt_buf, k_scale_buf},
                                  pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        const vk_op_flash_attn_split_k_reduce_push_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, split_k, (sinks != nullptr) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_split_k_reduce,
                                    {split_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_split_k_need_sync = true;
    } else {
        if (gqa_ratio > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        // D096-C: the native FA pipeline has parameter_count=8 (debug binding 7
        // is unused unless the kernel dump is armed); always pass 8 subbuffers.
        if (fa_pipeline_state.path == FA_F8_NATIVE && fa_f8_dbg_armed) {
            const vk_subbuffer dbg_sb{ ggml_vk_fa8_dbg_buf_ref(), 0, 512 * 1024 };
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                      {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, dbg_sb},
                                      pc, { workgroups_x, workgroups_y, workgroups_z });
        } else if (fa_pipeline_state.path == FA_F8_NATIVE) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                      {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, mask_opt_buf},
                                      pc, { workgroups_x, workgroups_y, workgroups_z });
        } else if (tuning_params.path == FA_SCALAR) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                      {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, k_scale_buf},
                                      pc, { workgroups_x, workgroups_y, workgroups_z });
        } else {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                      {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, k_scale_buf},
                                      pc, { workgroups_x, workgroups_y, workgroups_z });
        }
        fa_f8_dbg_write = false;  // single-write: bit 25 applies to the arm call only
    }

    // D096 v104: dump dst after the LAST non-armed native dispatch (final
    // layer output, after all preceding layers overwrote dst).
    static bool fa_out_final_dumped = false;
    if (fa_pipeline_state.path == FA_F8_NATIVE && !fa_out_final_dumped &&
        !fa_f8_dbg_armed && getenv("GGML_VK_FA_F8_DUMP") && neq1 > 4) {
        ggml_vk_sync_buffers(ctx, subctx);
        const size_t out_nelem = (size_t) ne0 * ne1 * ne2 * ne3;
        std::vector<float> out(out_nelem);
        ggml_vk_buffer_read(dst_buf.buffer, dst_buf.offset, out.data(), out.size() * sizeof(float));
        FILE * f = fopen("build_logs/agent-workload/fa_out_final.bin", "wb");
        if (f) {
            fwrite(out.data(), sizeof(float), out.size(), f);
            fclose(f);
            std::cerr << "ggml_vulkan: FA OUTDUMP final NQ=" << neq1 << " nelem=" << out_nelem << std::endl;
        }
        fa_out_final_dumped = true;
    }

    // D096 v106: dump dst on every real prefill call to a numbered file
    // (fa_out_native_<n>.bin / fa_out_fallback_<n>.bin). Pick the last
    // prefill layer from the log; avoids MTP/decode overwrites.
    static int fa_out_seq = 0;
    if (neq1 > 4 && getenv("GGML_VK_FA_OUTDUMP")) {
        ggml_vk_sync_buffers(ctx, subctx);
        const size_t out_nelem = (size_t) ne0 * ne1 * ne2 * ne3;
        std::vector<float> out(out_nelem);
        ggml_vk_buffer_read(dst_buf.buffer, dst_buf.offset, out.data(), out.size() * sizeof(float));
        const char * out_fmt = fa_pipeline_state.path == FA_F8_NATIVE
            ? "build_logs/agent-workload/fa_out_native_%d.bin"
            : "build_logs/agent-workload/fa_out_fallback_%d.bin";
        char out_path[256];
        snprintf(out_path, sizeof(out_path), out_fmt, fa_out_seq++);
        FILE * f = fopen(out_path, "wb");
        if (f) {
            fwrite(out.data(), sizeof(float), out.size(), f);
            fclose(f);
            std::cerr << "ggml_vulkan: FA OUTDUMP saved " << out_path
                      << " NQ=" << neq1 << " nelem=" << out_nelem << std::endl;
        }
    }

    if (fa_preconvert) {
        ctx->prealloc_fa_k16_need_sync = true;
        ctx->prealloc_fa_v16_need_sync = true;
    }
}
