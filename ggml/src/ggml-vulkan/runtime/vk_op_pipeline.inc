// Generic GGML operation support checks and operation-to-pipeline mapping.
static vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    auto n_tiles = [&](vk_conv_shapes s) {
        return CEIL_DIV(K, vk_conv_block_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_block_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    const uint32_t shader_core_count = ctx->device->shader_core_count > 0 ? ctx->device->shader_core_count : 32;

    if (K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_count * 2) {
        return CONV_SHAPE_128x128;
    } else if (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_count * 2) {
        return CONV_SHAPE_32x256;
    } else {
        return CONV_SHAPE_64x32;
    }
}

static vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * dst, ggml_op op) {
    switch (op) {
    case GGML_OP_GET_ROWS:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        if (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            return ctx->device->pipeline_get_rows[src0->type];
        }
        if (dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_get_rows[src0->type];
        }
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_get_rows_f32[src0->type];
        }
        return nullptr;
    case GGML_OP_ACC:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_acc_f32;
        }
        return nullptr;
    case GGML_OP_SET:
        if (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            return ctx->device->pipeline_set_f32;
        }
        return nullptr;
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            return nullptr;
        }
        switch (op) {
        case GGML_OP_ADD:
        {
            if (ctx->num_additional_fused_ops > 0) {
                if (ctx->do_add_rms_partials) {
                    return ctx->device->pipeline_multi_add_rms[ctx->num_additional_fused_ops];
                } else {
                    return ctx->device->pipeline_multi_add[ctx->num_additional_fused_ops];
                }
            }
            if (ctx->do_add_rms_partials) {
                auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_rms_norepeat : ctx->device->pipeline_add_rms;
                return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } else {
                auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_norepeat : ctx->device->pipeline_add;
                return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        case GGML_OP_SUB:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        case GGML_OP_MUL:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        case GGML_OP_DIV:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        default:
            break;
        }
        return nullptr;
    case GGML_OP_ADD_ID:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_add_id_f32;
        }
        return nullptr;
    case GGML_OP_CONCAT:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_concat_f32;
        }
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_concat_f16;
        }
        if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
            return ctx->device->pipeline_concat_i32;
        }
        return nullptr;
    case GGML_OP_UPSCALE:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            switch (mode) {
                case GGML_SCALE_MODE_NEAREST:
                    return ctx->device->pipeline_upscale_nearest_f32;
                case GGML_SCALE_MODE_BILINEAR:
                    return ctx->device->pipeline_upscale_bilinear_f32;
                case GGML_SCALE_MODE_BICUBIC:
                    return ctx->device->pipeline_upscale_bicubic_f32;
                case GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    return ctx->device->pipeline_upscale_bilinear_antialias_f32;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    case GGML_OP_SCALE:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_scale_f32;
        }
        return nullptr;
    case GGML_OP_SQR:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sqr_f32;
        }
        return nullptr;
    case GGML_OP_SQRT:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sqrt_f32;
        }
        return nullptr;
    case GGML_OP_SIN:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sin_f32;
        }
        return nullptr;
    case GGML_OP_COS:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_cos_f32;
        }
        return nullptr;
    case GGML_OP_LOG:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_TRI:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_DIAG:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_CLAMP:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_clamp_f32;
        }
        return nullptr;
    case GGML_OP_PAD:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_pad_f32;
        }
        return nullptr;
    case GGML_OP_ROLL:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_roll_f32;
        }
        return nullptr;
    case GGML_OP_REPEAT:
        if (ggml_type_size(src0->type) == sizeof(float) && ggml_type_size(dst->type) == sizeof(float)) {
            return ctx->device->pipeline_repeat_f32;
        }
        return nullptr;
    case GGML_OP_REPEAT_BACK:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_repeat_back_f32;
        }
        return nullptr;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        return ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    case GGML_OP_SET_ROWS:
        if (src1->type == GGML_TYPE_I64) {
            return ctx->device->pipeline_set_rows_i64[dst->type];
        } else {
            return ctx->device->pipeline_set_rows_i32[dst->type];
        }
    case GGML_OP_SILU_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_silu_back_f32;
        }
        return nullptr;
    case GGML_OP_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_norm_f32;
        }
        return nullptr;
    case GGML_OP_GROUP_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_group_norm_f32;
        }
        return nullptr;
    case GGML_OP_RMS_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            if (ctx->do_add_rms_partials) {
                return ctx->num_additional_fused_ops > 0 ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            } else {
                return ctx->num_additional_fused_ops > 0 ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
            }
        }
        return nullptr;
    case GGML_OP_RMS_NORM_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rms_norm_back_f32;
        }
        return nullptr;
    case GGML_OP_L2_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_l2_norm_f32;
        }
        return nullptr;
    case GGML_OP_TURBO_WHT:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_turbo_wht_f32;
        }
        return nullptr;
    case GGML_OP_UNARY:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            return nullptr;
        }

        switch (ggml_get_unary_op(dst)) {
            case GGML_UNARY_OP_EXP:
                return ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_ELU:
                return ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SILU:
                return ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU:
                return ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU_ERF:
                return ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU_QUICK:
                return ctx->device->pipeline_gelu_quick[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_RELU:
                return ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_XIELU:
                return ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_NEG:
                return ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_TANH:
                return ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SIGMOID:
                return ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_HARDSIGMOID:
                return ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_HARDSWISH:
                return ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_ABS:
                return ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SOFTPLUS:
                return ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_STEP:
                return ctx->device->pipeline_step[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_ROUND:
                return ctx->device->pipeline_round[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_CEIL:
                return ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_FLOOR:
                return ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_TRUNC:
                return ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SGN:
                return ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            default:
                break;
        }
        return nullptr;
    case GGML_OP_GLU:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            return nullptr;
        }

        switch (ggml_get_glu_op(dst)) {
            case GGML_GLU_OP_GEGLU:
                return ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_REGLU:
                return ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_SWIGLU:
                return ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_SWIGLU_OAI:
                return ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_GEGLU_ERF:
                return ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_GEGLU_QUICK:
                return ctx->device->pipeline_geglu_quick[dst->type == GGML_TYPE_F16];
            default:
                break;
        }
        return nullptr;
    case GGML_OP_DIAG_MASK_INF:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_diag_mask_inf_f32;
        }
        return nullptr;
    case GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        if (ctx->num_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(float(dst->ne[0])));
            GGML_ASSERT(idx < num_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            bool use_push = dst->ne[0] != (1u << idx);
            return ctx->device->pipeline_topk_moe[idx][use_push];
        }

        if (src0->type == GGML_TYPE_F32 && (src1 == nullptr || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            return src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            return src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        return nullptr;
    case GGML_OP_SOFT_MAX_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_soft_max_back_f32;
        }
        return nullptr;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        {
            const ggml_tensor *rope = ctx->num_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            const int mode = ((const int32_t *) rope->op_params)[2];
            const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
            const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

            if (is_neox) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_neox_f32;
                }
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_neox_f32_f16;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_neox_f16;
                }
            } else if (is_mrope && !is_vision) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_multi_f32;
                }
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_multi_f32_f16;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_multi_f16;
                }
            } else if (is_vision) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_vision_f32;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_vision_f16;
                }
            } else {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_norm_f32;
                }
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_norm_f32_f16;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_norm_f16;
                }
            }
            return nullptr;
        }
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_MEAN:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sum_rows_f32;
        }
        return nullptr;
    case GGML_OP_CUMSUM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            if (src0->ne[0] <= 512) {
                return ctx->device->pipeline_cumsum_small_f32;
            } else {
                return ctx->device->pipeline_cumsum_f32;
            }
        }
        return nullptr;
    case GGML_OP_SOLVE_TRI:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::recursive_mutex> guard(ctx->device->mutex);
                auto it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                if (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } else {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            return pipeline;
        }
        return nullptr;
    case GGML_OP_ARGMAX:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            return ctx->device->pipeline_argmax_f32;
        }
        return nullptr;
    case GGML_OP_COUNT_EQUAL:
        if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            return ctx->device->pipeline_count_equal_i32;
        }
        return nullptr;
    case GGML_OP_IM2COL:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_im2col_f32;
        }
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_im2col_f32_f16;
        }
        return nullptr;
    case GGML_OP_IM2COL_3D:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_im2col_3d_f32;
        }
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_im2col_3d_f32_f16;
        }
        return nullptr;
    case GGML_OP_TIMESTEP_EMBEDDING:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_timestep_embedding_f32;
        }
        return nullptr;
    case GGML_OP_CONV_TRANSPOSE_1D:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_conv_transpose_1d_f32;
        }
        return nullptr;
    case GGML_OP_POOL_2D:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_pool2d_f32;
        }
        return nullptr;
    case GGML_OP_RWKV_WKV6:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rwkv_wkv6_f32;
        }
        return nullptr;
    case GGML_OP_RWKV_WKV7:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rwkv_wkv7_f32;
        }
        return nullptr;
    case GGML_OP_GATED_DELTA_NET:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            const uint32_t S_v = dst->src[2]->ne[0];
            const uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            switch (S_v) {
                case 32:  si = 0; break;
                case 64:  si = 1; break;
                case 128: si = 2; break;
                default: return nullptr;
            }
            return ctx->device->pipeline_gated_delta_net[si][kda];
        }
        return nullptr;
    case GGML_OP_SSM_SCAN:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            const uint32_t d_state = src0->ne[0];
            if (d_state == 128) {
                return ctx->device->pipeline_ssm_scan_f32_d128;
            } else if (d_state == 256) {
                return ctx->device->pipeline_ssm_scan_f32_d256;
            }
        }
        return nullptr;
    case GGML_OP_SSM_CONV:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_ssm_conv_f32;
        }
        return nullptr;
    case GGML_OP_OPT_STEP_ADAMW:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_opt_step_adamw_f32;
        }
        return nullptr;
    case GGML_OP_OPT_STEP_SGD:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_opt_step_sgd_f32;
        }
        return nullptr;
    case GGML_OP_LEAKY_RELU:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_leaky_relu_f32;
        }
        return nullptr;
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_TRANSPOSE_2D:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            bool transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;
            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = nullptr;
            if (op == GGML_OP_CONV_2D) {
                if (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } else if (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } else if (op == GGML_OP_CONV_TRANSPOSE_2D) {
                if (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } else if (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::recursive_mutex> guard(ctx->device->mutex);
                auto it = pipelines->find(conv2d_pipeline_state);
                if (it != pipelines->end()) {
                    pipeline = it->second;
                } else {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            return pipeline;
        }
        return nullptr;
    case GGML_OP_CONV_2D_DW:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            if (ggml_is_contiguous(src1)) {
                return ctx->device->pipeline_conv2d_dw_whcn_f32;
            } else if (ggml_is_contiguous_channels(src1)) {
                return ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            if (ggml_is_contiguous(src1)) {
                return ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } else if (ggml_is_contiguous_channels(src1)) {
                return ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        return nullptr;
    case GGML_OP_ADD1:
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_add1_f16_f16;
        }
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_add1_f16_f32;
        }
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_add1_f32_f32;
        }
        return nullptr;
    case GGML_OP_ARANGE:
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_arange_f32;
        }
        return nullptr;
    case GGML_OP_FILL:
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_fill_f32;
        }
        if (dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_fill_f16;
        }
        return nullptr;
    default:
        return nullptr;
    }

    GGML_UNUSED(src2);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_vk_device_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_vk_device_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_sum_rows_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_vk_device_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_vk_device_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_pad_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_vk_device_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_vk_device_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_im2col_3d_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src1) / ggml_vk_device_type_size(src1->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_vk_device_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src0);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_binary_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_vk_device_type_size(src0->type);
    const uint32_t b_offset = get_misalign_bytes(ctx, src1) / ggml_vk_device_type_size(src1->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_vk_device_type_size(dst->type);

    GGML_ASSERT(dst->op != GGML_OP_GET_ROWS || (a_offset == 0 && b_offset == 0 && d_offset == 0));

    p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_vk_device_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_vk_device_type_size(dst->type);

    p.a_offset = a_offset;
    p.d_offset = d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template<typename PC>
static void ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    if (src1 != nullptr) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    if (src2 != nullptr) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    if (src3 != nullptr) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_ROWS || op == GGML_OP_CPY || (!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->buffer != nullptr);
    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const bool use_src1 = src1 != nullptr;
    const uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    const bool use_src2 = src2 != nullptr;
    const bool use_src3 = src3 != nullptr;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);

    if (pipeline == nullptr) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        if (src1 != nullptr) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::endl;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, true);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, true) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, true) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, true) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, true);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    switch (op) {
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM_BACK:
    case GGML_OP_L2_NORM:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SOFT_MAX_BACK:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_CUMSUM:
    case GGML_OP_MEAN:
    case GGML_OP_ARGMAX:
        {
            const uint32_t nr = ggml_nrows(src0);
            if (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } else if (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } else {
                elements = { nr, 1, 1 };
            }
        } break;
    case GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            if (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } else if (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } else {
                elements = { nr, 1, 1 };
            }
        }
        break;
    case GGML_OP_RMS_NORM:
        if (ctx->do_add_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } else {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
        }
        break;
    case GGML_OP_TURBO_WHT:
        elements = { (uint32_t)(ne00 / 128), (uint32_t)ne01, (uint32_t)(ne02 * ne03) };
        break;

    case GGML_OP_SUM:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        break;
    case GGML_OP_GROUP_NORM:
        {
            const uint32_t num_groups = dst->op_params[0];
            elements = { num_groups * (uint32_t)src0->ne[3], 1, 1 };
        } break;
    case GGML_OP_DIAG_MASK_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        break;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            if (nrows > ctx->device->properties.limits.maxComputeWorkGroupCount[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } break;
    case GGML_OP_GET_ROWS:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
        break;
    case GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        break;
    case GGML_OP_IM2COL:
        {
            const bool is_2D = dst->op_params[6] == 1;

            const uint32_t IC = src1->ne[is_2D ? 2 : 1];

            const uint32_t KH = is_2D ? src0->ne[1] : 1;
            const uint32_t KW =         src0->ne[0];

            const uint32_t OH = is_2D ? dst->ne[2] : 1;
            const uint32_t OW =         dst->ne[1];

            const uint32_t batch = src1->ne[is_2D ? 3 : 2];

            const uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            const uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            const uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
        } break;
    case GGML_OP_IM2COL_3D:
        {
            const uint32_t IC = ((const uint32_t *)(dst->op_params))[9];

            const uint32_t N  = ne13 / IC;

            const uint32_t KD = ne02;
            const uint32_t KH = ne01;
            const uint32_t KW = ne00;

            const uint32_t OD = dst->ne[3] / N;
            const uint32_t OH = dst->ne[2];
            const uint32_t OW = dst->ne[1];

            const uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            const uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
        } break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        {
            const uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } break;
    case GGML_OP_POOL_2D:
        {
            const uint32_t N = dst->ne[3];
            const uint32_t OC = dst->ne[2];
            const uint32_t OH = dst->ne[1];
            const uint32_t OW = dst->ne[0];
            elements = { N * OC * OH * OW, 1, 1};
        } break;
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_TRANSPOSE_2D:
        if constexpr (std::is_same_v<PC, vk_op_conv2d_push_constants>) {
            const uint32_t NPQ = pc.N * pc.OH * pc.OW;
            const vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            const uint32_t NPQ_blocks = CEIL_DIV(NPQ, vk_conv_block_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_blocks, 1 };
            if (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } else {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        break;
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_DIV:
    case GGML_OP_MUL:
    case GGML_OP_ADD1:
    case GGML_OP_ARANGE:
    case GGML_OP_FILL:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_LOG:
    case GGML_OP_TRI:
    case GGML_OP_DIAG:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_ROLL:
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_CPY:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_UNARY:
    case GGML_OP_GLU:
    case GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            if (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_size(src0->type);
                if ((ggml_vk_device_type_size(src0->type) % 4) == 0) {
                    ne *= ggml_vk_device_type_size(src0->type) / 4;
                } else {
                    ne *= ggml_vk_device_type_size(src0->type) / 2;
                }
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            if (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_size(dst->type));
            }
            if (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } else if (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } else {
                elements = { ne, 1, 1 };
            }

            if (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.maxComputeWorkGroupCount[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
            }
        } break;
    case GGML_OP_ADD_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } break;
    case GGML_OP_SET_ROWS:
        {
            uint32_t ne = ggml_nelements(src0);
            if (dst->type == GGML_TYPE_TKV4_0) {
                // TKV4 uses one 128-thread workgroup per quantization block.
                ne = CEIL_DIV(ne, ggml_blck_size(dst->type));
            } else if (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_size(dst->type));
            } else {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            if (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } else if (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } else {
                elements = { ne, 1, 1 };
            }
        }
        break;
    case GGML_OP_SSM_CONV:
        {
            const uint32_t nr  = src0->ne[1];
            const uint32_t n_t = dst->ne[1];
            const uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        break;
    default:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        break;
    }

    if (op == GGML_OP_ADD || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        if (ctx->do_add_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_add_rms_partials, ctx->prealloc_size_add_rms_partials_offset);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } else if (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } else if (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } else if (op == GGML_OP_ROPE || op == GGML_OP_ROPE_BACK) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } else if (op == GGML_OP_IM2COL || op == GGML_OP_IM2COL_3D) {
        if (ctx->device->shader_int64 && ctx->device->buffer_device_address) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } else if (op == GGML_OP_COUNT_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_buffer_memset_async(subctx, dst_buf.buffer, dst_buf.offset, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } else if (op == GGML_OP_OPT_STEP_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } else if (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } else if (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } else if (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } else {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}
