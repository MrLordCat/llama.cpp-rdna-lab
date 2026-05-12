#include "gated_delta_net.cuh"

#include <chrono>
#include <cstring>

static int64_t ggml_env_i64(const char * name, int64_t fallback, int64_t min_value, int64_t max_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    const int64_t parsed = atoll(value);
    if (parsed <= 0) {
        return fallback;
    }

    if (parsed < min_value) {
        return min_value;
    }
    if (parsed > max_value) {
        return max_value;
    }

    return parsed;
}

static int64_t ggml_pick_delta_net_chunk_adaptive(int64_t n_tokens, int64_t min_tail) {
    const int64_t candidates[] = {64, 96, 128};

    int64_t best_chunk = candidates[0];
    uint64_t best_score = (uint64_t) -1;

    for (int64_t candidate : candidates) {
        const int64_t tail = n_tokens % candidate;
        const int64_t pad = (candidate - tail) % candidate;
        const int64_t n_chunks = (n_tokens + pad) / candidate;

        uint64_t score = 0;
        // Keep consistency with model/planner contract scoring.
        score += (uint64_t) pad * 8192;
        score += (uint64_t) (n_chunks > 0 ? (n_chunks - 1) : 0) * 192;
        if (tail != 0 && tail < min_tail) {
            score += 1000000 + (uint64_t) (min_tail - tail) * 4096;
        }

        if (score < best_score) {
            best_score = score;
            best_chunk = candidate;
        }
    }

    return best_chunk;
}

template <bool fast_exp_t>
static __device__ __forceinline__ float ggml_gdn_exp(float x) {
    if constexpr (fast_exp_t) {
        return __expf(x);
    }
    return expf(x);
}

template <int S_v, bool KDA, bool keep_intermediates_t, bool fast_exp_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * __restrict__ q,
                                     const float * __restrict__ k,
                                     const float * __restrict__ v,
                                     const float * __restrict__ g,
                                     const float * __restrict__ beta,
                                     const float * __restrict__ curr_state,
                                     float *       __restrict__ dst,
                                     int64_t       H,
                                     int64_t       n_tokens_chunk,
                                     int64_t       n_tokens_total,
                                     int64_t       token_offset,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens_total * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset       = (sequence * H + h_idx) * S_v * S_v;
    const int64_t state_size_per_token = S_v * S_v * H * n_seqs; // keep_intermediates_t only
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens_total * H + token_offset * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    const float * q_t = q + iq3 * sq3 + iq1 * sq1;
    const float * k_t = k + iq3 * sq3 + iq1 * sq1;
    const float * v_t = v + sequence * sv3 + h_idx * sv1;

    const int64_t gb_base = sequence * sb3 + h_idx * sb1;
    const float * beta_t = beta + gb_base;
    const float * g_t    = g + gb_base * (KDA ? S_v : 1);

    for (int t = 0; t < n_tokens_chunk; t++) {
        float beta_val = lane == 0 ? *beta_t : 0.0f;
        beta_val = __shfl_sync(0xffffffff, beta_val, 0, warp_size);

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            // g is scalar per token for non-KDA; compute once per warp instead of per lane.
            float g_val = lane == 0 ? ggml_gdn_exp<fast_exp_t>(*g_t) : 0.0f;
            g_val = __shfl_sync(0xffffffff, g_val, 0, warp_size);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard = fmaf(s_shard[r], k_reg[r], kv_shard);
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = fmaf(-g_val, kv_col, v_t[col]) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const float s_next = fmaf(k_reg[r], delta_col, g_val * s_shard[r]);
                s_shard[r] = s_next;
                attn_partial = fmaf(s_next, q_reg[r], attn_partial);
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                g_reg[r] = ggml_gdn_exp<fast_exp_t>(g_t[i]);
                const float gs = g_reg[r] * s_shard[r];
                kv_shard = fmaf(gs, k_reg[r], kv_shard);
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const float s_next = fmaf(k_reg[r], delta_col, g_reg[r] * s_shard[r]);
                s_shard[r] = s_next;
                attn_partial = fmaf(s_next, q_reg[r], attn_partial);
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_intermediates_t) {
            float * curr_state = (dst + attn_score_elems) + (token_offset + t) * state_size_per_token + state_offset;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                curr_state[col * S_v + i] = s_shard[r];
            }
        }

        q_t += sq2;
        k_t += sq2;
        v_t += sv2;
        beta_t += sb2;
        if constexpr (KDA) {
            g_t += sb2 * S_v;
        } else {
            g_t += sb2;
        }
    }

    if constexpr (!keep_intermediates_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_intermediates_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    // RDNA4 prefill can benefit from shorter token loops in this kernel; keep_intermediates path
    // stays unchunked because it writes per-token state snapshots.
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const bool use_chunked_prefill = !keep_intermediates_t && GGML_CUDA_CC_IS_RDNA4(cc) && n_tokens >= 128;
    // Adaptive chunk size: target ~3-4 kernel launches to minimise dispatch overhead without
    // excessive register pressure per launch.  Sweep (2026-05-08, gfx1201):
    //   n_tokens<=256 => chunk=96  (3 launches) => ~32.5 TPS  (optimal)
    //   n_tokens> 256 => chunk=128 (4 launches) => ~31.5 TPS  (optimal for larger batches)
    int64_t chunk_size = (n_tokens > 256) ? 128 : 96;
    const char * chunk_source = "adaptive-default";

    const char * chunk_policy = std::getenv("LLAMA_DELTA_NET_CHUNK_POLICY");
    const bool use_adaptive_policy = chunk_policy != nullptr && std::strcmp(chunk_policy, "adaptive") == 0;
    if (use_adaptive_policy && !KDA) {
        const int64_t min_tail = ggml_env_i64("LLAMA_DELTA_NET_CHUNK_MIN_TAIL", 32, 1, 127);
        chunk_size = ggml_pick_delta_net_chunk_adaptive(n_tokens, min_tail);
        chunk_source = "adaptive-policy";
    }

    const char * gdn_chunk_override = std::getenv("GGML_GDN_CHUNK_SIZE");
    if (gdn_chunk_override != nullptr) {
        const int64_t parsed = atoll(gdn_chunk_override);
        if (parsed > 0 && parsed % 16 == 0) {
            chunk_size = parsed;
            chunk_source = "ggml-override";
        }
    } else {
        const char * model_chunk_override = std::getenv("LLAMA_DELTA_NET_CHUNK_SIZE");
        if (model_chunk_override != nullptr) {
            const int64_t parsed = atoll(model_chunk_override);
            if (parsed > 0 && parsed % 16 == 0) {
                chunk_size = parsed;
                chunk_source = "model-override";
            }
        }
    }

    const int64_t chunk_tail = n_tokens % chunk_size;
    const int64_t chunk_pad = (chunk_size - chunk_tail) % chunk_size;
    const int64_t n_chunks = (n_tokens + chunk_pad) / chunk_size;

    const bool trace_contract = std::getenv("LLAMA_TRACE_DELTA_NET_CONTRACT") != nullptr;
    const bool trace_path = std::getenv("GGML_TRACE_GDN_PATH") != nullptr;
    const bool trace_timing = std::getenv("GGML_TRACE_GDN_TIMING") != nullptr;
    const bool use_fast_exp = !KDA && ggml_env_i64("GGML_GDN_FAST_EXP", 0, 0, 1) == 1;
    if (trace_contract) {
        GGML_LOG_INFO(
            "%s: contract kda=%d keep_intermediates=%d source=%s cc=%d chunk_size=%lld n_tokens=%lld pad=%lld tail=%lld n_chunks=%lld chunked_prefill=%d fast_exp=%d\n",
            __func__,
            (int) KDA,
            (int) keep_intermediates_t,
            chunk_source,
            cc,
            (long long) chunk_size,
            (long long) n_tokens,
            (long long) chunk_pad,
            (long long) chunk_tail,
            (long long) n_chunks,
                (int) use_chunked_prefill,
                (int) use_fast_exp);
    }

    if (trace_path) {
        GGML_LOG_INFO(
            "%s: KDA=%d keep_intermediates=%d cc=%d n_tokens=%lld n_seqs=%lld S_v=%lld chunked_prefill=%d chunk_size=%lld fast_exp=%d\n",
            __func__,
            (int) KDA,
            (int) keep_intermediates_t,
            cc,
            (long long) n_tokens,
            (long long) n_seqs,
            (long long) S_v,
            (int) use_chunked_prefill,
                (long long) chunk_size,
                (int) use_fast_exp);
    }

    auto launch_once = [&](const float * q_ptr, const float * k_ptr, const float * v_ptr,
                           const float * g_ptr, const float * b_ptr, const float * state_ptr,
                           int64_t n_tokens_chunk, int64_t token_offset) {
        if (trace_path) {
            GGML_LOG_INFO(
                "%s: launch chunk token_offset=%lld n_tokens_chunk=%lld/%lld\n",
                __func__,
                (long long) token_offset,
                (long long) n_tokens_chunk,
                (long long) n_tokens);
        }
        const auto timing_start = trace_timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

        if constexpr (!KDA) {
            if (use_fast_exp) {
                switch (S_v) {
                    case 16:
                        gated_delta_net_cuda<16, KDA, keep_intermediates_t, true><<<grid_dims, block_dims, 0, stream>>>(
                            q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                            n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                        break;
                    case 32:
                        gated_delta_net_cuda<32, KDA, keep_intermediates_t, true><<<grid_dims, block_dims, 0, stream>>>(
                            q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                            n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                        break;
                    case 64:
                        gated_delta_net_cuda<64, KDA, keep_intermediates_t, true><<<grid_dims, block_dims, 0, stream>>>(
                            q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                            n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                        break;
                    case 128:
                        gated_delta_net_cuda<128, KDA, keep_intermediates_t, true><<<grid_dims, block_dims, 0, stream>>>(
                            q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                            n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                        break;
                    default:
                        GGML_ABORT("fatal error");
                        break;
                }
            } else {
                switch (S_v) {
                    case 16:
                        gated_delta_net_cuda<16, KDA, keep_intermediates_t, false><<<grid_dims, block_dims, 0, stream>>>(
                            q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                            n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                        break;
                    case 32:
                        gated_delta_net_cuda<32, KDA, keep_intermediates_t, false><<<grid_dims, block_dims, 0, stream>>>(
                            q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                            n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                        break;
                    case 64:
                        gated_delta_net_cuda<64, KDA, keep_intermediates_t, false><<<grid_dims, block_dims, 0, stream>>>(
                            q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                            n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                        break;
                    case 128:
                        gated_delta_net_cuda<128, KDA, keep_intermediates_t, false><<<grid_dims, block_dims, 0, stream>>>(
                            q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                            n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                        break;
                    default:
                        GGML_ABORT("fatal error");
                        break;
                }
            }
        } else {
            switch (S_v) {
                case 16:
                    gated_delta_net_cuda<16, KDA, keep_intermediates_t, false><<<grid_dims, block_dims, 0, stream>>>(
                        q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                        n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    break;
                case 32:
                    gated_delta_net_cuda<32, KDA, keep_intermediates_t, false><<<grid_dims, block_dims, 0, stream>>>(
                        q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                        n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    break;
                case 64:
                    gated_delta_net_cuda<64, KDA, keep_intermediates_t, false><<<grid_dims, block_dims, 0, stream>>>(
                        q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                        n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    break;
                case 128:
                    gated_delta_net_cuda<128, KDA, keep_intermediates_t, false><<<grid_dims, block_dims, 0, stream>>>(
                        q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, state_ptr, dst_d, H,
                        n_tokens_chunk, n_tokens, token_offset, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    break;
                default:
                    GGML_ABORT("fatal error");
                    break;
            }
        }

        CUDA_CHECK(cudaGetLastError());

        if (trace_timing) {
#ifdef GGML_USE_HIP
            GGML_UNUSED(timing_start);
#else
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto timing_stop = std::chrono::high_resolution_clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(timing_stop - timing_start).count();
            GGML_LOG_INFO(
                "%s: timing token_offset=%lld n_tokens_chunk=%lld ms=%.3f fast_exp=%d\n",
                __func__,
                (long long) token_offset,
                (long long) n_tokens_chunk,
                elapsed_ms,
                (int) use_fast_exp);
#endif
        }
    };

    if (use_chunked_prefill) {
        float * state_base = dst_d + S_v * H * n_tokens * n_seqs;
        const float * state_src = s_d;

        for (int64_t token_offset = 0; token_offset < n_tokens; token_offset += chunk_size) {
            const int64_t n_tokens_chunk = (n_tokens - token_offset < chunk_size) ? (n_tokens - token_offset) : chunk_size;

            launch_once(
                q_d + token_offset * sq2,
                k_d + token_offset * sq2,
                v_d + token_offset * sv2,
                g_d + token_offset * sb2,
                b_d + token_offset * sb2,
                state_src,
                n_tokens_chunk,
                token_offset);
            state_src = state_base;
        }
    } else {
        launch_once(
            q_d,
            k_d,
            v_d,
            g_d,
            b_d,
            s_d,
            n_tokens,
            0);
    }

}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    const bool keep_intermediates = (((const int32_t *) dst->op_params)[0] != 0);

    if (kda) {
        if (keep_intermediates) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, stream);
        }
    } else {
        if (keep_intermediates) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, stream);
        }
    }
}
