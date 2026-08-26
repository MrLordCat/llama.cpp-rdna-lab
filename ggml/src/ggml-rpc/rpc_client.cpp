#include "rpc_internal.h"

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

// Structural hash of a computation graph. Only fields that are identical on
// the RPC client and the server are mixed in (names, shapes, ops, op params);
// pointer fields (data, src, view_src, buffer, extra) are excluded so the
// hashes match across the wire and stay stable across decode steps that reuse
// the same graph structure with fresh tensor data.
struct graph_cache {

    bool is_cached(const ggml_cgraph * cgraph) {
        if ((int)last_graph.size() != cgraph->n_nodes) {
            return false;
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (memcmp(&last_graph[i], cgraph->nodes[i], sizeof(ggml_tensor)) != 0) {
                return false;
            }
        }
        return true;
    }

    void add(const ggml_cgraph * cgraph) {
        last_graph.resize(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            memcpy(&last_graph[i], cgraph->nodes[i], sizeof(ggml_tensor));
        }
        last_hash = graph_structure_hash(cgraph);
    }

    std::vector<ggml_tensor> last_graph;
    uint64_t last_hash = 0;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    graph_cache gc;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void * base_ptr;
    uint64_t remote_ptr;
};

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

// Single-sequence causal attention masks are regenerated on the server and
// must never be transferred. The scheduler decorates cross-backend copy
// tensors as "<backend>#<name>#<idx>", so match both the plain names and the
// decorated form ("#attn_inp_kq_mask#N" / "#attn_inp_kq_mask (copy)#N").
static bool is_rpc_activation_name(const char * name) {
    return strstr(name, "l_out-") != nullptr || strstr(name, "result_output") != nullptr;
}

static bool is_rpc_layer_output_name(const char * name) {
    return strstr(name, "l_out-") != nullptr;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    memset(&result.rpc_flags, 0, sizeof(result.rpc_flags) + sizeof(result.padding));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);

    // Single-sequence causal attention masks are regenerated on the server
    // (n_kv x n_tokens x 2 bytes per batch is never transmitted).
    // The F16 cast of the mask ("(copy)") must be skipped as well: when the
    // mask is pinned to a local backend, the scheduler copies the cast into
    // the RPC buffer for the remote flash-attention split, and that copy is
    // discarded on the server (FA src[3] is substituted with NULL).
    if (is_causal_mask_name(result.name)) {
        result.rpc_flags |= GGML_RPC_TENSOR_FLAG_CAUSAL_MASK;
    }
    if (result.type == GGML_TYPE_F32 && is_rpc_activation_name(result.name)) {
        // F8 is deliberately opt-in and limited to intermediate layer
        // outputs. Keep result_output/logits at F16 to avoid amplifying
        // sampling sensitivity at the final boundary.
        if (std::getenv("GGML_RPC_ACT_F8") != nullptr && is_rpc_layer_output_name(result.name)) {
            result.rpc_flags |= GGML_RPC_TENSOR_FLAG_ACT_F8;
        } else {
            result.rpc_flags |= GGML_RPC_TENSOR_FLAG_ACT_F16;
        }
    }
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

// Bit-pack a causal attention mask (0.0 / -inf) for RPC_CMD_SET_TENSOR_MASK.
// Returns false when the data contains values other than 0.0 and -inf (alibi
// masks etc.); the caller then falls back to a plain F16/F32 transfer.
static bool pack_causal_mask(const void * data, size_t n_elems, ggml_type type, std::vector<uint8_t> & bits) {
    bits.resize((n_elems + 7) / 8, 0);
    uint8_t * dst = bits.data();
    if (type == GGML_TYPE_F16) {
        const uint16_t * src = (const uint16_t *) data;
        for (size_t i = 0; i < n_elems; ++i) {
            uint16_t h = src[i];
            if (h == 0x0000) {
                // bit 0
            } else if (h == 0xFC00) { // -inf in f16
                dst[i >> 3] |= (uint8_t) (1u << (i & 7));
            } else {
                return false;
            }
        }
        return true;
    }
    if (type == GGML_TYPE_F32) {
        const float * src = (const float *) data;
        for (size_t i = 0; i < n_elems; ++i) {
            float f = src[i];
            if (f == 0.0f) {
                // bit 0
            } else if (f == -INFINITY) {
                dst[i >> 3] |= (uint8_t) (1u << (i & 7));
            } else {
                return false;
            }
        }
        return true;
    }
    return false;
}

// Unpack a 1-bit causal mask into 0.0 / -inf values of the tensor type.
static bool rpc_send_tensor_data(const std::shared_ptr<socket_t> & sock,
        const rpc_tensor & tmeta, const void * data, size_t offset, size_t size) {
    auto t0 = std::chrono::steady_clock::now();
    const ggml_type type = (ggml_type) tmeta.type;
    const char * name = tmeta.name;
    if ((tmeta.rpc_flags & GGML_RPC_TENSOR_FLAG_CAUSAL_MASK) != 0 &&
        (getenv("GGML_RPC_ENABLE_MASK_NULL") != nullptr)) {
        // causal mask regenerated on the server; nothing to transmit
        return true;
    }
    const bool act_f16 = (tmeta.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F16) != 0;
    const bool act_f8  = (tmeta.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F8)  != 0;
    if ((tmeta.rpc_flags & GGML_RPC_TENSOR_FLAG_CAUSAL_MASK) != 0) {
        // Single-sequence causal masks are binary (0.0 / -inf): transmit as a
        // bitmask instead of raw F16. Fall back to the plain transfer when the
        // data is not binary (e.g. alibi masks).
        const size_t elem_size = ggml_type_size(type);
        if (offset % elem_size == 0 && size % elem_size == 0) {
            std::vector<uint8_t> bits;
            if (pack_causal_mask(data, size / elem_size, type, bits)) {
                if (std::getenv("GGML_RPC_SERVER_MAKE_MASK") != nullptr &&
                    tmeta.ne[3] == 1 && tmeta.ne[1] > 0 &&
                    tmeta.ne[0] >= tmeta.ne[1]) {
                    // Server-side generation: the binary mask is fully defined
                    // by n_past + shape, so only metadata is transmitted.
                    const int64_t n_past = (int64_t) tmeta.ne[0] - (int64_t) tmeta.ne[1];
                    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + sizeof(uint64_t);
                    std::vector<uint8_t> input(input_size, 0);
                    memcpy(input.data(), &tmeta, sizeof(rpc_tensor));
                    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
                    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(uint64_t), &n_past, sizeof(n_past));
                    bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_MASK_NPAST, input.data(), input.size());
                    RPC_STATUS_ASSERT(status);
                    if (RPC_DEBUG) {
                        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        fprintf(stderr, "[rpc-client] mask_npast '%s' n_kv=%u n_tokens=%u n_past=%" PRId64 " time=%.2fms t+=%.1fms\n",
                                name, tmeta.ne[0], tmeta.ne[1], n_past, ms, rpc_wall_ms());
                    }
                    return true;
                }
                size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + sizeof(uint64_t) + bits.size();
                std::vector<uint8_t> input(input_size, 0);
                memcpy(input.data(), &tmeta, sizeof(rpc_tensor));
                memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
                uint64_t n_elems = size / elem_size;
                memcpy(input.data() + sizeof(rpc_tensor) + sizeof(uint64_t), &n_elems, sizeof(n_elems));
                memcpy(input.data() + sizeof(rpc_tensor) + 2 * sizeof(uint64_t), bits.data(), bits.size());
                bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_MASK, input.data(), input.size());
                RPC_STATUS_ASSERT(status);
                if (RPC_DEBUG) {
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    fprintf(stderr, "[rpc-client] mask '%s' size=%zu -> packed %zu bytes time=%.2fms t+=%.1fms\n",
                            name, size, bits.size(), ms, rpc_wall_ms());
                }
                return true;
            }
        }
        // fall through to the plain transfer below
    }
    if (size > HASH_THRESHOLD && !act_f16 && !act_f8) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = tmeta;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        if (RPC_TIMELINE) {
            fprintf(stderr, "RPC_TL|name|SET_TENSOR_HASH|%s|%zu|t=%.1f\n",
                    name, size, rpc_wall_ms());
        }
        bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (response.result) {
            // the server already has the data, nothing to send
            return true;
        }
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t data_size = size;
    std::vector<uint8_t> tmp;
    const void * data_ptr = data;
    if (act_f8) {
        // F8 E4M3 is one byte per F32 element, reducing layer-boundary
        // traffic by 4x versus raw F32 and 2x versus the default F16 path.
        data_size = size / 4;
        tmp.resize(data_size);
        ggml_fp32_to_fp8_e4m3_row((const float *) data, tmp.data(), (int64_t) size / sizeof(float));
        data_ptr = tmp.data();
    } else if (act_f16) {
        // transmit F32 activations as F16 (halves the LAN traffic)
        data_size = size / 2;
        tmp.resize(data_size);
        ggml_fp32_to_fp16_row((const float *) data, (ggml_fp16_t *) tmp.data(), (int64_t) size / sizeof(float));
        data_ptr = tmp.data();
    }
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + data_size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &tmeta, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data_ptr, data_size);
    bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    RPC_STATUS_ASSERT(status);
    if (RPC_TIMELINE) {
        fprintf(stderr, "RPC_TL|name|SET_TENSOR|%s|%zu|t=%.1f\n",
                name, size, rpc_wall_ms());
    }
    if (RPC_DEBUG) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[rpc-client] tensor '%s' size=%zu time=%.2fms t+=%.1fms\n", name, size, ms, rpc_wall_ms());
    }
    return true;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (!rpc_send_tensor_data(ctx->sock, rpc_tensor, data, offset, size)) {
        RPC_STATUS_ASSERT(false);
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    auto t0 = std::chrono::steady_clock::now();
    const bool act_f16 = (request.tensor.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F16) != 0;
    const bool act_f8  = (request.tensor.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F8)  != 0;
    std::vector<uint8_t> tmp;
    void * rsp_ptr = data;
    size_t rsp_size = size;
    if (act_f8) {
        rsp_size = size / 4;
        tmp.resize(rsp_size);
        rsp_ptr = tmp.data();
    } else if (act_f16) {
        // server sends F32 activations as F16; convert back after receive
        rsp_size = size / 2;
        tmp.resize(rsp_size);
        rsp_ptr = tmp.data();
    }
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), rsp_ptr, rsp_size);
    RPC_STATUS_ASSERT(status);
    if (act_f8) {
        ggml_fp8_e4m3_to_fp32_row((const uint8_t *) rsp_ptr, (float *) data, (int64_t) size / sizeof(float));
    } else if (act_f16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) rsp_ptr, (float *) data, (int64_t) size / sizeof(float));
    }
    if (RPC_DEBUG) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[rpc-client] get_tensor '%s' size=%zu time=%.1fms\n", tensor->name, size, ms);
    }
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->sock != dst_ctx->sock) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

// RPC: cache of server-reported alloc sizes, keyed by the endpoint plus a
// deterministic fingerprint of the tensor layout (type/ne/nb/op/op_params).
// The server-side size is a pure function of the tensor layout on the server
// backend (e.g. Vulkan device block padding for quantized types), so a
// hit is safe across graph rebuilds. Without this cache every
// sched reserve/allocation round re-queries the server once per node
// (quantized tensors, FLASH_ATTN_EXT, MUL_MAT_ID), which costs ~0.5 s per
// prefill ubatch on the 12K RPC lane (measured 537 ms alloc in
// process_ubatch).
static std::mutex g_rpc_alloc_size_cache_mu;
static std::unordered_map<uint64_t, size_t> g_rpc_alloc_size_cache;

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // Some backends (e.g. Vulkan q3_K/q6_K) use a device layout wider than the host layout,
    // so ggml_nbytes is not a valid allocation size for quantized tensors on the server.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // fingerprint: endpoint + deterministic tensor layout fields only
        // (no pointers - tensor instances are recreated on graph rebuilds)
        // NOTE: srcs are intentionally NOT part of the key: the supported
        // server backends (Vulkan, CPU) return a size that depends only on
        // the tensor's own layout, and FA kv-cache view srcs grow with n_past
        // every prefill ubatch, which would defeat the cache.
        std::array<uint8_t, sizeof(uint64_t) + 1 + GGML_MAX_DIMS*2*sizeof(int64_t) + sizeof(int) + GGML_MAX_OP_PARAMS + 1> fp;
        size_t fp_len = 0;
        const uint64_t endpoint_hash = fnv_hash((const uint8_t *) buft_ctx->endpoint.c_str(), buft_ctx->endpoint.size());
        memcpy(fp.data() + fp_len, &endpoint_hash, sizeof(endpoint_hash)); fp_len += sizeof(endpoint_hash);
        fp[fp_len++] = (uint8_t) tensor->type;
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            memcpy(fp.data() + fp_len, &tensor->ne[i], sizeof(tensor->ne[i])); fp_len += sizeof(tensor->ne[i]);
        }
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            memcpy(fp.data() + fp_len, &tensor->nb[i], sizeof(tensor->nb[i])); fp_len += sizeof(tensor->nb[i]);
        }
        memcpy(fp.data() + fp_len, &tensor->op, sizeof(tensor->op)); fp_len += sizeof(tensor->op);
        memcpy(fp.data() + fp_len, tensor->op_params, sizeof(tensor->op_params)); fp_len += sizeof(tensor->op_params);
        fp[fp_len++] = tensor->view_src != nullptr ? 1 : 0;
        const uint64_t key = fnv_hash(fp.data(), fp_len);

        {
            std::lock_guard<std::mutex> lock(g_rpc_alloc_size_cache_mu);
            auto it = g_rpc_alloc_size_cache.find(key);
            if (it != g_rpc_alloc_size_cache.end()) {
                return it->second;
            }
        }

        auto sock = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);

        {
            std::lock_guard<std::mutex> lock(g_rpc_alloc_size_cache_mu);
            g_rpc_alloc_size_cache[key] = response.alloc_size;
        }

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

// Async outbound copy (l_out-* and other split inputs): the data is
// snapshotted HERE, in the scheduler thread, before the job reaches the
// per-socket worker. The worker no longer touches the producing GPU backend:
// ggml_vk_synchronize is not thread-safe against a concurrent graph compute
// on the same backend (it shares compute_ctx/fence/submit_pending), and the
// previous design (worker synchronizing + reading the source) deadlocked
// deterministically on the 3rd-4th decode ubatch. The scheduler owns the
// backends, so synchronize/get here is safe; the queued task only streams
// the captured payload to the server, letting the next ubatch local layers
// start while the transfer runs.
static bool rpc_async_copy_submit(ggml_backend_rpc_context * rpc_ctx,
        ggml_backend_t src_backend, const ggml_tensor * src,
        ggml_backend_t dst_backend, ggml_tensor * dst) {
    (void) dst_backend;
    auto sock = get_socket(rpc_ctx->endpoint);
    if (sock == nullptr) {
        return false;
    }
    const size_t nbytes = ggml_nbytes(src);
    rpc_tensor dst_meta = serialize_tensor(dst);
    const bool src_host = src->buffer != nullptr && ggml_backend_buffer_is_host(src->buffer);
    const auto t_snap0 = std::chrono::steady_clock::now();
    if (!src_host) {
        // wait for the producing GPU graph in the scheduler thread only
        ggml_backend_synchronize(src_backend);
    }
    const auto t_snap1 = std::chrono::steady_clock::now();
    std::vector<uint8_t> host(nbytes, 0);
    ggml_backend_tensor_get(src, host.data(), 0, host.size());
    const auto t_snap2 = std::chrono::steady_clock::now();
    if (RPC_DEBUG) {
        const double ms0 = std::chrono::duration<double, std::milli>(t_snap1 - t_snap0).count();
        const double ms1 = std::chrono::duration<double, std::milli>(t_snap2 - t_snap1).count();
        fprintf(stderr, "[rpc-worker] snapshot src='%s' bytes=%zu host=%d sync=%.2fms get=%.2fms\n",
                src->name, nbytes, (int) src_host, ms0, ms1);
    }
    return rpc_send_submit(sock, [sock, dst_meta = std::move(dst_meta), host = std::move(host)]() {
        if (RPC_DEBUG) {
            fprintf(stderr, "[rpc-worker] send '%s' bytes=%zu\n", dst_meta.name, host.size());
        }
        bool ok = rpc_send_tensor_data(sock, dst_meta, host.data(), 0, host.size());
        if (RPC_DEBUG) {
            fprintf(stderr, "[rpc-worker] sent '%s' ok=%d\n", dst_meta.name, (int) ok);
        }
        return ok;
    }, true);
}

static void rpc_wait_pending_copies(ggml_backend_rpc_context * rpc_ctx) {
    // Barrier: queue a no-op behind every previously submitted command (both
    // async transfers and graph submissions) and wait for it to execute.
    auto sock = get_socket(rpc_ctx->endpoint);
    if (sock == nullptr) {
        return;
    }
    rpc_send_submit(sock, []() { return true; }, false);
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    // barrier: the per-socket worker must finish all queued transfers and
    // graphs before the context (and its buffers) are torn down
    rpc_wait_pending_copies(rpc_ctx);
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    if (std::getenv("GGML_RPC_BARRIER_DISABLE") != nullptr) {
        return;
    }
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    // Full barrier: wait for every command already submitted on this socket,
    // including queued async copies and graph computes. Without this, a
    // pending copy can still be streaming l_out when the scheduler resets or
    // reallocates the buffers (GGML_ASSERT "tensor buffer not set"). The
    // scheduler calls synchronize only on boundaries (reserve/switch), not in
    // the hot ubatch loop, so this does not serialize per-ubatch server work.
    rpc_wait_pending_copies(rpc_ctx);
}

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_rpc_event_context * event_ctx = (ggml_backend_rpc_event_context *) event->context;
    // Commands on one RPC connection are ordered. Record is therefore a
    // logical fence after the graph command already sent on this socket.
    event_ctx->recorded = true;
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_rpc_event_context * event_ctx = (ggml_backend_rpc_event_context *) event->context;
    if (event_ctx->recorded) {
        // RPC has no cross-device semaphore. A server-side worker drain is a
        // conservative event fence and only occurs when a scheduler copy is
        // about to be reused (normally after GGML_SCHED_MAX_COPIES ubatches).
        ggml_backend_rpc_wait_endpoint(event_ctx->endpoint);
        event_ctx->recorded = false;
    }
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    // Pipeline mode: submit the graph without waiting for the server to
    // finish it - the server runs it in a worker and drains it before the
    // next data command, so the client's local work (the previous/next
    // ubatch layers) overlaps with the server pass.
    const bool async_pipeline = std::getenv("GGML_RPC_ASYNC_GRAPH") != nullptr;
    // RECOMPUTE is a round-trip command (the client waits for the response),
    // which would serialize the server work back into the client critical
    // path. In async pipeline mode always send the full graph instead - the
    // server caches it and a remiss full send costs a few ms on the transport,
    // far less than the blocking round trip.
    auto t0 = std::chrono::steady_clock::now();
    bool reuse = !async_pipeline && std::getenv("GGML_RPC_NO_GRAPH_CACHE") == nullptr && rpc_ctx->gc.is_cached(cgraph);
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        request.hash = rpc_ctx->gc.last_hash;
        rpc_msg_graph_recompute_rsp response;
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        const double recompute_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (response.result) {
            if (RPC_DEBUG) {
                fprintf(stderr, "[rpc-client] graph_recompute nodes=%d hash=%016" PRIx64
                    " time=%.1fms t+=%.1fms first='%s' op=%d src0='%s'\n",
                        cgraph->n_nodes, request.hash,
                    recompute_ms, rpc_wall_ms(),
                        cgraph->nodes[0] ? cgraph->nodes[0]->name : "-",
                        cgraph->nodes[0] ? (int) cgraph->nodes[0]->op : -1,
                        (cgraph->nodes[0] && cgraph->nodes[0]->src[0]) ? cgraph->nodes[0]->src[0]->name : "-");
            }
            return GGML_STATUS_SUCCESS;
        }
        // the server does not have this graph cached - fall through and send it
        // in full (another context may have overwritten the server-side cache)
        if (RPC_DEBUG) {
            fprintf(stderr, "[rpc-client] graph_recompute MISS hash=%016" PRIx64 " - resending full graph\n", request.hash);
        }
        reuse = false;
    }
    if (!reuse) {
        rpc_ctx->gc.add(cgraph);
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        auto sock = get_socket(rpc_ctx->endpoint);
        const enum rpc_cmd cmd = async_pipeline ? RPC_CMD_GRAPH_COMPUTE_ASYNC : RPC_CMD_GRAPH_COMPUTE;
        bool status = async_pipeline
            ? send_rpc_cmd_async(sock, cmd, input.data(), input.size())
            : send_rpc_cmd(sock, cmd, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
        if (RPC_DEBUG) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[rpc-client] graph_compute%s nodes=%zu ser=%zu time=%.1fms t+=%.1fms\n",
                    async_pipeline ? " async" : "", cgraph->n_nodes, input.size(), ms, rpc_wall_ms());
            if (cgraph->n_nodes <= 48) {
                for (uint32_t i = 0; i < cgraph->n_nodes; i++) {
                    const ggml_tensor * n = cgraph->nodes[i];
                    fprintf(stderr, "[rpc-client]   node[%u] name='%s' op=%d src0='%s' src1='%s' src2='%s'\n", i, n->name, (int) n->op,
                            n->src[0] ? n->src[0]->name : "-", n->src[1] ? n->src[1]->name : "-", n->src[2] ? n->src[2]->name : "-");
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_rpc_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
        const ggml_tensor * src, ggml_tensor * dst) {
    // Async outbound: only for GPU->RPC split inputs, and only when the
    // pipeline is explicitly enabled. Remote->remote and CPU->RPC transfers
    // use the synchronous path (they never block on a local GPU graph).
    if (std::getenv("GGML_RPC_ASYNC_GRAPH") == nullptr ||
        std::getenv("GGML_RPC_NO_ASYNC_COPY") != nullptr) {
        return false;
    }
    if (ggml_backend_is_rpc(backend_src)) {
        return false;
    }
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend_dst->context;
    return rpc_async_copy_submit(rpc_ctx, backend_src, src, backend_dst, dst);
}

static void ggml_backend_rpc_get_async(ggml_backend_t backend, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    // Non-blocking GET: the ordered per-socket worker streams the tensor into
    // the caller's host buffer after the preceding graph command. The caller
    // must not read `data` before ggml_backend_synchronize()/llama_synchronize()
    // (the RPC synchronize queues a no-op behind this command, so a barrier
    // also waits for any in-flight async get). This keeps the scheduler thread
    // free to start the next ubatch local layers while logits/embeddings are
    // still being received - previously get_tensor_async==NULL forced a full
    // RPC barrier plus a synchronous GET block per ubatch.
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    auto sock = get_socket(rpc_ctx->endpoint);
    if (sock == nullptr) {
        GGML_ABORT("rpc: get_async: no socket for endpoint %s", rpc_ctx->endpoint);
    }
    // Prototype gate: only the final logits (result_output) are safe to defer
    // - their host buffer is consumed only after llama_synchronize before
    // sampling. Embeddings/NextN rows feed the next graph immediately in some
    // paths, so those stay on the synchronous path until proven safe.
    if (strstr(tensor->name, "result_output") == nullptr) {
        ggml_backend_tensor_get(tensor, data, offset, size);
        return;
    }
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    const bool act_f16 = (request.tensor.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F16) != 0;
    const bool act_f8  = (request.tensor.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F8)  != 0;
    rpc_send_submit(sock, [sock, request = std::move(request), data, offset, size, act_f16, act_f8]() {
        std::vector<uint8_t> tmp;
        void * rsp_ptr = data;
        size_t rsp_size = size;
        if (act_f8) {
            rsp_size = size / 4;
            tmp.resize(rsp_size);
            rsp_ptr = tmp.data();
        } else if (act_f16) {
            // server sends F32 activations as F16; convert back after receive
            rsp_size = size / 2;
            tmp.resize(rsp_size);
            rsp_ptr = tmp.data();
        }
        const bool status = send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), rsp_ptr, rsp_size);
        RPC_STATUS_ASSERT(status);
        if (act_f8) {
            ggml_fp8_e4m3_to_fp32_row((const uint8_t *) rsp_ptr, (float *) data, (int64_t) size / sizeof(float));
        } else if (act_f16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) rsp_ptr, (float *) data, (int64_t) size / sizeof(float));
        }
        if (RPC_DEBUG) {
            fprintf(stderr, "[rpc-worker] get_async done '%s' bytes=%zu\n", request.tensor.name, size);
        }
        return true;
    }, true);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    // Async outbound copy of split inputs, routed through the ordered
    // per-socket queue (rpc_async_copy_submit): the copy task waits for the
    // producing backend graph, streams the tensor and only then the graph
    // command is submitted on the same socket, so command order is preserved
    // on the server. The scheduler thread never blocks on the source GPU,
    // allowing the next ubatch local layers to start while the copy runs.
    // Enabled with GGML_RPC_ASYNC_GRAPH=1 (same gate as the async graph).
    // NOTE: the earlier detached-thread approach in ggml-backend.cpp
    // (GGML_SCHED_ASYNC_SPLIT_COPY) is NOT used here and remains disabled -
    // it has no ordering with the RPC queue and caused buffer-reuse races.
    /* .cpy_tensor_async        = */ ggml_backend_rpc_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint = */ endpoint,
        /* .device   = */ device,
        /* .name     = */ dev_name,
        /* .gc       = */ {},
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

// RPC server-side implementation

