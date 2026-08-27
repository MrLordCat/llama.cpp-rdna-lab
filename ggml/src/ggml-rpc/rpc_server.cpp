#include "rpc_internal.h"

#include "rpc_internal.h"

#include "rpc_internal.h"

static bool unpack_causal_mask(const std::vector<uint8_t> & bits, size_t n_elems, ggml_type type, std::vector<uint8_t> & out) {
    const size_t elem_size = ggml_type_size(type);
    if (elem_size != 2 && elem_size != 4) {
        return false;
    }
    out.resize(n_elems * elem_size);
    uint8_t * dst = out.data();
    for (size_t i = 0; i < n_elems; ++i) {
        uint8_t bit = (bits[i >> 3] >> (i & 7)) & 1;
        if (elem_size == 2) {
            uint16_t v = bit ? (uint16_t) 0xFC00 : (uint16_t) 0x0000;
            memcpy(dst + i * 2, &v, 2);
        } else {
            uint32_t v = bit ? 0xFF800000u : 0u; // -inf / 0.0f
            memcpy(dst + i * 4, &v, 4);
        }
    }
    return true;
}

// Send one tensor payload to the server using metadata captured at submit
// time (serialized rpc_tensor: remote buffer id/offsets/shape/name/flags).
// Used by both the synchronous buffer set path and the async outbound
// worker, so the worker never dereferences live ggml tensors or buffers -
// it only holds the captured identity and the data (host snapshot taken in
// the scheduler thread).
class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        worker = std::thread(&rpc_server::worker_loop, this);
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_mask(const std::vector<uint8_t> & input);
    bool set_tensor_mask_npast(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_compute_async(const std::vector<uint8_t> & input);
    void graph_compute_wait();
    bool graph_recompute(const rpc_msg_graph_recompute_req & request, rpc_msg_graph_recompute_rsp & response);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
    // P2 pipeline: used by the connection handler to wait only for the
    // requested graph and to release the next one after the GET is served.
    bool p2_pipeline() const { return worker_pipeline; }
    bool wait_graph_seq(uint64_t seq);
    void wait_enqueued_graphs();
    void release_graph_seq(uint64_t seq);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);
    void worker_loop();

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // store recently computed graphs for each backend, keyed by structural hash
    // (multiple contexts share one device, so a single stored graph per device
    // would let recompute hit the wrong graph)
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;

    // async graph compute pipeline state
    std::mutex              worker_mutex;
    std::condition_variable worker_cv;
    bool                    worker_busy = false;
    bool                    worker_shutdown = false;
    std::vector<uint8_t>    worker_input;
    std::thread             worker;
    // P2 pipeline (GGML_RPC_PREFILL_PIPELINE=1): a small queue allows the
    // client to enqueue graph N+1 right after graph N. The worker runs each
    // graph in turn, but a later graph is not allowed to start before the
    // connection thread has served the GET for the previous graph
    // (worker_release gate) - the server result buffers are shared between
    // consecutive graphs, so this keeps GET reads race-free without
    // double-buffering the outputs.
    bool                    worker_pipeline = std::getenv("GGML_RPC_PREFILL_PIPELINE") != nullptr;
    std::deque<std::pair<uint64_t, std::vector<uint8_t>>> worker_pending;
    uint64_t worker_seq      = 0; // last enqueued graph sequence (1-based)
    uint64_t worker_done     = 0; // last completed graph sequence
    uint64_t worker_release  = 0; // graphs whose GET has been served; worker may start seq <= release+1
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    const bool act_f16 = (in_tensor->rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F16) != 0;
    const bool act_f8  = (in_tensor->rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F8)  != 0;
    const bool act_q8  = (in_tensor->rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_Q8_0) != 0;
    if ((int) act_f16 + (int) act_f8 + (int) act_q8 > 1) {
        GGML_LOG_ERROR("[%s] activation payload has conflicting transport flags\n", __func__);
        return false;
    }
    if ((act_f16 && size > SIZE_MAX / 2) || (act_f8 && size > SIZE_MAX / 4)) {
        GGML_LOG_ERROR("[%s] activation payload size overflows destination size\n", __func__);
        return false;
    }
    const size_t f32_size = act_q8 ? rpc_q8_0_f32_size(size) :
                            (act_f8 ? size * 4 : (act_f16 ? size * 2 : size));

    // sanitize tensor->data against the expanded destination size, not the
    // compressed wire payload size.
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || f32_size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, f32_size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    std::vector<uint8_t> f32_data;
    if (act_q8) {
        f32_data.resize(f32_size);
        rpc_q8_0_to_f32(data, size, f32_data.data());
        data = f32_data.data();
    } else if (act_f8) {
        f32_data.resize(f32_size);
        ggml_fp8_e4m3_to_fp32_row((const uint8_t *) data, (float *) f32_data.data(), (int64_t) size);
        data = f32_data.data();
    } else if (act_f16) {
        // client sent the F32 activation as F16; convert back before storing
        f32_data.resize(f32_size);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) data, (float *) f32_data.data(), (int64_t) size / sizeof(ggml_fp16_t));
        data = f32_data.data();
    }
    if (cache_dir && size > HASH_THRESHOLD && !act_f16 && !act_f8 && !act_q8) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, f32_size);
    return true;
}

bool rpc_server::set_tensor_mask(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | n_elems (8 bytes) | bits ((n_elems+7)/8) |
    if (input.size() < sizeof(rpc_tensor) + 2 * sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *) input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    uint64_t n_elems;
    memcpy(&n_elems, input.data() + sizeof(rpc_tensor) + sizeof(uint64_t), sizeof(n_elems));
    const size_t bits_size = input.size() - sizeof(rpc_tensor) - 2 * sizeof(uint64_t);
    if (bits_size != (size_t) (n_elems + 7) / 8) {
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const size_t elem_size = ggml_type_size(tensor->type);
    if (elem_size != 2 && elem_size != 4) {
        GGML_LOG_ERROR("[%s] mask tensor type %s is not f16/f32\n", __func__, ggml_type_name(tensor->type));
        return false;
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        const uint64_t size = n_elems * elem_size;
        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] mask tensor data region out of buffer bounds\n", __func__);
            return false;
        }
    }

    const uint8_t * bits = input.data() + sizeof(rpc_tensor) + 2 * sizeof(uint64_t);
    std::vector<uint8_t> bits_vec(bits, bits + bits_size);
    std::vector<uint8_t> out;
    if (!unpack_causal_mask(bits_vec, n_elems, tensor->type, out)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, out.data(), offset, out.size());
    return true;
}

bool rpc_server::set_tensor_mask_npast(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | n_past (8 bytes) |
    if (input.size() < sizeof(rpc_tensor) + 2 * sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *) input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    int64_t n_past;
    memcpy(&n_past, input.data() + sizeof(rpc_tensor) + sizeof(uint64_t), sizeof(n_past));

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->ne[1] <= 0 || tensor->ne[0] <= 0) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const size_t elem_size = ggml_type_size(tensor->type);
    if (elem_size != 2 && elem_size != 4) {
        GGML_LOG_ERROR("[%s] mask tensor type %s is not f16/f32\n", __func__, ggml_type_name(tensor->type));
        return false;
    }
    if (tensor->ne[3] != 1) {
        GGML_LOG_ERROR("[%s] multi-stream mask is not regenerated\n", __func__);
        return false;
    }

    const uint64_t n_kv     = tensor->ne[0];
    const uint64_t n_tokens = tensor->ne[1];
    const uint64_t n_elems  = n_kv * n_tokens;
    if (n_past < 0 || (uint64_t) n_past + n_tokens > n_kv) {
        GGML_LOG_ERROR("[%s] invalid n_past=%" PRId64 " (n_kv=%" PRIu64 ", n_tokens=%" PRIu64 ")\n",
                       __func__, n_past, n_kv, n_tokens);
        return false;
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        const uint64_t size = n_elems * elem_size;
        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] mask tensor data region out of buffer bounds\n", __func__);
            return false;
        }
    }

    // Regenerate the exact client mask: M[i][j] = 0.0 when position j is not
    // in the future (j <= n_past + i), else -inf. Single-sequence causal
    // prefill/decode with a contiguous KV cache: cell position == j.
    std::vector<uint8_t> out(n_elems * elem_size);
    uint8_t * dst = out.data();
    for (uint64_t i = 0; i < n_tokens; i++) {
        for (uint64_t j = 0; j < n_kv; j++) {
            const bool attend = (int64_t) j <= n_past + (int64_t) i;
            if (elem_size == 2) {
                uint16_t v = attend ? (uint16_t) 0x0000 : (uint16_t) 0xFC00; // 0.0 / -inf f16
                memcpy(dst, &v, 2);
            } else {
                float v = attend ? 0.0f : -INFINITY;
                memcpy(dst, &v, 4);
            }
            dst += elem_size;
        }
    }
    ggml_backend_tensor_set(tensor, out.data(), offset, out.size());
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    const bool act_f16 = (request.tensor.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F16) != 0;
    const bool act_f8  = (request.tensor.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_F8)  != 0;
    const bool act_q8  = (request.tensor.rpc_flags & GGML_RPC_TENSOR_FLAG_ACT_Q8_0) != 0;
    if ((int) act_f16 + (int) act_f8 + (int) act_q8 > 1) {
        GGML_LOG_ERROR("[%s] activation request has conflicting transport flags\n", __func__);
        return false;
    }
    if (act_q8) {
        std::vector<uint8_t> f32_buf(request.size, 0);
        ggml_backend_tensor_get(tensor, f32_buf.data(), request.offset, request.size);
        rpc_f32_to_q8_0(f32_buf.data(), request.size, response);
    } else if (act_f8) {
        std::vector<uint8_t> f32_buf(request.size, 0);
        ggml_backend_tensor_get(tensor, f32_buf.data(), request.offset, request.size);
        response.resize(request.size / 4, 0);
        ggml_fp32_to_fp8_e4m3_row((const float *) f32_buf.data(), response.data(), (int64_t) request.size / sizeof(float));
    } else if (act_f16) {
        // send F32 activations as F16 (halves the LAN traffic)
        std::vector<uint8_t> f32_buf(request.size, 0);
        ggml_backend_tensor_get(tensor, f32_buf.data(), request.offset, request.size);
        response.resize(request.size / 2, 0);
        ggml_fp32_to_fp16_row((const float *) f32_buf.data(), (ggml_fp16_t *) response.data(), (int64_t) request.size / sizeof(float));
    } else {
        response.resize(request.size, 0);
        ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    }
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    // The backend copy (e.g. Vulkan ggml_vk_buffer_copy) may run on a
    // transfer queue with no cross-queue dependency on the compute queue,
    // so a copy issued right after an asynchronous graph_compute can read
    // stale data. Drain the source backend's pending work first. This is
    // what the client-side fallback of ggml_backend_tensor_copy_async
    // expects from ggml_backend_synchronize, but the RPC synchronize is a
    // no-op - the ordering must be enforced server-side. Measured on the
    // RTX 3080 (separate transfer queue): MTP draft acceptance 0.4% -> ~70%
    // with this sync.
    for (auto & backend : backends) {
        if (ggml_backend_get_device(backend) == ggml_backend_buft_get_device(src->buffer->buft)) {
            ggml_backend_synchronize(backend);
            break;
        }
    }

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Causal attention masks (single-seq prefill/decode) are regenerated on the
    // server: substitute NULL so ggml_flash_attn_ext uses the causal path.
    // The mask may be the F32 tensor, its F16 cast ("(copy)"), or a decorated
    // scheduler copy ("<backend>#<name>#<idx>") when the cast is pinned to a
    // local backend.
    if (result->op == GGML_OP_FLASH_ATTN_EXT && result->src[3] != nullptr &&
        (getenv("GGML_RPC_ENABLE_MASK_NULL") != nullptr)) {
        const char * mask_name = result->src[3]->name;
        if (is_causal_mask_name(mask_name)) {
            result->src[3] = nullptr;
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    std::vector<uint8_t> graph_buf(buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ graph_buf.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
    }
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    auto & device_graphs = stored_graphs[device];
    const uint64_t hash = graph_structure_hash(graph);
    // bounded cache: if too many distinct graph structures are in flight, drop
    // them all - the client falls back to full sends on recompute miss
    if (device_graphs.size() >= 16) {
        device_graphs.clear();
    }
    stored_graph sg;
    sg.buffer = std::move(graph_buf);
    sg.graph = graph;
    device_graphs[hash] = std::move(sg);
    LOG_DBG("[%s] device: %u, n_nodes: %u, hash: %016" PRIx64 " (cached: %zu)\n",
            __func__, device, n_nodes, hash, device_graphs.size());
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request, rpc_msg_graph_recompute_rsp & response) {
    uint32_t device = request.device;
    response.result = 0;
    if (device >= backends.size()) {
        return true;
    }
    auto & device_graphs = stored_graphs[device];
    auto it = device_graphs.find(request.hash);
    if (it == device_graphs.end()) {
        LOG_DBG("[%s] device: %u, hash: %016" PRIx64 " - cache miss\n",
                __func__, device, request.hash);
        return true;
    }
    ggml_cgraph * graph = it->second.graph;
    if (graph == nullptr) {
        return true;
    }
    LOG_DBG("[%s] device: %u, hash: %016" PRIx64 " (cached: %zu)\n",
            __func__, device, request.hash, device_graphs.size());
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    response.result = 1;
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

rpc_server::~rpc_server() {
    {
        std::lock_guard<std::mutex> lock(worker_mutex);
        worker_shutdown = true;
        worker_busy = false;
    }
    worker_cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

bool rpc_server::graph_compute_async(const std::vector<uint8_t> & input) {
    if (worker_pipeline) {
        {
            std::unique_lock<std::mutex> lock(worker_mutex);
            if (worker_shutdown) {
                return false;
            }
            worker_pending.emplace_back(++worker_seq, input);
        }
        worker_cv.notify_one();
        return true;
    }
    {
        std::unique_lock<std::mutex> lock(worker_mutex);
        // The client sends at most one in-flight async graph followed by a
        // command that drains it; if a previous async graph is still running
        // (e.g. a second RPC segment in the same ubatch), preserve strict
        // ordering by waiting for it to finish before queueing the next one.
        worker_cv.wait(lock, [this] { return !worker_busy || worker_shutdown; });
        if (worker_shutdown) {
            return false;
        }
        worker_input = input;
        worker_busy = true;
    }
    worker_cv.notify_one();
    return true;
}

void rpc_server::graph_compute_wait() {
    std::unique_lock<std::mutex> lock(worker_mutex);
    if (worker_pipeline) {
        worker_cv.wait(lock, [this] { return (worker_pending.empty() && !worker_busy) || worker_shutdown; });
    } else {
        worker_cv.wait(lock, [this] { return !worker_busy || worker_shutdown; });
    }
}

void rpc_server::wait_enqueued_graphs() {
    if (!worker_pipeline) {
        graph_compute_wait();
        return;
    }
    std::unique_lock<std::mutex> lock(worker_mutex);
    const uint64_t seq = worker_seq;
    if (seq == 0) {
        return;
    }
    worker_cv.wait(lock, [this, seq] { return worker_done >= seq || worker_shutdown; });
}

// P2 pipeline: wait until graph number seq has completed; after the GET
// payload is served, release the next queued graph (its buffers are shared).
bool rpc_server::wait_graph_seq(uint64_t seq) {
    if (!worker_pipeline || seq == 0) {
        return false;
    }
    std::unique_lock<std::mutex> lock(worker_mutex);
    worker_cv.wait(lock, [this, seq] { return worker_done >= seq || worker_shutdown; });
    return !worker_shutdown;
}

void rpc_server::release_graph_seq(uint64_t seq) {
    if (!worker_pipeline || seq == 0) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(worker_mutex);
        if (seq > worker_release) {
            worker_release = seq;
        }
    }
    worker_cv.notify_all();
}

void rpc_server::worker_loop() {
    for (;;) {
        std::vector<uint8_t> input;
        uint64_t seq = 0;
        {
            std::unique_lock<std::mutex> lock(worker_mutex);
            worker_cv.wait(lock, [this] { return !worker_pending.empty() || worker_shutdown; });
            if (worker_shutdown) {
                return;
            }
            const auto & front = worker_pending.front();
            seq = front.first;
            input = front.second;
            worker_pending.pop_front();
            worker_busy = true;
        }
        if (worker_pipeline && seq > 1) {
            // do not overwrite the result buffers of the previous graph until
            // its GET has been served by the connection thread
            std::unique_lock<std::mutex> lock(worker_mutex);
            worker_cv.wait(lock, [this, seq] { return worker_release >= seq - 1 || worker_shutdown; });
            if (worker_shutdown) {
                return;
            }
        }
        const bool ok = graph_compute(input);
        if (!ok) {
            GGML_LOG_ERROR("[%s] async graph compute failed\n", __func__);
        }
        // The graph compute may submit asynchronously on the device queue;
        // drain it before the next buffer write/read is allowed to touch the
        // same device buffers (the connection handler awaits this before any
        // non-async command).
        for (const auto & backend : backends) {
            ggml_backend_synchronize(backend);
        }
        {
            std::lock_guard<std::mutex> lock(worker_mutex);
            worker_busy = false;
            if (worker_pipeline && seq > worker_done) {
                worker_done = seq;
            }
        }
        worker_cv.notify_all();
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    double tl_prev_end = 0.0;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        const double t_cmd = RPC_TIMELINE ? rpc_wall_ms() : 0.0;
        double tl_flush_ms = 0.0;
        // Only command classes that can race with an in-flight async graph must
        // drain it first:
        //   - SET_TENSOR (all data writes: KV, activation inputs like l_out-*,
        //     mask): the buffer may still be read by the running worker (l_out
        //     and PREV kv are reused between ubatches) — without a drain the
        //     next ubatch input can be overwritten while the previous graph
        //     still computes, corrupting the result and dropping the socket
        //     when the worker hits a bad shape/assert;
        //   - SET_TENSOR_MASK/_NPAST: the mask buffer is reused between ubatches
        //     and the previous graph may still be reading it;
        //   - GRAPH_COMPUTE/RECOMPUTE, COPY_TENSOR, GET_TENSOR: need the results
        //     of the previous graph (ordering / data ready);
        //   - BUFFER_CLEAR, FREE_BUFFER: may touch buffers the worker uses.
        // SET_TENSOR_NOFLUSH is intentionally NOT in this list: KV-cache
        // inputs (attn_inp_k_rot / attn_inp_v_rot) are written to the offset
        // of the *current* ubatch, a region the in-flight graph never reads
        // (causal attention only ever looks back).
        const bool needs_graph_drain =
            cmd == RPC_CMD_SET_TENSOR ||
            cmd == RPC_CMD_SET_TENSOR_MASK ||
            cmd == RPC_CMD_SET_TENSOR_MASK_NPAST ||
            cmd == RPC_CMD_GRAPH_COMPUTE ||
            cmd == RPC_CMD_GRAPH_RECOMPUTE ||
            cmd == RPC_CMD_GRAPH_WAIT ||
            cmd == RPC_CMD_COPY_TENSOR ||
            cmd == RPC_CMD_BUFFER_CLEAR ||
            cmd == RPC_CMD_FREE_BUFFER;
        if (needs_graph_drain) {
            // All data commands (buffer writes/reads, graph compute, recompute)
            // must observe the result of any in-flight async graph, otherwise
            // they could race with the worker on the device buffers.
            // Timeline: tl_flush_ms shows how long this wait really blocks.
            const double t_flush = RPC_TIMELINE ? rpc_wall_ms() : 0.0;
            if (server.p2_pipeline()) {
                // P2: only the graphs enqueued before this command matter; the
                // next queued graph (e.g. N+1, already pending) may still be
                // using the input buffer, so its inputs reach the server BEFORE
                // the graph is released and must not wait for it.
                server.wait_enqueued_graphs();
            } else {
                server.graph_compute_wait();
            }
            if (RPC_TIMELINE) {
                tl_flush_ms = rpc_wall_ms() - t_flush;
            }
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_NOFLUSH: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_MASK: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_mask(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_MASK_NPAST: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_mask_npast(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (server.p2_pipeline() && request.wait_seq > 0) {
                    // P2: wait only for the requested graph, not the whole
                    // queue - the next queued graph may already be pending.
                    server.wait_graph_seq(request.wait_seq);
                } else {
                    server.graph_compute_wait();
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                // the result buffers of graph wait_seq are no longer needed;
                // allow the next queued graph to start
                server.release_graph_seq(request.wait_seq);
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE_ASYNC: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute_async(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_WAIT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_graph_recompute_rsp response;
                if (!server.graph_recompute(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
        if (RPC_TIMELINE) {
            const double now = rpc_wall_ms();
            const double idle_ms = tl_prev_end > 0.0 ? t_cmd - tl_prev_end : 0.0;
            fprintf(stderr, "RPC_TL|srv|%d|%s|0|%.3f|%.3f|%.3f|t=%.1f\n",
                    (int) cmd, rpc_cmd_name((enum rpc_cmd) cmd), idle_ms,
                    now - t_cmd, tl_flush_ms, now);
            tl_prev_end = now;
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

// device interface

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    // Async graph submission is advertised when the pipeline env is set (the
    // server worker + ordered send queue keep the protocol consistent), but
    // events are NOT advertised: scheduler pipeline copies (n_copies>1) add
    // double-buffered activation memory and their per-copy event fences
    // serialize the very transfer the async outbound lane is meant to
    // overlap. Without events llama keeps pipeline_parallel off and the
    // single ordered queue still overlaps l_out streaming with the next
    // ubatch local layers.
    const bool async_pipeline = std::getenv("GGML_RPC_ASYNC_GRAPH") != nullptr;
    props->caps = {
        /* .async                 = */ async_pipeline,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *) dev->context;
    auto * event_ctx = new ggml_backend_rpc_event_context {
        /* .endpoint = */ dev_ctx->endpoint,
        /* .recorded = */ false,
    };
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event_ctx,
    };
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete (ggml_backend_rpc_event_context *) event->context;
    delete event;
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_rpc_event_context * event_ctx = (ggml_backend_rpc_event_context *) event->context;
    if (event_ctx->recorded) {
        ggml_backend_rpc_wait_endpoint(event_ctx->endpoint);
        event_ctx->recorded = false;
    }
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */ endpoint,
            /* .device      = */ ind,
            /* .name        = */ dev_name,
            /* .description = */ dev_desc
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

