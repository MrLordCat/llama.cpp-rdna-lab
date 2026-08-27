#pragma once
// RPC wire protocol types (structs are packed and sent verbatim).
#include "ggml.h"
#include <array>
#include <cstdint>
#include <cstring>

#pragma once
// RPC wire protocol types (structs are packed and sent verbatim).
#include "ggml.h"
#include <array>
#include <cstdint>
#include <cstring>

#pragma once
// RPC wire protocol types (structs are packed and sent verbatim).
#include "ggml.h"
#include <array>
#include <cstdint>
#include <cstring>

#pragma pack(push, 1)

struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    uint8_t rpc_flags;
    char padding[3];
};

enum {
    // Single-sequence causal attention mask: the server regenerates it locally
    // (ggml flash-attn with mask == NULL is causal), so data is never transmitted.
    GGML_RPC_TENSOR_FLAG_CAUSAL_MASK = 1 << 0,
    // F32 activation tensors that cross the RPC boundary (layer-split output
    // "l_out-<il>" and "result_output" logits): transmitted as F16 to halve
    // the network traffic and converted back at the receiving end.
    GGML_RPC_TENSOR_FLAG_ACT_F16 = 1 << 1,
    // Optional F8 E4M3 transport for layer-split outputs. This is kept
    // separate from ACT_F16 so mixed-version peers fail closed instead of
    // silently interpreting one-byte payloads as half precision.
    GGML_RPC_TENSOR_FLAG_ACT_F8 = 1 << 2,
    // Optional block-Q8_0 transport for layer-split F32 activations. It uses
    // one fp16 scale plus 32 signed bytes per block (1.0625 B/value): nearly
    // the F8 wire size, but with a per-block scale and materially more
    // mantissa precision for MTP-sensitive boundaries.
    GGML_RPC_TENSOR_FLAG_ACT_Q8_0 = 1 << 3,
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    // Causal attention mask transmitted as 1 bit per element (0.0 vs -inf).
    // Format: | rpc_tensor | offset (8) | n_elems (8) | bits ((n_elems+7)/8) |
    RPC_CMD_SET_TENSOR_MASK,
    // Causal attention mask regenerated on the server from n_past (client
    // sends only tensor metadata + offset + n_past, no payload).
    // Format: | rpc_tensor | offset (8) | n_past (8) |
    RPC_CMD_SET_TENSOR_MASK_NPAST,
    // Async graph compute (pipeline): the server queues the graph and returns
    // immediately; the client can keep computing local layers while the
    // server runs. The server drains pending async work before processing
    // any non-async command (buffer data / recompute), so the protocol stays
    // strictly ordered without a client-side wait.
    RPC_CMD_GRAPH_COMPUTE_ASYNC,
    // Wait until the server-side async graph worker and its device queue are
    // drained. Used by backend synchronize and coarse RPC scheduler events.
    RPC_CMD_GRAPH_WAIT,
    // Set-tensor without draining the in-flight async graph. Only legal for
    // KV-cache inputs (attn_inp_k_rot / attn_inp_v_rot): each ubatch writes to
    // its own non-overlapping KV region, so the data can be consumed by the
    // server immediately even while the previous graph is still computing.
    // Kept as a separate command so mixed-version peers fail closed.
    RPC_CMD_SET_TENSOR_NOFLUSH,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

static const char * rpc_cmd_name(enum rpc_cmd cmd) {
    switch (cmd) {
        case RPC_CMD_HELLO:               return "HELLO";
        case RPC_CMD_DEVICE_COUNT:        return "DEVICE_COUNT";
        case RPC_CMD_ALLOC_BUFFER:        return "ALLOC_BUFFER";
        case RPC_CMD_GET_ALLOC_SIZE:      return "GET_ALLOC_SIZE";
        case RPC_CMD_GET_ALIGNMENT:       return "GET_ALIGNMENT";
        case RPC_CMD_GET_MAX_SIZE:        return "GET_MAX_SIZE";
        case RPC_CMD_BUFFER_GET_BASE:     return "BUFFER_GET_BASE";
        case RPC_CMD_FREE_BUFFER:         return "FREE_BUFFER";
        case RPC_CMD_BUFFER_CLEAR:        return "BUFFER_CLEAR";
        case RPC_CMD_SET_TENSOR:          return "SET_TENSOR";
        case RPC_CMD_SET_TENSOR_MASK:     return "SET_TENSOR_MASK";
        case RPC_CMD_SET_TENSOR_MASK_NPAST: return "SET_TENSOR_MASK_NPAST";
        case RPC_CMD_SET_TENSOR_HASH:     return "SET_TENSOR_HASH";
        case RPC_CMD_INIT_TENSOR:         return "INIT_TENSOR";
        case RPC_CMD_GET_TENSOR:          return "GET_TENSOR";
        case RPC_CMD_COPY_TENSOR:         return "COPY_TENSOR";
        case RPC_CMD_GRAPH_COMPUTE:       return "GRAPH_COMPUTE";
        case RPC_CMD_GRAPH_COMPUTE_ASYNC: return "GRAPH_COMPUTE_ASYNC";
        case RPC_CMD_GRAPH_WAIT:          return "GRAPH_WAIT";
        case RPC_CMD_GRAPH_RECOMPUTE:     return "GRAPH_RECOMPUTE";
        case RPC_CMD_SET_TENSOR_NOFLUSH:  return "SET_TENSOR_NOFLUSH";
        case RPC_CMD_GET_DEVICE_MEMORY:   return "GET_DEVICE_MEMORY";
        default:                          return "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    // P2 pipeline (GGML_RPC_PREFILL_PIPELINE=1): the client may have queued
    // several async graphs; this GET must wait only for graph number
    // wait_seq (1-based) to finish, not for the whole worker queue.
    uint64_t wait_seq;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t hash;
};

struct rpc_msg_graph_recompute_rsp {
    uint8_t result;
};

#pragma pack(pop)

// RPC data structures

