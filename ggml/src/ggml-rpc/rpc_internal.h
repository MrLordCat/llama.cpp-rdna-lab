#pragma once
// Internal shared header for the ggml-rpc file stack.
// Split from ggml-rpc.cpp (2026-08-26): protocol types (rpc_types.h),
// common transport helpers (rpc_common.cpp), client backend (rpc_client.cpp),
// server (rpc_server.cpp).

#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "transport.h"
#include "rpc_types.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// per-TU copies of the env gates (same process-wide value, avoids cross-TU state)
static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");
static const char * RPC_TIMELINE = std::getenv("GGML_RPC_TIMELINE");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

static inline size_t rpc_q8_0_wire_size(size_t f32_size) {
    GGML_ASSERT(f32_size % sizeof(float) == 0);
    const int64_t n = (int64_t) f32_size / sizeof(float);
    GGML_ASSERT(n % QK8_0 == 0);
    return ggml_row_size(GGML_TYPE_Q8_0, n);
}

static inline size_t rpc_q8_0_f32_size(size_t wire_size) {
    GGML_ASSERT(wire_size % sizeof(block_q8_0) == 0);
    return (wire_size / sizeof(block_q8_0)) * QK8_0 * sizeof(float);
}

static inline size_t rpc_activation_threads(size_t n_blocks) {
    const char * value = std::getenv("GGML_RPC_ACT_THREADS");
    const size_t requested = value != nullptr ? (size_t) std::max(1, atoi(value)) : 8;
    return std::min(requested, std::max<size_t>(1, n_blocks));
}

static inline void rpc_f32_to_q8_0(const void * src, size_t f32_size, std::vector<uint8_t> & dst) {
    const int64_t n = (int64_t) f32_size / sizeof(float);
    dst.resize(rpc_q8_0_wire_size(f32_size));
    const size_t n_blocks = (size_t) n / QK8_0;
    const size_t n_threads = rpc_activation_threads(n_blocks);
    std::vector<std::thread> workers;
    workers.reserve(n_threads > 0 ? n_threads - 1 : 0);

    auto convert = [src, &dst, n_blocks, n_threads](size_t tid) {
        const size_t b0 = n_blocks * tid / n_threads;
        const size_t b1 = n_blocks * (tid + 1) / n_threads;
        const int64_t chunk_n = (int64_t) (b1 - b0) * QK8_0;
        if (chunk_n == 0) {
            return;
        }
        const size_t written = ggml_quantize_chunk(
                GGML_TYPE_Q8_0,
                (const float *) src + b0 * QK8_0,
                dst.data() + b0 * sizeof(block_q8_0),
                0, 1, chunk_n, nullptr);
        GGML_ASSERT(written == (b1 - b0) * sizeof(block_q8_0));
    };

    for (size_t t = 1; t < n_threads; ++t) {
        workers.emplace_back(convert, t);
    }
    convert(0);
    for (auto & worker : workers) {
        worker.join();
    }
}

// Multi-threaded F32 <-> F16 conversion for activation transfers. The stock
// ggml_fp32_to_fp16_row/fp16_to_fp32_row helpers are single-threaded; on the
// serial RPC server side a 20 MB layer output costs tens of milliseconds,
// which is the dominant part of the per-ubatch GET_TENSOR/ SET_TENSOR round
// trip on the local loopback lane.
static inline void rpc_f32_to_f16(const void * src, size_t f32_size, std::vector<uint8_t> & dst) {
    const int64_t n = (int64_t) f32_size / sizeof(float);
    GGML_ASSERT(n > 0 && f32_size % sizeof(float) == 0);
    dst.resize(size_t(n) * sizeof(ggml_fp16_t));
    const size_t n_threads = rpc_activation_threads((size_t) n);
    std::vector<std::thread> workers;
    workers.reserve(n_threads > 0 ? n_threads - 1 : 0);

    auto convert = [src, n, n_threads](std::vector<uint8_t> & dst, size_t tid) {
        const int64_t c0 = n * (int64_t) tid / (int64_t) n_threads;
        const int64_t c1 = n * (int64_t) (tid + 1) / (int64_t) n_threads;
        if (c1 == c0) {
            return;
        }
        ggml_fp32_to_fp16_row((const float *) src + c0, (ggml_fp16_t *) dst.data() + c0, c1 - c0);
    };

    for (size_t t = 1; t < n_threads; ++t) {
        workers.emplace_back(convert, std::ref(dst), t);
    }
    convert(dst, 0);
    for (auto & worker : workers) {
        worker.join();
    }
}

static inline void rpc_f16_to_f32(const void * src, size_t f16_size, void * dst) {
    const int64_t n = (int64_t) f16_size / sizeof(ggml_fp16_t);
    GGML_ASSERT(n > 0 && f16_size % sizeof(ggml_fp16_t) == 0);
    const size_t n_threads = rpc_activation_threads((size_t) n);
    std::vector<std::thread> workers;
    workers.reserve(n_threads > 0 ? n_threads - 1 : 0);

    auto convert = [src, dst, n, n_threads](size_t tid) {
        const int64_t c0 = n * (int64_t) tid / (int64_t) n_threads;
        const int64_t c1 = n * (int64_t) (tid + 1) / (int64_t) n_threads;
        if (c1 == c0) {
            return;
        }
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src + c0, (float *) dst + c0, c1 - c0);
    };

    for (size_t t = 1; t < n_threads; ++t) {
        workers.emplace_back(convert, t);
    }
    convert(0);
    for (auto & worker : workers) {
        worker.join();
    }
}

static inline void rpc_q8_0_to_f32(const void * src, size_t wire_size, void * dst) {
    const size_t n_blocks = wire_size / sizeof(block_q8_0);
    const size_t n_threads = rpc_activation_threads(n_blocks);
    std::vector<std::thread> workers;
    workers.reserve(n_threads > 0 ? n_threads - 1 : 0);

    auto convert = [src, dst, n_blocks, n_threads](size_t tid) {
        const size_t b0 = n_blocks * tid / n_threads;
        const size_t b1 = n_blocks * (tid + 1) / n_threads;
        const int64_t chunk_n = (int64_t) (b1 - b0) * QK8_0;
        if (chunk_n == 0) {
            return;
        }
        dequantize_row_q8_0(
                (const block_q8_0 *) src + b0,
                (float *) dst + b0 * QK8_0,
                chunk_n);
    };

    for (size_t t = 1; t < n_threads; ++t) {
        workers.emplace_back(convert, t);
    }
    convert(0);
    for (auto & worker : workers) {
        worker.join();
    }
}

// ---- common transport / helpers (defined in rpc_common.cpp) ----
double rpc_wall_ms();

bool send_msg(socket_ptr sock, const void * msg, size_t msg_size);
bool recv_msg(socket_ptr sock, void * msg, size_t msg_size);
bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input);

bool parse_endpoint(const std::string & endpoint, std::string & host, int & port);
bool negotiate_hello(const std::shared_ptr<socket_t> & sock);
std::shared_ptr<socket_t> get_socket(const std::string & endpoint);

// ordered per-socket outbound queue with optional fire-and-forget mode
bool rpc_send_submit(const socket_ptr & sock, std::function<bool()> fn, bool fire_and_forget);

bool send_rpc_cmd_direct(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size);
bool send_rpc_cmd_direct(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size);
bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size);
bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size);
bool send_rpc_cmd_async(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size);

bool is_causal_mask_name(const char * name);
uint64_t graph_structure_hash(const ggml_cgraph * cgraph);

// ---- shared between rpc_client.cpp and rpc_server.cpp ----
struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft);
void ggml_backend_rpc_wait_endpoint(const std::string & endpoint);

uint64_t fnv_hash(const uint8_t * data, size_t len);

struct ggml_backend_rpc_event_context {
    std::string endpoint;
    bool recorded = false;
};
