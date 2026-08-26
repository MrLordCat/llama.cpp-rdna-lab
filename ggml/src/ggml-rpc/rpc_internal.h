#pragma once
// Internal shared header for the ggml-rpc file stack.
// Split from ggml-rpc.cpp (2026-08-26): protocol types (rpc_types.h),
// common transport helpers (rpc_common.cpp), client backend (rpc_client.cpp),
// server (rpc_server.cpp).

#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"
#include "rpc_types.h"

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
