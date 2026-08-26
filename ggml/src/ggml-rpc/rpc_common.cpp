#include "rpc_internal.h"

#include <array>
#include <cinttypes>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>

// Machine-readable RPC timeline (both client and server print one line per
// command, all in microseconds of their own steady clock):
//   client: RPC_TL|cli|<cmd_id>|<name>|<bytes>|<send_ms>|<rsp_ms>|<gap_ms>|t=<wall_ms>
//   server: RPC_TL|srv|<cmd_id>|<name>|<bytes>|<idle_ms>|<proc_ms>|<flush_ms>|t=<wall_ms>
// gap_ms = time since the previous RPC command completed (client side: time
// spent in local work; server side: idle waiting for the next command).
// flush_ms = server time spent draining an in-flight async graph before a
// non-async command (the suspected prefill serialization bottleneck).
static const auto RPC_T0 = std::chrono::steady_clock::now();
double rpc_wall_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - RPC_T0).count();
}

uint64_t graph_structure_hash(const ggml_cgraph * cgraph) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint64_t fnv_prime = 0x100000001b3ULL;
    auto mix = [&hash](const void * data, size_t len) {
        const uint8_t * p = (const uint8_t *) data;
        for (size_t i = 0; i < len; i++) {
            hash ^= p[i];
            hash *= fnv_prime;
        }
    };
    const uint64_t zero = 0;
    for (uint32_t i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        mix(&n->type, sizeof(n->type));
        mix(&n->op, sizeof(n->op));
        mix(n->ne, sizeof(n->ne));
        mix(n->nb, sizeof(n->nb));
        mix(n->op_params, sizeof(n->op_params));
        mix(n->name, strnlen(n->name, GGML_MAX_NAME));
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = n->src[s];
            if (src) {
                mix(src->name, strnlen(src->name, GGML_MAX_NAME));
            } else {
                mix(&zero, sizeof(zero));
            }
        }
    }
    return hash;
}

bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
// thread_local guards: the response variant reuses this one; the inner call
// must not print a duplicate line.
static thread_local bool rpc_tl_in_rsp = false;
static thread_local double rpc_tl_prev_end = 0.0; // wall ms of last printed command end

// ---- ordered client send queue -------------------------------------------
//
// Every client RPC command is executed by one per-socket worker thread in
// submission order. Split-input transfers (l_out-*) may block this worker
// (waiting for the producing GPU graph, then streaming the tensor), but the
// scheduler thread only submits work and never blocks on the GPU: the next
// ubatch local layers are submitted while the worker still streams the
// previous ubatch. Linear commands (get, sync, data writes) are submitted and
// awaited, so protocol order is always preserved.

struct rpc_send_task {
    std::function<bool()> fn;
    bool fire_and_forget = false;
};

struct rpc_send_queue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<rpc_send_task> tasks;
    std::thread thread;
    std::thread::id thread_id;
    bool shutdown = false;
    // task status of the latest fire-and-forget task, signalled to callers
    bool last_ok = true;
    bool has_last = false;
};

static std::mutex g_rpc_send_queues_mu;
static std::unordered_map<const socket_t *, std::shared_ptr<rpc_send_queue>> g_rpc_send_queues;
static thread_local bool tls_rpc_send_thread = false;

static std::shared_ptr<rpc_send_queue> rpc_get_send_queue(const socket_ptr & sock) {
    std::lock_guard<std::mutex> lock(g_rpc_send_queues_mu);
    auto & slot = g_rpc_send_queues[sock.get()];
    if (!slot) {
        slot = std::make_shared<rpc_send_queue>();
        slot->thread = std::thread([slot]() {
            tls_rpc_send_thread = true;
            slot->thread_id = std::this_thread::get_id();
            for (;;) {
                rpc_send_task task;
                {
                    std::unique_lock<std::mutex> lock(slot->mu);
                    slot->cv.wait(lock, [&] { return !slot->tasks.empty() || slot->shutdown; });
                    if (slot->tasks.empty() && slot->shutdown) {
                        return;
                    }
                    task = std::move(slot->tasks.front());
                    slot->tasks.pop_front();
                }
                const bool ok = task.fn ? task.fn() : false;
                {
                    std::lock_guard<std::mutex> lock(slot->mu);
                    slot->last_ok = ok;
                    slot->has_last = true;
                }
                slot->cv.notify_all();
            }
        });
    }
    return slot;
}

bool rpc_send_submit(const socket_ptr & sock, std::function<bool()> fn, bool fire_and_forget) {
    auto q = rpc_get_send_queue(sock);
    if (tls_rpc_send_thread) {
        // re-entrant call from inside the queue worker (e.g. an async copy
        // task streaming via ggml_backend_tensor_set): run inline, the worker
        // already owns the ordered execution context.
        return fn();
    }
    if (fire_and_forget) {
        {
            std::lock_guard<std::mutex> lock(q->mu);
            q->tasks.push_back({ std::move(fn), true });
        }
        q->cv.notify_one();
        return true;
    }
    // Blocking submit: wait for THIS task to complete (the worker pops the
    // task from the deque *before* running it, so "queue empty" is NOT a
    // completion signal - it can fire while the previous task still runs).
    auto done = std::make_shared<std::promise<bool>>();
    auto fut = done->get_future();
    {
        std::lock_guard<std::mutex> lock(q->mu);
        q->tasks.push_back({ [fn = std::move(fn), done]() {
            const bool ok = fn();
            done->set_value(ok);
            return ok;
        }, false });
    }
    q->cv.notify_one();
    return fut.get();
}

bool send_rpc_cmd_direct(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    auto lock = sock->lock(); // keep the whole command atomic w.r.t. other threads
    auto t0 = std::chrono::steady_clock::now();
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    if (RPC_TIMELINE && !rpc_tl_in_rsp) {
        const double send_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double now = rpc_wall_ms();
        const double gap_ms = rpc_tl_prev_end > 0.0 ? now - send_ms - rpc_tl_prev_end : 0.0;
        fprintf(stderr, "RPC_TL|cli|%d|%s|%zu|%.3f|0.000|%.3f|t=%.1f\n",
                (int) cmd, rpc_cmd_name(cmd), input_size, send_ms, gap_ms, now);
        rpc_tl_prev_end = now;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
bool send_rpc_cmd_direct(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    auto lock = sock->lock(); // hold the socket for the whole request/response pair
    if (RPC_TIMELINE) {
        rpc_tl_in_rsp = true;
    }
    auto t_start = std::chrono::steady_clock::now();
    if (!send_rpc_cmd_direct(sock, cmd, input, input_size)) {
        rpc_tl_in_rsp = false;
        return false;
    }
    const double send_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        fprintf(stderr, "[rpc-client] ERROR recv response size for cmd %s failed\n", rpc_cmd_name(cmd));
        rpc_tl_in_rsp = false;
        return false;
    }
    if (out_size != output_size) {
        fprintf(stderr, "[rpc-client] ERROR cmd %s response size mismatch: got %llu expected %llu\n",
                rpc_cmd_name(cmd), (unsigned long long) out_size, (unsigned long long) output_size);
        rpc_tl_in_rsp = false;
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        fprintf(stderr, "[rpc-client] ERROR recv response body for cmd %s failed\n", rpc_cmd_name(cmd));
        rpc_tl_in_rsp = false;
        return false;
    }
    if (RPC_TIMELINE) {
        const double rsp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count() - send_ms;
        const double now = rpc_wall_ms();
        const double gap_ms = rpc_tl_prev_end > 0.0 ? now - rsp_ms - rpc_tl_prev_end : 0.0;
        fprintf(stderr, "RPC_TL|cli|%d|%s-rsp|%zu|%.3f|%.3f|%.3f|t=%.1f\n",
                (int) cmd, rpc_cmd_name(cmd), input_size, send_ms, rsp_ms, gap_ms, now);
        rpc_tl_prev_end = now;
    }
    rpc_tl_in_rsp = false;
    return true;
}

// Blocking wrapper: preserves protocol order through the per-socket queue.
// Enabled only with GGML_RPC_ASYNC_GRAPH=1 (the async outbound pipeline);
// otherwise commands go straight to the socket as before.
static const bool rpc_send_queue_enabled = std::getenv("GGML_RPC_ASYNC_GRAPH") != nullptr;

bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    if (!rpc_send_queue_enabled) {
        return send_rpc_cmd_direct(sock, cmd, input, input_size);
    }
    std::vector<uint8_t> in((const uint8_t *) input, (const uint8_t *) input + input_size);
    return rpc_send_submit(sock, [sock, cmd, in = std::move(in)]() {
        return send_rpc_cmd_direct(sock, cmd, in.data(), in.size());
    }, false);
}

bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!rpc_send_queue_enabled) {
        return send_rpc_cmd_direct(sock, cmd, input, input_size, output, output_size);
    }
    std::vector<uint8_t> in((const uint8_t *) input, (const uint8_t *) input + input_size);
    std::vector<uint8_t> out(output_size, 0);
    const bool ok = rpc_send_submit(sock, [sock, cmd, in = std::move(in), &out]() {
        return send_rpc_cmd_direct(sock, cmd, in.data(), in.size(), out.data(), out.size());
    }, false);
    if (ok && output_size > 0) {
        memcpy(output, out.data(), output_size);
    }
    return ok;
}

// Fire-and-forget wrapper (used by the async graph path): the command is
// queued after any pending transfers but the caller does not wait for it.
bool send_rpc_cmd_async(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    if (!rpc_send_queue_enabled) {
        return send_rpc_cmd_direct(sock, cmd, input, input_size);
    }
    std::vector<uint8_t> in((const uint8_t *) input, (const uint8_t *) input + input_size);
    return rpc_send_submit(sock, [sock, cmd, in = std::move(in)]() {
        return send_rpc_cmd_direct(sock, cmd, in.data(), in.size());
    }, true);
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    return true;
}

std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!negotiate_hello(sock)) {
        return nullptr;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = sock;
    return sock;
}

bool is_causal_mask_name(const char * name) {
    if (strcmp(name, "attn_inp_kq_mask") == 0 ||
        strcmp(name, "attn_inp_kq_mask (copy)") == 0) {
        return true;
    }
    const char * p = strstr(name, "#attn_inp_kq_mask");
    if (p == nullptr) {
        return false;
    }
    p += strlen("#attn_inp_kq_mask");
    // F32 mask: "#attn_inp_kq_mask#<idx>"; F16 cast: "#attn_inp_kq_mask (copy)#<idx>";
    // multi-seq/SWA mask keeps the "_ms" suffix and is excluded.
    return p[0] == '#' || p[0] == ' ';
}

// F32 activations that cross the RPC boundary are transmitted as F16 by
// default. Match the scheduler-decorated names of the layer-split output and
// the logits: "RPC0[host]#l_out-16#0" and "result_output".

const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

void ggml_backend_rpc_wait_endpoint(const std::string & endpoint) {
    auto sock = get_socket(endpoint);
    if (RPC_TIMELINE) {
        fprintf(stderr, "RPC_TL|cli|SYNC_ENTER|%s|t=%.1f\n", endpoint.c_str(), rpc_wall_ms());
    }
    bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_WAIT, nullptr, 0, nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}
