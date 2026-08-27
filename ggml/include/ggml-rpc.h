#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_RPC_SHARED
#ifdef _WIN32
#ifdef GGML_RPC_BUILD
#define GGML_RPC_API __declspec(dllexport)
#else
#define GGML_RPC_API __declspec(dllimport)
#endif
#else
#define GGML_RPC_API __attribute__((visibility("default")))
#endif
#else
#define GGML_RPC_API
#endif

// P2 prefill pipeline (GGML_RPC_PREFILL_PIPELINE=1): tell the RPC client
// which queued graph (1-based sequence) the next GET_TENSOR should wait for
// and read. 0 restores legacy behavior (wait for the whole worker queue).
GGML_RPC_API void ggml_backend_rpc_set_p2_get_seq(ggml_backend_t backend, uint64_t seq);

#define RPC_PROTO_MAJOR_VERSION    5
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    2

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 96, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
