// The Vulkan runtime is kept as one translation unit to preserve static linkage,
// initialization order, and hot-path inlining. The ordered implementation modules
// below are documented in docs/vulkan/ and listed in the backend CMake target.
#include "runtime/vk_common.inc"
#include "runtime/vk_pipeline.inc"
#include "runtime/vk_shaders.inc"
#include "runtime/vk_device.inc"
#include "runtime/vk_transfer.inc"
#include "runtime/vk_dispatch.inc"
#include "runtime/vk_op_pipeline.inc"
#include "runtime/vk_ops.inc"
#include "runtime/vk_tests.inc"
#include "runtime/vk_graph.inc"
#include "runtime/vk_backend_buffers.inc"
#include "runtime/vk_backend_execution.inc"
#include "runtime/vk_backend_registry.inc"
#include "runtime/vk_device_helpers.inc"
#include "runtime/vk_checks.inc"
