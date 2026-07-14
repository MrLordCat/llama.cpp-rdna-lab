# Vulkan Runtime Documentation

This directory documents the fork's Vulkan backend implementation. It is aimed
at maintainers working on RDNA4 prompt evaluation, decode, MTP, long context,
dual-GPU execution, and vision support.

The public backend entry point remains
`ggml/src/ggml-vulkan/ggml-vulkan.cpp`. The former 18,597-line implementation is
now split into ordered modules under `ggml/src/ggml-vulkan/runtime/`.

## Documents

- `ARCHITECTURE.md`: compilation model, runtime lifecycle, state ownership, and
  synchronization boundaries.
- `QWEN35_PATHS.md`: paths used by Qwen3.6 text, MTP, and vision workloads.
- `CHANGE_GUIDE.md`: where changes belong and the validation required before a
  Vulkan change can be accepted.
- `VALIDATION.md`: structural, build, operation, and model-output evidence for
  the modular split.

## Module Map

| Module | Responsibility |
| --- | --- |
| `vk_common.inc` | Includes, Vulkan-Hpp dispatcher, shared types, architecture classification, logging |
| `vk_pipeline.inc` | Pipeline creation, command pools, synchronization primitives, allocation |
| `vk_shaders.inc` | Shader registry and device-specific specialization constants |
| `vk_device.inc` | Instance/device probing, extensions, features, queues, context initialization |
| `vk_transfer.inc` | Pipeline lookup, staging, host/device copies, buffer fills |
| `vk_dispatch.inc` | Matmul and attention route selection and command encoding |
| `vk_op_pipeline.inc` | Generic operation support checks and operation-to-pipeline mapping |
| `vk_ops.inc` | Concrete operation encoders, including GDN and vision operations |
| `vk_tests.inc` | Optional kernel diagnostics under `GGML_VULKAN_RUN_TESTS` |
| `vk_graph.inc` | Graph recording, preallocation, submission boundaries, cleanup |
| `vk_backend_buffers.inc` | GGML buffer and buffer-type interfaces |
| `vk_backend_execution.inc` | Graph execution, fusion, optimization, synchronization, events |
| `vk_backend_registry.inc` | Public API, device registry, multi-device communication |
| `vk_device_helpers.inc` | Late-bound vendor and extension helpers |
| `vk_checks.inc` | Optional CPU-reference checks under `GGML_VULKAN_CHECK_RESULTS` |

## Why Ordered Include Modules

The modules are included by one translation unit. This first-stage design is
intentional:

- it preserves the original `static` linkage and initialization order;
- hot helper inlining and generated shader symbol access remain unchanged;
- the mechanical split can be validated independently of an ABI refactor;
- each subsystem now has a stable ownership boundary for future work.

Moving modules into independently compiled `.cpp` files is a later refactor. It
requires an internal header for shared state and must be benchmarked because
moving hot helpers across translation units can change inlining without LTO.
