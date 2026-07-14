# Vulkan Change Guide

## Where A Change Belongs

| Change | Primary module |
| --- | --- |
| Vendor/driver capability or device enumeration | `vk_device.inc`, `vk_device_helpers.inc` |
| Shader specialization or pipeline registration | `vk_shaders.inc` |
| Matmul, MMVQ/MMQ, Flash Attention route choice | `vk_dispatch.inc` |
| Generic operation support or pipeline mapping | `vk_op_pipeline.inc` |
| GGML operation command encoding | `vk_ops.inc` |
| Allocation, staging, transfer, host memory | `vk_pipeline.inc`, `vk_transfer.inc` |
| Graph submission or temporary buffers | `vk_graph.inc` |
| Fusion, graph ordering, synchronization | `vk_backend_execution.inc` |
| Public device/backend API or multi-GPU communication | `vk_backend_registry.inc` |
| Diagnostics only | `vk_tests.inc`, `vk_checks.inc` |

Do not put a route workaround into an unrelated operation merely because that
location is convenient. Keep capability discovery, policy selection, and command
encoding separate.

## Required Development Sequence

1. Record an adjacent baseline with the exact intended workload.
2. State the expected mechanism and affected graph/shader shape.
3. Make the smallest source change that tests that mechanism.
4. Build `ggml-vulkan` before launching a model.
5. Run a short correctness/output smoke.
6. Run matched prompt/decode measurements.
7. Keep accepted changes and remove rejected runtime experiments.
8. Update the relevant research note when behavior or policy changes.

## Build Gates

From the repository root:

```powershell
cmake --build build-vulkan --target ggml-vulkan -j 8
cmake --build build-vulkan --target llama-server test-backend-ops -j 8
```

A configure from a clean build directory is required after changing CMake source
lists or feature definitions. Existing binaries are not evidence that a new
module is present.

## Runtime Gates

Minimum validation for a structural refactor:

- server reaches readiness on Vulkan;
- the target model produces coherent text rather than empty/repeated output;
- prompt and decode timings are present;
- baseline and MTP requests both complete when MTP code is touched;
- dual-GPU startup uses both intended devices without shared-memory spill in the
  chosen short smoke;
- vision returns an image-grounded response when vision paths are touched.

Performance equivalence should use the canonical benchmark runner and an
adjacent pre-change result. A compile success alone cannot validate graph order,
shader selection, synchronization, or generated output.

## Refactor Invariants

- Keep runtime module include order stable unless dependencies are explicitly
  moved into a private header.
- Do not compile `.inc` modules as standalone sources.
- Do not duplicate Vulkan-Hpp dispatcher storage or configuration.
- Do not remove vendor-named extensions based only on their prefix.
- Preserve public symbols declared by `ggml-vulkan.h`.
- Preserve both text and vision operation support in the default build.
- Treat hot-path translation-unit moves as performance changes until measured.
