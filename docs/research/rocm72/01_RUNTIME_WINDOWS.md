# Runtime and the Windows ROCm 7.2 stack

## Product boundary

The HIP SDK is the native-Windows portion of ROCm. The distinction matters
because many attractive ROCm features are Linux-only.

| Capability | Windows HIP SDK 7.2 | Relevance here |
| --- | --- | --- |
| `hipcc` / `clang++` | yes | builds `ggml-hip` for `gfx1201` |
| HIP runtime | yes, closed source | streams, graphs, allocations, kernels |
| Math and primitive libraries | yes | rocBLAS, rocWMMA/rocPRIM family, conditional hipBLASLt |
| Radeon GPU Profiler | yes | supported profiler for HIP compute on RX 9000 |
| ROCProfiler / `rocprofv3` | no | do not use Linux capture instructions |
| Communication libraries | no | RCCL is not a native-Windows solution |
| MIOpen / MIGraphX / AI frameworks | no in this SDK matrix | not a llama.cpp backend shortcut |
| ROCm SMI / `rocminfo` | no | use `hipInfo.exe`; use driver tools for telemetry |
| CMake HIP language | unsupported | this project correctly builds HIP through its CUDA-compatible path |

The official system-requirements table marks the RX 9070 XT (`gfx1201`) as
supported for both runtime and HIP SDK on Windows 11.

## Local version boundary

Both SDKs are installed:

- `C:/Program Files/AMD/ROCm/7.1`;
- `C:/Program Files/AMD/ROCm/7.2`.

However, `build-rocm/CMakeCache.txt` points to the 7.1 toolchain. The first
7.2 experiment therefore needs a new build directory such as `build-rocm72`.
It must not overwrite the current production/control build.

## HIP execution model

A HIP program submits work from CPU threads into GPU streams. Operations in
one stream are ordered; independent streams may overlap if the runtime maps
them to independent hardware queues and dependencies allow it. Events express
cross-stream ordering without a full device synchronization.

The relevant costs are:

- CPU launch and graph submission overhead;
- queue dependencies and explicit synchronization;
- kernel device time and occupancy;
- memory traffic and inter-device transfers.

The fork already makes broad use of streams, asynchronous copies, events, and
per-thread streams in [ggml-cuda](../../../ggml/src/ggml-cuda/). Adding more
streams cannot parallelize the two layer-split halves of one token because the
second half consumes the first half's output.

## HIP graphs

HIP graphs capture kernels, copies, and their dependencies, instantiate an
executable graph, then replay it with lower repeated launch overhead. They can
be created manually or through stream capture and can be updated between
launches.

This is not an unused feature. The local runtime implements graph capture,
instantiation, update, and replay in
[runtime_graph.inc](../../../ggml/src/ggml-cuda/runtime/runtime_graph.inc).
Existing decode evidence already shows stable graph replay. Generic graph
rewrites or graph-upload flags are not admitted without a measured launch gap.

## Environment variables worth knowing

The HIP 7.2 environment reference lists controls that are useful for
diagnosis or bounded A/B tests:

| Variable | Purpose | Use in this lane |
| --- | --- | --- |
| `HIP_VISIBLE_DEVICES` | Windows GPU isolation/order | useful for single-GPU controls |
| `GPU_MAX_HW_QUEUES` | hardware queues per process/device; default 4 | a 1/2/4 queue A/B is safe after the 7.2 baseline, but expected decode upside is low |
| `AMD_SERIALIZE_KERNEL` | force waits around kernel enqueue | debugging only; invalid for performance results |
| `AMD_SERIALIZE_COPY` | force waits around copies | debugging only; invalid for performance results |
| `GPU_DUMP_CODE_OBJECT` | dump GPU code objects | ISA/debug evidence, not an optimization |
| `HIP_MEM_POOL_USE_VM` | Windows memory-pool implementation control | startup/allocation research only |

`AMD_DIRECT_DISPATCH` is documented as Linux-current and still under
development for Windows in HIP 7.2. It is not a valid Windows tuning lever.
Serialization variables often make failures reproducible, but any timing
captured with them is instrumentation-only.

## Runtime features not currently used

Source inventory found no use of stream-ordered `hipMallocAsync`, HIP memory
pools, or stream priorities in `ggml-cuda`. HIP 7.2 marks the stream-ordered
allocator API Beta, implemented on Linux and under development on Windows, so
it is not an implementation candidate for the current production lane. Their
eventual roles are limited:

- async allocation / memory pools: model load, scratch allocation, and
  fragmentation;
- stream priorities: latency preference when multiple independent workloads
  compete;
- neither removes the serial dependency between two GPUs for one token.

Before implementing any of them, the Windows SDK must report support and RGP
or a targeted trace must show the corresponding overhead in the steady-state
decode path.

## Sources

- [HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/understand/programming_model.html)
- [HIP asynchronous execution](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/asynchronous.html)
- [HIP graph API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/hipgraph.html)
- [HIP graph tutorial](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/tutorial/graph_api.html)
- [HIP environment variables](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/env_variables.html)
- [Windows component support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html)
