# ROCm runtime layout

The Windows ROCm backend reuses the CUDA-compatible runtime under
`ggml/src/ggml-cuda`. The directory name is historical: CMake compiles the
same source with HIP/Clang through `ggml/src/ggml-hip`.

## Module map

`ggml-cuda.cu` is an ordered aggregator. Its implementation is split into
`ggml/src/ggml-cuda/runtime` without creating additional translation units:

| Module | Responsibility |
| --- | --- |
| `runtime_prelude.inc` | Includes, global helpers, error handling, and shared declarations |
| `runtime_device.inc` | Device discovery, initialization, capabilities, streams, and handles |
| `runtime_buffers.inc` | Device, host, and split buffers plus transfer paths |
| `runtime_compute.inc` | Operation dispatch, matrix multiplication routing, and compute helpers |
| `runtime_backend.inc` | Backend tensor I/O and copy operations |
| `runtime_graph.inc` | Graph execution, graph replay, fusion, synchronization, and backend interface |
| `runtime_device_api.inc` | Public device-facing backend API |
| `runtime_registry.inc` | Backend registration and exported entry points |

The include order in `ggml-cuda.cu` is significant. Modules may use private
symbols declared by an earlier module. Keep code in the narrowest owning
module, but do not reorder the includes without a full dependency review.

## Why one translation unit

The split is for navigation and ownership, not a runtime optimization. Keeping
one translation unit preserves internal symbol visibility, inlining, template
instantiation behavior, and generated HIP code. A conversion to separately
compiled `.cu` files would be a different architectural change and needs its
own performance validation.

## Windows HIP toolset

ROCm 7.1 for Windows currently needs compatible MSVC 14.44 standard library
headers when newer MSVC `<cmath>` declarations conflict with HIP device
overloads. CMake discovers installed 14.44 toolsets automatically. Set
`GGML_HIP_MSVC_COMPAT_DIR` only when the toolset is installed in a location
outside the normal Visual Studio layout.

## Validation

Configure the ROCm build for the target GPU, then build the complete server:

```powershell
cmake -B build-rocm-full -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-full --target llama-server -j 4
```

For the initial split, concatenating the eight modules in aggregator order was
verified byte-for-byte against the previous `ggml-cuda.cu` Git blob. The full
ROCm `llama-server` build also passed.

## Performance work

The layout gives ROCm profiling findings a clear owner:

1. Prompt-evaluation matrix route selection belongs in `runtime_compute.inc`
   and the selected MMQ/rocBLAS kernels.
2. Decode dispatch, graph replay, and synchronization belong in
   `runtime_graph.inc` and `runtime_compute.inc`.
3. Dual-GPU staging and transfer behavior belongs in `runtime_buffers.inc` and
   the scheduler/backend copy path.
4. Device capability gates belong in `runtime_device.inc`.

Always measure prompt evaluation and decode separately. A routing change that
helps broad prompt matrices can regress the small-matrix decode path.
