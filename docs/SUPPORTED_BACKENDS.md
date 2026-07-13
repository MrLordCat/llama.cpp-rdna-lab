# Supported Backends

This fork intentionally supports only CPU, Vulkan and ROCm/HIP.

## Build targets

| CMake option | Target | Notes |
| --- | --- | --- |
| `GGML_CPU=ON` | `ggml-cpu` | Always enabled by default |
| `GGML_BLAS=ON` | `ggml-blas` | Optional CPU acceleration |
| `GGML_VULKAN=ON` | `ggml-vulkan` | Primary AMD prompt-eval path |
| `GGML_HIP=ON` | `ggml-hip` | ROCm runtime and AMD kernel path |

Unsupported backend options and source directories are removed instead of left
as dormant build choices. This keeps CMake, GUI and CI aligned.

## Why `ggml-cuda` remains

Upstream implements HIP through a CUDA-compatible kernel source layer. The
local `ggml-hip/hip-source-bundles.cmake` compiles `.cu` and `.cuh` files from
`ggml-cuda` with the ROCm compiler and `GGML_USE_HIP` enabled. The native CUDA
`CMakeLists.txt` is removed, so this directory cannot produce a native NVIDIA
backend in this fork.

Renaming or mechanically stripping this layer would touch hundreds of shared
kernels without changing runtime behavior. Treat its names and public
`ggml-cuda.h` ABI as implementation compatibility until a dedicated upstream
HIP ABI exists.

## Upstream imports

When importing upstream changes:

1. accept CPU, Vulkan and HIP changes;
2. port shared kernel changes only when they compile under HIP;
3. do not restore deleted backend directories or CMake options;
4. update this document when the supported set changes intentionally;
5. build at least CPU and the affected GPU backend before committing.
