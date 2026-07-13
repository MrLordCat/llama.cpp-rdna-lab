# Build Guide

This fork supports CPU, Vulkan and ROCm/HIP builds.

## Requirements

- CMake;
- Ninja;
- a C/C++ compiler;
- Vulkan SDK for Vulkan builds;
- AMD ROCm/HIP SDK for HIP builds;
- MSVC Build Tools with Desktop C++ and Windows SDK for ROCm host linking;
- OpenSSL development files when HTTPS support is required.

## CPU

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4
```

Optional CPU BLAS:

```powershell
cmake -S . -B build-cpu-blas -G Ninja -DGGML_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu-blas -j 4
```

## Vulkan

```powershell
cmake -S . -B build-vulkan -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 4
```

On the local dual-RX 9070 XT system, use explicit device order at runtime and
take the layer split from current autotune history.

## ROCm/HIP on Windows

```powershell
cmake -S . -B build-rocm -G Ninja `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS=gfx1201 `
  -DGGML_HIP_MMQ_MFMA=ON `
  -DGGML_HIP_NO_VMM=ON `
  -DGGML_OPENMP=OFF `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm -j 4
```

Use the clang toolchain shipped with HIP SDK. ROCm compilation is
memory-intensive, so `-j 4` is the normal upper bound on this machine. The
compiler is ROCm clang, but it still links against the Windows SDK/MSVC host
libraries; missing `kernel32.lib` or `msvcrtd.lib` means that Build Tools or the
developer environment is absent.

## Focused targets

```powershell
cmake --build build-vulkan -j 4 --target llama-server
cmake --build build-vulkan -j 4 --target test-backend-ops
```

## Validation safety

Do not use `llama-server --version`/`--help`, `hipMemGetInfo`, hard process
termination or `bash scripts/stage-vulkan-dlls.sh` as build checks. Prefer CMake
target success, Python syntax checks, `ctest` and a user-approved server run.

See [Supported Backends](SUPPORTED_BACKENDS.md) for the backend allowlist.
