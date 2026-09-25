# Supported Backends

This fork intentionally supports only CPU, Vulkan and ROCm/HIP as local
compute backends. RPC remains supported as a Vulkan-profile transport.

## Build targets

| CMake option | Target | Notes |
| --- | --- | --- |
| `GGML_CPU=ON` | `ggml-cpu` | Always enabled by default |
| `GGML_BLAS=ON` | `ggml-blas` | Optional CPU acceleration |
| `GGML_VULKAN=ON` | `ggml-vulkan` | Primary AMD prompt-eval path |
| `GGML_HIP=ON` | `ggml-hip` | ROCm runtime and AMD kernel path |
| `GGML_RPC=ON` | `ggml-rpc` / `rpc-server` | Remote-device transport used by the Vulkan profile; disabled in ROCm builds |

Unsupported backend options and source directories are removed instead of left
as dormant build choices. RPC is not a local compute backend: it exposes
remote workers to the Vulkan client as `RPC0`, `RPC1`, and so on. This keeps
CMake, GUI and CI aligned.

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

## Local fp8 E4M3 support

This fork carries a local fp8 implementation layered on the supported
backends. See `FP8_ATTENTION.md` for the full description; the essentials:

- **KV cache type `f8_e4m3`** (`--cache-type-k f8_e4m3 --cache-type-v f8_e4m3`)
  with a bit-level Vulkan encoder (`types.glsl` `f32_to_fp8_e4m3`) and
  byte-compatible HIP conversion on ROCm.
- **Native fp8 attention paths** P2–P5 (Vulkan coopmat), selected by env
  (`GGML_VK_FA_F8_P5` is the default for Vulkan in this fork; the kernel
  ignores it for non-f8 KV). Plain f16-preconvert remains the fallback.
- **ROCm fp8 attention** consumes eligible F8 phases with gfx12 WMMA and uses
  device-resident F8-to-f16 conversion everywhere else. Native KQ owns
  D=64/80/96/112/128/256; native P*V owns D=64/128/256, while other supported
  head sizes, mixed K/V types and pre-RDNA4 builds use the portable fallback.
  Exact K/V aliases are raw only when both phases are native and otherwise
  share one converted allocation. The spill-free D=256 F8/F8 route remains
  the production default after D098/D099; set
  `GGML_ROCM_FATTN_F8_NATIVE_KQ=0` for full rollback or
  `GGML_ROCM_FATTN_F8_NATIVE_V=0` for the KQ-only bisect.
- **Hybrid KV cache**: `LLAMA_VK_MTP_KV_LAST_F16=N` keeps the last N KV
  layers in f16 under MTP + f8/q8 KV. It is **opt-in**: unset means the cache
  stays uniformly quantized. Before 2026-09-14 `common.cpp` set N=8 (12 at
  `n_ctx >= 98304`) automatically; E348 measured on ROCm that the tail is never
  read by the draft head on that backend (NextN arrives by device handoff) while
  costing `+3552 MiB` and `-13.5%` prefill at `ctx 151552`, and that it broke
  tool-using agent sessions on f8_e4m3 KV. The automatic policy was removed; the
  D096/D097/W30 Vulkan lanes that measured an acceptance win pass N explicitly.
- P5/native ROCm/hybrid are local performance work; when importing upstream
  GPU changes, keep the `GGML_VK_FA_F8_*`, `GGML_ROCM_FATTN_F8_NATIVE_*` and
  `LLAMA_VK_MTP_KV_LAST_F16` rollback contracts plus both byte-compatible
  encoders intact.
