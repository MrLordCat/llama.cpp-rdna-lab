# Vulkan Modular Split Validation

Date: 2026-07-13

## Structural Evidence

The pre-refactor `ggml-vulkan.cpp` contained 18,597 implementation lines. The
modular runtime has the following verified properties:

- `ggml-vulkan.cpp` is an 18-line ordered aggregator;
- all 15 runtime modules are present in the same order in CMake and the
  aggregator;
- the largest module is `vk_dispatch.inc` at 2,392 lines;
- every module ends at preprocessor depth zero;
- concatenating the modules and removing only their new module-description
  comments exactly matches `HEAD:ggml-vulkan.cpp` line for line.

This proves the initial split changed source organization, not implementation
logic.

## Build Evidence

The following targets built successfully with the MinGW Vulkan build:

```powershell
cmake --build build-vulkan --target ggml-vulkan -j 8
cmake --build build-vulkan --target llama-server test-backend-ops -j 8
```

The existing compiler warnings came from `ggml-backend-meta.cpp`,
`ggml-quants.c`, and MinGW format checking in `test-backend-ops.cpp`; no warning
or error originated in the modular Vulkan runtime.

## Operation Correctness

`test-backend-ops` passed the Q3_K decode-shape `MUL_MAT` check independently on
both RX 9070 XT devices:

```text
type_a=q3_K,type_b=f32,m=16,n=1,k=256
Vulkan0: supported=1, no error
Vulkan1: supported=1, no error
```

Both devices were detected through the AMD proprietary driver with fp16, bf16,
integer-dot, and KHR cooperative-matrix support.

## Model Output

All smokes used the modular `build-vulkan/bin/llama-server.exe`, dual device
order `Vulkan1,Vulkan0`, layer split `1,1`, q4 KV, and graceful shutdown.

### Baseline text

- model: `Qwen3.6-27B-Q3_K_S_mtp.gguf`;
- `spec=none`, `ctx=4096`, `b512/ub128`;
- prompt: `159/159` tokens;
- decode: `40.25 tok/s`;
- output: coherent analysis of the requested repository task;
- server/request errors: none.

### MTP text

- same model and runtime shape;
- `draft-mtp`, `n_max=2`, 128 completion tokens;
- decode: `70.76 tok/s`;
- acceptance: `82/90`, or `91.11%`;
- output: coherent continuation of the same repository analysis;
- server/request errors: none.

These short runs are correctness smokes, not formal performance claims. Their
generation lengths differ, so the throughput values must not be used as a
matched benchmark ratio.

### Vision

- projector: `mmproj-F16.gguf`;
- `ctx=8192`, thinking disabled, `spec=none`;
- input image: `media/matmul.png`;
- prompt/completion: `4074/64` tokens;
- output correctly identified matrix multiplication, transposed matrices, and
  row-major/column-major storage shown in the image;
- graceful shutdown succeeded.

The first `ctx=4096` vision attempt consumed 4,072 of 4,096 tokens in the image
prompt and produced no final visible content after hidden reasoning. It was not
accepted as a successful output check; increasing context to 8,192 resolved the
test without a backend change.

## Process State

After all runtime checks, no `llama-server`, `llama-cli`, or `llama-bench`
process remained. No hard process termination was enabled or used.
