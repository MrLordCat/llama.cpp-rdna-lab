# Upstream Sync Policy

The fork imports useful `ggml-org/llama.cpp` core/runtime work without becoming
a full mirror.

## Local contract

Keep these local systems intact:

- PyQt6 GUI and build registry;
- benchmark/autotune tooling and generated history format;
- MTP/DFlash integration;
- dual-RX 9070 XT Vulkan and ROCm optimizations;
- local documentation, CI and agent rules;
- the CPU, Vulkan and ROCm/HIP-only backend allowlist.

Protected paths include `.github/`, `docs/`, `gui/`, root project documents and
local benchmark scripts. Do not replace them wholesale with upstream versions.

## Import procedure

1. Check `git status --short --branch` and identify unrelated local changes.
2. Fetch upstream and inspect commit/file scope.
3. Prefer a focused manual port or small cherry-pick over a broad merge.
4. Import only the required core/runtime behavior.
5. Resolve conflicts in sympathy with local MTP, Vulkan, ROCm and GUI changes.
6. Confirm that removed backends were not restored.
7. Build CPU and every affected GPU backend.
8. Update local research/docs when behavior or performance changed.

Useful inspection commands:

```powershell
git fetch upstream
git show --stat <commit>
git show --name-only <commit>
git diff HEAD...upstream/master -- src common include ggml tools
```

## Backend filter

Accept directly when relevant:

- generic ggml/llama core;
- CPU;
- Vulkan;
- HIP/ROCm;
- shared `ggml-cuda` kernel changes required by HIP.

Do not import:

- native CUDA build integration;
- Metal, SYCL, OpenCL/OpenVINO, CANN, MUSA, WebGPU, VirtGPU, Hexagon,
  zDNN/ZenDNN or RPC backends;
- their CI, docs, examples, packages or public headers.

`ggml/src/ggml-cuda` remains a HIP implementation dependency. Its native
`CMakeLists.txt` must stay absent. See `docs/SUPPORTED_BACKENDS.md`.

## Validation

```powershell
python -m compileall -q gui scripts run.py
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4 --target llama-server
git diff --check
```

For Vulkan or ROCm changes, build the corresponding target. Do not use
`llama-server --version`/`--help`, `hipMemGetInfo`, hard process termination or
the forbidden Vulkan DLL staging script as validation.

## MTP changes

Port MTP as a coherent pipeline: graph inputs, NextN extraction, speculative
state and server integration must match. Validate an adjacent `spec=none`
baseline and MTP-enabled GGUF. Keep prompt and decode metrics separate.
