# llama.cpp-with-GUI

`llama.cpp-with-GUI` is a hardware-focused fork of
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) for local AI on
Windows with two AMD RDNA4 GPUs. It combines a maintained subset of the
`llama.cpp` runtime with a PyQt6 desktop application, reproducible
benchmark/autotune tooling, long-context work, and AMD-specific Vulkan and
ROCm/HIP optimizations.

The primary workload is agentic coding with Qwen3.6-27B: large cold prompts,
single-user requests, long contexts, tool use, vision, and speculative decode.
The main performance priority is prompt evaluation. MTP is kept only when its
decode gain does not impose an unacceptable prefill cost.

> This is a specialized research and production fork, not a drop-in replacement
> for every upstream platform. Results and defaults are tuned for the reference
> dual-RX 9070 XT machine described below.

## Project Goals

- Maximize Qwen3.6 prompt-evaluation throughput for agent workloads.
- Use both GPUs without moving the active working set into system RAM.
- Make MTP improve decode while keeping long-prompt prefill close to baseline.
- Provide a practical GUI for building, launching, monitoring, and autotuning.
- Keep performance claims reproducible through cold, lane-locked benchmarks.
- Keep the fork maintainable by carrying only the backends and upstream changes
  that are useful on this machine.

Current non-goals include broad accelerator portability and native support for
NVIDIA CUDA, Metal, SYCL, OpenCL, CANN, or other removed upstream backends.

## Supported Backends

| Backend | Role | Status |
| --- | --- | --- |
| Vulkan | Primary prompt-eval and general AMD runtime | Supported and preferred for prompt-heavy work |
| ROCm/HIP | MTP, RDNA4 kernel research, and AMD runtime comparison | Supported |
| CPU | Fallback, conversion, sanity checks, and tests | Supported |

ROCm still builds HIP-compatible kernels from `ggml/src/ggml-cuda`. That is an
internal HIP implementation detail and does not mean that this fork supports
NVIDIA hardware. See [Supported Backends](docs/SUPPORTED_BACKENDS.md).

## Reference System

- Windows 11
- AMD Ryzen 7 5800X3D, 8 cores / 16 threads
- 64 GB system RAM
- 2x AMD Radeon RX 9070 XT, 16 GB VRAM each, RDNA4 `gfx1201`
- AMD ROCm/HIP SDK 7.1 for Windows
- AMD proprietary Vulkan driver
- Main model: `Qwen3.6-27B-Q3_K_S_mtp.gguf`
- Vision projector: `mmproj-F16.gguf`

The two GPUs are normally used with layer split, not tensor split. GPU1 is the
preferred output device because GPU0 also drives the desktop. Device order is
backend- and workload-sensitive even with two identical cards; the reason and
the measured orders are documented below. Exact PCIe topology, driver version,
background GPU load, KV type, and model residency can materially change the
numbers below.

## Current Performance

Snapshot date: **2026-07-14**.

All headline rows use the Qwen3.6-27B Q3_K_S MTP-enabled GGUF, FlashAttention,
one server slot, cold prompt processing, no prompt-cache reuse, and no prime
pass. Prompt and decode TPS come from server timings; aggregate TPS includes the
whole request wall time. Compare `none` and `MTP` only inside the same backend
and lane. The table was rerun after rebuilding both backends, with no foreground
GPU workload active.

### Benchmark Launch Parameters

| Lane | Context | Actual prompt | Output | Batch / UBatch | KV | Repeats |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Vulkan short | 12,288 | 7,842 | 128 | 8192 / 1024 | q8_0 / q8_0 | 3 |
| ROCm short | 12,288 | 7,729 | 256 | 8192 / 1024 | q8_0 / q8_0 | 3 |
| Matched long, both backends | 49,152 | 31,997 | 128 | 8192 / 1024 | q8_0 / q8_0 | 2 |

Every row also uses `-np 1 -ngl 999 --flash-attn on --no-warmup -fit off`, seed
42, temperature 0.2, top-p 0.9, `--cache-ram 0`, `--ctx-checkpoints 0`, and no
prompt reuse.
The matched long lane requests 147,456 repository-snapshot characters; the
runner's 0.88 safe-fill policy caps this to 106,800 characters, producing the
same 31,997-token prompt on both backends. The GUI name now says
`Long ctx 49K - ~32K actual prompt` because 49,152 is the context capacity, not
the measured prompt length.

Vulkan uses `-dev Vulkan0,Vulkan1 -sm layer -ts 1,1`,
`LLAMA_OUTPUT_DEVICE=Vulkan1`, and `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`. ROCm uses
`-dev ROCm1,ROCm0 -sm layer -ts 1,1` with direct peer copy disabled. MTP rows
add `--spec-type draft-mtp --spec-draft-n-max 3`; the prefill window is the
server default of 256 tokens.

### Short Prompt Lanes

| Backend | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | `none`, r3 mean | 7,842 / 128 | **1783.49** | 38.17 | 16.42 | `Vulkan0,Vulkan1`, `ctx=12288`, q8/q8 KV |
| Vulkan | MTP n3, r3 mean | 7,842 / 128 | 1724.73 | **51.82** | **17.99** | 60.05% acceptance; backend-resident NextN |
| ROCm | `none`, r3 mean | 7,729 / 256 | **1706.57** | 25.74 | 17.61 | `ROCm1,ROCm0`, `ctx=12288`, q8/q8 KV |
| ROCm | MTP n3, r3 mean | 7,729 / 256 | 1656.53 | **34.95** | **21.26** | 44.65% acceptance; backend-resident NextN |

In this lane, Vulkan MTP changes prompt/decode/aggregate throughput by
`-3.29% / +35.78% / +9.55%`. ROCm MTP changes them by
`-2.93% / +35.76% / +20.71%`.

### Long Prompt Lanes

| Backend | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | `none`, r2 mean | 31,997 / 128 | 1488.47 | 32.89 | 5.02 | `ctx=49152`, `b8192/ub1024`, q8/q8 KV |
| Vulkan | MTP n3, r2 mean | 31,997 / 128 | **1519.98** | **41.97** | **5.29** | 48.70% acceptance; 256-token prefill window |
| ROCm | `none`, r2 mean | 31,997 / 128 | **1338.10** | 22.30 | 4.30 | `ctx=49152`, `b8192/ub1024`, q8/q8 KV |
| ROCm | MTP n3, r2 mean | 31,997 / 128 | 1337.23 | **32.24** | **4.57** | 50.33% acceptance; 256-token prefill window |

In the long lanes, Vulkan MTP changes prompt/decode/aggregate throughput by
`+2.12% / +27.59% / +5.28%`. ROCm MTP changes them by
`-0.07% / +44.61% / +6.27%`. The prefill tax is therefore within the current
acceptance target, while generation remains materially faster. Longer generated
answers benefit more because prompt evaluation dominates these 128-token runs.

Q4_K_M and UD-Q4_K_XL are supported, but the 27B Q4 long-context working set
currently enters WDDM shared memory on this 2x16 GB system; Q3_K_S remains the
practical performance model. The active Q3 prompt-evaluation research target is
2000 prompt tok/s.

Evidence:

- [E284: matched 49K-context README lane](docs/research/experiments/E284_matched_49k_context_readme_lane.md)
- [E283: clean README revalidation](docs/research/experiments/E283_clean_readme_revalidation.md)
- [E282: MTP device hidden-state handoff](docs/research/experiments/E282_mtp_device_hidden_handoff.md)
- [D078: ROCm RDNA4 small-N DP4A MTP](docs/research/major-topology/D078_P002_ROCM_MTP_SMALLN_DP4A_MMQ.md)
- [D080: Vulkan layer-stage balance](docs/research/major-topology/D080_P003_VULKAN_LAYER_STAGE_BALANCE.md)
- [Canonical benchmark history](build_logs/agent-workload/BENCH_RUNS.csv)
- [Benchmark notes](BENCHMARKS.md)

## Key Fork Features

- PyQt6 GUI for dependency checks, builds, server launch, monitoring, and logs.
- Vulkan/ROCm-aware benchmark and autotune UI with live prompt progress.
- OpenAI-compatible `llama-server` for local applications and coding agents.
- Dual-GPU layer placement and explicit output-device controls.
- Upstream-style Qwen3.6 MTP pipeline with backend-resident NextN handoff.
- Long-prompt MTP recent-window processing; built-in window is 256 tokens.
- RDNA4 Q3_K prompt and small-N decode kernel specializations.
- Vision support through a compatible `mmproj-*.gguf` projector.
- Prompt checkpoints, cache controls, benchmark history, and diagnostic traces.
- DFlash integration for research; it is not currently the recommended runtime
  profile.

## Quick Start

Install Python GUI dependencies and launch the application from the repository
root:

```powershell
python -m pip install -r gui/requirements-gui.txt
python run.py
```

`run.bat` and `start-gui.bat` are also available. In the GUI:

1. Open **Build & Setup** and configure Vulkan, ROCm/HIP, or CPU.
2. Build `llama-server` or select an existing compatible build.
3. Open **Launch Server** and select a local GGUF model.
4. Start with `Spec: None` to establish a baseline.
5. For an MTP-enabled GGUF, select MTP and use depth 3 as the current general
   Vulkan/ROCm starting point.
6. For vision, enable the projector and select `models/mmproj-F16.gguf`.
7. Use **Benchmark / Autotune** to validate batch, ubatch, KV, split, and spec
   settings for the intended context length.

Model files are not part of the source tree history. Put local GGUF files in
`models/` or select them from another local directory.

## Build Requirements

The reference builds are Windows x64 builds. A clean machine needs:

- Git and 64-bit Python 3.11 or newer with `pip`;
- CMake 3.14 or newer and Ninja (tested with CMake 3.29 and Ninja 1.12);
- Visual Studio Build Tools 2022 with **Desktop development with C++**, the
  MSVC v143 toolset, and a Windows 10 or 11 SDK;
- the current AMD display driver, including the Vulkan runtime;
- LunarG Vulkan SDK with `glslc` for Vulkan builds;
- AMD ROCm/HIP SDK 7.1 for Windows for ROCm builds;
- Strawberry Perl for Windows ROCm configuration and the reference MinGW
  Vulkan toolchain;
- OpenSSL development files. HTTPS is enabled by default; use
  `-DLLAMA_OPENSSL=OFF` only when HTTPS/model downloads are not required.

The tested Vulkan build uses the GCC 13.2 MinGW-w64 toolchain bundled with
Strawberry Perl. A MinGW executable also needs `libgcc_s_seh-1.dll`,
`libstdc++-6.dll`, and `libwinpthread-1.dll` either beside the executable or on
`PATH`. The GUI launch environment handles the configured toolchain; for a
manual launch, put `C:\Strawberry\c\bin` before other MinGW installations on
`PATH` to avoid loading incompatible runtime DLLs.

The tested ROCm build uses `clang.exe` and `clang++.exe` from HIP SDK 7.1, not
MSVC as the compiler, but still links against MSVC v143 and Windows SDK host
libraries. Strawberry Perl is also required. A full HIP compilation is memory
intensive; 64 GB RAM and `-j 4` are recommended for this fork. Allow roughly
30 GB of free disk space for source, two build trees, and one local model.

Install the Python side and verify the native tools before opening the GUI:

```powershell
python -m pip install --upgrade pip
python -m pip install -r gui/requirements-gui.txt
cmake --version
ninja --version
glslc --version
```

The GUI's **Build & Setup** tab checks the configured dependencies and creates
backend-specific build directories. Manual equivalents are shown below.

### CPU

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4 --target llama-server
```

### Vulkan

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\<version>"
$env:PATH = "$env:VULKAN_SDK\Bin;C:\Strawberry\c\bin;$env:PATH"

cmake -S . -B build-vulkan -G Ninja `
  -DGGML_VULKAN=ON `
  -DCMAKE_C_COMPILER=C:\Strawberry\c\bin\gcc.exe `
  -DCMAKE_CXX_COMPILER=C:\Strawberry\c\bin\g++.exe `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 4 --target llama-server
```

### ROCm/HIP on Windows RDNA4

```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.1"
$env:ROCM_PATH = $env:HIP_PATH
$env:CMAKE_PREFIX_PATH = "$env:HIP_PATH\lib\cmake"
$env:PATH = "$env:HIP_PATH\bin;C:\Strawberry\perl\bin;C:\Strawberry\c\bin;$env:PATH"

cmake -S . -B build-rocm -G Ninja `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS=gfx1201 `
  -DGGML_HIP_MMQ_MFMA=ON `
  -DGGML_HIP_NO_VMM=ON `
  -DGGML_OPENMP=OFF `
  -DCMAKE_C_COMPILER="$env:HIP_PATH\bin\clang.exe" `
  -DCMAKE_CXX_COMPILER="$env:HIP_PATH\bin\clang++.exe" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm -j 4 --target llama-server
```

ROCm uses clang from the HIP SDK but still needs the Windows SDK and MSVC host
libraries. Missing `kernel32.lib`, `msvcrtd.lib`, or similar files indicates an
incomplete Build Tools environment. See the full [Build Guide](docs/build.md).

## Recommended Runtime Profiles

### Vulkan Dual GPU

Use GPU1 as the output device, but keep the measured MTP device order shown
below:

```powershell
$env:LLAMA_OUTPUT_DEVICE = "Vulkan1"
$env:GGML_VK_FORCE_AMD_LARGE_MATMUL = "1"

build-vulkan\bin\llama-server.exe `
  -m models\Qwen3.6-27B-Q3_K_S_mtp.gguf `
  -c 131072 -b 8192 -ub 1024 -ngl 999 `
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on `
  -dev Vulkan0,Vulkan1 -sm layer -ts 1,1 `
  --spec-type none
```

Equal split is the current conservative general default; use autotune for a
specific context and residency target. For MTP, replace the final line with:

```powershell
--spec-type draft-mtp --spec-draft-n-max 3
```

The server's built-in MTP prefill window is 256 tokens. Override it only for a
controlled comparison:

```powershell
$env:LLAMA_SPEC_PREFILL_WINDOW = "512"
```

### ROCm Dual GPU

The reference MTP device order is:

```text
-dev ROCm1,ROCm0 -sm layer -ts 1,1
```

Direct HIP peer copy remains disabled by default on Windows/RDNA4. The safe
host-staged split route is used instead. Do not enable
`GGML_ROCM_ENABLE_PEER_COPY=1` as a production default without a fresh
correctness and driver-stability validation.

### Why GPU Order Matters

`-sm layer` is pipeline/layer placement, not symmetric tensor parallelism. The
first and second entries do not receive identical work: token embeddings,
repeating-layer ranges, recurrent state, output tensors, MTP NextN staging, and
scheduler copy boundaries are placed according to graph ownership and device
order. `LLAMA_OUTPUT_DEVICE` changes output placement but does not make the
rest of that topology symmetric.

Consequently, swapping two identical GPUs can change both transfer direction
and which device owns a synchronization-heavy graph boundary. On this machine,
`Vulkan0,Vulkan1` with `LLAMA_OUTPUT_DEVICE=Vulkan1` preserves MTP prefill
throughput; the reversed order was nearly equivalent for `spec=none` but much
slower during MTP prefill. ROCm's measured general order remains
`ROCm1,ROCm0`. A mature tensor-parallel implementation would reduce this
asymmetry, but the current supported production mode is layer split.

## MTP Behavior

MTP accelerates token generation; it does not make the target model's prompt
prefill free. This fork limits draft-context prefill to the recent prompt tail
and keeps NextN hidden states on their backend, avoiding the previous
GPU-to-RAM-to-GPU round trip.

Practical rules:

- Use at least 128 output tokens when benchmarking MTP. Very short runs are
  dominated by the first target-verification graph.
- Compare MTP and `none` with the same model, prompt, output length, KV type,
  batch/ubatch, device split, and background load.
- Depth 3 is the current robust starting point. Higher depth is not
  automatically faster because acceptance falls and verification batches grow.
- For prompt-dominated requests with short answers, `none` can still win wall
  time even when MTP decode is much faster.
- Set `LLAMA_MTP_DEVICE_HANDOFF=0` only as a diagnostic rollback to the old host
  hidden-state path.

## Vision

Qwen3.6 vision requires a projector that matches the text model architecture
and embedding dimension. In the GUI, enable Vision and select
`models/mmproj-F16.gguf`. The equivalent server argument is:

```text
--mmproj models/mmproj-F16.gguf
```

Use `Spec: None` for the first image request so vision-pipeline issues can be
separated from speculative decoding.

## Benchmarking

The canonical runner starts an isolated OpenAI-compatible server, injects a
real repository snapshot, records prompt/decode timings, and updates the live
history files:

```powershell
python scripts/agent_workload_bench.py --help
```

Important history files:

- `build_logs/agent-workload/BENCH_RUNS.csv`
- `build_logs/agent-workload/BENCH_RECENT.md`
- `build_logs/agent-workload/BENCH_LANES.md`
- `docs/research/RESULTS_LOG.md`

Performance work should use neighboring controls. Background GPU applications,
driver power state, warm shader caches, prompt-cache reuse, or a different
output length can otherwise create a false improvement.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `gui/` | PyQt6 desktop application |
| `src/`, `common/`, `include/` | llama runtime and speculative pipeline |
| `ggml/src/ggml-vulkan/` | Vulkan backend and generated shaders |
| `ggml/src/ggml-hip/` | ROCm/HIP build integration |
| `ggml/src/ggml-cuda/` | Shared HIP-compatible kernel implementation |
| `ggml/src/ggml-cpu/` | CPU backend |
| `scripts/agent_workload_bench.py` | Benchmark and autotune runner |
| `docs/research/` | Accepted, rejected, and diagnostic performance work |
| `docs/vulkan/` | Vulkan architecture and validation rules |

## Development

Read [AGENTS.md](AGENTS.md) before changing the fork. Upstream changes are
ported selectively according to [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md); removed
backends are not restored automatically during synchronization.

When reporting performance, include the model, backend, device order, split,
context, actual prompt tokens, output tokens, batch/ubatch, KV types, speculative
mode, cache policy, and background load. A faster isolated number is useful only
when its lane and tradeoffs are visible.
