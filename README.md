# llama.cpp-rdna-lab

`llama.cpp-rdna-lab` is a hardware-focused fork of
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) for local AI on
Windows with two AMD RDNA4 GPUs. It combines a maintained subset of the
`llama.cpp` runtime with a PyQt6 desktop application, reproducible
benchmark/autotune tooling, long-context work, and AMD-specific Vulkan and
ROCm/HIP optimizations.

The desktop application is branded **RDNA LLM Studio**. The repository keeps
upstream `llama-*` executable names and source structure so that runtime
compatibility and upstream synchronization remain straightforward.

The primary workload is agentic coding with Qwen3.8-27B: large cold prompts,
single-user requests, long contexts, tool use, vision, and speculative decode.
The main performance priority is prompt evaluation. MTP is kept only when its
decode gain does not impose an unacceptable prefill cost.

> This is a specialized research and production fork, not a drop-in replacement
> for every upstream platform. Results and defaults are tuned for the reference
> dual-RX 9070 XT machine described below.

## Documents

- [README](README.md) — this file: overview, quick start, and build guide
- [Contributing](CONTRIBUTING.md) — contribution rules and workflow
- [License & Security](LICENSE) — MIT license, with private vulnerability
  reporting and secure-use guidance in [SECURITY.md](SECURITY.md)
- [Fork Details](FORK_DETAILS.md) — fork-only features, backend fixes, and
  recommended runtime profiles
- [Performance](PERFORMANCE.md) — current benchmark tables and matched lane
  contracts
- [Benchmarking](BENCHMARKS.md) — canonical benchmark methodology and history
- [Supported backends](docs/SUPPORTED_BACKENDS.md) — backend policy for this
  fork

## At a Glance

| Area | Current focus |
| --- | --- |
| Host platform | Windows 11 on AMD AM4 |
| Accelerators | 2x Radeon RX 9070 XT 16 GB (`gfx1201`) |
| Backends | ROCm/HIP, Vulkan, and CPU |
| Primary model | Qwen3.8-27B Q4_K_M with MTP |
| Experimental model | Ternary Bonsai 27B `PQ2_0` on CPU and ROCm |
| Main objective | Maximum cold prompt evaluation without sacrificing useful decode speed |
| Serving | OpenAI-compatible `llama-server` plus a PyQt6 desktop GUI |

The fork is currently substantially faster than the measured stock upstream
checkout on the same long-prompt contract. See
[Fork vs Stock Upstream](PERFORMANCE.md#fork-vs-stock-upstream) for the exact
matched runs.

## Active Branches

Production and daily work stay on `master`; larger experiments run on named
branches and are merged back selectively. Snapshot: 2026-08-26.

| Branch | Status | Latest state |
| --- | --- | --- |
| `master` | Stable baseline | D131 R9 MTP window audit PASS, multi-stream scale-view fix (`276121b7e`). |
| `rpc-vulkan` | **Active** (current work) | Recovers the RPC backend (removed upstream) for Vulkan offload, 11 commits ahead of `master`. `eb0d3c5ec` fixed the quantized `alloc_size` OOB crash (q3_K/q6_K). The 3-GPU 12K RPC lane reached **1314 ptps / 23.6 t/s** on 2026-08-23 (target ≥1277 ptps; PPL 4.0148 ≈ baseline), up from 839 ptps after alloc-size caching, a 16 MB send buffer, and `-ts 0.9,0.6,1.5`. Latest: async outbound queue plus split-timing diagnosis (94K lane 1067 t/s). See [RPC resume playbook](docs/research/rpc-vulkan/RPC_PREFILL_RESUME_PLAYBOOK.md). |
| `dflash2` | Paused, to be resumed | DFlash2 block-diffusion drafter port for Qwen3.8-27B (local 2-tap depthwise conv + candidate selector, `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` in `models/`), 5 commits ahead of `master`. Paused at a measured ROCm checkpoint (2026-08-21): opt-in multi-scheduler reuse gives 29.59 aggregate / 36.90 decode / 785.58 prompt tok/s on 498+128 vs 29.00/36.33/749.70 for the single-cache control, with the strict n=3 boundary/parity gate bit-exact. Open items include the upstream batched-greedy divergence and the Vulkan long-decode crash. Resume: [ROCm playbook](docs/research/dflash/ROCM_DFLASH2_RESUME_PLAYBOOK.md), [Vulkan playbook](docs/research/dflash/VULKAN_DFLASH2_RESUME_PLAYBOOK.md). |
| `research/vulkan-decode` | Backlog, to be resumed | D104 (Q6_K prefill dispatch) and D105 (decode bandwidth) closed; Q4_K16 port log reached Vulkan PPL parity with bf16 (6.6124 vs 6.6202 on 256 chunks); MTP n=3 measured decode optimum 1.83x. 15 commits ahead, 10 behind `master` (rebase needed before continuation). |
| `research/vulkan-fp8-kv` | Backlog, to be resumed | D131 R9: fp8 K per-block scale (`LLAMA_VK_F8_K_SCALE`), K-scale broadcast fix, MTP window audit PASS, multi-stream scale-view fix. C2 closed: f8_direct prefill lost 23.5% to preconvert. 20 commits ahead, 10 behind `master`. |

## Contents

- [Documents](#documents)
- [At a Glance](#at-a-glance)
- [Active Branches](#active-branches)
- [Project Goals](#project-goals)
- [Supported Backends and Models](#supported-backends-and-models)
- [Reference System](#reference-system)
- [Performance Summary](#performance-summary)
- [Fork Highlights](#fork-highlights)
- [Quick Start](#quick-start)
- [Build Requirements](#build-requirements)
- [MTP Behavior](#mtp-behavior)
- [Vision](#vision)
- [Benchmarking](#benchmarking)
- [Repository Layout](#repository-layout)
- [Development](#development)
- [License and Security](#license-and-security)

## Project Goals

- Maximize Qwen3.8 prompt-evaluation throughput for agent workloads.
- Use both GPUs without moving the active working set into system RAM.
- Make MTP improve decode while keeping long-prompt prefill close to baseline.
- Provide a practical GUI for building, launching, monitoring, and autotuning.
- Keep performance claims reproducible through cold, lane-locked benchmarks.
- Keep the fork maintainable by carrying only the backends and upstream changes
  that are useful on this machine.

Current non-goals include broad accelerator portability and native support for
NVIDIA CUDA, Metal, SYCL, OpenCL, CANN, or other removed upstream backends.

## Supported Backends and Models

| Backend | Role | Status |
| --- | --- | --- |
| ROCm/HIP | Primary prompt-eval, long-context MTP, and RDNA4 runtime | Supported and preferred for prompt-heavy MTP work |
| Vulkan | General AMD runtime and backend comparison | Supported; competitive for decode-heavy work; q8/MTP path fixed (D094) |
| CPU | Fallback, conversion, sanity checks, and tests | Supported |

ROCm still builds HIP-compatible kernels from `ggml/src/ggml-cuda`. That is an
internal HIP implementation detail and does not mean that this fork supports
NVIDIA hardware. See [Supported Backends](docs/SUPPORTED_BACKENDS.md).

### Model and Format Matrix

| Model / feature | CPU | ROCm/HIP | Vulkan | Notes |
| --- | --- | --- | --- | --- |
| Qwen3.8 GGUF (Q4_K_M primary; Qwen3.6 family also supported) | Yes | Yes | Yes | Primary supported family |
| Qwen3.8 NextN MTP | Yes | Yes | Yes | Requires an MTP-enabled GGUF |
| Ternary Bonsai 27B `PQ2_0` | Yes | Yes | Not yet | Native loader, CPU kernels, and HIP MMQ/MMVQ path |
| Qwen3.5/3.6/3.8 vision projector | Yes | Yes | Yes | Use a matching `mmproj-*.gguf` |
| DFlash | Research | Research | Research | Not a recommended production profile |

D094 (2026-08-07, tested on `Qwen3.6-27B-Q4_K_M.gguf` and
`Qwen3.6-27B-Q3_K_S_mtp.gguf`, 2x RX 9070 XT): the Vulkan q8_0 vec/mmq
numerical divergence vs ROCm was root-caused and fixed (CUDA-style dp4a
accumulation, round-half-away q8_1 quantize, mmq variant-B math). MTP
acceptance recovered from 0.33 to 0.80+ (52k-token drafts; target 0.53) and
the f16-KV 49K lane now beats ROCm
(prompt 1719.92 vs 1679.20 tok/s, decode 43.10 vs 32.33, aggregate 6.1358 vs
5.7655 TPS). See [BENCHMARKS.md](BENCHMARKS.md) and
[Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md).

Qwen3.8-27B-Q4_K_M is the primary practical Qwen model on this 2x16 GB
machine (rebased 2026-08-14; it shares the qwen35 architecture family with
Qwen3.6 and runs the same MTP/vision paths). The one-copy ROCm scheduler and
bounded Q8 Flash Attention route make its measured 49K and 98K lanes viable.
Q3_K_S (Qwen3.6) remains the secondary choice for maximum
context/VRAM headroom, vision, and Q3-specific kernel research. `PQ2_0` is an
experimental Prism format and should not be confused with conventional `Q2_0`
quantization.

## Reference System

- Windows 11
- AMD Ryzen 7 5800X3D, 8 cores / 16 threads
- 64 GB system RAM
- 2x AMD Radeon RX 9070 XT, 16 GB VRAM each, RDNA4 `gfx1201`
- AMD ROCm/HIP SDK 7.1 for Windows
- AMD proprietary Vulkan driver
- Main model: `Qwen3.8-27B-Q4_K_M.gguf`
- Vision projector: `mmproj-F16.gguf`

The two GPUs are normally used with layer split, not tensor split. GPU1 is the
preferred output device because GPU0 also drives the desktop. Device order is
backend- and workload-sensitive even with two identical cards, and the best
Vulkan order is not identical for every lane. Exact routes are recorded with
each benchmark instead of being presented as a universal default. PCIe
topology, driver version, background GPU load, KV type, and model residency can
materially change the numbers below.

## Performance Summary

Full tables, lane contracts, and evidence links live in
[PERFORMANCE.md](PERFORMANCE.md). Headline snapshot (2026-08-14,
Qwen3.8-27B-Q4_K_M rebaseline; FlashAttention, cold prompts, no reuse):

| Backend | Lane | Mode | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Vulkan | 49,152 | q8_0 none | 1532.79 | 26.05 | 5.1016 | - |
| Vulkan | 49,152 | q8_0 MTP n2 | 1637.44 | **48.56** | 5.9465 | 81.8% |
| ROCm | 49,152 | **f8_e4m3 native** none | **1713.67** | 22.20 | **8.6528** | - |
| ROCm | 49,152 | f8_e4m3 native MTP n2 | **1716.79** | 39.96 | 6.0332 | **78.2%** |

On Qwen3.8 the Vulkan FP8 prompt advantage over q8 is roughly parity at
12K/49K/98K; ROCm native FP8 still holds `+4.0%` prompt, `+3.1%` decode and
`+3.7%` aggregate at 49K. The 98K Vulkan last-12-f16 MTP profile is
context-research material (60.8% acceptance vs q8's 71.5%), not a default
recommendation.

## Fork Highlights

The fork's main differences from stock, beyond benchmark tooling and the GUI:

- **Dual-GPU control** — explicit device order, layer split, and
  `LLAMA_OUTPUT_DEVICE` placement instead of relying on automatic defaults.
- **Qwen MTP on both backends** — backend-resident NextN handoff, ROCm KV-only
  sparse history, warm Vulkan verification topology.
- **AMD kernel work** — RDNA4 Q3_K/PQ2_0 HIP kernels, rocWMMA FlashAttention,
  native E4M3 FP8 KV routes, Vulkan q8/mmq correctness fixes.
- **Reproducible benchmarking** — `scripts/agent_workload_bench.py` with
  canonical history files and lane contracts.

See [Fork Details](FORK_DETAILS.md) for the complete feature list, diagnostics
and rollback controls.

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
7. In **Benchmark / Autotune**, use the recommended explicit device order for
   reproducible dual-GPU tests. `Auto` remains useful for discovery, but it is
   not a stable benchmark contract.
8. Validate batch, ubatch, KV, split, and spec settings at the intended context
   length. Short-prompt winners do not automatically remain best at 49K.

Model files are not part of the source tree history. Put local GGUF files in
`models/` or select them from another local directory.

## Build Requirements

The reference builds are Windows x64 builds. A clean machine needs:

- Git and 64-bit Python 3.11 or newer with `pip`;
- CMake 3.14 or newer and Ninja (tested with CMake 3.29 and Ninja 1.12);
- Visual Studio Build Tools 2022 with **Desktop development with C++**, the
  MSVC v143 toolset, and a Windows 10 or 11 SDK;
- the current AMD display driver, including the Vulkan runtime;
- full LunarG Vulkan SDK with `glslc`, `spirv-as`, `spirv-dis`, and
  `spirv-val` for Vulkan/FP8 shader builds;
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
spirv-as --version
spirv-dis --version
spirv-val --version
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
  -DGGML_HIP_ROCWMMA_FATTN=ON `
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
The fork includes rocWMMA 7.1 headers under `third_party/rocwmma`; no separate
rocWMMA SDK install is required for the command above.

## MTP Behavior

MTP accelerates token generation; it does not make the target model's prompt
prefill free. ROCm uses selected long-range KV blocks plus the recent prompt
tail and keeps NextN hidden states on their backend, avoiding a complete draft
prefill and the previous GPU-to-RAM-to-GPU round trip. Vulkan uses a host
handoff by default because keeping unmasked NextN output resident over the
whole Vulkan prompt was substantially slower.

Practical rules:

- Use at least 128 output tokens when benchmarking MTP. Very short runs are
  dominated by the first target-verification graph.
- Compare MTP and `none` with the same model, prompt, output length, KV type,
  batch/ubatch, device split, and background load.
- Depth 3 is the current robust starting point. Higher depth is not
  automatically faster because acceptance falls and verification batches grow.
- For prompt-dominated requests with short answers, `none` can still win wall
  time even when MTP decode is much faster.
- Non-zero Windows Shared memory is not by itself proof that MTP is reading KV
  from RAM. Check for a throughput cliff and compare process Dedicated/Shared;
  the current 72K lane adds only about 62 MiB Shared during MTP prefill.
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
output length can otherwise create a false improvement. Record an explicit
`-dev` route for every dual-GPU result; the GUI now defaults new ROCm and Vulkan
benchmark configurations to the measured recommended order instead of `Auto`.

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
| `PERFORMANCE.md` | Current benchmark tables and lane contracts |
| `FORK_DETAILS.md` | Fork-only features, fixes, and runtime profiles |
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

## License and Security

The runtime is derived from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
Upstream changes are reviewed and ported selectively so they do not silently
restore removed backends or invalidate AMD-specific behavior. This repository
is distributed under the [MIT License](LICENSE); bundled third-party components
retain their own notices and licenses.

Security issues are handled privately. See [SECURITY.md](SECURITY.md) for the
reporting policy, covered topics (runtime, ggml, and GGUF tooling), and
secure-use guidance for untrusted models, inputs, and networks. Do not report
vulnerabilities as public issues before the disclosure window closes.
