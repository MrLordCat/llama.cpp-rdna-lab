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
preferred primary/output device because GPU0 also drives the desktop. Exact
PCIe topology, driver version, background GPU load, KV type, and model residency
can materially change the numbers below.

## Current Performance

Snapshot date: **2026-07-14**.

All headline rows use the Qwen3.6-27B Q3_K_S MTP-enabled GGUF, FlashAttention,
one server slot, cold prompt processing, no prompt-cache reuse, and no prime
pass. Prompt and decode TPS come from server timings; aggregate TPS includes the
whole request wall time. Compare `none` and `MTP` only inside the same backend
and lane because prompt/output lengths differ between Vulkan and ROCm.

### Short Prompt Lanes

| Backend | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | `none` | 7,842 / 128 | **1770.64** | 38.03 | **16.34** | `ctx=12288`, `b8192/ub1024`, q8/q8 KV |
| Vulkan | MTP n3 | 7,842 / 128 | 1647.97 | **41.64** | 16.23 | GPU-resident NextN path; explicit 512-token prefill window |
| ROCm | `none`, r3 mean | 7,188 / 256 | **1694.82** | 25.02 | 17.61 | `ctx=12288`, `b8192/ub1024`, q8/q8 KV |
| ROCm | MTP n3, r3 mean | 7,188 / 256 | 1547.21 | **41.25** | **23.45** | RDNA4 Q3_K small-N DP4A route |

ROCm MTP reaches about `1.65x` its same-build decode baseline in this lane.
Vulkan remains the stronger prompt-eval route. A generation-only Vulkan smoke
with no injected repository prompt has reached `70.76` decode tok/s, but it is
not used as a prompt-heavy headline.

### Long Prompt Lanes

| Backend | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | `none`, two-run mean | 29,540 / 128 | **1420.11** | 29.04 | 5.06 | `ctx=49152`, `b8192/ub1024`, q8/q8 KV |
| Vulkan | MTP n3, two-run mean | 29,540 / 128 | 1404.66 | **38.28** | **5.23** | Built-in 256-token prefill window |
| ROCm | `none` | 56,305 / 128 | **1088.67** | 19.02 | **2.1859** | `ctx=131072`, `b8192/ub1024`, q8/q8 KV |
| ROCm | MTP n3 | 56,305 / 128 | 1045.62 | **26.85** | 2.1799 | 68.55% acceptance |

The Vulkan long-prompt pair was bracketed and measured while League of Legends
was active. Against the mean of the two neighboring controls, the final
GPU-resident MTP path changes prompt eval by `-1.09%`, decode by `+31.83%`, and
aggregate throughput by `+3.36%`. Both MTP repeats produced the same `76/151`
accepted/generated count.

The ROCm 56k lane shows the long-context tradeoff clearly: MTP improves decode
by `1.41x`, but a roughly 4% prompt tax cancels the wall-time benefit for a
128-token response. Longer generated answers benefit more; prompt-dominated
requests should still compare against `spec=none`.

For a pure Vulkan Q3 prompt-eval target at 56,456 prompt tokens, the current
balanced layer-split reference is `1327.82` prompt tok/s as an r3 mean, with a
best cold run of `1350.01`. The active research target remains 2000 prompt
tok/s. Q4_K_M and UD-Q4_K_XL are supported, but the 27B Q4 long-context working
set currently enters WDDM shared memory on this 2x16 GB system; Q3_K_S remains
the practical performance model.

Evidence:

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

- Python 3 and the packages in `gui/requirements-gui.txt`
- CMake and Ninja
- MSVC Build Tools with **Desktop development with C++** and a Windows SDK
- Vulkan SDK for Vulkan builds
- AMD ROCm/HIP SDK 7.1 for ROCm builds
- OpenSSL development files for HTTPS-enabled builds

The GUI can configure the local Windows toolchains automatically. Manual builds
are shown below.

### CPU

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4 --target llama-server
```

### Vulkan

```powershell
cmake -S . -B build-vulkan -G Ninja `
  -DGGML_VULKAN=ON `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 4 --target llama-server
```

### ROCm/HIP on Windows RDNA4

```powershell
cmake -S . -B build-rocm -G Ninja `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS=gfx1201 `
  -DGGML_HIP_MMQ_MFMA=ON `
  -DGGML_HIP_NO_VMM=ON `
  -DGGML_OPENMP=OFF `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm -j 4 --target llama-server
```

ROCm uses clang from the HIP SDK but still needs the Windows SDK and MSVC host
libraries. Missing `kernel32.lib`, `msvcrtd.lib`, or similar files indicates an
incomplete Build Tools environment. See the full [Build Guide](docs/build.md).

## Recommended Runtime Profiles

### Vulkan Dual GPU

Use GPU1 as the primary/output device on the reference desktop:

```powershell
$env:LLAMA_OUTPUT_DEVICE = "Vulkan1"
$env:GGML_VK_FORCE_AMD_LARGE_MATMUL = "1"

build-vulkan\bin\llama-server.exe `
  -m models\Qwen3.6-27B-Q3_K_S_mtp.gguf `
  -c 131072 -b 8192 -ub 1024 -ngl 999 `
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on `
  -dev Vulkan1,Vulkan0 -sm layer -ts 1,1 `
  --spec-type none
```

Use current autotune evidence for the final layer ratio. `-ts 5,6` is the
measured 56k prompt-eval profile, while equal split is the conservative general
default. For MTP, replace the final line with:

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
