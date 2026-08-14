# Agent Instructions

## Project identity

This is `llama.cpp-rdna-lab`, a local fork for Windows and two AMD Radeon RX
9070 XT GPUs. It combines a PyQt6 GUI, long-context benchmark/autotune tooling,
MTP/DFlash and local ggml performance work. Do not treat it as a clean upstream
checkout.

The canonical repository root on this machine is:

```text
D:\GitHub\llama.cpp-with-GUI
```

Do not use or infer the retired `C:\Users\Chris\Documents\GitHub\llama.cpp-with-GUI`
path. Models, builds, logs, scripts and GUI launches are all resolved from the
`D:` repository unless a command explicitly names another checkout.

## Supported backends

Only these backends are supported:

- CPU, including optional BLAS and CPU SIMD paths;
- Vulkan;
- ROCm/HIP.

Do not restore CUDA, Metal, SYCL, OpenCL, CANN, MUSA, WebGPU, RPC or other
removed backends during upstream work. `ggml/src/ggml-cuda` is retained only as
the CUDA-compatible source layer compiled by `ggml-hip`; its native CUDA CMake
entry point is intentionally removed. Read `docs/SUPPORTED_BACKENDS.md`.

## Local hardware

- Windows 11, AMD Ryzen 7 5800X3D, 64 GB RAM.
- Two AMD Radeon RX 9070 XT 16 GB, target `gfx1201`.
- ROCm/HIP SDK 7.1.
- Vulkan normally uses `Vulkan1,Vulkan0`, with GPU1 first because GPU0 handles
  display/system load.
- ROCm production and long-context benchmarks use both GPUs as
  `-dev ROCm1,ROCm0 -sm layer`; single-GPU runs are valid only for isolated
  kernel diagnosis because the long-prompt lane can spill into shared RAM.
- Do not set `LLAMA_OUTPUT_DEVICE=ROCm1` for that ROCm order. The default output
  placement on the last device (`ROCm0`) keeps the layer pipeline monotonic;
  forcing output back to `ROCm1` adds a second cross-device boundary and can
  nearly halve long-prompt evaluation throughput.
- ROCm on Windows must use Ninja and ROCm clang, not the Visual Studio
  generator.

## Driver safety

This machine has experienced driver drops during discovery and process teardown.
The following rules are mandatory:

- call `hipMemGetInfo` only while no GPU server, model load, benchmark or other
  HIP/Vulkan discovery is active; an idle direct probe passed on both GPUs
  after the clean Windows installation on 2026-07-14, but concurrent use
  remains unvalidated;
- never run `bash scripts/stage-vulkan-dlls.sh`;
- never use `llama-server --version` or `llama-server --help` as a build probe;
- stop `llama-server` gracefully and wait for exit before considering a forceful
  fallback;
- do not hard-kill a GPU server during load, prompt evaluation or decode;
- do not run hardware discovery commands while a benchmark is active;
- do not start a benchmark when another `llama-server` is still listening.

Use process existence, file timestamps, CMake targets and HTTP readiness for
validation without launching extra backend discovery paths.

## Worktree and editing

- The worktree may contain changes from another agent. Never revert changes you
  did not make.
- Check `git status --short --branch` before editing.
- Use `apply_patch` for manual edits.
- Never use `git reset --hard`, destructive checkout, or broad cleanup commands.
- Keep local GUI, ROCm, Vulkan, benchmark and research changes during upstream
  sync.
- Avoid `cmd.exe` wrappers for long builds and benchmarks; use PowerShell or
  direct executables.

Protected local paths:

```text
.github/**
docs/**
README.md
AGENTS.md
AGENT_WORKFLOW.md
BENCHMARKS.md
MTP.md
PROJECT_PROFILE.md
QWEN_SPEED_RESEARCH.md
UPSTREAM_SYNC.md
gui/**
scripts/agent_workload_bench.py
```

## Instruction precedence and multi-agent work

- This file contains global mandatory rules. Task-specific instructions may
  tighten them, but never relax backend, driver-safety, worktree, or validation
  requirements.
- For subagents, parallel research, independent review, repository cleanup, or
  BYOK model routing, read and follow `AGENT_WORKFLOW.md`.
- The coordinating agent owns the plan, shared integration files, final
  validation, and user-facing answer. Subagents receive bounded scopes and
  explicit file ownership.
- Read-only investigation may run in parallel. GPU discovery, model servers,
  and benchmarks always have exactly one sequential owner.
- Shared agent definitions remain model-neutral. At dispatch time the
  coordinator must actively choose and pass an explicit BYOK model for each
  subagent; never rely on `auto`, and never store provider credentials or
  endpoints here.

## Performance policy

- The primary production and performance model is Qwen3.8-27B Q4_K_M
  (rebased from Qwen3.6 2026-08-14; same qwen35 architecture family). The
  safe baseline is the dual-ROCm 49K lane from D089; Q3_K_S remains a
  secondary headroom and model-specific kernel-research lane.
- Preserve model-scoped targets. The Q3 `2000 prompt tok/s` objective remains
  historical/open for that lane and is not a Q4 baseline claim.
- MTP performance is measured against an adjacent `spec=none` decode baseline.
- MTP is expected to accelerate decode, not prompt evaluation.
- Compare only equal backend, model, context, batch/ubatch, KV, split, cache
  policy, prompt scale and background load.
- When a game or other GPU workload is active, record it and use an adjacent
  baseline under the same load.
- Use short runs for diagnosis and a long run only for final confirmation.
- Keep cold-first and steady-session results separate.
- Use `build_logs/agent-workload/BENCH_RUNS.csv`, `BENCH_RECENT.md` and
  `BENCH_LANES.md` as canonical generated history.
- Keep only benchmark records and artifacts dated `2026-07-01` or newer. Raw
  server logs, traces, screenshots and superseded aggregate files stay ignored.
- Record accepted/rejected changes in `docs/research/` and do not leave a
  rejected runtime experiment enabled by default.

Before new performance work, read:

- `BENCHMARKS.md`;
- `docs/research/major-topology/README.md`;
- `docs/research/decode-hotspots/C01_RESUME_PLAYBOOK.md` when present;
- the latest relevant D### design note.

## MTP status

MTP is implemented through the upstream NextN extraction pipeline. Vulkan has
local warm-cache and safe small-row dispatch optimizations. Verified dual-GPU
Vulkan lanes currently show about `1.29-1.42x` decode speedup depending on lane,
acceptance and load. Do not document `1.6-2x` as achieved until measured.

For vision requests, validate with `Spec: None` first. MTP-enabled GGUF is
required for `--spec-type draft-mtp`.

## Minimum validation

Python and GUI changes:

```powershell
python -m compileall -q gui scripts run.py
```

CPU/build-system changes:

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4 --target llama-server
```

Vulkan changes:

```powershell
cmake -S . -B build-vulkan -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 4 --target llama-server
```

ROCm configure:

```powershell
cmake -S . -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
```

Always finish with `git diff --check`. Do not run GPU server probes merely to
print version/help output.

## Upstream sync

Follow `UPSTREAM_SYNC.md`. Inspect upstream commits, then manually port the
smallest useful core/runtime portion. Reject backend reintroduction. Shared
`ggml-cuda` kernel changes may be imported only when needed by HIP and verified
with a ROCm build.

## Pause/resume

Before pausing a performance branch, update the active resume playbook with the
lane contract, current baseline, open hypothesis, artifacts and next command.
On resume, repeat the adjacent baseline before drawing a new speed conclusion.
