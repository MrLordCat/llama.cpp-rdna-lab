# Agent Instructions

## Project identity

This is `llama.cpp-with-GUI`, a local fork for Windows and two AMD Radeon RX
9070 XT GPUs. It combines a PyQt6 GUI, long-context benchmark/autotune tooling,
MTP/DFlash and local ggml performance work. Do not treat it as a clean upstream
checkout.

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
- ROCm on Windows must use Ninja and ROCm clang, not the Visual Studio
  generator.

## Driver safety

This machine has experienced driver drops during discovery and process teardown.
The following rules are mandatory:

- never call `hipMemGetInfo` or add a probe that calls it;
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
BENCHMARKS.md
MTP.md
PROJECT_PROFILE.md
QWEN_SPEED_RESEARCH.md
UPSTREAM_SYNC.md
gui/**
scripts/agent_workload_bench.py
```

## Performance policy

- The active prompt target is Qwen3.6-27B Q3_K_S on realistic long prompts;
  `2000 prompt tok/s` is a research target, not a current claim.
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
