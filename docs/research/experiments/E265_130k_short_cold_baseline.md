# E265: 130k Short Cold Baseline

Date: 2026-05-26

## Goal

Create a correct dense 27B `ctx=131072` cold benchmark that still exercises real
repo-context load but completes in roughly 10-20 seconds. The lane is for quick
iteration and speed claims; heavier full-fill/residency runs are separate.

## Kept Contract

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- Context: `ctx=131072`
- Shape: `batch=512`, `ubatch=128`
- KV: `q4_0/q4_0`, FlashAttention on
- Workload: `quick:triage_diff`, `max_tokens=16`
- Real context: `--real-context-mode repo-snapshot --real-context-chars 24576`
- Cold policy: `--no-reuse --no-v2-prime-pass`, thinking enabled
- Runtime: `--spec-type none`; Vulkan also uses `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` and `--no-mmap`

The current snapshot injected `23531` chars from 2 files and produced `7904`
prompt tokens.

## Results

| Backend | Label | Wall | TPS | Prompt tok/s | Decode tok/s | Prompt tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Vulkan | `scout-vulkan130k-quick-c24k-b512-ub128-r1` | `9.93s` | `1.6115` | `830.98` | `42.53` | `7904` |
| ROCm | `scout-rocm130k-quick-c24k-b512-ub128-r1` | `11.44s` | `1.3984` | `725.21` | `31.44` | `7904` |

## Shape Evidence

- Vulkan `real-context-chars=32768`, `b1024/ub256`: `13.94s`, `1.1475 TPS`, prompt `835.52 tok/s`.
- Vulkan `real-context-chars=32768`, `b2048/ub512`: `70.92s`, `0.2256 TPS`, prompt `160.21 tok/s`.
- ROCm `real-context-chars=32768`, `b1024/ub256`: hard timeout at `90.01s` during prompt processing.
- ROCm `real-context-chars=16384`, `b512/ub128`: `7.43s`, `2.1546 TPS`, too short for the requested quick baseline window.

## Decision

Keep `b512/ub128`, `real-context-chars=24576`, and `max_tokens=16` as the primary
dense 27B 130k cold quick baseline. Do not promote larger `ubatch` shapes as
defaults until they beat this same lane on the same backend. Use full safe-fill
only for explicit heavy residency stress tests.

## Artifacts

- `build_logs/agent-workload/scout-vulkan130k-quick-c24k-b512-ub128-r1.diagnostics.md`
- `build_logs/agent-workload/scout-rocm130k-quick-c24k-b512-ub128-r1.diagnostics.md`
- `build_logs/agent-workload/scout-vulkan130k-quick-c32k-b1024-ub256-r2.diagnostics.md`
- `build_logs/agent-workload/scout-vulkan130k-quick-c32k-b2048-ub512-r1.diagnostics.md`
- `build_logs/agent-workload/scout-rocm130k-quick-c32k-b1024-ub256-r1.diagnostics.md`