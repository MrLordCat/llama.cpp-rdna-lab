# E122 Real-Context 128-Token Backend Boundary

## Metadata

- Experiment ID: E122
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ 9a7e3663a
- Hypothesis ID: H34
- Target lane: Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4 KV, real repo-snapshot context, `max_tokens=128`, thinking on, reuse enabled

## Hypothesis

- Statement: Vulkan q4 may still beat ROCm q4 at 128 generated tokens in a repeated/session real-context workload.
- Mechanism: prompt cache/checkpoints reduce repeated prefill enough that Vulkan's faster decode may win earlier than expected.
- Why now: E121 showed Vulkan q4 already wins at 256 tokens. E122 checks whether the boundary is near the existing 120-token session lane.

## Benchmark Plan

- ROCm q4, `spec=none`, r3.
- Vulkan q4, `spec=none`, r3.
- Same real-context/reuse setup as E121, with `max_tokens=128`.

## Result

- Outcome: Vulkan q4 still wins for repeated/session at `max_tokens=128`, but cold-first is effectively tied and ROCm slightly wins the first task.
- ROCm q4 r3:
  - aggregate `18.3480 TPS`
  - warm-only `20.55 TPS`
  - cold-only `15.11 TPS`
  - prompt eval `1171.5600 tok/s`
  - decode eval `28.8350 tok/s`
- Vulkan q4 r3:
  - aggregate `19.6365 TPS`
  - warm-only `23.15 TPS`
  - cold-only `15.06 TPS`
  - prompt eval `881.7917 tok/s`
  - decode eval `40.2400 tok/s`
- Delta:
  - aggregate `+7.02%`
  - warm-only `+12.65%`
  - cold-only `-0.33%`

## Interpretation

- Route boundary:
  - repeated/session at `128+` generated tokens: Vulkan q4 is better.
  - cold-first short prompt-heavy: ROCm q4 remains better/safer.
- This explains why E113/E114 short-session ROCm ngram work and E120/E121 Vulkan long-answer work are both valid, but for different usage shapes.

## Artifacts

- `build_logs/agent-workload/e122-realctx128-rocm-q4-specnone-r3.csv`
- `build_logs/agent-workload/e122-realctx128-rocm-q4-specnone-r3.diagnostics.md`
- `build_logs/agent-workload/e122-realctx128-vulkan-q4-specnone-r3.csv`
- `build_logs/agent-workload/e122-realctx128-vulkan-q4-specnone-r3.diagnostics.md`
