# E121 Real-Context 256-Token Backend Boundary

## Metadata

- Experiment ID: E121
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ 8e08fb608
- Hypothesis ID: H34
- Target lane: Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4 KV, real repo-snapshot context, `max_tokens=256`, thinking on, reuse enabled

## Hypothesis

- Statement: Vulkan q4 may already beat ROCm q4 at 256 generated tokens, not only at 512.
- Mechanism: ROCm still wins prompt eval, but Vulkan decode is much faster. The break-even point depends on output length and prompt-cache/checkpoint reuse.
- Why now: E120 established Vulkan q4 as the 512-token long-answer route. E121 finds whether 256-token answers should use the same route.

## Benchmark Plan

- ROCm q4, `spec=none`, r3.
- Vulkan q4, `spec=none`, r3.
- Same real-context prompt injection and reuse behavior as E119/E120.

## Result

- Outcome: Vulkan q4 already wins at `max_tokens=256`.
- ROCm q4 r3:
  - aggregate `22.3563 TPS`
  - warm-only `23.88 TPS`
  - prompt eval `1167.3883 tok/s`
  - decode eval `28.7067 tok/s`
- Vulkan q4 r3:
  - aggregate `26.6050 TPS`
  - warm-only `29.43 TPS`
  - prompt eval `905.5500 tok/s`
  - decode eval `40.0600 tok/s`
- Delta:
  - aggregate `+19.00%`
  - warm-only `+23.24%`

## Interpretation

- The practical routing boundary is below 256 generated tokens for this repeated/session real-context workload.
- Vulkan q4 should be preferred for medium/long answers where decode dominates enough to repay slower prompt eval.
- ROCm q4 remains the prompt-heavy/cold-first default, especially when generated output is short or when benchmark claims need no-reuse controls.

## Artifacts

- `build_logs/agent-workload/e121-realctx256-rocm-q4-specnone-r3.csv`
- `build_logs/agent-workload/e121-realctx256-rocm-q4-specnone-r3.diagnostics.md`
- `build_logs/agent-workload/e121-realctx256-vulkan-q4-specnone-r3.csv`
- `build_logs/agent-workload/e121-realctx256-vulkan-q4-specnone-r3.diagnostics.md`
