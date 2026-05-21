# E119 Real-Context 512-Token Vulkan Decode Session Route

## Metadata

- Experiment ID: E119
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ f045a2fbf
- Driver: AMD `32.0.31007.5012`
- Target lane: Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, real repo-snapshot context, `max_tokens=512`, thinking on, reuse enabled

## Hypothesis

- Statement: E116's Vulkan f16 decode route should become a practical long-answer/session route when output length is large enough for decode to dominate.
- Mechanism: Vulkan has slower prompt eval than ROCm on this model, but much faster decode (`~39-41 tok/s` vs ROCm `~28-29 tok/s`). With real context plus 512 generated tokens, decode time dominates wall time.
- Why now: E118 live-server sanity accepted the Vulkan route as correct; E119 quantifies it on a more realistic long-answer workload.

## Benchmark Plan

- Compare:
  - ROCm q4 KV, `spec=none`
  - Vulkan f16 KV, `spec=none`
- Same workload:
  - `tasks=quick`, `triage_diff,review_bug`
  - `runs=3`
  - `real-context-mode=repo-snapshot`
  - prompt cache/checkpoints enabled
  - `max_tokens=512`
  - no v2 prime, thinking on

## Result

- Outcome: keep Vulkan f16 as the current real-context long-answer/decode-heavy session route.
- Delta:
  - ROCm q4: `24.9524 TPS` aggregate, `25.82 TPS` warm-only, decode eval `28.4483 tok/s`.
  - Vulkan f16: `32.0298 TPS` aggregate, `33.89 TPS` warm-only, decode eval `39.4483 tok/s`.
  - Aggregate speedup: `+28.36%`.
  - Warm-only speedup: `+31.25%`.
- Prompt tradeoff:
  - ROCm prompt eval mean: `1152.5583 tok/s`.
  - Vulkan prompt eval mean: `966.9283 tok/s`.
  - Vulkan loses prompt speed but wins overall once the answer is long enough.

## Interpretation

- This is a session/long-answer route, not a cold-first prompt-heavy default.
- The decision boundary is output length and prefix reuse:
  - short prompt-heavy/cold-first: keep ROCm q4;
  - long generation / repeated session: Vulkan f16 is now the better route.
- Correctness is covered by E118: live-server output was manually checked and did not show the old `wm32-wn32` corruption pattern.

## Artifacts

- `build_logs/agent-workload/e119-realctx512-rocm-q4-specnone-r3.csv`
- `build_logs/agent-workload/e119-realctx512-rocm-q4-specnone-r3.diagnostics.md`
- `build_logs/agent-workload/e119-realctx512-vulkan-f16-specnone-r3.csv`
- `build_logs/agent-workload/e119-realctx512-vulkan-f16-specnone-r3.diagnostics.md`
