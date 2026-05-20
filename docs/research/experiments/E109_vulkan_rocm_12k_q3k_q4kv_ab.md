# E109 Vulkan vs ROCm 12k Q3_K q4-KV A/B

## Metadata

- Experiment ID: E109
- Date: 2026-05-20
- Owner: Codex
- Branch/Commit: master @ 533bb5ed2 plus local experiment docs
- Target lane: Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: current Vulkan may be a practical route candidate for the 12k q4-KV lane after the E102 AMD large-matmul default, even if ROCm remains the preferred backend generally.
- Mechanism: Vulkan's Q3_K coopmat route can avoid some ROCm Q3_K staging costs, but may lose elsewhere in prompt or decode. A fresh same-lane A/B is needed because prior Vulkan results targeted different context/shape states.
- Why now: E106/E107/E108 did not produce a ROCm cold-first gain. The user's map request explicitly includes Vulkan as a fallback route, so route-level TPS should be measured before deeper ROCm-only work.

## Math / Theory

- Assumptions: use existing `build-vulkan/bin/llama-server.exe`; no extra Vulkan env unless needed for a negative-control rerun.
- Expected speedup corridor: `+1%` to `+5%` if Vulkan's current Q3_K path offsets any decode/prefill losses; otherwise ROCm stays default.
- Failure conditions: prompt eval below ROCm enough to offset any decode gain, or server route/feature mismatch.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: compare against E106/E108 ROCm controls; do not change GUI default unless r3 confirms a route win.
3. Rollback path: no code changes.

## Benchmark Plan

- Baseline command: ROCm E106/E108 q4-KV no-spec controls.
- Candidate command: same lane with `build-vulkan/bin/llama-server.exe`.
- Number of runs: one-run gate; r3 only if Vulkan beats ROCm by more than noise.
- Artifacts path:
  - `build_logs/agent-workload/e109-vulkan12k-q3k-q4kv-r1.*`

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s
- decode eval tok/s
- errors / route support problems

## Result

- Outcome: reject Vulkan as a route win for this exact 12k q4-KV Q3_K_S lane.
- Delta: ROCm controls are around `11.76-11.85 TPS`; Vulkan candidate timed out before first task completion and recorded `0.0000 TPS`.
- Confidence: high for "not a practical replacement route" under this benchmark contract; low for deeper Vulkan root-cause ranking because the run did not reach prompt timing.
- Recommendation: keep ROCm default for the active q4-KV lane. Use Vulkan route map for fallback/debug and for its own validated Vulkan-specific lanes, not as a replacement for this ROCm profile.

## Notes

- Server setup succeeded: Vulkan full offload fit with `65/65` layers, model buffer about `11434 MiB`, KV `216 MiB`, compute buffer `728 MiB`, graph nodes `3849`, splits `2`.
- The first prompt did not finish before the benchmark timeout, so the failure is practical throughput/latency rather than a simple load/offload error.
- Why the hypothesis missed: the current Vulkan Q3_K route improvements do not overcome ROCm's prompt-heavy advantage at `ctx=12288`, `b=6144`, `ub=2048`, q4 KV.
- Follow-up action: if Vulkan is revisited for this lane, start with a prompt-only pp7488 gate and timeout budget sized to observe prompt eval, not a full agent workload first.
