# E115 ROCm Driver 5012 Prefill Shape Sweep

## Metadata

- Experiment ID: E115
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ 054bccb00
- Driver: AMD `32.0.31007.5012`
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, KV `q4_0/q4_0`, `ctx=12288`, `triage_diff,review_bug`, cold-first, no reuse, `spec=none`, thinking on

## Hypothesis

- Statement: after the driver update, the best cold-first prefill shape may have moved away from `batch=6144`, `ubatch=2048`.
- Mechanism: driver/runtime scheduling and residency can shift the tradeoff between larger prompt chunks, compute-buffer shape, and Q3_K hipBLAS staging. The previous best shape was selected before driver `32.0.31007.5012`.
- Why now: E113 refreshed the cold baseline at `11.9858 TPS`; new acceleration work should compare against this post-driver baseline, not older pre-driver shape results.

## Math / Theory

- Baseline: `e113-driver5012-rocm-cold-specnone-r3`, `11.9858 TPS`, prompt eval mean `1272.84 tok/s`, decode mean `28.9183 tok/s`.
- Expected speedup corridor: `+0.5%` to `+3%` if a neighboring shape reduces prompt chunks or avoids a driver-specific residency pocket.
- Failure conditions: prompt eval does not improve, decode regresses enough to offset prefill, or task wall is unchanged.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: cold-first only; keep `--no-reuse --cache-ram 0 --ctx-checkpoints 0`.
3. Rollback path: no code changes.

## Benchmark Plan

- One-run shape gates:
  - `b6144/ub1024`
  - `b6144/ub1536`
  - `b6144/ub3072`
  - `b6144/ub4096`
  - `b8192/ub2048`
  - `b8192/ub4096`
- Confirm with r3 only if a gate beats `11.9858 TPS` by more than noise.

## Metrics

- aggregate completion TPS
- prompt eval tok/s and ms
- decode eval tok/s
- per-task wall TPS
- timeout/error status

## Result

- Outcome: reject all tested shape alternatives.
- Delta: post-driver baseline `b6144/ub2048` is `11.9858 TPS`; best candidate was `b8192/ub2048` at `11.8144 TPS`.
- Confidence: high enough for no-code shape gates; no candidate showed a positive signal.
- Recommendation: keep `batch=6144`, `ubatch=2048` as the active post-driver cold-first prefill shape. Continue prefill work in route/kernel space, not batch/ubatch retuning.

## Notes

- Results:
  - `b6144/ub1024`: `11.4616 TPS`
  - `b6144/ub1536`: `11.7577 TPS`
  - `b6144/ub3072`: `11.6965 TPS`
  - `b6144/ub4096`: `11.6627 TPS`
  - `b8192/ub2048`: `11.8144 TPS`
  - `b8192/ub4096`: `11.7728 TPS`
- Why it failed: smaller ubatch lowers prompt eval; larger ubatch or batch changes do not reduce enough overhead to beat the current driver/runtime residency point. `b8192` processes the whole prompt as one outer batch, but prompt eval still trails the baseline.
- Follow-up action: keep E113 cold baseline for shape comparisons; next prefill work should return to H35 large-Q3_K staging/fused-route analysis.
