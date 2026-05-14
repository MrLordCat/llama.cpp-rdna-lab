# E017 H18 C01 Q3_K theory gate and k-pair8 probe

## Metadata

- Experiment ID: E017
- Date: 2026-05-14
- Owner: Codex
- Branch/Commit: post `16ac1039e`
- Target lane: `Qwen3.6-27B-Q3_K_S`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `review_bug,patch_sim`, `no-reuse`, thinking on

## Hypothesis

- Statement: after E015, further C01 work should pass a small geometry/math gate before code changes.
- Mechanism: the active Q3_K MMQ bucket can be screened by tile count, shared memory, and loop structure. Candidates that do not change a limiting term or require too large a compensating win should be rejected before benchmarking.
- Why now: E015 changed the geometry to `mmq_y=64/nwarps=4`, invalidating some older intuitions from the `mmq_y=128/nwarps=8` path.

## Math / Theory

Analytic gate:

```powershell
python scripts/research/c01_mmq_q3_theory_gate.py build_logs/agent-workload/c01-e015-rdna4-y64w4-trace-r1.server.log --ncols 192
```

Current active bucket:
- `mmq_x=96`, `mmq_y=64`, `threads=128`
- shared: `35712` bytes (`54.49%`)
- inferred split: Q3 x tile `21504`, Q8 y tile `13824`, misc `384`
- x tile count at `ncols=192`: `2`

Gate results:
- pack Q3 scales to half at `x96`: projected shared `33664` bytes; still above `32 KiB`, so no second block/SM unlock.
- pack Q3 scales + force `x80`: projected shared `31360` bytes and could unlock <=32 KiB, but x tile count rises from `2` to `3`; it needs a very large occupancy win to compensate.
- pair two MMA k steps (`k4 -> k8`): shared and tile count unchanged; halves outer k-loop iterations and dB loads, so it was the cheapest theory-positive probe.

## Implementation Plan

1. Minimal code surface to change:
   - temporary RDNA4-only branch in `ggml/src/ggml-cuda/mmq.cuh` inside the AMD MMA path of `vec_dot_q8_0_16_q8_1_mma`.
2. Guard rails:
   - build first,
   - one `runs=1` runtime gate,
   - revert unless runtime beats E015 reference.
3. Rollback path:
   - remove the temporary RDNA4 `k01 += 8` branch and restore `k01 += 4`.

## Benchmark Plan

- Baseline reference: E015 `c01-e015-rdna4-y64w4-r3` -> `9.6080 TPS`.
- Candidate: `c01-e017-rdna4-q3-kpair8-r1`.
- Number of runs: `1` gate only unless clearly positive.

## Result

- Outcome: reject.
- Candidate r1:
  - `review_bug`: `9.6002 TPS`
  - `patch_sim`: `9.5752 TPS`
  - aggregate: `9.59 TPS`
- Delta vs E015 r3 reference: negative/slightly below.
- Confidence: medium; enough to reject a low-expected-gain loop-overhead idea without trace/r3.
- Recommendation: keep E015 geometry unchanged; do not pursue k-pair8 in this form.

## Notes

- The theory gate utility is kept because it gives a cheap repeatable screen for the next C01 ideas.
- The rejected code probe was reverted and `llama-server` rebuilt.
