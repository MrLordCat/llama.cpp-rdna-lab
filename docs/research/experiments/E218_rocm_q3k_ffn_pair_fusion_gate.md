# E218 ROCm Q3_K FFN Gate/Up Pair Fusion Gate

## Metadata

- Experiment ID: E218
- Date: 2026-05-24
- Owner: Codex
- Hypothesis ID: H42
- Target lane: ROCm `build-rocm-vec`, Qwen3.6-27B-Q3_K_S, active repo-snapshot lane

## Hypothesis

- Statement: a structural H42 candidate could fuse or pair the adjacent `ffn_gate` and `ffn_up` Q3_K prefill matmuls because they share the same activation input and dominate the `17408x5120@2048` cuBLAS bucket.
- Mechanism under review: reuse/merge the shared `src1` staging and possibly execute one larger gate+up route instead of two independent cuBLAS calls.
- Why now: E217 shows the simple large-Q3_K cublas hotspot clearly enough to screen complex graph-fusion ideas before implementing them.

## Math / Theory

- E217 top bucket `(17408,5120,2048)`:
  - `ffn_gate`: `189` calls, `706.623 ms`
  - `ffn_up`: `189` calls, `708.488 ms`
  - combined: `1415.111 ms`
  - GEMM: `1147.612 ms`
  - `src1` staging: `82.721 ms`
- Simple shared-activation staging ceiling:
  - best case if all duplicated `src1` staging disappears: `82.721 ms`
  - trace wall context: `~9150 ms`
  - projected wall gain: `~0.90%` before graph/fusion overhead.
- To reach a meaningful `+2%` wall gain on this lane, the route would need roughly `183 ms` saved at the same trace wall. After `src1` staging, it still needs about `100 ms` from GEMM/epilogue efficiency, i.e. a nontrivial new body rather than pairing alone.

## Analytical Gate

- Pass condition for coding:
  - either prove that one larger gate+up GEMM is materially faster than two `17408x5120` GEMMs at the same `ncols`, or
  - design a fused Q3_K body that reduces real GEMM/dot work or avoids intermediate traffic beyond `src1` staging.
- Fail condition:
  - only shared `src1` staging, launch-count reduction, or graph adjacency reuse.

## Result

- Outcome: reject the simple pair-fusion/cache variant; keep only the broader H42 body idea.
- Delta: projected `~0.9%` wall ceiling for shared `src1` staging alone, below the threshold for a risky graph-fusion implementation.
- Confidence: medium. This is an analytical screen from fresh E217 measured point sums, not a runtime candidate.
- Recommendation: do not implement a gate/up pair route that only caches or shares `src1`. A real H42 continuation must attack GEMM/body efficiency or fused epilogue/intermediate traffic with a measurable microbench proof first.

## Notes

- This is not the same as rejecting all FFN fusion. It rejects the cheap subcase that only removes duplicate activation staging.
- The useful insight is that E217's top prefill bucket is `81.1%` GEMM time, so conversion/staging-only work cannot carry the next breakthrough.
