# E276 ROCm MTP Small-N MMQ Pair Reuse

## Metadata

- Experiment ID: E276
- Date: 2026-07-11
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`
- Target lane: dual RX 9070 XT ROCm, Qwen3.6-27B-Q3_K_S MTP, `ctx=12288`, `b=8192`, `ub=1024`, q8/q8 KV

## Hypothesis

- Statement: target verify at `N=2..5` can reuse the Q8_1 activation prepared for adjacent dense Q3_K `ffn_gate` and `ffn_up` MMQ calls.
- Mechanism: recognize the existing `{ MUL_MAT, MUL_MAT, GLU }` subgraph, quantize its shared f32 activation once in a call-local pool buffer, run both MMQ bodies from that buffer, then launch the existing SWIGLU epilogue.
- Why now: E275 moved RDNA4 Q3_K multi-column verify from MMVQ to MMQ and raised decode to `35.58 tok/s`, exposing repeated MMQ activation staging as the next narrow removable cost.

## Math / Theory

- Synchronized E276 component trace, target `N=5`: `130` gate conversions summed to `10.72 ms`; `130` up conversions summed to `6.44 ms`.
- The second conversion is therefore a measured `~6 ms` upper bound per target round in this trace, roughly `8%` of a warmed `74-77 ms` target verify pass.
- E218 rejected the same idea for large prefill because its projected wall ceiling was only `~0.9%`. Small-N verify is a distinct route where conversion and dispatch are a larger share.
- Expected speedup corridor: `1.03-1.09x` for target verify, smaller end-to-end decode gain after draft overhead.
- Failure conditions: no graph fusion activation, stale-buffer reuse, output mismatch, or clean decode gain below noise.

## Implementation Plan

1. Refactor dense MMQ launch so an already prepared Q8_1 activation can be consumed by two matrices.
2. Add a call-local pair entry point; do not add persistent cache state.
3. Activate only with `GGML_RDNA4_Q3K_MMQ_PAIR_REUSE=1`, HIP RDNA4, dense Q3_K, SWIGLU, same activation, same matrix shape, and `N=2..5`.
4. Keep the ordinary MMQ and all prefill routes unchanged when the env is absent.

## Benchmark Plan

- Baseline: current E275 default, MTP `n_max=4`, cold-first, no reuse/no prime, thinking on.
- Candidate: identical command plus `GGML_RDNA4_Q3K_MMQ_PAIR_REUSE=1`.
- First gate: build, route trace, one short correctness run.
- Performance gate: `max_tokens=256`, one run each; promote only if the candidate beats noise without acceptance regression.
- Artifacts path: `build_logs/agent-workload/e276-*`.

## Metrics

- target verify ubatch time
- decode and aggregate TPS
- local acceptance and coverage
- fused pair activation count
- errors or output divergence

## Result

- Outcome: inconclusive under external GPU contention; no measured win
- Delta: candidate `16.7051` vs control `16.7764` aggregate TPS (`-0.43%`),
  `26.10` vs `26.17` decode tok/s (`-0.27%`), with identical
  `177/311 = 56.91%` local acceptance
- Confidence: low for the small delta because League of Legends was active, but
  high that activation reuse alone did not produce a large gain in this run
- Recommendation: remove the runtime prototype from source and do not promote
  the knob. A clean rerun is optional after both GPUs are idle; continue with a
  true small-N compute topology instead of iterating graph-level reuse.

## Notes

- Current upstream `master` at `76f279805` still restricts CUDA/HIP matmul fusion to vector routes with `ncols_dst=1`; MMQ has no equivalent pair API.
- This experiment is deliberately not the final 1.6x body. A positive result removes measured duplicate work; the remaining gap still requires a true small-N Q3_K MMQ/body improvement.
- The candidate activated `256` graph pairs in the smoke run (`128` at `N=2`,
  `128` at `N=5`) and preserved output/acceptance, so the result is not a route
  activation failure. The source prototype was reverted after the neutral A/B;
  the already-built local binary may still contain the default-off diagnostic
  specialization until the next ROCm rebuild.
- Performance artifacts: `e276-rocm-n4-control-256.*` and
  `e276-rocm-n4-pair-256.*` under `build_logs/agent-workload/`.
