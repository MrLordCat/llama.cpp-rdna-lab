# E014 H16 C01 post-E013 MMQ selector/resource pressure

## Metadata

- Experiment ID: E014
- Date: 2026-05-14
- Owner: Codex
- Branch/Commit: `master`, base `0cd0c1f05`
- Target lane: `Qwen3.6-27B-Q3_K_S`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `review_bug,patch_sim`, `no-reuse`, thinking on

## Hypothesis

- Statement: after E013, the remaining C01 cost may be reducible by route-local MMQ selector/resource pressure tweaks on the active `Q3_K/ncols_max=192` bucket.
- Mechanism: changing stream-k threshold, forced `mmq_x`, launch bounds, or `mmq_y` could reduce occupancy/resource pressure or improve the selected tile shape for RDNA4.
- Why now: the fresh post-E013 trace still shows steady decode dominated by `mul_mat_q_direct|q3_K`, so non-MMVQ C01 work must target that route directly.

## Math / Theory

- Assumptions: the active bucket is `type=11 ncols_max=192`; resource trace shows `mmq_x_best=96`, `mmq_y=128`, `nwarps=8`, `regs=160`, dynamic shared about `57728` bytes, one block per SM.
- Expected speedup corridor: small selector/resource changes are expected to be low-single-digit at best unless they reduce the q3 compute core itself.
- Failure conditions: runtime-only improvements without target hotspot improvement are treated as noise or cold-start redistribution, not causal wins.

## Implementation Plan

1. Minimal code surface to change:
   - temporary env or compile-time local probes in `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh`.
2. Guard rails:
   - paired fresh default control,
   - trace target check for `MMQ type=11 ncols_max=192`,
   - revert all non-winning code probes.
3. Rollback path:
   - restore source after each failed probe and rebuild `llama-server`.

## Benchmark Plan

- Baseline command: `scripts/agent_workload_bench.py` on the lane above, `--runs 1`, `--real-context-mode repo-snapshot`, `--no-reuse`.
- Candidate command: same lane with one selector/resource change at a time.
- Number of runs: `1` for gate; `3` only if both runtime and target hotspot are positive.
- Artifacts path: `build_logs/agent-workload/c01-poste013-*`

## Metrics

- aggregate completion TPS (wall)
- target trace bucket timing: `MMQ type=11 ncols_max=192`
- `CUDA_NODE op=MUL_MAT kind=forward`
- timeout/error rate

## Result

- Outcome: regression/tie; no keep candidate.
- Delta:
  - fresh default no-trace: `9.41 TPS`
  - force-x valid points: `64=8.90`, `80=8.34`, `112=8.76`, `128=8.59 TPS`
  - apparent `88/104` no-trace points were invalid as force overrides; trace showed `mmq_x_best=96`, `mmq_x_forced=0`
  - real `x104` with temporary granularity-8 probe hard-timed-out at `30.01s`
  - stream-k post-E013 retest: `192=9.43`, `144=9.41 TPS`, indistinguishable from default
  - `mmq_y=64` does not compile because the write-back MMA static assert requires `nwarps * tile_C::I == mmq_y`
  - RDNA4 `launch_bounds(..., 1)` reached `9.48 TPS`, but trace target worsened: `MMQ type=11 ncols_max=192` from `9949.928` to `10005.326 ms`
- Confidence: medium for rejecting these selector/resource levers; all code probes were reverted.
- Recommendation: stop spending C01 time on scalar selector knobs for this bucket; move next to a Q3_K compute/load specialization or deeper tensor/node split.

## Notes

- Fresh post-E013 resource trace:
  - artifact: `build_logs/agent-workload/c01-poste013-r1-resources.server.log`
  - aggregate trace TPS: `6.61`
  - shape gate: `qtype=11 ncols=192` PASS, histogram `192:26524`, `91:349`, `90:349`
  - steady `MUL_MAT forward`: `15616.091 ms`
  - steady `mul_mat_q_direct|q3_K`: `12325.249 ms` (`78.93%`)
  - q3 coarse split: `compute_core_q3=12325.249 ms` (`84.72%`), `fallback_cublas=2047.566 ms` (`14.07%`), `dequant_load_vec_q3=176.057 ms` (`1.21%`)
- Follow-up action:
  - inspect the Q3_K MMQ load/scale/unpack path rather than only route selectors.
  - keep target-positive requirement: a candidate must improve `MUL_MAT forward` and `MMQ type=11 ncols_max=192`, not merely total wall TPS.
