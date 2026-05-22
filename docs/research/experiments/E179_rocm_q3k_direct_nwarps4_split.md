# E179 ROCm Q3_K Direct Nwarps4 Split

## Metadata

- Experiment ID: E179
- Date: 2026-05-22
- Owner: Codex
- Hypothesis ID: H39
- Branch/Commit: temporary worktree patch after `51e7ca31c`, reverted
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: RDNA4 Q3_K direct one-token MMVQ might prefer `nwarps=4/rpb=4`, while fused FFN Q3_K should remain on the kept E151 `nwarps=2/rpb=2` policy.
- Mechanism: E151 kept `nwarps=2` for all Q3_K one-token MMVQ, but E169/E177 split the residual trace into fused and direct buckets. Historical broad `nwarps=4` could have failed because the larger fused FFN route was affected. A fused/direct split tests the direct branch without changing fused FFN math or launch shape.
- Why now: E178 proved that broad `small_k` rollback is wrong. The next cheap topology gate is whether direct-only row parallelism can be increased without touching fused FFN.

## Math / Theory

- Same-session control from E177: `29.5387 TPS` aggregate, `31.97 tok/s` decode.
- E177 short resource trace parsed Q3_K direct ncols1 at `271.708 ms` and fused at `534.579 ms`.
- If only direct Q3_K improves, the local speedup must be large: against roughly `9-20%` decode trace share, a `+1%` wall target needs about `1.06x-1.13x` local depending on whether node-share or parsed-MMVQ share is used.
- Failure condition: direct local time does not improve, or wall TPS falls below same-session control.

## Implementation Plan

Temporary code in `ggml/src/ggml-cuda/mmvq.cu`:

1. Add a `calc_kernel_nwarps(..., has_fusion)` helper.
2. For `MMVQ_PARAMETERS_RDNA4`, `type == GGML_TYPE_Q3_K`, `ncols_dst == 1`, and `has_fusion == false`, return `4`.
3. Keep fused Q3_K on `nwarps=2`.
4. Compute launch dims with the matching fused/direct template branch.
5. Revert if wall or direct bucket regresses.

## Benchmark Plan

- Baseline: E177 same-session control `e177-h39-rocm-current-control-r1`.
- Candidate: `e179-h39-rocm-q3-direct-nwarps4-r1`.
- Trace: `e179-h39-rocm-q3-direct-nwarps4-trace-r1`, graph disabled, sync timing and resource trace, `max_tokens=16`.
- Runs: `1` for gate.

## Metrics

| Run | Aggregate TPS | Decode eval | Prompt eval | Errors |
| --- | ---: | ---: | ---: | ---: |
| E177 control `e177-h39-rocm-current-control-r1` | `29.5387` | `31.97 tok/s` | `511.67 tok/s` | `0` |
| Candidate `e179-h39-rocm-q3-direct-nwarps4-r1` | `28.2361` | `30.46 tok/s` | `510.83 tok/s` | `0` |

Delta:

- aggregate wall: `-4.41%`;
- decode eval: `-4.72%`.

Trace comparison:

| Bucket | Control E177 | Candidate E179 | Interpretation |
| --- | ---: | ---: | --- |
| Q3_K direct ncols1 | `271.708 ms`, block `(32,2,1)`, regs `88`, occ `87.5%`, waves `56` | `315.459 ms`, block `(32,4,1)`, regs `45`, occ `100%`, waves `64` | `+16.1%` slower despite lower VGPR and higher nominal occupancy |
| Q3_K fused ncols1 | `534.579 ms`, unchanged block `(32,2,1)` | `580.699 ms`, unchanged block `(32,2,1)` | unchanged route also slower in this trace, consistent with global scheduling/cache disturbance and/or trace noise, not a local win |

The route activated correctly for direct kernels: the direct ncols1 grid changed from
`(5120,1,1)/(3072,1,1)` style buckets to `(2560,1,1)/(1536,1,1)` with
`block=(32,4,1)`. The lowered register count did not translate to faster
decode; halving row-grid granularity and changing scheduling cadence outweighed
the occupancy improvement.

## Result

- Outcome: regression.
- Confidence: high enough to reject; both wall and local direct bucket moved the wrong way.
- Recommendation: reject and revert. Do not pursue Q3_K direct-only `nwarps=4/rpb=4` as the next parity route. Future Q3_K work should not assume lower VGPR or higher nominal occupancy is sufficient; it must preserve enough row/grid granularity or reduce A/B work directly.

## Notes

- This updates the workflow heuristic: for one-token Q3_K MMVQ, resource stats alone are misleading. A candidate can halve VGPR and hit 100% occupancy while losing because the grid becomes too coarse.
- This also narrows the next viable route: packed/q8 activation layout or another topology that reduces memory/load work without reducing row grid width is more plausible than another warp-count split.

## Artifacts

- `build_logs/agent-workload/e179-h39-rocm-q3-direct-nwarps4-r1.diagnostics.md`
- `build_logs/agent-workload/e179-h39-rocm-q3-direct-nwarps4-trace-r1.server.log`
