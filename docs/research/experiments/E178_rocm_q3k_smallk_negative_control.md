# E178 ROCm Q3_K Small-K Negative Control

## Metadata

- Experiment ID: E178
- Date: 2026-05-22
- Owner: Codex
- Hypothesis ID: H39
- Branch/Commit: master after `51e7ca31c`
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: The residual ROCm decode gap might be partly caused by the kept RDNA4 Qwen-hot `small_k` branch, which batches `rows_per_block = nwarps` for `Q3_K/Q4_K/Q6_K`.
- Mechanism: E151 improved Q3_K by moving RDNA4 one-token decode to `nwarps=2`, which also enables `rows_per_block=2`. If row batching hurts the post-E151 route, disabling `small_k` should recover speed without code changes.
- Why now: E169/E177 traces show Q3_K still dominates, so the cheapest route-level negative control is to isolate the row-batching part before designing a larger Q3_K topology.

## Math / Theory

- Same-session control from E177: `29.5387 TPS` aggregate, `31.97 tok/s` decode.
- E177 short sync trace puts combined Q3_K forward+fused node share near `39.36%` of decode node time.
- A `+1%` wall target from Q3_K alone needs about `1.026x` local speedup if all Q3_K buckets improve; a row-batching regression should be obvious in an r1 gate.
- Failure condition: aggregate and decode eval below same-session control means row batching is not the active bottleneck.

Analytic tooling:

```powershell
python scripts\research\formula_sanity_checks.py
python scripts\research\speedup_model.py --baseline-tps 29.5387 --prefill-share 0.0643 --draft-len 1 --accept-rate 0 --spec-overhead 0 --flash-prefill-speedup 1.0 --decode-kernel-speedup 1.02
python scripts\research\required_acceptance.py --target-wall 1.01,1.02 --draft-len 1 --prefill-share 0.0643 --prefill-speedup 1.0 --decode-kernel-speedup 1.0 --spec-overhead 0.0
```

The speculative acceptance solver is not applicable to this non-speculative route; it reports unreachable under the no-spec assumptions, as expected.

## Implementation Plan

No code change. Set `GGML_MMVQ_QWEN_DISABLE_SMALL_K=1` for one run, then clear it.

## Benchmark Plan

- Baseline: E177 same-session control `e177-h39-rocm-current-control-r1`.
- Candidate: `e178-h39-rocm-disable-smallk-r1`.
- Runs: `1`.
- Artifacts path: `build_logs/agent-workload/`.

## Metrics

| Run | Aggregate TPS | Decode eval | Prompt eval | Errors |
| --- | ---: | ---: | ---: | ---: |
| E177 control `e177-h39-rocm-current-control-r1` | `29.5387` | `31.97 tok/s` | `511.67 tok/s` | `0` |
| Candidate `e178-h39-rocm-disable-smallk-r1` | `26.2521` | `28.16 tok/s` | `508.89 tok/s` | `0` |

Delta:

- aggregate wall: `-11.12%`;
- decode eval: `-11.92%`.

## Result

- Outcome: regression.
- Confidence: high enough for a negative control; the drop is much larger than normal r1 noise.
- Recommendation: keep the E151 RDNA4 Qwen-hot `small_k` row batching. Do not pursue a broad `rows_per_block=1` rollback. The next branch should change Q3_K topology or split fused/direct policies, not disable the current small-k batching.

## Notes

- Prompt eval remained close to control, while decode collapsed, matching the targeted route.
- This also explains why a simple "less row batching" intuition is wrong on this lane: the kernel needs the extra row work to amortize the RDNA4 one-token Q3_K route.

## Artifacts

- `build_logs/agent-workload/e178-h39-rocm-disable-smallk-r1.diagnostics.md`
- `build_logs/agent-workload/e178-h39-rocm-disable-smallk-r1.server.log`
