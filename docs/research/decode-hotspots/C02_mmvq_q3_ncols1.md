# C02 - MMVQ type=11/q3_K ncols_dst=1

## Current cost snapshot

- Center: `MMVQ type=11/q3_K ncols_dst=1`
- sum_ms: `339.110`
- count: `4618`
- avg_ms: `0.073`
- Priority: `P2`

Source trace:
- `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`

## Initial notes

- This center showed immediate route sensitivity via `small_k` switch in A/B trace.
- Likely directly impacts decode matvec path and can shift `MUL_MAT` aggregate wall time.

## Planned trace steps

1. Baseline trace with default MMVQ route. Done in E013 as `e013-h15-baseline-r1`.
2. Route-forced trace (`GGML_MMVQ_QWEN_FORCE_SMALL_K=1`). Superseded by default Qwen-hot `small_k` policy already present locally.
3. Compare per-key deltas and pre-sync share. Done in E013.
4. Decide if route control can be formalized into guarded policy. Done: keep RDNA4 `Q3_K/ncols_dst=1` at `nwarps=2`.

## E013 result

- Baseline trace: `6.3251 TPS`.
- Candidate trace: `6.5513 TPS`, `+3.58%`.
- Paired non-trace control: `9.1629 TPS`.
- Candidate non-trace: `9.3847 TPS`, `+2.42%`.
- Bootstrap candidate-control 95% CI: `[+0.2019, +0.2442]` TPS.
- Hotspot shift: `MMQ type=11 ncols_max=192 -96.409 ms`; `MUL_MAT forward -252.983 ms`.
- Decision: keep `GGML_TYPE_Q3_K -> nwarps=2` for RDNA4 `ncols_dst=1`; do not alter Q4_K without a Q4-heavy lane.
