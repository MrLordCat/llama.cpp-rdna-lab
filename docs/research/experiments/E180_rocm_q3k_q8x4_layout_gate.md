# E180 ROCm Q3_K Q8X4 Layout Gate

## Metadata

- Experiment ID: E180
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: local working tree
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: Q3_K `ncols_dst=1` may benefit from a temporary `q8_1 x4` layout if warp/grid policy is unchanged.
- Mechanism: keep existing MMVQ launch policy (`block=(32,2,1)` for hot Q3_K direct/fused route) and only change transient q8_1 packing/read path under env gate.
- Why now: E179 showed grid-width/warp-policy changes are strongly negative; this probe isolates layout effects without changing launch geometry.

## Implementation Plan

1. Add env-gated candidate path: `GGML_MMVQ_Q3K_Q8X4=1`.
2. Limit activation to Q3_K and `ncols_dst=1` path.
3. Keep non-candidate/default path unchanged.

## Benchmark Plan

- Baseline command: active H39 quick lane (`triage_diff`, `--max-tokens 128`, `--runs 1/3`, no reuse).
- Candidate command: same lane with `GGML_MMVQ_Q3K_Q8X4=1`.
- Artifacts path: `build_logs/agent-workload/e180-rocm-q3k-q8x4-*`.

## Results

### r1 gate

- control: `11.7534 TPS`, decode `28.04 tok/s`
- candidate: `11.8823 TPS`, decode `28.17 tok/s`
- initial signal: `+1.10%` aggregate (promising but borderline)

### r3 confirmation

- control: `12.1378 TPS`, decode mean `27.93 tok/s`
- candidate: `12.0940 TPS`, decode mean `28.00 tok/s`
- pair delta: `aggregate_tps_delta=-0.0438`, speedup `0.9964x`

## Decision

- Outcome: reject and revert
- Rationale: r1 uplift did not survive r3 confirmation. Candidate is slightly worse on wall TPS in lane-matched comparison.
- Action taken: reverted all E180 runtime/code-path edits to restore pre-experiment behavior.

## Notes

- The candidate slightly improved decode micro-rate while slightly regressing prompt/prefill timing, resulting in net wall loss.
- Keep this as evidence that isolated transient q8_1 packing changes are below current H39 ceiling unless combined with a larger route-level improvement.

## Artifacts

- `build_logs/agent-workload/e180-rocm-q3k-q8x4-control-r1.diagnostics.md`
- `build_logs/agent-workload/e180-rocm-q3k-q8x4-cand-r1.diagnostics.md`
- `build_logs/agent-workload/e180-rocm-q3k-q8x4-control-r3.diagnostics.md`
- `build_logs/agent-workload/e180-rocm-q3k-q8x4-cand-r3.diagnostics.md`
- `build_logs/agent-workload/e180-rocm-q3k-q8x4-control-r3.server.log`
- `build_logs/agent-workload/e180-rocm-q3k-q8x4-cand-r3.server.log`