# E110 ROCm Q3_K fit-off Gate

## Metadata

- Experiment ID: E110
- Date: 2026-05-20
- Owner: Codex
- Branch/Commit: master @ 533bb5ed2
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: disabling fit (`-fit off`) might avoid a conservative memory/offload adjustment and improve the q4-KV Q3_K lane.
- Mechanism: E070 needed `-fit off` for a Q4_K_S route recovery; this gate checks whether that lesson transfers to Q3_K_S.
- Why now: E109 showed Vulkan is not a practical route for this exact lane, so this was the last cheap ROCm route flag before returning to kernel work.

## Math / Theory

- Assumptions: baseline E108 post-build control is `11.76 TPS`.
- Expected speedup corridor: small, `0-2%`, only if fit changes backend memory/layout decisions.
- Failure conditions: offload and compute route remain unchanged; wall TPS ties control.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: no default recommendation unless same-session A/B is positive.
3. Rollback path: no code changes.

## Benchmark Plan

- Baseline command: `e108-rocm-gdn-control-r1`.
- Candidate command: same lane with `-fit off` in `server-extra`.
- Number of runs: one-run gate.
- Artifacts path: `build_logs/agent-workload/e110-rocm-q3k-fitoff-r1.*`

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s
- decode eval tok/s
- offload / memory notes from server log

## Result

- Outcome: tie/reject
- Delta: `11.76 TPS` control vs `11.76 TPS` fit-off candidate.
- Confidence: medium; one-run is enough because there is no positive signal.
- Recommendation: do not apply the Q4_K_S `-fit off` lesson to the Q3_K_S q4-KV lane without a fresh A/B.

## Notes

- Surprises: none; fit did not explain current Q3_K_S bottleneck.
- Follow-up action: focus on kernel-level route work or separate repeated/steady-session optimizations, with baselines kept separate from cold-first.
