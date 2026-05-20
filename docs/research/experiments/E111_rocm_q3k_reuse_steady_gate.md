# E111 ROCm Q3_K Reuse Steady Gate

## Metadata

- Experiment ID: E111
- Date: 2026-05-20
- Owner: Codex
- Branch/Commit: master @ 533bb5ed2
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, thinking on

## Hypothesis

- Statement: server prompt cache/checkpoints may provide a practical repeated/steady-session TPS gain on sequential repo tasks that share a large context prefix.
- Mechanism: cold-first measurements intentionally disable `cache-ram` and `ctx-checkpoints`; repeated GUI/agent use may reuse the repo snapshot prefix and reduce prompt work on later tasks.
- Why now: cold-first no-code and small kernel gates did not produce a gain. The project protocol requires cold-first and repeated/steady metrics to stay separate, so this gate measures the steady route explicitly.

## Math / Theory

- Assumptions: same two quick tasks in one server session, no `--no-reuse`, default cache/checkpoint behavior.
- Expected speedup corridor: `+5%` to `+30%` aggregate if the second task reuses a large prefix; tie if prompts diverge too early or cache restore overhead cancels reuse.
- Failure conditions: prompt eval ms remains near cold-first for both tasks; aggregate TPS ties cold-first.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: do not compare this as a cold-first kernel/default claim; label as repeated/steady only.
3. Rollback path: no code changes.

## Benchmark Plan

- Baseline command: cold-first E106/E108 controls.
- Candidate command: same lane, omit `--no-reuse`, keep `--spec-type none`, and do not force `--cache-ram 0 --ctx-checkpoints 0`.
- Number of runs: one-run gate; r3 if clear improvement appears.
- Artifacts path:
  - `build_logs/agent-workload/e111-rocm-q3k-reuse-steady-r1.*`
  - `build_logs/agent-workload/e111-rocm-q3k-reuse-steady-r3.*`

## Metrics

- aggregate completion TPS (wall)
- per-task wall time
- prompt eval ms/tok per task
- cache/checkpoint log evidence

## Result

- Outcome: keep as a confirmed repeated/steady-session route.
- Delta: cold-first control `e106-rocm-q3k-control-r1` was `11.8464 TPS`; repeated/session r1 measured `14.6132 TPS` (`+23.36%` vs cold-first reference); r3 measured `17.7984 TPS` aggregate across six tasks (`+50.24%` vs cold-first reference). This is not a cold-first kernel claim.
- Confidence: high for the measured session effect under this two-task repo-snapshot workload; mechanism is directly visible in server logs.
- Recommendation: keep prompt cache/checkpoints enabled for practical repeated GUI/agent sessions. Continue using `--no-reuse --cache-ram 0 --ctx-checkpoints 0` only for cold-first kernel/default claims.

## Notes

- r3 per-task wall TPS:
  - first cold task: `11.4837`
  - subsequent cached/checkpointed tasks: `20.0298`, `20.0116`, `20.0675`, `19.8866`, `19.9940`
- Steady after-first task throughput is about `20.00 TPS`, roughly `+68.8%` vs the `11.8464 TPS` cold-first reference.
- Log evidence: prompt cache enabled with `8192 MiB` limit; checkpoints were created at about `5370` and `7418` tokens; later tasks selected the slot by LCP similarity (`sim_best=0.982-0.984`) and restored the `5370`-token checkpoint.
- Mechanism: prompt eval token rate stayed near `1151-1191 tok/s`, but reused tasks only reprocessed about `2033-2052` prompt tokens instead of the full `7403-7422`; decode stayed unchanged around `28.5-28.7 tok/s`.
- Follow-up action: add this to the route atlas as a separate repeated/steady route and keep it out of cold-first TPS claims.
