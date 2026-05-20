# E112 ROCm Q3_K Reuse + Ngram Gate

## Metadata

- Experiment ID: E112
- Date: 2026-05-20
- Owner: Codex
- Branch/Commit: master @ 5014549b4
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, reuse enabled, thinking on

## Hypothesis

- Statement: after E111 removes most repeated prefill, `ngram-mod` may improve the repeated/session route by reducing decode work on the cached tasks.
- Mechanism: prompt-cache/checkpoints reduce repeated prompt processing from about `7400` tokens to about `2033-2052` tokens, so decode becomes a larger share of after-first wall time. If ngram coverage appears in repeated requests, speculative acceptance could stack with the prompt-cache route.
- Why now: E107 rejected ngram for cold-first because coverage was near zero, but E111 changes the session regime and makes repeated tasks the main practical route.

## Math / Theory

- Assumptions: E111 after-first tasks run about `20.00 TPS`, with decode around `4.20 s` per 120 tokens and prompt tail around `1.75 s`.
- Expected speedup corridor: `+3%` to `+12%` after-first if coverage/effective acceptance rises above the E107 near-zero values.
- Failure conditions: generated draft count remains near zero, effective acceptance stays below about `0.02`, or speculative bookkeeping offsets accepted-token savings.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: compare only against E111 repeated/session route, not against cold-first no-reuse baseline.
3. Rollback path: no code changes.

## Benchmark Plan

- Baseline command: `e111-rocm-q3k-reuse-steady-r3`.
- Candidate command: same lane with reuse enabled and `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`.
- Number of runs: r3 directly, because E111 r3 is the comparable repeated/session baseline and the run is cheap.
- Artifacts path: `build_logs/agent-workload/e112-rocm-q3k-reuse-ngram244864-r3.*`

## Metrics

- aggregate completion TPS (wall)
- after-first task TPS
- speculative generated drafts/tokens
- local acceptance, coverage, effective acceptance
- prompt-cache/checkpoint restore evidence

## Result

- Outcome: keep as an opt-in stacked repeated/session route.
- Delta: E111 reuse baseline r3 `17.7984 TPS`; E112 reuse + `ngram-mod 24/48/64` r3 `18.7194 TPS`, `+5.17%` aggregate. After-first task throughput improved from about `20.00 TPS` to about `21.40 TPS`, `+7.00%`.
- Confidence: medium-high for this exact repeated two-task workload. The gain is visible in wall TPS and decode timing, but ngram benefit is bursty and task-dependent.
- Recommendation: keep `ngram-mod 24/48/64` as an opt-in stacked session route on top of prompt cache/checkpoints; do not promote it as a cold-first default.

## Notes

- Per-task wall TPS: first cold task `11.5122`; reused tasks `20.0960`, `20.0756`, `23.4347`, `19.9994`, `24.1472`.
- Prompt reuse remained active: each later task restored the `5370`-token checkpoint and processed only about `2033-2052` prompt tokens.
- Decode mean improved from E111 `28.575 tok/s` to E112 `31.3567 tok/s`; prompt eval stayed comparable (`1164.9` vs `1168.77 tok/s` mean).
- Spec stats: `gen_drafts=2`, `acc_drafts=2`, `gen_tokens=126`, `acc_tokens=102`, local acceptance `0.809524`.
- The coverage script reports low overall coverage (`0.006173`) because drafts occurred only in two bursts, but those bursts were long enough to reduce wall time on repeated `review_bug` tasks.
- Why this worked while E107 failed: E107 cold-first ngram had no useful draft bursts; E112 reused session state and repeated task structure, making rare long draft bursts possible after cache/checkpoint reuse had already removed most shared-prefix prefill.
- Workflow update: speculative stacking should be evaluated after the active session/reuse route is established, and reports should include both aggregate coverage and accepted-token bursts by task.
