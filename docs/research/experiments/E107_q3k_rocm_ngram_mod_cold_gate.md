# E107 Q3_K ROCm ngram-mod Cold Gate

## Metadata

- Experiment ID: E107
- Date: 2026-05-20
- Owner: Codex
- Branch/Commit: master @ 533bb5ed2
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, no reuse, thinking on

## Hypothesis

- Statement: `ngram-mod 24/48/64` may improve practical completion TPS on the current q4 KV lane because decode is a substantial share of wall time even in the prompt-heavy cold-first benchmark.
- Mechanism: ngram speculative decoding can accept multiple draft tokens per verification step when the generated answer reuses prompt/repository phrases. This reduces target-model decode work without changing ROCm kernels.
- Why now: E106 refreshed the no-spec baseline at `11.8464 TPS` and shows decode is not negligible: baseline diagnostics report prompt eval mean `5926.61 ms` and decode eval mean `4171.11 ms` per task.

## Math / Theory

- Assumptions: q4 KV, no reuse, same prompt/task mix as E106. Any speed claim must separate cold-first no-reuse behavior from repeated/steady-session speculative behavior.
- Expected speedup corridor: `+3%` to `+12%` aggregate if ngram coverage is non-zero; tie/regression if coverage is sparse or speculative bookkeeping exceeds accepted-token savings.
- Failure conditions: generated draft count near zero, low effective acceptance, or improved `decode_eval_tps` without improved aggregate wall TPS.

## Implementation Plan

1. Minimal code surface to change: none; run current server with opt-in `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`.
2. Guard rails: do not promote as default kernel improvement; treat as q4 KV opt-in/session route unless cold-first r3 confirms real aggregate gain.
3. Rollback path: no code changes.

## Benchmark Plan

- Baseline command: E106 no-spec cold-first control `e106-rocm-q3k-control-r1`.
- Candidate command: same lane with `ngram-mod 24/48/64`, cache disabled.
- Number of runs: one-run gate first; if clearly positive, run 3-run confirmation.
- Artifacts path:
  - `build_logs/agent-workload/e107-rocm-q3k-ngrammod-r1.*`
  - optional confirmation `build_logs/agent-workload/e107-rocm-q3k-ngrammod-r3.*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- prompt/decode eval tok/s
- speculative generated drafts/tokens
- local acceptance, coverage, effective acceptance

## Result

- Outcome: reject for this cold-first q4-KV lane.
- Delta: baseline `11.8464 TPS`; `ngram-mod 24/48/64` measured `11.7838 TPS`; smaller `ngram-mod 12/16/32` measured `11.3471 TPS`; `ngram-simple n8/m16` measured `11.2810 TPS`.
- Confidence: high that the tested ngram routes are not cold-first wins here because draft coverage was almost zero.
- Recommendation: do not promote ngram for the current `ctx=12288`, no-reuse, q4-KV Q3_K_S cold-first baseline. Keep speculative work separate as session/repeated-task research.

## Notes

- `ngram-mod 24/48/64`: `gen_drafts=0`, `acc_tokens=0`, coverage `0.000000`, effective acceptance `0.000000`.
- `ngram-mod 12/16/32`: local acceptance `0.281250`, but coverage only `0.005076`, effective acceptance `0.001428`.
- `ngram-simple n8/m16`: local acceptance `0.468750`, but coverage only `0.010471`, effective acceptance `0.004908`.
- Why the hypothesis missed: decode share was real, but the benchmark did not enter draft-enabled spans often enough. Local acceptance looked acceptable in the smaller probes, yet coverage was too sparse to pay for speculative overhead.
- Workflow change: speculative candidates must first report coverage and effective acceptance, not just local acceptance, before they are promoted to r3 or default/profile discussion.
