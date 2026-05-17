# E029 C01 Cold-First Ngram-Mod 24/48/64 Recheck

## Metadata

- Experiment ID: E029
- Date: 2026-05-16
- Owner: Codex
- Branch/Commit: local working tree after E028 docs policy update
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, no-reuse

## Hypothesis

- Statement: `ngram-mod 24/48/64` can produce a measurable gain even under cold-first execution, not only in repeated/steady slices.
- Mechanism: decode-side speculative draft acceptance cuts generated-token work during cold tasks with recurring prompt segments.
- Why now: E028 was positive but classified as repeated/steady opt-in; cold adoption needed a direct cold gate.

## Plan

1. Run strict cold r1 pair to check for obvious signal/noise.
2. If inconclusive, run powered r3 pair.
3. Confirm with extended r6 pair.
4. Use `decision_stats.py` for each pair.

## Commands

- control r1: `python scripts/agent_workload_bench.py --label c01-e029-cold-control-r1 ... --runs 1`
- ngram r1: `python scripts/agent_workload_bench.py --label c01-e029-cold-ngram244864-r1 ... --runs 1 --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64"`
- control r3/r6 and ngram r3/r6: same lane with `--runs 3` and `--runs 6`
- stats: `python scripts/research/decision_stats.py --baseline <control.csv> --candidate <ngram.csv>`

## Result

- r1: `9.4381 -> 9.4476 TPS` (`+0.10%`), verdict `inconclusive`.
- r3: `9.3031 -> 10.0948 TPS` (`+8.51%`), CI `[+0.2943,+1.3488]`, verdict `positive`.
- r6: `9.2468 -> 10.2456 TPS` (`+10.80%`), CI `[+0.6980,+1.3441]`, verdict `positive`.
- Candidate diagnostics show decode-led gain with higher variance:
  - decode eval mean around `42.97 tok/s` vs control `29.69 tok/s` on r3,
  - prompt eval remains neutral/slightly lower.
- Spec log stats (`r3`): `gen_drafts=4`, `acc_drafts=4`, `gen_tokens=246`, `acc_tokens=218`, `token_accept_ratio=0.8862`.

## Decision

- Keep as opt-in accelerated profile.
- Keep `spec=none` as conservative default for kernel/default cold claims.
- Use `ngram-mod 24/48/64` as explicit opt-in when practical throughput is prioritized and variance is acceptable.

## Artifacts

- `build_logs/agent-workload/c01-e029-cold-control-r1.csv`
- `build_logs/agent-workload/c01-e029-cold-ngram244864-r1.csv`
- `build_logs/agent-workload/c01-e029-cold-control-r3.csv`
- `build_logs/agent-workload/c01-e029-cold-ngram244864-r3.csv`
- `build_logs/agent-workload/c01-e029-cold-control-r6.csv`
- `build_logs/agent-workload/c01-e029-cold-ngram244864-r6.csv`
- `build_logs/agent-workload/c01-e029-cold-ngram244864-r6.server.log`
