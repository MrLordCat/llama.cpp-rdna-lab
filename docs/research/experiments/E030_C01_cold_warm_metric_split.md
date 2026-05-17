# E030 C01 Cold/Warm Metric Split

## Metadata

- Experiment ID: E030
- Date: 2026-05-16
- Owner: Codex
- Branch/Commit: local working tree after E029 docs
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, no-reuse, thinking on

## Hypothesis

- Statement: The C01 benchmark needs an explicit split between true first-run cold TPS and repeated/session-warm TPS.
- Mechanism: `agent_workload_bench.py` writes `run` in JSONL, but the CSV did not include it and headline aggregate TPS mixes run #1 with later runs. Speculative modes can look like cold wins when the gain is actually concentrated in later repeated runs.
- Why now: E029 reported `ngram-mod 24/48/64` as positive on an aggregate `r6` headline, but the changed measurement contract now treats cold-first as primary and warm/session as secondary.

## Math / Theory

- Cold-first metric: aggregate completion TPS over rows with `run == 1`.
- Warm/session metric: aggregate completion TPS over rows with `run > 1`.
- Aggregate-all metric remains useful for practical repeated sessions, but it must not be labeled as cold-first.

Expected interpretation:

- default/kernel/runtime claims must improve cold run #1,
- opt-in repeated/session profiles can be kept when warm TPS improves and cold does not regress materially,
- speculative speedups need acceptance/coverage context because sparse coverage can create high variance.

## Implementation Plan

1. Add `run` to per-run CSV output in `scripts/agent_workload_bench.py`.
2. When `--stats-ignore-first-run` is enabled, print cold-only run #1 stats in addition to warm-only stats.
3. Re-run clean and `ngram-mod 24/48/64` on the same C01 lane with `runs=2`.

## Benchmark Plan

- Baseline command:
  - `python scripts/agent_workload_bench.py --label c01-e030-clean-split-r2 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug,patch_sim --runs 2 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --stats-ignore-first-run`
- Candidate command:
  - same as baseline plus `--server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64"`
- Number of runs: `2` for the split sanity check
- Artifacts path: `build_logs/agent-workload/`

## Metrics

- aggregate completion TPS over all rows
- cold-only aggregate TPS (`run == 1`)
- warm-only aggregate TPS (`run > 1`)
- prompt/decode split
- speculative local acceptance, coverage, effective acceptance

## Result

- Outcome: measurement contract corrected; ngram remains opt-in warm/session accelerator, not a cold-first default win.
- Clean split:
  - all: `9.4569 TPS`
  - cold run #1: `9.47 TPS`
  - warm excluding run #1: `9.45 TPS`
- `ngram-mod 24/48/64` split:
  - all: `10.0476 TPS` (`+6.25%` vs clean all)
  - cold run #1: `9.46 TPS` (neutral vs clean cold)
  - warm excluding run #1: `10.72 TPS` (`+13.4%` vs clean warm)
- Decision stats on all rows: bootstrap 95% CI `[-0.0034,+1.2763]` TPS, verdict `inconclusive` because `n=4` and variance is high.
- Spec stats:
  - local acceptance: `0.868852`
  - coverage: `0.011236`
  - effective acceptance: `0.009762`

Additional probes:

- Server warmup enabled via `--no-no-warmup` did not improve cold or warm (`9.47/9.44 TPS` split), so keep benchmark default `--no-warmup`.
- `ubatch=224` regressed to `8.03 TPS`.
- `ubatch=160` regressed to `8.88 TPS`.
- Fresh C01 resource trace `c01-e030-resume-r1-resources` still shows the active target:
  - shape gate PASS for `Q3_K type=11 ncols_max=192` (`26524` hits),
  - steady `mul_mat_q_direct|q3_K = 12268.144 ms` (`78.52%` of steady `MUL_MAT forward`),
  - active geometry remains `mmq_x=96`, `mmq_y=64`, shared `35712`, regs `160`, waves `4.0`.

## Notes

- E029 aggregate `r6` remains useful as a repeated/session artifact, but it should not be described as a pure cold-first result.
- For future C01 reports, always quote at least:
  - cold run #1 TPS,
  - warm excluding run #1 TPS,
  - aggregate-all TPS only as a secondary practical/session number.
- Next cold-first work should remain no-spec and target the Q3_K MMQ prefill path; simple `ubatch` movement around `192` is not promising under the current lane.
