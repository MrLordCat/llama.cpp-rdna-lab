# E028 C01 Ngram-Mod 24/48/64 Confirmation

## Metadata

- Experiment ID: E028
- Date: 2026-05-16
- Owner: Codex
- Branch/Commit: local working tree after E027, with negative quant/MMF probes reverted
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, no-reuse, thinking on

## Hypothesis

- Statement: `ngram-mod` with a long repeated-span profile (`n_match=24`, `n_min=48`, `n_max=64`) can provide an opt-in C01 speedup even though the current cold-first lane has sparse draft coverage.
- Mechanism: When the repeated repository-context tail is hit, accepted ngram drafts reduce decode work; the effect is bursty, so it needs more than a single r1/r3 sample.
- Why now: E026 showed a promising but inconclusive `+3.31%` aggregate result with low effective acceptance. E028 reruns the same lane with a clean paired control and a larger candidate sample.

## Math / Theory

- E026 effective acceptance was only `0.00675`, so a default promotion was not justified.
- The speedup should appear mostly in decode metrics, not prompt eval, because ngram speculation does not change prompt prefill kernels.
- Failure condition: if the gain depends on one lucky repeated task, bootstrap CI should cross zero after more samples.

## Implementation Plan

1. Keep runtime code unchanged; negative quant/MMF code probes were reverted before confirmation.
2. Run clean control r3 on the C01 lane.
3. Run `ngram-mod 24/48/64` as an opt-in server-extra candidate with r3, then r6 if r3 remains promising but noisy.

## Benchmark Plan

- Baseline command: `python scripts/agent_workload_bench.py --label c01-e028-clean-control-r3 ... --runs 3`
- Candidate command: `python scripts/agent_workload_bench.py --label c01-e028-ngram244864-r6 ... --runs 6 --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64"`
- Artifacts path: `build_logs/agent-workload/`

## Metrics

- aggregate completion TPS
- mean task TPS
- prompt/decode eval TPS
- local acceptance, coverage, effective acceptance
- bootstrap decision stats

## Result

- Outcome: opt-in win / no default
- Baseline: `c01-e028-clean-control-r3 = 9.4890 TPS`
- Candidate: `c01-e028-ngram244864-r6 = 10.3689 TPS`
- Delta: `+0.8799 TPS` (`+9.27%`)
- Decision stats: bootstrap 95% CI `[+0.5192, +1.3106]` TPS, verdict `positive`
- Prompt eval: `855.5400 -> 851.3758 TPS` (`0.9951x`)
- Decode eval: `30.1433 -> 45.1508 TPS` (`1.4979x`)
- Spec stats: local acceptance `0.581422`, coverage `0.040580`, effective acceptance `0.023594`
- Errors: `0` in both control and candidate

## Notes

- This is a real measured speedup for the current C01 task pair, but it is not a kernel/default win. The benefit is decode/speculative and coverage-dependent.
- Keep `ngram-mod 24/48/64` as an opt-in repeated/steady-task preset. Do not make it the cold-first default unless a broader task suite shows stable coverage.
- Negative code probes during this cycle:
  - `GGML_QUANT_MMQ_BLOCK_SIZE=64/32/256` did not beat the default `128`.
  - Q8_1 quant `__float2int_rn` did not beat `roundf`.
  - RDNA4 F32 MMF threshold `32/64` did not beat the cuBLAS route.
  - All negative runtime code changes were reverted before the final ngram confirmation.
