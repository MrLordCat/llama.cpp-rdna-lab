# Research Hub: Next Efficiency Wave

## Archive Status

The current Qwen3.6/RDNA4 acceleration cycle is archived as of 2026-05-18. Start with `PERFORMANCE_ARCHIVE_2026-05-18.md` before reopening any performance work.

No active default speedup branch remains. Parked leads are H28 selector parity, H29 gfx12 direct quantized prefill design, future MTP with an MTP-enabled GGUF, and any new upstream/RDNA4 evidence that changes the current route mix.

This folder is a local R&D workspace for ideas beyond current ngram speculative decoding and Flash Attention baselines.

For VS Code agent setup, fixed tasks, tool budgets, and benchmark workflow, start with `PERF_WORKSPACE.md`.

## Goal

Find reproducible changes that can improve wall TPS in the active prompt-heavy lane, not only synthetic microbenchmarks.

Primary lane (project policy):

- context below 16k (reference: ctx=12288)
- prompt-heavy, no reuse
- reproducible runs and artifact logging

## Benchmark Baseline Policy

- For A/B research, compare candidates against the current best config from autotune/history under analogous parameters, not against a freshly invented baseline.
- Match ctx, batch, ubatch, KV types, spec mode, extra preset, real-context mode, real-context size, reuse state, thinking mode, and max token budget.
- Do not use `--v2-prime-pass` for cold-first speed claims. Priming is only an explicit steady-state diagnostic and must be labeled as such.
- If the historical best used repo-snapshot context, preserve the effective prompt size with `--real-context-chars` when needed; default safe-fill may drift as the repo changes.

## What Goes Here

- hypothesis docs with math and expected effect size
- experiment plans and acceptance criteria
- run logs and postmortems
- links to implementation PRs or patches

## Structure

- HYPOTHESES.md: prioritized candidate ideas and why they might work
- PERF_WORKSPACE.md: VS Code agent/tool/task workflow for reproducible TPS work
- PERFORMANCE_ARCHIVE_2026-05-18.md: final pause/archive summary for the current cycle
- EXPERIMENT_TEMPLATE.md: standard template for each experiment
- RESULTS_LOG.md: compact ledger of executed experiments
- R0_post_ngram_flashattention.md: first deep-dive note with concrete discovery directions
- experiments/: per-experiment notes (E001, E002, ...)

## Related Tooling

Use the lightweight estimator before expensive runs:

- scripts/research/speedup_model.py
- scripts/research/required_acceptance.py
- scripts/research/formula_sanity_checks.py
- scripts/research/bench_pair_compare.py
- scripts/research/formula_vs_observed.py
- scripts/research/spec_log_stats.py
- scripts/research/spec_effective_acceptance.py
- scripts/research/spec_model_compare.py
- scripts/research/spec_model_batch_compare.py
- scripts/research/required_spec_overhead.py

Purpose:

- speedup_model.py: quick projection and sensitivity grid
- required_acceptance.py: minimum acceptance needed for target wall speedup
- formula_sanity_checks.py: monotonicity and inverse-solver consistency checks
- bench_pair_compare.py: measured baseline/candidate comparison from CSV (+ optional server logs)
- formula_vs_observed.py: observed speedup vs model assumptions with implied acceptance backsolve
- spec_log_stats.py: extract draft/accept statistics from llama-server logs
- spec_effective_acceptance.py: coverage-weighted acceptance from log stats
- spec_model_compare.py: side-by-side error comparison of naive vs coverage-aware formulas
- spec_model_batch_compare.py: batch validation of naive vs coverage-aware formulas across multiple measured cases
- required_spec_overhead.py: backsolve required speculative overhead for observed wall speedup

Example:

```bash
python scripts/research/speedup_model.py \
  --baseline-tps 9.85 \
  --prefill-share 0.70 \
  --draft-len 48 \
  --accept-rate 0.60 \
  --spec-overhead 0.08 \
  --flash-prefill-speedup 1.30 \
  --decode-kernel-speedup 1.05 \
  --sweep-accept 0.45,0.55,0.65 \
  --sweep-flash 1.1,1.2,1.3,1.4

python scripts/research/formula_sanity_checks.py --samples 3000 --seed 9070

python scripts/research/required_acceptance.py \
  --target-wall 1.10,1.20,1.30 \
  --draft-len 16,24,32,48 \
  --prefill-share 0.70 \
  --prefill-speedup 1.20 \
  --decode-kernel-speedup 1.00 \
  --spec-overhead 0.08

python scripts/research/bench_pair_compare.py \
  --baseline-name none \
  --baseline-csv build_logs/agent-workload/scan16k-vec-b6144-ub512-none.csv \
  --candidate-name ngram \
  --candidate-csv build_logs/agent-workload/scan16k-vec-b6144-ub512-ngrammod-noprime-postrebuild.csv

python scripts/research/spec_log_stats.py \
  --log build_logs/agent-workload/postrebuild-vec-b6144-ub512-ngram-prime.server.log --json

python scripts/research/spec_effective_acceptance.py \
  --log build_logs/agent-workload/postrebuild-vec-b6144-ub512-ngram-prime.server.log --json
```

## Current Status

- E059 external RDNA4 research completed: `experiments/E059_external_rdna4_llama_research.md`.
- Current acceleration cycle archived: `PERFORMANCE_ARCHIVE_2026-05-18.md`.
- Do not continue low/medium-risk local probing by inertia; reopen only via the archive protocol.

- E001 analytic gate completed: `experiments/E001_H02_analytic_gate.md`
- E002 measured ubatch cliff completed: `experiments/E002_H08_measured_ubatch_cliff.md`
- E003 ngram formula/observed cross-check completed: `experiments/E003_ngram_formula_observed_crosscheck.md`
- E004 coverage-aware formula validation completed: `experiments/E004_coverage_aware_formula_validation.md`
- E005 multi-case speculative model validation completed: `experiments/E005_spec_model_batch_validation.md`
- E006 full retest and scientific audit completed: `experiments/E006_full_retest_and_scientific_audit.md`
- E007 ub490+ root-cause + 32k ceiling-break follow-up documented: `experiments/E007_H08_current_ub480_490_recon.md`
- E008 ROCm compute-vbuffer residency cliff fix documented: `experiments/E008_H11_rocm_compute_vbuffer_residency.md`

Retest artifacts snapshot:

- `build_logs/agent-workload/retests-20260512/`

Latest root-cause lesson:

- A sharp RDNA4 ROCm `ubatch` cliff can happen with identical kernel routes and node counts. For `Qwen3.6-27B`, the final fix was ROCm graph compute vbuffer chunking, not a smaller physical `ubatch` or a GDN/FATTN selector change.
- Keep `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` as the negative control when validating future allocator/residency hypotheses.

## R&D Loop

1. Define hypothesis and expected speedup corridor.
2. Estimate upper bound with the analytic model.
3. Run minimal microbench to validate mechanism.
4. Run real lane benchmark against the current autotune/history best baseline without `--v2-prime-pass`.
5. Keep only reproducible wins, archive regressions with reason.

## If You Are New To Research

1. Pick one hypothesis from HYPOTHESES.md.
2. Fill EXPERIMENT_TEMPLATE.md.
3. Run formula_sanity_checks.py and required_acceptance.py first.
4. Only after that run microbench and lane benchmark.
5. Record the result in RESULTS_LOG.md and add one experiment note.
