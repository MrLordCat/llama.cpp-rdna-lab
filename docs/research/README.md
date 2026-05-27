# Research Hub: Next Efficiency Wave

## Current Status

The 2026-05-18 acceleration cycle remains archived in
`archive/2026-05-fast-probe-cycle/PERFORMANCE_ARCHIVE_2026-05-18.md`, but a new
post-E264 major-topology research mode is open for dense
`Qwen3.6-27B-Q3_K_S` at `ctx=131072` (~130k) on RDNA4/Vulkan and ROCm.

Start with `CONTEXT_130K_WORKFLOW.md`, then `MAJOR_TOPOLOGY_WORKFLOW.md` before opening new backend prototypes.
The short version: the quick E### loop has exhausted nearby no-code, f16,
helper, and simple layout probes. New work must begin as a design/topology note
under `major-topology/`, with route evidence and a ceiling model before code.

This folder is a local R&D workspace for ideas beyond current ngram speculative decoding and Flash Attention baselines.

For VS Code agent setup, fixed tasks, tool budgets, and benchmark workflow, start with `PERF_WORKSPACE.md`.

## Goal

Find reproducible changes that can improve wall TPS in the active 130k long-context lane, not only synthetic microbenchmarks.

Primary lane (project policy):

- `ctx=131072` (~130k), dense `Qwen3.6-27B-Q3_K_S`
- cold-first, repo-snapshot real context, no reuse, no v2 prime pass
- thinking enabled, q4_0/q4_0 KV, Vulkan and ROCm baselines measured separately
- expected RAM-spill/residency pressure on 16 GB VRAM; diagnostics are part of the result
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
- CONTEXT_130K_WORKFLOW.md: active 130k lane contract and baseline commands
- PERF_WORKSPACE.md: VS Code agent/tool/task workflow for reproducible TPS work
- EXPERIMENTS_DIGEST.md: compact historical base grouped by route family
- BENCH_HISTORY_POLICY.md: canonical benchmark history file contract
- MAJOR_TOPOLOGY_WORKFLOW.md: post-E264 workflow for large architecture changes
- major-topology/: program board and design notes before source prototypes
- EXPERIMENT_TEMPLATE.md: standard template for each experiment
- RESULTS_LOG.md: compact ledger of executed experiments
- DFLASH_IMPLEMENTATION_PREP.md: staged DFlash integration plan for this fork
- experiments/: per-experiment notes (E001, E002, ...)
- dflash/: source vendor manifest and DFlash-specific planning artifacts
- archive/: historical plans and audits no longer used as active entry points

DFlash planning directory currently includes:

- `dflash/VENDOR_MANIFEST.md`
- `dflash/PHASE_PLAYBOOK.md`
- `dflash/BRANCH_AND_COMMIT_PLAN.md`
- `dflash/COMPATIBILITY_MATRIX.md`
- `dflash/IMPLEMENTATION_RUNBOOK.md`
- `dflash/FUTURE_WORKFLOW.md`
- `dflash/KICKOFF_PACKET.md`

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

## Recent Status

- E059 external RDNA4 research completed: `experiments/E059_external_rdna4_llama_research.md`.
- E249-E264 close the latest archived ROCm/Vulkan tail of short-context cold-lane gates.
- Current active lane is 130k: `ctx=131072,b=512,q4_0/q4_0,spec=none`,
  `real-context-chars=24576`, `max_tokens=16`; current Vulkan best is D005 `ub=256` with `--no-mmap` at `1.7898 TPS` r3, while ROCm is `ub=128` at `1.5200 TPS` r3.
- E257 is the archived dense Vulkan 12k reference: `7.0319 TPS` r3 at
  `ctx=12288,b=7168,ub=1024,q4_0/q4_0,spec=none`.
- E258/E259/E260/E264 reject nearby Vulkan Q3_K transfer routes; do not continue
  low/medium-risk local probing by inertia.

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
2. Read EXPERIMENTS_DIGEST.md to avoid repeating closed route families.
3. Fill EXPERIMENT_TEMPLATE.md, or a major-topology design note when the change is broad.
4. Run formula_sanity_checks.py and required_acceptance.py first when applicable.
5. Only after that run microbench and lane benchmark.
6. Record the result in RESULTS_LOG.md, refresh EXPERIMENTS_DIGEST.md, and add one experiment note.
