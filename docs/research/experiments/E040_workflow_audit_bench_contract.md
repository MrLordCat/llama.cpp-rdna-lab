# E040 Workflow Audit: Why Progress Stalls Despite Many Experiments

## Metadata

- Experiment ID: E040
- Date: 2026-05-17
- Owner: Copilot
- Type: process/instrumentation audit
- Scope: C01 benchmark workflow quality

## Question

Why do many experiments fail to produce reliable progress signals?

## Findings (measured)

Using `scripts/research/bench_history_audit.py` on C01 labels:

- rows analyzed: `126`
- unique comparability signatures: `8`
- verdict: `MISMATCHED`

Main drift dimensions inside one C01 prefix:

- `max_tokens`: `120/160/256/1`
- `no_reuse`: `1/0`
- `ubatch`: `192/224/160`
- `spec_mode`: `none/ngram-mod`
- `extra_args`: multiple variants (`-`, `--spec-type none`, ngram profile, no-reuse explicit flags)

Interpretation:

- A/B outcomes were often compared across non-identical workloads.
- This creates false positives/negatives larger than many candidate deltas.

## Root Cause Summary

1. Analysis issue:
   - mixed lane signatures were treated as comparable baseline/candidate pairs.
2. Measurement issue:
   - trace and non-trace runs were occasionally mixed in conclusions.
3. Implementation issue:
   - many probes were graph/selector-level while the dominant steady center stayed unchanged (`mul_mat_q_direct|q3_K`, `ncols_max=192`).

## Instrumentation Added

New tool:

- `scripts/research/bench_history_audit.py`

Capabilities:

- audits `BENCH_HISTORY.csv` by label prefix,
- computes comparability signatures,
- reports heterogeneity per field,
- emits recommended dominant lane signature,
- supports `--strict` non-zero exit for automation.

## New Operating Contract (effective immediately)

For C01 runtime claims, baseline and candidate must match all fields:

- `tasks`
- `ctx`
- `batch`
- `ubatch`
- `kv_k`
- `kv_v`
- `max_tokens`
- `no_reuse`
- `spec_mode`
- `extra_args`

And additionally:

- do not compare trace-on runs against trace-off runs,
- do not compare warm/repeated results against cold-first claims,
- reject any claim where dominant signature mismatch exists.

## Next Step

- keep the workflow gate in every C01 cycle:
  1. run `bench_history_audit.py` before interpreting deltas,
  2. if mismatch -> re-run proper control with the dominant signature first,
  3. only then evaluate candidate deltas.
