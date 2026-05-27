# Benchmark History Policy

This repo keeps raw benchmark artifacts for forensic work, but agents should use
three canonical history files for day-to-day navigation. All three are updated by
`scripts/agent_workload_bench.py`.

## Canonical Files

| File | Role | Use |
| --- | --- | --- |
| `build_logs/agent-workload/BENCH_RUNS.csv` | Structured run ledger | Machine-readable source for labels, lane keys, TPS, prompt/decode metrics, errors, and artifacts |
| `build_logs/agent-workload/BENCH_RECENT.md` | Human recent log | Quick status check before running or comparing a new benchmark |
| `build_logs/agent-workload/BENCH_LANES.md` | Best-by-lane table | Find the best comparable run for a lane before A/B claims |

Legacy `BENCH_HISTORY*.csv/.md` files are still preserved for compatibility and
for old GUI/autotune workflows, but they are no longer the main agent entry
point.

## Refresh Command

After pulling old logs, changing history schema, or before a cleanup commit, run:

```bash
python scripts/agent_workload_bench.py --refresh-canonical-history
```

This rebuilds the canonical files from existing `BENCH_HISTORY*.csv` plus any
current canonical rows. It does not start `llama-server` and does not run a
benchmark.

## New Benchmark Rule

Use `scripts/agent_workload_bench.py` for agent workload benchmarks. Do not add
new ad hoc history markdown files unless a specific experiment needs a separate
analysis note. The runner records every run into the canonical files and keeps
per-run CSV/JSONL/server logs when `--artifact-mode full` is used.

For cold-first claims, labels and history rows must preserve:

- backend and build;
- model;
- ctx, batch, ubatch;
- KV types;
- spec mode and extra server args;
- reuse/prime state;
- thinking mode;
- task set, task IDs, max tokens, and real-context mode.

## Cleanup Rule

Never delete fresh raw logs just to make the folder smaller. Use
`--cleanup-legacy-artifacts` only after the canonical history has been refreshed,
and keep the protected files in the default keep pattern.
