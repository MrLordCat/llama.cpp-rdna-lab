---
description: "Use when doing llama.cpp-with-GUI TPS, performance, ROCm, Vulkan, RDNA4, Qwen, benchmark, autotune, or research documentation work."
---
# Performance Workspace Instructions

For TPS/performance work in this fork, start with:

1. `AGENTS.md`
2. `docs/research/CONTEXT_130K_WORKFLOW.md`
3. `docs/research/PERF_WORKSPACE.md`
4. `docs/research/MAJOR_TOPOLOGY_WORKFLOW.md`
5. `docs/research/major-topology/README.md`
6. `docs/research/EXPERIMENTS_DIGEST.md`
7. `docs/research/HYPOTHESES.md`
8. `docs/research/RESULTS_LOG.md`
9. Relevant notes in `docs/research/major-topology/` for the active program
10. `docs/research/experiments/` only for archived or narrow legacy comparisons

Use the narrowest useful tool set: read/search, focused edits, terminal builds or
benchmarks, and todo tracking. Avoid browser/UI/notebook/Java/debug/extension
search tools unless the user explicitly asks for that class of work.

Benchmark claims must preserve lane shape: context, batch, ubatch, KV types,
spec mode, reuse, thinking mode, max tokens, real-context settings, and backend.
For quick iteration use `--runs 1`; use `--runs 3` only for final confirmation of
borderline or promising deltas.

Current active dense Qwen target is `Qwen3.6-27B-Q3_K_S` at `ctx=131072`
(~130k), cold-first, repo-snapshot real context, thinking enabled, no reuse and
no prime pass. The quick baseline shape is `real-context-chars=24576`,
`max_tokens=16`: Vulkan D012 `b512/ub256` = `2.0013 TPS` r3 after
q3quad/GLU opt-in stack with `--no-mmap` is now the baseline; the active Vulkan
target is `2.4 TPS` (D028 gate: `1.1992x` wall, about `1.387x` local on dense
FFN or `1.260x` on all-Q3). D029-D033 reject activation-only/naive-streaming
whole-FFN, old all-Q3 storage/helper/Q8/tile families, compact Q3S layout-body
work, an FA-only pivot, and q3-octa/LOAD_VEC_A=8 repeats; the next Vulkan speed
route needs a true Q3_K compute body or compressed-dot route, not layout-only
unpack simplification or wider per-invocation dequant. FA can stack only after
Q3 has about `1.18-1.20x` local point/static evidence. ROCm baseline
`b512/ub128` = `1.5200 TPS` r3 and ROCm is paused after D013-D027. Treat older 12k/32k/64k/128k runs as historical references unless
the user explicitly asks for a short-context lane. At 130k, RX 9070 XT 16 GB is
expected to spill KV/context/working set into system RAM; diagnostics about
residency, mmap/no-mmap, startup time, and RAM pressure are part of the result.

Primary workflow policy: open or update a major-topology P/D/S note first, then
run gates and measured A/B. Do not start new performance work from a standalone
E### note unless it is an explicit narrow ledger update.

Negative shader/runtime probes should be reverted unless they are intentionally
kept behind a documented opt-in gate. Update the experiment note and
`docs/research/RESULTS_LOG.md` before closing the task.

Benchmark history uses three canonical files written by
`scripts/agent_workload_bench.py`: `build_logs/agent-workload/BENCH_RUNS.csv`,
`build_logs/agent-workload/BENCH_RECENT.md`, and
`build_logs/agent-workload/BENCH_LANES.md`. Refresh them with
`python scripts/agent_workload_bench.py --refresh-canonical-history` after
schema or archive changes.
