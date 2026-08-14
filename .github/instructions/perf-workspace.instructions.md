---
description: "Use when doing llama.cpp-rdna-lab TPS, performance, ROCm, Vulkan, RDNA4, Qwen, benchmark, autotune, or research documentation work."
---
# Performance Workspace Instructions

This file inherits all safety, dirty-worktree, ownership, delegation, and
handoff rules from `AGENTS.md` and `AGENT_WORKFLOW.md`. It may tighten those
rules for performance work but never relax them. In delegated workflows,
`tps-research` owns the hypothesis/source patch, `bench-runner` alone owns the
hardware lane, and `research-docs` writes prose from accepted artifacts.

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

Current primary dense Qwen baseline is `Qwen3.8-27B-Q4_K_M.gguf` (rebased from Qwen3.6 2026-08-14). Start generic
production/performance work from the safe ROCm lane
`ctx=49152,b=8192,ub=1024,q8_0/q8_0,-dev ROCm1,ROCm0,-sm layer,-ts 1,1`, one
slot, cold/no-reuse/no-warmup. The adjacent `spec=none` control is
`1778.59 prompt tok/s`; MTP n3 is the production agent profile when answer
length amortizes its `2.64%` prompt cost (`39.58 decode tok/s`, `6.2802`
aggregate TPS, `74.36%` acceptance on the measured 29.5K/128 lane). The
one-copy ROCm scheduler is validated at `ctx=98304`; `ctx=131072` remains a
residency stress lane requiring fresh placement evidence. Keep q8 as the Q4 KV
baseline. D088 TKV4 is an opt-in residency route pending Q4 quality/perplexity.

The prior `Qwen3.6-27B-Q3_K_S` 130K Vulkan P002/P003 programs remain valid
model-scoped history and a secondary headroom/Q3-kernel lane. Do not compare
Q3 and Q4 speed rows as one baseline or carry Q3-specific target math into a
Q4 experiment.

Primary workflow policy: open or update a major-topology P/D/S note first, then
run gates and measured A/B. Do not start new performance work from a standalone
E### note unless it is an explicit narrow ledger update.

Negative shader/runtime probes should be reverted unless they are intentionally
kept behind a documented opt-in gate. Revert only the patch created for that
probe, by its owner or the coordinator, using a narrow reverse patch. Never use
`git checkout`, `git reset`, or whole-file restoration over user/other-agent
changes; `bench-runner` never edits or reverts source. Update the experiment
note and `docs/research/RESULTS_LOG.md` before closing the task.

Benchmark history uses three canonical files written by
`scripts/agent_workload_bench.py`: `build_logs/agent-workload/BENCH_RUNS.csv`,
`build_logs/agent-workload/BENCH_RECENT.md`, and
`build_logs/agent-workload/BENCH_LANES.md`. Refresh them with
`python scripts/agent_workload_bench.py --refresh-canonical-history` after
schema or archive changes.
