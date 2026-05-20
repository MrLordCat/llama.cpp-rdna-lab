---
description: "Use when doing llama.cpp-with-GUI TPS, performance, ROCm, Vulkan, RDNA4, Qwen, benchmark, autotune, or research documentation work."
---
# Performance Workspace Instructions

For TPS/performance work in this fork, start with:

1. `AGENTS.md`
2. `docs/research/PERF_WORKSPACE.md`
3. `docs/research/HYPOTHESES.md`
4. `docs/research/RESULTS_LOG.md`
5. Relevant notes in `docs/research/experiments/`

Use the narrowest useful tool set: read/search, focused edits, terminal builds or
benchmarks, and todo tracking. Avoid browser/UI/notebook/Java/debug/extension
search tools unless the user explicitly asks for that class of work.

Benchmark claims must preserve lane shape: context, batch, ubatch, KV types,
spec mode, reuse, thinking mode, max tokens, real-context settings, and backend.
For quick iteration use `--runs 1`; use `--runs 3` only for final confirmation of
borderline or promising deltas.

Negative shader/runtime probes should be reverted unless they are intentionally
kept behind a documented opt-in gate. Update the experiment note and
`docs/research/RESULTS_LOG.md` before closing the task.
