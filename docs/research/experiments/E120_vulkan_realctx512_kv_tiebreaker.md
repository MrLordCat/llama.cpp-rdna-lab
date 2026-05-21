# E120 Vulkan Real-Context 512-Token KV Tie-Breaker

## Metadata

- Experiment ID: E120
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ cfa2667d3
- Hypothesis ID: H34
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, `ctx=12288`, `batch=6144`, `ubatch=2048`, real repo-snapshot context, `max_tokens=512`, thinking on, reuse enabled

## Hypothesis

- Statement: Vulkan q4 KV may match or beat Vulkan f16 KV on the E119 real-context long-answer route while using less VRAM.
- Mechanism: E116 decode-only showed Vulkan q4 and f16 were very close (`39.8801` vs `40.2753 TPS`). On real-context mixed runs, lower KV memory pressure might offset any f16 decode edge.
- Why now: before promoting Vulkan f16 as the long-answer route, we should check whether q4 is an equal-or-better practical route.

## Benchmark Plan

- Baseline: E119 Vulkan f16 r3, `32.0298 TPS` aggregate, `33.89 TPS` warm-only, decode `39.4483 tok/s`.
- Candidate: Vulkan q4 KV, same workload and run count.
- Decision:
  - keep q4 if it is within noise or faster, because it saves KV memory;
  - keep f16 if it remains materially faster.

## Result

- Outcome: keep Vulkan q4 KV as the practical real-context long-answer route.
- Baseline: E119 Vulkan f16 r3, `32.0298 TPS` aggregate, `33.89 TPS` warm-only, decode `39.4483 tok/s`, prompt `966.9283 tok/s`.
- Candidate: Vulkan q4 r3, `32.1668 TPS` aggregate, `33.96 TPS` warm-only, decode `40.1350 tok/s`, prompt `894.0517 tok/s`.
- Delta vs f16:
  - aggregate: `+0.43%`
  - warm-only: `+0.21%`
  - decode eval: `+1.74%`
- KV memory: q4 KV uses `216 MiB` at `ctx=12288`; f16 KV uses `768 MiB`. Because speed is effectively tied and q4 saves `552 MiB`, q4 is the better default for this route.

## Interpretation

- E119's backend conclusion stands: Vulkan is the long-answer/session route.
- E120 refines the KV choice: prefer Vulkan q4 KV over Vulkan f16 KV unless a future lane specifically proves f16 faster.
- This also makes the route safer for larger contexts and busier VRAM budgets.

## Artifacts

- `build_logs/agent-workload/e120-realctx512-vulkan-q4-specnone-r3.csv`
- `build_logs/agent-workload/e120-realctx512-vulkan-q4-specnone-r3.diagnostics.md`
- `build_logs/agent-workload/e120-realctx512-vulkan-q4-specnone-r3.server.log`
