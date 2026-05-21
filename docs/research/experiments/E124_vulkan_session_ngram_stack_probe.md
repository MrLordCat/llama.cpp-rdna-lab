# E124 Vulkan Session Ngram Stack Probe

## Metadata

- Experiment ID: E124
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ b9462608b
- Hypothesis ID: H36 / H09
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan q4, `ctx=12288`, `batch=6144`, `ubatch=2048`, real repo-snapshot context, `max_tokens=128`, thinking on, reuse enabled

## Hypothesis

- Statement: `ngram-mod 12/16/32` may stack on the Vulkan q4 repeated/session route.
- Mechanism: ROCm E113 showed ngram can help when prompt cache/checkpoints create repeated-session structure. Vulkan E116 rejected ngram in decode-only short-prompt mode, but real-context/reuse may improve coverage.
- Why now: E122 established Vulkan q4 as the 128+ repeated/session backend. Before freezing the route as `spec=none`, test whether the current ROCm session ngram can transfer.

## Benchmark Plan

- Baseline: E122 Vulkan q4 `spec=none`, r3, `19.6365 TPS`, warm-only `23.15 TPS`.
- Candidate: Vulkan q4 + `ngram-mod 12/16/32`, r3.
- Inspect speculative acceptance; reject if effective acceptance/coverage is too low or wall TPS regresses.

## Result

- Outcome: reject Vulkan q4 + `ngram-mod 12/16/32`.
- Baseline: E122 Vulkan q4 `spec=none`, `19.6365 TPS`, warm-only `23.15 TPS`, decode `40.2400 tok/s`.
- Candidate: E124 Vulkan q4 `ngram-mod 12/16/32`, `14.3229 TPS`, warm-only `16.72 TPS`, decode `23.6333 tok/s`.
- Delta:
  - aggregate speedup `0.7294x` (`-27.06%`)
  - aggregate TPS delta `-5.3135`
- Spec stats:
  - generated draft tokens: `296`
  - accepted draft tokens: `64`
  - local acceptance: `0.216216`
  - coverage: `0.022403`
  - effective acceptance: `0.004844`

## Interpretation

- The ROCm session ngram win does not transfer to Vulkan.
- Cause: draft coverage is too sparse on this route, so local acceptance is not enough to pay for speculative bookkeeping and verification overhead.
- Keep the Vulkan session route as `spec=none`.
- Keep `ngram-mod 12/16/32` only in the ROCm repeated/session profile where E113 measured a positive after-first route.

## Artifacts

- `build_logs/agent-workload/e124-realctx128-vulkan-q4-ngram121632-r3.csv`
- `build_logs/agent-workload/e124-realctx128-vulkan-q4-ngram121632-r3.diagnostics.md`
- `build_logs/agent-workload/e124-realctx128-vulkan-q4-ngram121632-r3.server.log`
