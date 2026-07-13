# E280 Vulkan GPU1 Primary Residency

## Metadata

- Experiment ID: E280
- Date: 2026-07-11
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`
- Target lane: dual RX 9070 XT Vulkan, Qwen3.6-27B, long context

## Hypothesis

- Statement: Physical GPU0 enters WDDM Shared GPU Memory because it owns first-device compute/output overhead in addition to its layer-local model and KV allocations, while physical GPU1 retains dedicated VRAM headroom.
- Mechanism: Explicit device order `Vulkan1,Vulkan0` makes GPU1 the first model/context backend. Layer split and equal tensor proportions remain enabled, so KV continues to follow its transformer layers without forced cross-device attention reads.
- Why now: The live long-prompt workload fills GPU0 dedicated VRAM while GPU1 reports only about 11 GiB resident.

## Math / Theory

- Assumptions: both cards have the same capacity and performance; the observed imbalance is primarily the roughly 0.9 GiB first-device compute/output delta, not unequal model weights.
- Expected speedup corridor: no raw-kernel speedup is claimed; the goal is to remove GPU0 shared-memory residency and its associated latency cliff without reducing prompt or decode throughput.
- Failure conditions: GPU0 still enters Shared GPU Memory, GPU1 overcommits instead, or prompt/decode throughput regresses materially.

## Implementation Plan

1. Add a stable GUI device profile for `Vulkan1,Vulkan0 -sm layer -ts 1,1`.
2. Add `LLAMA_OUTPUT_DEVICE=Vulkan1` so the large output/vocabulary tensor leaves GPU0 without changing layer-local attention.
3. Add opt-in `LLAMA_KV_DEVICE=Vulkan1` for literal all-KV residency and make the auto-FA check use the actual KV device.
4. Preserve the old GPU0-primary order and layer-local KV as rollback paths.

Guard rails: no backend memory query or post-build executable probe; server shutdown must remain graceful.

Rollback path: select `Vulkan0,Vulkan1 — GPU0 primary (diagnostics)` in the GUI.

## Benchmark Plan

- Baseline command: current long-context server with `-dev Vulkan0,Vulkan1 -sm layer -ts 1,1`.
- Candidate command: identical launch with `-dev Vulkan1,Vulkan0 -sm layer -ts 1,1`.
- Number of runs: one startup/residency smoke, then one representative long prompt.
- Artifacts path: `build_logs/agent-workload/e280-*`

## Metrics

- per-device model, context/KV, and compute allocation from the server memory breakdown
- WDDM dedicated/shared GPU memory during the long prompt
- prompt and decode TPS
- server and driver stability

## Result

- Outcome: keep GPU1-primary/output placement; reject all-KV as the performance default
- Delta: equal GPU1-primary control measured `1860.21 tok/s` prompt with accounted memory `Vulkan1 7915 MiB / Vulkan0 8586 MiB`. Adding `LLAMA_OUTPUT_DEVICE=Vulkan1` measured `1793.75 tok/s` and moved `1004 MiB` from GPU0 (`Vulkan1 8920 / Vulkan0 7582 MiB`). Literal all-KV placed one `4352 MiB` KV buffer on Vulkan1 and reduced Vulkan0 to `6347 MiB`, but prompt collapsed to `814.82 tok/s`.
- Confidence: high for placement and the all-KV regression; medium for the output-route `-3.6%` r1 prompt delta.
- Recommendation: keep reverse device order plus GPU1 output placement as the GUI default for this rig. Keep all-KV available but off by default until Vulkan gains real row/tensor parallel access to remote KV without layer-scheduler copies.

## Notes

- `--main-gpu 1` is intentionally not used: its CLI contract affects split mode `none`, or intermediate/KV placement with split mode `row`; it does not choose a primary device for layer split.
- Device order is authoritative: explicit `-dev` order is copied into `model.devices`, backends are initialized in that order, and layer KV allocation uses `model.dev_layer(il)`.
- Uneven layer controls also lost prompt throughput: `-ts 3,2` reached `1564.88 tok/s`, `5,4` reached `1648.69`, and `17,15` reached `1694.11`.
- Moving remote attention weights (`795.80 tok/s`) or whole remote attention blocks (`751.52 tok/s`) did not recover the all-KV route.
- Long validation at `54757` prompt tokens with `b8192/ub1024`, q8 KV completed cleanly at `1313.68 tok/s`; accounted memory was `Vulkan1 9604 MiB / Vulkan0 7972 MiB`, with no server left behind.
- Artifacts: `e280-vulkan-gpu1-primary-smoke.server.log`, `e280-vulkan-all-kv-gpu1-smoke.server.log`, and `e280-vulkan-gpu1-output-smoke.server.log` plus their diagnostics.
