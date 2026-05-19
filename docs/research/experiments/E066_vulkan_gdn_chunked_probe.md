# E066 Vulkan GDN chunked probe

## Metadata

- Experiment ID: E066
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master @ 8c1195ab4 plus local Vulkan prototypes
- Target lane: RX 9070 XT, Windows Vulkan proprietary driver, `build-vulkan`, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `q4_0/q4_0`, thinking on, no reuse

## Hypothesis

Qwen3.6 uses Gated DeltaNet layers, and Vulkan prompt-heavy performance is still behind ROCm after E065. Upstream `ggml-org/llama.cpp#20377` adds a chunked Vulkan GATED_DELTA_NET path intended to reduce sequential work for long prompts. A narrow, env-gated local prototype might improve prompt eval on the active `triage_diff` lane.

## Implementation

- Pulled the upstream `#20377` patch for inspection.
- Applied the chunked GDN shaders and wired a temporary `GGML_VK_GDN_CHUNKED=1` prototype for `S_v=128`, non-KDA, no-intermediates GDN.
- Rebuilt `llama-server` and `llama-bench` successfully after fixing a local merge slip in the nearby WKV pipeline setup.
- Ran the active prompt-heavy lane with E064/E065 still enabled.
- Reverted the chunked GDN source and shader additions after the negative result.

## Results

Prompt-heavy workload (`triage_diff`, repo-snapshot, 7489 prompt tokens, 64 generated, `ctx=12288`, `b4096/ub1024`, `q4_0/q4_0`, `flash-attn=on`, `spec=none`, no reuse, thinking on):

| Config | Runs | Wall TPS | Prompt eval TPS | Decode eval TPS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| E065 large tile + Q3_K alignment | 3 | `6.4180` aggregate / `6.38` median | `897.63` | `40.35` | reference |
| E066 `GGML_VK_GDN_CHUNKED=1` | 1 | `5.4760` | `745.49` | `40.33` | `-14.7%` vs E065 |

The regression is in prompt eval. Decode stayed essentially flat, so the chunked path did not address the remaining prefill gap on this shape.

## Decision

Reject and revert for the active lane. Do not keep `GGML_VK_GDN_CHUNKED` or chunked GDN shader additions in the tree. The next Vulkan work should continue from E065 and focus on other prompt/prefill bottlenecks.

## Artifacts

- `build_logs/agent-workload/e066-vulkan-gdnchunk-large-b4096-ub1024-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e066-vulkan-gdnchunk-large-b4096-ub1024-ctx12288-q3ks.server.log`
- `build_logs/agent-workload/e066-vulkan-q3k-align-large-highub-pp7488.md`
- `build_logs/agent-workload/e066-pr21024-q3k-summary.txt`