# E130 Vulkan 64k ubatch midpoint sweep

## Metadata

- Experiment ID: E130
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master, local working tree
- Hypothesis ID: H38 / H08
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, real server repo-snapshot, no reuse

## Hypothesis

- Statement: Vulkan 64k may have a midpoint `ubatch` sweet spot between the already measured `512`, `1024`, and `2048` values.
- Mechanism: E128 found `ub=1024` better than `512` and `2048`. Because the hot route combines Q3_K large matmul and long-KV FlashAttention, intermediate physical batches may trade matmul efficiency, FA tile count, and scheduler/memory pressure differently.
- Why now: no-code route tuning is safe, and E129 rejected simple FA tile retuning.

## Probe

- Kept E128 best stack:
  - `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`
  - `--no-mmap`
  - `b=8192`
  - q4/q4 KV, FlashAttention on
  - `--spec-type none --cache-ram 0 --ctx-checkpoints 0`
- Screened `ub=768` and `ub=1280` with real 64k `max_tokens=1`.
- Compared against E129 default `ub=1024` prefill screen: prompt eval `670.70 tok/s`.

## Decision Gate

- Keep only if prompt eval improves by at least about `2%` or if a follow-up full 120-token run confirms a meaningful wall TPS gain.

## Metrics

| Route | Prompt Tokens | Prompt Eval TPS | Delta vs `ub=1024` | Decision |
| --- | ---: | ---: | ---: | --- |
| `b8192/ub1024` baseline | `57522` | `670.70` | baseline | keep |
| `b8192/ub768` | `57522` | `664.37` | `-0.94%` | reject |
| `b8192/ub1280` | `57522` | `661.06` | `-1.44%` | reject |

## Result

- Outcome: reject midpoint ubatch alternatives.
- Interpretation: the 64k Vulkan no-code shape has a real local optimum around `ub=1024`; `512`, `768`, `1280`, and `2048` are all below it in the current measurements.
- Workflow correction: stop spending 64k cycles on nearby `ubatch` sweeps unless a code change alters the active Q3_K/FA route shape. Use `b8192/ub1024` for the next H38 probes.

## Artifacts

- `build_logs/agent-workload/e130-vulkan64k-ub768-prefill1-repo-summary.md`
- `build_logs/agent-workload/e130-vulkan64k-ub1280-prefill1-repo-summary.md`
