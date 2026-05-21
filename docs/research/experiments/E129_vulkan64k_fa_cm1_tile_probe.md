# E129 Vulkan 64k FA coopmat1 tile probe

## Metadata

- Experiment ID: E129
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master, local working tree
- Hypothesis ID: H38 / H05
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, real server repo-snapshot, no reuse

## Hypothesis

- Statement: Vulkan q4 FlashAttention at 64k may benefit from a different coopmat1 column tile.
- Mechanism: E128 shows `FLASH_ATTN_EXT` is about `38.03%` of traced Vulkan 64k time. The local AMD path uses coopmat1 FA with `Br=16`, default `row_split=4`, `Bc=64`, and `workgroup_size=256`. Long KV may prefer either a smaller tile for occupancy/cache behavior or a larger tile to reduce KV block loop overhead.
- Why now: short/32k traces previously made FA secondary, but E128 made FA a first-class 64k bottleneck.

## Probe

- Added a temporary env gate:
  - `GGML_VK_FA_CM1_SUBGROUPS=1` -> `Bc=16`
  - `GGML_VK_FA_CM1_SUBGROUPS=2` -> `Bc=32`
  - default / `4` -> `Bc=64`
  - `GGML_VK_FA_CM1_SUBGROUPS=8` -> `Bc=128`
- Keep default unchanged without the env.
- Built only `llama-server`.
- Used the E128 64k real-server lane with `max_tokens=1` for prefill screening.
- The code probe was reverted after measurement because no variant improved the default.

## Baseline

- E128 best safe full route: `1.3406 TPS`, prompt `666.62 tok/s`, decode `36.58 tok/s`.
- E128 `b4096/ub1024` no-code route: `1.3106 TPS`, prompt `651.59 tok/s`.
- E128 perf trace: `FLASH_ATTN_EXT` `33965.16 ms`, `38.03%`.

## Decision Gate

- Keep only if a 64k real-server prefill screen improves prompt eval by at least about `2%` without breaking output.
- If a tile variant regresses or only shifts noise, revert the code probe and record the negative result.

## Metrics

| Route | Effective tile | Prompt Tokens | Prompt Eval TPS | Delta vs sg4 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| default / `sg4` | `Br=16,Bc=64,wg=256` | `57522` | `670.70` | baseline | keep default |
| `GGML_VK_FA_CM1_SUBGROUPS=2` | `Br=16,Bc=32,wg=128` | `57522` | `635.94` | `-5.18%` | reject |
| `GGML_VK_FA_CM1_SUBGROUPS=8` | `Br=16,Bc=128,wg=512` | `57522` | `515.29` | `-23.17%` | reject |

## Interpretation

- Smaller `Bc=32` likely pays too much extra KV-block loop/mask overhead for any occupancy/cache gain.
- Larger `Bc=128` likely overuses shared memory/registers and reduces occupancy enough to dominate the lower loop count.
- The default coopmat1 `Bc=64` is the local sweet spot for this 64k q4 FlashAttention path.
- This explains why the earlier FA scalar-only probe also failed: the route is hot, but nearby tiling changes are not automatically positive.

## Result

- Outcome: reject and revert the env-gated code probe.
- Workflow correction: do not assume high FA trace share means a simple FA tile-size retune will help. Future FA work needs either pipeline/resource proof or a more structural memory/dequant reduction, not just `Bc` sweep.
- Next focus returns to Q3_K large-prefill or deeper FA instrumentation, with E128/E129 as the current guardrail.

## Artifacts

- `build_logs/agent-workload/e129-vulkan64k-fa-cm1sg4-prefill1-repo-summary.md`
- `build_logs/agent-workload/e129-vulkan64k-fa-cm1sg2-prefill1-repo-summary.md`
- `build_logs/agent-workload/e129-vulkan64k-fa-cm1sg8-prefill1-repo-summary.md`
