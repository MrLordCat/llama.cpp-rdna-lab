# E131 Vulkan 64k FlashAttention Route Trace and Gates

## Metadata

- Experiment ID: E131
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master, local working tree
- Hypothesis ID: H38 / H05
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, real server repo-snapshot, no reuse

## Hypothesis

- Statement: after E128-E130, the Vulkan 64k route needs exact FlashAttention route visibility before more FA tuning.
- Mechanism: E128 attributes about `38.03%` of traced Vulkan time to `FLASH_ATTN_EXT`, but the existing Vulkan route trace only covered matmul. A default-off FA trace can reveal whether the active long-context route is scalar/coopmat, which tile it uses, whether split-k is active, and whether mask optimization is active.
- Why now: E129 rejected simple coopmat1 `Bc` retuning and E130 rejected nearby `ubatch` sweeps, so the next probe should close route uncertainty rather than guess another tile.

## Implementation

- Kept code:
  - `GGML_VK_FA_ROUTE_TRACE=1` prints unique `ggml_vk_flash_attn(...)` routes.
  - Default behavior is unchanged when the env var is absent.
- Temporary probes, both reverted:
  - `GGML_VK_FA_DISABLE_MASK_OPT=1`
  - `GGML_VK_FA_FORCE_F16ACC=1`

## Route Findings

Real 64k server trace:

- Main long-context FA route: `flash_attn_f32_f16_aligned_f32accq4_0`
- Path: `coopmat1`
- KV types: `k=q4_0`, `v=q4_0`
- Geometry: `HSK=256`, `HSV=256`, `Br=16`, `Bc=64`, `D_split=8`, `row_split=4`
- Workgroup: `workgroup_size=256`, `subgroup_size=64`
- Main prefill chunks: `N=1024`, `KV=1024..57344`, `Tr=64`, `wg_y=24`
- Tail chunk: `N=178`, `KV=57600`, `Tr=12`
- Mask route: `use_mask_opt=1` for the main/tail chunks
- Split-k: `split_k=1` on main long-context chunks; only the small GQA warmup route used `split_k=4`

Interpretation: for the active 64k lane, simple split-k forcing is unlikely to be the missing lever. The active FA route is already coopmat1, aligned, q4/q4, and mask-optimized. Future FA work needs shader/resource analysis or a structural long-KV mechanism, not another nearby `Bc`/split toggle.

## Metrics

Baseline references:

- E128 best full run: `1.3406 TPS`, prompt `666.62`, decode `36.58`
- E129 default screen: prompt `670.70`
- E131 same-build default screen: prompt `665.44`

| Probe | Tokens | Wall TPS | Prompt Eval TPS | Decode Eval TPS | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| FA route trace, default route | `57522 + 1` | screen only | `668.28` | n/a | keep diagnostic |
| Disable mask opt | `57522 + 1` | screen only | `639.44` | n/a | reject |
| Force FA f16acc screen | `57522 + 1` | screen only | `671.72` | n/a | confirm before claim |
| Same-build default screen | `57522 + 1` | screen only | `665.44` | n/a | control |
| Force FA f16acc full run | `57522 + 120` | `1.3380` | `666.86` | `36.16` | reject |

## Result

- Outcome: keep route trace diagnostic; reject both FA speed probes.
- Delta:
  - Disabling mask-opt regressed prompt eval by about `-4.3%` vs the traced default screen.
  - Forced f16acc did not beat the full-run best (`1.3380` vs `1.3406`) and slightly reduced decode eval.
- Confidence: medium-high for rejecting these narrow probes; low for declaring any broader FA ceiling.
- Recommendation: keep the active Vulkan 64k profile from E128 (`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `--no-mmap`, `b8192/ub1024`, q4/q4 KV). Do not revisit mask-opt disable or FA f16acc unless a future route change makes FA arithmetic rather than memory/mask handling the limiting factor.

## Workflow Correction

- `max_tokens=1` screens can overstate tiny FA changes. The f16acc screen looked slightly positive, but the full real-server run did not improve wall TPS.
- For H38, any sub-2% screen result needs either a same-build control or a full `max_tokens=120` confirmation before promotion.

## Artifacts

- `build_logs/agent-workload/e131-vulkan64k-fa-route-trace-c152k-b8192-ub1024-q4-repo-summary.md`
- `build_logs/agent-workload/e131-vulkan64k-fa-route-trace-c152k-b8192-ub1024-q4-ctx64k.server.log`
- `build_logs/agent-workload/e131-vulkan64k-fa-disable-maskopt-c152k-b8192-ub1024-q4-screen-repo-summary.md`
- `build_logs/agent-workload/e131-vulkan64k-fa-force-f16acc-c152k-b8192-ub1024-q4-screen-repo-summary.md`
- `build_logs/agent-workload/e131-vulkan64k-default-samebuild-c152k-b8192-ub1024-q4-screen-repo-summary.md`
- `build_logs/agent-workload/e131-vulkan64k-fa-force-f16acc-c152k-b8192-ub1024-q4-confirm120-repo-summary.md`
