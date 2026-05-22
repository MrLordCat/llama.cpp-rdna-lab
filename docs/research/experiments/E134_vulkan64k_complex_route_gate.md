# E134 Vulkan 64k Complex Route Gate

## Metadata

- Experiment ID: E134
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E133
- Hypothesis ID: H38 / H31
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, real server repo-snapshot, no reuse

## Hypothesis

- Statement: small Vulkan 64k tweaks are exhausted; the next useful work must change a whole route family, not only a tile constant or one shader expression.
- Mechanism: E133 shows almost all parsed time is Q3_K prefill matmul plus q4 FlashAttention. A candidate should either move the whole Q3_K prefill family, move the whole long-KV FA route, or combine both with smaller but coordinated gains.
- Why now: E129-E132 rejected simple FA retunes, E130 rejected nearby ubatches, and E079-E101/E098 already rejected the local Q3_K arithmetic/tile neighbors.

## Tooling

Added:

```powershell
python scripts\research\vulkan_route_ceiling.py build_logs\agent-workload\e128-vulkan64k-c152k-b4096-ub1024-q4-perf1-ctx64k.server.log --baseline-tps 1.3406 --target-tps 1.5545
```

The tool parses Vulkan perf rows and estimates Amdahl ceilings for route-sized changes. The target is the E128 Vulkan best `1.3406 TPS` vs same-lane ROCm `1.5545 TPS`, so the required wall speedup is `1.1596x`.

Checks:

```powershell
python -m py_compile scripts\research\vulkan_route_ceiling.py
python scripts\research\formula_sanity_checks.py
```

Both passed.

## Metrics

Parsed route shares:

| Route | Total ms | Parsed share | Required local speedup to reach ROCm target alone |
| --- | ---: | ---: | ---: |
| Dense FFN gate/up Q3_K | `20338.69` | `24.91%` | `2.234x` |
| Dense FFN down Q3_K | `11289.87` | `13.83%` | `207.959x` |
| Dense FFN gate/up + down Q3_K | `31628.56` | `38.74%` | `1.551x` |
| Other Q3_K prefill shapes | `11055.89` | `13.54%` | unreachable |
| All Q3_K `MUL_MAT` | `42684.45` | `52.28%` | `1.357x` |
| All `FLASH_ATTN_EXT` | `33965.16` | `41.60%` | `1.494x` |
| All Q3_K `MUL_MAT` + FA | `76649.61` | `93.87%` | `1.172x` |
| All `GLU` | `1271.25` | `1.56%` | unreachable |

Local-speedup corridors:

| Route | `1.10x` local | `1.20x` local | `1.35x` local | `1.50x` local |
| --- | ---: | ---: | ---: | ---: |
| Dense FFN gate/up Q3_K | `1.3717 TPS` | `1.3987 TPS` | `1.4331 TPS` | `1.4620 TPS` |
| Dense FFN gate/up + down Q3_K | `1.3895 TPS` | `1.4331 TPS` | `1.4903 TPS` | `1.5394 TPS` |
| All Q3_K `MUL_MAT` | `1.4075 TPS` | `1.4685 TPS` | `1.5508 TPS` | `1.6235 TPS` |
| All `FLASH_ATTN_EXT` | `1.3933 TPS` | `1.4405 TPS` | `1.5027 TPS` | `1.5564 TPS` |
| All Q3_K `MUL_MAT` + FA | `1.4657 TPS` | `1.5892 TPS` | `1.7718 TPS` | `1.9511 TPS` |

## Code Route Findings

CUDA/ROCm already has a graph-level `MUL_MAT + MUL_MAT + GLU` fusion detector and executor in `ggml/src/ggml-cuda/ggml-cuda.cu`. That route is limited to `mul_mat_vec`/`ncols_dst=1`, so it is mainly a decode fusion and does not solve the current `n=1024` prefill wall. It is still valuable as a reference for graph matching:

- it uses `ggml_can_fuse_subgraph` for sibling `MUL_MAT` nodes feeding one `GLU`;
- it validates same weight shapes, same activation source, and supported GLU ops;
- it rejects split buffers and non-vector destinations.

Vulkan has a fusion framework, but currently only covers routes such as `MUL_MAT_ADD`, RMS/RoPE, multi-add, and top-k MoE. There is no dense FFN `MUL_MAT + MUL_MAT + GLU` prefill route.

## Route Decision

Rejected as primary route:

- Pure `GLU` fusion: only `1.56%` parsed share, so even a perfect post-op fusion cannot matter.
- Launch-only FFN gate/up fusion: the route is large, but a launch-only or post-op-only fusion would not approach the required `2.234x` local speedup.
- Another single Q3_K tile variant: E098 already shows the nearby large-tile family loses from LDS/register pressure, and E133/E134 say the required local win is now too high for blind tile probing.

Promoted as the next complex branch:

1. Vulkan Q3_K large-prefill route, not one shader expression.
2. First proof: add/validate a Vulkan graph detector for dense `MUL_MAT + MUL_MAT + GLU` patterns at `n>1`, using CUDA's constraints as the reference.
3. First real candidate only if resource proof is sane: a default-off dual-A/same-B Q3_K SwiGLU prefill shader for the `m=17408,n=1024,k=5120` pair, where one B/activation tile feeds both gate and up accumulators and writes the activated hidden result directly.
4. Expected risk: doubled accumulators can exceed the current `113 VGPR / 20480 B LDS` Q3_K route budget. A valid prototype likely needs a smaller `WM/WN` profile and must prove it remains coopmat with no scratch.
5. If dual-A resource proof fails, switch the Q3_K branch to backend-private Q3_K repack/layout instead of more arithmetic rewrites.

FA remains co-primary, but a standalone FA fix must be roughly `1.494x` local to close the whole gap alone. That is too high for another `Bc`/mask/f16acc toggle; FA work should resume only as a shader/resource redesign for long-KV tail chunks while staying on coopmat1.

## Workflow Correction

Before any new Vulkan 64k server A/B:

- classify the candidate as route-level or reject it as a micro-probe;
- require a route ceiling from `vulkan_route_ceiling.py`;
- for FFN fusion, require graph-pattern proof plus pipeline resource proof before full 64k real-server timing;
- compare combined-route candidates against the E128 Vulkan best, not against a fresh noisy sweep.

## Result

- Outcome: diagnostic keep; no speed claim.
- Decision: stop spending this lane on isolated flags/tiles. The next coding branch should be Vulkan dense FFN route detection and resource-gated Q3_K gate/up prefill fusion, with FA long-KV redesign as the second branch.
- Confidence: high for route selection, medium for the dual-A shader feasibility because register pressure may kill the mechanism.

## Artifacts

- `scripts/research/vulkan_route_ceiling.py`
- `build_logs/agent-workload/e134-vulkan64k-route-ceiling.md`
- Source perf log: `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q4-perf1-ctx64k.server.log`
