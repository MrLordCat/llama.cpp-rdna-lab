# D005 P002 Vulkan FFN Down Split-K

Date: 2026-05-26

Status: kept as default for the guarded Q3_K FFN-down shape.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, cold-first, no reuse, no v2 prime, thinking on.
- Prior default anchor: `d005-vulkan-control-confirm3`, `1.6679 TPS`, prompt `867.95 tok/s`, decode `43.30 tok/s`.
- New default anchor: `d005-vulkan-default-splitk-confirm3`, `1.7898 TPS`, prompt `934.81 tok/s`, decode `43.59 tok/s`.

## Hypothesis

D004 showed that the dense FFN down projection remains a large route after the
`ub=256` recenter: `m=5120,n=256,k=17408` had fewer M/N workgroups than the
gate/up projection while carrying the same FLOP count. The existing split-K
heuristic did not split this shape because the tile count was above its
low-occupancy threshold, but the trace showed the down projection running much
slower per FLOP than gate/up.

The cheap probe was to reuse the existing Vulkan split-K/reduce route only for
Q3_K FFN down, rather than building a new shader body.

## Point Gate

Reference corrected trace:

- `build_logs/agent-workload/d004-vulkan130k-route-ceiling-corrected.md`.

Split-K point probes:

| Candidate | Parsed total ms | Dense FFN down ms | Decision |
| --- | ---: | ---: | --- |
| control | `8893.65` | `2188.84` | baseline |
| split-K 2 | `8488.95` | `1817.54` | positive, but not best |
| split-K 3 | `8245.34` | `1626.31` | best local route |
| split-K 4 | `28214.79` | `6050.53` | reject, reduce/temp overhead cliff |

Split-K 3 reduces the targeted FFN-down bucket by `25.7%` and parsed total time
by `7.3%`. Split-K 4 confirms the route has a sharp over-splitting cliff.

Artifacts:

- `build_logs/agent-workload/d005-vulkan-ffndown-splitk2-route-ceiling.md`.
- `build_logs/agent-workload/d005-vulkan-ffndown-splitk3-route-ceiling.md`.
- `build_logs/agent-workload/d005-vulkan-ffndown-splitk4-route-ceiling.md`.
- `build_logs/agent-workload/d005-vulkan-postsplit-route-ceiling.md`.

## Wall Validation

| Variant | Runs | TPS | Prompt tok/s | Decode tok/s | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 3 | `1.6679` | `867.95` | `43.30` | baseline |
| split-K 3 opt-in | 3 | `1.7774` | `927.75` | `43.35` | `+6.56%` |
| split-K 3 default | 3 | `1.7898` | `934.81` | `43.59` | `+7.31%` vs control |

Single-run companion:

- `d005-vulkan-control-full-r1`: `1.6743 TPS`, prompt `871.34`.
- `d005-vulkan-ffndown-splitk3-full-r1`: `1.8032 TPS`, prompt `941.59`.
- `d005-vulkan-default-splitk-full-r1`: `1.7866 TPS`, prompt `932.81`.

The response previews remained normal thinking text, with no symbol-spam or
corruption signal.

## Code Policy

The default path now uses split-K 3 only when all of these are true:

- `src0_type == GGML_TYPE_Q3_K`;
- `m == 5120`;
- `k == 17408`;
- `n >= 128`;
- the caller did not disable split-K for the operation.

Rollback and further probes:

- `GGML_VK_Q3K_FFN_DOWN_SPLIT_K=0` or `1`: disable this shape-specific default.
- `GGML_VK_Q3K_FFN_DOWN_SPLIT_K=2..8`: force another split count for this shape.
- `GGML_VK_MATMUL_ROUTE_TRACE=1`: print the selected split-K route once per shape.

## Post-D005 Ceiling

Using the split-K 3 point trace and the new default baseline `1.7898 TPS`, the
remaining path to `2 TPS` requires:

| Route | Parsed share after D005 | Required local speedup |
| --- | ---: | ---: |
| Dense FFN gate/up Q3_K | `37.16%` | `1.394x` |
| Dense FFN down Q3_K | `19.72%` | `2.141x` |
| Dense FFN gate/up + down Q3_K | `56.88%` | `1.227x` |
| All Q3_K MUL_MAT | `78.63%` | `1.154x` |
| All Q3_K MUL_MAT + FA | `87.04%` | `1.137x` |

The next route should not keep pushing split count. The remaining useful path is
another structural Q3_K change, most likely gate/up or all-Q3 body/layout work
that can stack with this down-projection split.
