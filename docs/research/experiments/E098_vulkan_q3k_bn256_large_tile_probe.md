# E098 Vulkan Q3_K BN256 Large Tile Probe

## Metadata

- Experiment ID: E098
- Date: 2026-05-20
- Owner: Copilot
- Type: env-gated Vulkan tile/resource probe
- Hypothesis: H31
- Target lane: Qwen3.6-27B-Q3_K_S Vulkan prefill, KHR coopmat route, ctx=12288, b=4096, ub=1024, q4_0/q4_0 KV, FlashAttention on

## Hypothesis

Increasing the active Q3_K large matmul tile from `BN=128` to `BN=256` can reduce repeated A/Q3 dequant work across N tiles. For the active shapes with `N=1024`, the static model reduces N-block count from `8` to `4`, cutting the full-K A pair dequant proxy by about half.

## Gate Evidence

Corrected E097 scout model uses the real RX 9070 XT KHR coopmat specialization: subgroup `64`, cooperative matrix `16x16x16`.

Static scout comparison:

| Variant | BMxBNxBK | Prepared block | Q3 shader LDS | Workgroups | Full B reload | Full A pair dequants |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| base | `128x128x32` | `256` | `20480 B` | `1088/320` | `1760 MiB` | `461.37M` |
| `bn256` | `128x256x32` | `512` | `31744 B` | `544/160` | `1760 MiB` | `230.69M` |

The candidate is close to the 32 KiB LDS limit but does not exceed it in the corrected shader model. `bm256-bn256` is modeled above the limit and is not a candidate.

## Implementation Plan

1. Add an opt-in `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn256` branch in `ggml-vulkan.cpp`.
2. Build Vulkan `llama-server` and `llama-bench`.
3. Run driver pipeline stats on pp7488 before full workload.
4. If stats show scratch or severe VGPR pressure, reject without full workload.
5. If pp7488 is promising, run one cold workload A/B against current base.

## Benchmark Plan

Baseline resource command: `GGML_VK_FORCE_AMD_LARGE_MATMUL=1 GGML_VK_PIPELINE_STATS=matmul_q3_k llama-bench -p 7488 -n 0 -r 1 --no-warmup ...`

Candidate resource command: same plus `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn256`.

Full workload only if candidate clears the resource and pp gate.

## Result

Rejected and reverted from runtime variant list.

Driver resource gates and pp7488 results:

| Variant | VGPR | SGPR | LDS | Scratch | pp7488 tok/s | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base recheck | `113` | `45` | `20480 B` | `0` | `983.21` | reference |
| `bn256` | `86` | `46` | `31744 B` | `0` | `947.12` | reject |
| `bm256` | `94` | `45` | `31744 B` | `0` | `909.59` | reject |
| `bn256-wn128` | `165` | `58` | `29696 B` | `0` | `921.84` | reject |
| `bn256-wm128` | `165` | `43` | `29696 B` | `0` | `940.21` | reject |

The original `bn256` theory was directionally plausible because it halves the N-block count and A-dequant proxy, but on this AMD proprietary driver the extra LDS footprint near 32 KiB and/or reduced occupancy dominates. Retile variants reduced LDS slightly but exploded VGPR pressure to `165`, so they were rejected without full workload promotion.

Implementation cleanup: the `bm256`, `bn256`, `bn256-wn128`, and `bn256-wm128` env branches were removed from `ggml-vulkan.cpp` after measurement. Do not retry this large-tile family unless a new model includes occupancy/driver resource behavior, not only full-K work proxies.
