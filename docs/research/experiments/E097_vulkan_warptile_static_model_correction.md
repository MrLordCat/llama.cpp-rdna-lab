# E097 Vulkan Warptile Static Model Correction

## Metadata

- Experiment ID: E097
- Date: 2026-05-20
- Owner: Copilot
- Type: no-build tooling correction + resource gate
- Hypothesis: H31
- Target lane: Vulkan Q3_K prompt-heavy prefill, KHR coopmat route

## Hypothesis

The H31 static warptile scout should model the real RX 9070 XT KHR cooperative matrix specialization (`subgroup=64`, `TM/TN/TK=16/16/16`). The previous scout used a subgroup-8/default-shader approximation, which made prepared workgroup sizes and LDS estimates too optimistic for some variants.

## Gate Plan

1. Capture baseline driver pipeline stats with `GGML_VK_PIPELINE_STATS=matmul_q3_k` on pp7488.
2. Correct `scripts/research/vulkan_warptile_static_scout.py` to match runtime specialization and include the coopmat staging LDS.
3. Re-run the scout and compare its base LDS estimate against the driver-reported pipeline stats.
4. Use corrected output to decide whether any BK/tile candidate has enough resource headroom for a build/benchmark.

## Baseline Evidence

Command shape: `llama-bench -p 7488 -n 0 -r 1 --no-warmup -b 4096 -ub 1024 -ctk q4_0 -ctv q4_0 -ngl 999 -fa 1` with `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` and `GGML_VK_PIPELINE_STATS=matmul_q3_k`.

Driver pipeline stats for `matmul_q3_k_f32_f16acc_aligned_l`:

| metric | value |
| --- | ---: |
| `numUsedVgprs` | `113` |
| `numUsedSgprs` | `45` |
| `ldsSizePerLocalWorkGroup` | `20480` |
| `scratchMemUsageInBytes` | `0` |

## Result

Corrected static model was kept.

The scout now matches the actual RX 9070 XT KHR coopmat runtime constants:

| field | corrected value |
| --- | ---: |
| subgroup/warp | `64` |
| cooperative matrix | `TM=16`, `TN=16`, `TK=16` |
| base tile | `BM=128`, `BN=128`, `BK=32` |
| base prepared block | `256` |
| base Q3 shader LDS | `20480 B` |

The base Q3 LDS estimate exactly matches the driver pipeline stats (`20480 B`), so the scout can be used as a prebuild gate for large-tile proposals. It also marks `wn48`/`wn96` invalid for current `BN=128`, explaining why E091 cannot be promoted as a valid tile profile. `BK=64` remains a `needs-resource-proof` idea because it raises Q3 shader LDS above the 32 KiB limit in the static model (`34816 B`) even though it reduces K-loop/barrier count.

Follow-up resource gates from E098 confirmed the corrected model's warning: larger tile families can reduce a dequant proxy but lose on LDS/register/occupancy.

Decision: keep the corrected scout and require it before new H31 warptile/env probes.
