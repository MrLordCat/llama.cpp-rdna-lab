# E133 Vulkan 64k Perf Shape Summary

## Metadata

- Experiment ID: E133
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E132
- Hypothesis ID: H38 / H31
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, real server repo-snapshot, no reuse

## Hypothesis

- Statement: the 64k Vulkan bottleneck should be tracked by exact matmul and attention shapes, not only by op family.
- Mechanism: E128 already showed `MUL_MAT q3_K` plus `FLASH_ATTN_EXT` dominate the trace. Splitting the perf log by shape tells which Q3_K matrices and which long-KV FA chunks deserve code work.
- Why now: E129-E132 closed the easy FA route toggles, and the next Q3_K work needs a higher-ceiling target than broad warptile retuning.

## Tooling

Added:

```powershell
python scripts\research\vulkan_perf_shape_summary.py build_logs\agent-workload\e128-vulkan64k-c152k-b4096-ub1024-q4-perf1-ctx64k.server.log --top 30
```

The parser aggregates Vulkan perf logger rows for:

- `MUL_MAT`, `MUL_MAT_VEC`, and `MUL_MAT_ADD MUL_MAT_VEC` by type and `m/n/k`;
- `FLASH_ATTN_EXT` by `N/KV` chunk.

## Metrics

Parsed from the E128 intrusive perf log:

| Bucket | Calls | Total ms | Parsed share |
| --- | ---: | ---: | ---: |
| `MUL_MAT q3_K` | `19893` | `42684.45` | `53.10%` |
| `FLASH_ATTN_EXT` | `928` | `33965.16` | `42.25%` |
| `MUL_MAT q4_K` | `2736` | `2824.24` | `3.51%` |
| `MUL_MAT f32` | `9168` | `868.42` | `1.08%` |

Top shape contributors:

| Shape | Calls | Total ms | Parsed share |
| --- | ---: | ---: | ---: |
| `MUL_MAT q3_K m=17408 n=1024 k=5120` | `7056` | `20338.69` | `25.30%` |
| `MUL_MAT q3_K m=5120 n=1024 k=17408` | `3528` | `11289.87` | `14.05%` |
| `MUL_MAT q3_K m=10240 n=1024 k=5120` | `2688` | `4698.14` | `5.84%` |
| `MUL_MAT q3_K m=6144 n=1024 k=5120` | `2688` | `2901.79` | `3.61%` |
| `MUL_MAT q4_K m=5120 n=1024 k=6144` | `2688` | `2814.43` | `3.50%` |
| `MUL_MAT q3_K m=12288 n=1024 k=5120` | `896` | `1983.91` | `2.47%` |

The top two Q3_K forms contribute `31628.56 ms`, or about `74.1%` of all parsed Q3_K matmul time.

Long-KV FA rows are individually smaller than the largest Q3_K shapes, but their tail accumulates heavily:

- `N=1024,KV=57344`: `1168.85 ms`;
- `N=1024,KV=56320`: `1136.25 ms`;
- `N=1024,KV=55296`: `1122.66 ms`;
- `N=1024,KV=54272`: `1097.38 ms`;
- total `FLASH_ATTN_EXT`: `33965.16 ms`.

## Result

- Outcome: diagnostic keep; no speed claim.
- Decision: use shape-level timing before new Vulkan 64k code probes.
- Interpretation: the Q3_K half of the bottleneck is concentrated in feed-forward up/gate style `m=17408,k=5120` and down style `m=5120,k=17408` forms. The old global large-tile ideas (`bn256`, `bm256`, `BK=64`) touched these shapes, but E098/E132 already show that simple tile-resource changes are low-confidence or negative without a new mechanism.
- FA interpretation: long-KV FA is genuinely co-primary at 64k, but the best route is not a single huge shape; it is a growing KV series where the tail chunks dominate. Any FA candidate should report per-KV timing and prove it remains on `coopmat1`.

## Workflow Correction

- Future Vulkan 64k experiments must include either:
  - shape-level perf evidence showing the target bucket moved, or
  - a route/resource trace proving the active route changed in a useful way.
- Do not promote a candidate from aggregate prompt TPS alone when the measured shape bucket did not move.

## Artifacts

- `scripts/research/vulkan_perf_shape_summary.py`
- `build_logs/agent-workload/e133-vulkan64k-perf-shape-summary.md`
- Source perf log: `build_logs/agent-workload/e128-vulkan64k-c152k-b4096-ub1024-q4-perf1-ctx64k.server.log`
