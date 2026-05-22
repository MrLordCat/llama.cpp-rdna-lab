# E144 Vulkan 64k Q3_K BK16 Route Gate

## Metadata

- Experiment ID: E144
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master @ 256ee0648
- Target lane: Vulkan 64k, Qwen3.6-27B-Q3_K_S, q4_0/q4_0 KV, FlashAttention on, `b8192/ub1024`, full offload, no reuse

## Hypothesis

- Statement: Reducing the active Q3_K matmul K tile from `BK=32` to `BK=16` may improve occupancy/resource pressure enough to offset doubled K-loop/barrier count.
- Mechanism: `BK16` reduces Q3_K shader LDS from `20480 B` to `12288 B` at the same `BMxBN=128x128`, while keeping valid load mapping and coverage. It does not reduce full-K A/B traffic or dequant count.
- Why now: E143 closed larger-N tiles because VGPR/LDS/live-state pressure dominated static A-dequant savings. `BK16` probes the opposite direction: lower resource footprint, higher loop count.

## Math / Theory

- Static scout:
  - `base`: `128x128x32`, `20480 B` Q3 LDS, `160` K blocks, barrier proxy `0.45M`.
  - `bk16`: `128x128x16`, `12288 B` Q3 LDS, `320` K blocks, barrier proxy `0.90M`.
  - `bk64`: `128x128x64`, `36864 B` Q3 LDS, above the 32 KiB local shared-memory budget.
- Expected speedup corridor:
  - Likely negative unless baseline is occupancy/LDS-limited.
  - A small positive would imply LDS/resource pressure is a hidden limiter.
- Failure conditions:
  - pp7488 regresses due doubled barriers and K-loop overhead.
  - Pipeline resources do not materially improve.
  - Route falls back or changes away from the active Q3_K coopmat path.

## Implementation Plan

1. Minimal code surface to change:
   - Add temporary env-gated `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bk16` branch in AMD large matmul setup.
2. Guard rails:
   - Use `GGML_VK_MATMUL_ROUTE_TRACE=1` and `GGML_VK_PIPELINE_STATS=matmul_q3_k`.
   - Do not build `BK64`; static LDS exceeds the current device limit.
3. Rollback path:
   - If negative, revert the env branch and rebuild clean `llama-bench`/`llama-server`.

## Benchmark Plan

- Baseline command:
  - `llama-bench -p 7488 -n 0 -r 1 --no-warmup -b 8192 -ub 1024 -ctk q4_0 -ctv q4_0 -ngl 999 -fa 1 -mmp 0`
- Candidate command:
  - Same command with `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bk16`.
- Number of runs:
  - `1` for the route/resource gate.
- Artifacts path:
  - `build_logs/agent-workload/e144-vulkan-q3k-bk-depth-static-scout.md`
  - `build_logs/agent-workload/e144-vulkan-q3k-bk16-prebuild-gate.txt`
  - `build_logs/agent-workload/e144-vulkan-q3k-bk16-*.log`

## Metrics

- pp7488 prompt TPS
- Q3_K pipeline resource stats
- route trace active tile/path

## Result

- Outcome: regression; temporary runtime branch reverted.
- Delta:
  - Baseline default: `972.77 tok/s`, `113 VGPR / 45 SGPR / 20480 B LDS / 0 scratch`.
  - `bk16`: `587.52 tok/s`, `70 VGPR / 46 SGPR / 12288 B LDS / 0 scratch`, `-39.60%`.
- Confidence: high for rejecting BK16 as a speed route; the resource drop is real, but K-loop/barrier overhead dominates.
- Recommendation: do not pursue smaller BK in the current Q3_K coopmat shader. The active default is not primarily limited by LDS/VGPR occupancy in a way that can justify doubling K-loop iterations.

## Notes

- This is a resource-direction negative/positive control for the active Q3_K coopmat route, not a full target-closing candidate.
- Workflow correction: a lower-resource shader can be much slower if it increases barrier cadence and loop overhead. Future BK-depth ideas need a model that includes both pipeline resources and barrier/K-loop cost before build.
