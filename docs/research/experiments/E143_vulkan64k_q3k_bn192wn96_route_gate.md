# E143 Vulkan 64k Q3_K BN192/WN96 Route Gate

## Metadata

- Experiment ID: E143
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master @ b6c883abb
- Target lane: Vulkan 64k, Qwen3.6-27B-Q3_K_S, q4_0/q4_0 KV, FlashAttention on, `b8192/ub1024`, full offload, no reuse

## Hypothesis

- Statement: A Q3_K large-matmul route using `BMxBNxBK=128x192x32` with `WMxWN=64x96` can improve 64k prefill by reducing repeated A-side Q3_K dequant work and workgroup count for N=1024 shapes.
- Mechanism: The active base route uses `BN=128`, so N=1024 needs 8 N tiles. `BN=192` reduces this to 6 tiles, cutting A-side Q3_K dequant proxy by about 25% for the hot dense FFN shapes. `WN=96` keeps the prepared workgroup at 256 and avoids the unsafe A-load overshoot of plain `BN192`.
- Why now: E128/E134 showed Vulkan 64k is bottlenecked by Q3_K matmul plus q4 FlashAttention. E141/E142 rejected KV dtype and larger FA Br routes, so the next route-level target is the Q3_K large matmul tile family.

## Math / Theory

- Assumptions:
  - Current best Vulkan 64k wall TPS: `1.3406`.
  - Q3_K prefill hotspot share from route traces: `0.5228`.
  - Baseline pp7488 Q3_K/FA local proxy: `971.09 tok/s`.
  - Target wall uplift corridor: at least a measurable positive pp7488 win, then real 64k server confirmation before promotion.
- Static scout:
  - `base`: valid, `128x128x32`, `WMxWN=64x64`, workgroups `1088/320`, A pair dequants `461.37M`, B reload `1760 MiB`.
  - `bn192`: valid layout but invalid load map; prepared block `384`, A load loop can overshoot the shared A tile.
  - `bn192-wn96`: valid layout and valid load map; prepared block `256`, workgroups `816/240`, A pair dequants `346.03M`, B reload `1980 MiB`.
  - `bn192-wm128-wn96`: valid, prepared block `128`, same tile counts but likely weaker occupancy/parallelism.
- Expected speedup corridor:
  - A-side proxy reduction is `25%`, but B reload rises `12.5%` and `WN=96` increases accumulator/live-state pressure.
  - If local Q3_K route gain is only `+10%`, Amdahl projects roughly `+4.99%` wall TPS.
  - Required local speedup to solve the whole 64k gap via Q3_K alone is `+35.73%`, so this is a partial-route probe, not a complete answer.
- Failure conditions:
  - Pipeline resources show large VGPR/LDS pressure.
  - pp7488 regresses despite lower workgroup count.
  - Route logs show fallback to base or non-active tile.
  - Real server output sanity fails even if benchmark improves.

## Implementation Plan

1. Minimal code surface to change:
   - Add env-gated `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn192-wn96` branch in AMD large matmul setup.
   - Keep the route experimental unless pp7488 and real 64k server both confirm a win.
2. Guard rails:
   - Use `GGML_VK_MATMUL_ROUTE_TRACE=1` and `GGML_VK_PIPELINE_STATS=matmul_q3_k`.
   - Do not benchmark plain `bn192` because static gate marks its current load map unsafe.
   - Clear `HSA_OVERRIDE_GFX_VERSION`; ensure no background `llama-server`.
3. Rollback path:
   - If negative or unsafe, revert the env branch and rebuild clean `llama-bench`/`llama-server`.

## Benchmark Plan

- Baseline command:
  - `llama-bench -p 7488 -n 0 -r 1 --no-warmup -b 8192 -ub 1024 -ctk q4_0 -ctv q4_0 -ngl 999 -fa 1 -mmp 0`
- Candidate command:
  - Same command with `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn192-wn96`.
- Number of runs:
  - `1` for the initial route gate; repeat and real 64k server only if promising.
- Artifacts path:
  - `build_logs/agent-workload/e143-vulkan-q3k-bn192wn96-static-scout.md`
  - `build_logs/agent-workload/e143-vulkan-q3k-bn192wn96-prebuild-gate.txt`
  - `build_logs/agent-workload/e143-vulkan-q3k-bn192wn96-*.log`

## Metrics

- pp7488 prompt TPS
- Q3_K pipeline resource stats
- route trace active tile
- 64k real server wall/prompt/decode TPS if pp7488 is positive

## Result

- Outcome: regression; temporary runtime branches reverted.
- Delta:
  - Baseline default: `974.19 tok/s`, `113 VGPR / 45 SGPR / 20480 B LDS / 0 scratch`.
  - `bn192-wn96`: `760.78 tok/s`, `139 VGPR / 48 SGPR / 25088 B LDS / 0 scratch`, `-21.90%`.
  - `bn192-wm128-wn96`: `137.71 tok/s`, `171 VGPR / 54 SGPR / 24064 B LDS / 784 B scratch`, `-85.86%`.
  - `bn256-wn128`: `659.02 tok/s`, `165 VGPR / 58 SGPR / 29696 B LDS / 0 scratch`, `-32.35%`.
  - `bn256-wm128`: `660.97 tok/s`, `165 VGPR / 43 SGPR / 29696 B LDS / 0 scratch`, `-32.15%`.
- Confidence: high for rejecting this route family; all valid large-N variants regressed far beyond noise.
- Recommendation: do not pursue larger-N warptile retuning inside the current `mul_mm.comp` topology. The next Q3_K route must change the data/layout topology enough to reduce repeated A-side work without increasing per-warp accumulator/live state this sharply.

## Notes

- Plain `BN192` is a useful negative-control shape but should not be run in the current shader because the A load loop does not evenly tile `BM=128`.
- The failed prediction was clear after measurement: reducing N tiles did reduce the static A-dequant proxy, but `WN=96/128` or `WM=128` increased live fragments/register pressure enough to dominate. The workflow correction is to treat `A pair dequant reduction` as insufficient unless paired with a VGPR/occupancy estimate or measured pipeline stats.
