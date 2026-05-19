# E071 Vulkan Q3_K Prefill Tile Retarget

## Metadata

- Experiment ID: E071
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master after Q4 ROCm fix `172fa02c8`
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, ctx=12288, b=4096, ub=1024, q4_0/q4_0 KV, FlashAttention on, thinking on, no reuse, `--spec-type none`

## Hypothesis

- Statement: For raw Q3_K Vulkan prefill, the existing RDNA4 large-MMQ tile variant `wn16` may beat the E068 wall-best `wm32-wn32` on the active prompt-heavy lane.
- Mechanism: E068 selected `wm32-wn32` by total wall TPS, but the pp7488 gate had `wn16` slightly ahead. Q3_K large `MUL_MAT` dominates prompt time; a smaller N tile can reduce register/LDS pressure enough to improve prompt eval even if decode or overhead changes.
- Why now: The user shifted the target back from Q4 practicality to Q3 Vulkan prefill and explicitly wants internet/codebase research before new changes. External PRs #22951/#23056 are already represented locally for MMVQ, #22970 is mostly Q4/Q5/Q6 A-transpose, and #21024 repack is mixed or regressive on RX 9070 XT prompt tests.

## Math / Theory

- Assumptions: Prompt eval is dominated by Q3_K `MUL_MAT` chunks with `n=1024`; E068 tile knobs affect those MMQ kernels without changing correctness.
- Expected speedup corridor: +2% to +8% prompt eval if `wn16` lowers occupancy/register pressure; no-code result could be enough for a GUI/autotune profile hint.
- Failure conditions: The pp7488 advantage was noise, full active-lane prompt uses a different mix, or wall overhead/decode offsets the prefill gain.

## Implementation Plan

1. Minimal code surface to change: none for the first pass; use existing `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` and `GGML_VK_AMD_LARGE_MATMUL_VARIANT`. A temporary source patch added extra WM/WN variants after the no-code pass, then was reverted after active-lane gates showed no win.
2. Guard rails: compare only clean no-reuse/no-prime active lane against E068 `wm32-wn32` and same-session ROCm prompt baseline.
3. Rollback path: unset `GGML_VK_AMD_LARGE_MATMUL_VARIANT`; no source rollback needed.

## Benchmark Plan

- Baseline command: E068 `wm32-wn32` active lane, `--runs 1` for refresh if needed.
- Candidate command: same active lane with `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wn16`; optionally gate nearby variants only if `wn16` is promising.
- Number of runs: 1 for iteration, 3 only if prompt eval approaches or beats ROCm `1173.2367 tok/s`.
- Artifacts path: `build_logs/agent-workload/e071-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split

## Result

- Outcome: near miss / no new source kept
- Delta: Fresh `wm32-wn32` refresh reached `1165.33` prompt tok/s and `7.9505` wall TPS over 3 runs, vs fresh ROCm `1172.45` prompt tok/s and `7.3832` wall TPS. Vulkan is now `+7.7%` wall from faster decode, but still `-0.6%` prompt eval vs ROCm. `wn16` reached `1155.97` prompt tok/s over 3 runs. Temporary `wm16-wn32` and `wm32-wn16` source variants improved pp7488 gates but active-lane prompt stayed around `1162.91` and `1164.98` tok/s.
- Confidence: medium-high for rejecting the extra tile variants; synthetic pp gains did not transfer to active prompt-heavy lane, and the best current profile remains E068 `wm32-wn32`.
- Recommendation: keep using `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` + `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32` for Q3 Vulkan. Do not keep the temporary extra WM/WN variants. To significantly exceed ROCm raw prefill, move beyond tile constants into Q3_K `mul_mm`/dequant or activation-side work.

## Notes

- Surprises: Fresh E071 runs are higher than E068 despite the same source path: `wm32-wn32` prompt moved from `1110.09` to `1165.33` tok/s, while ROCm refresh stayed effectively unchanged (`1172.45` vs old `1173.24`). The first E071 `wn16` run without `--real-context-mode repo-snapshot` produced a short invalid `35.58 TPS` artifact and is discarded.
- Follow-up action: Source-level work should not repeat E067's simple Q3_K packed32 `mul_mm` load rewrite. The next useful probe is instrumentation or a narrowly guarded Q3_K matmul/dequant micro-optimization that explains the remaining ~0.6% prompt gap first.

## Key Measurements

| Config | Runs | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| ROCm refresh | 3 | `7.3832` | `1172.4467` | `28.6967` | current fair prompt target |
| Vulkan `wm32-wn32` refresh | 3 | `7.9505` | `1165.33` | `40.3967` | best kept Vulkan profile |
| Vulkan `wn16` refresh | 3 | `7.8972` | `1155.9667` | `40.36` | no-code candidate, below `wm32-wn32` |
| Vulkan temp `wm16-wn32` | 1 | `7.9327` | `1162.91` | `40.32` | source variant rejected |
| Vulkan temp `wm32-wn16` | 1 | `7.9415` | `1164.98` | `40.30` | source variant rejected |

Important pp7488 gates after the temporary variant patch:

| Variant | pp7488 tok/s |
| --- | ---: |
| `wm32-wn32` | `1128.62` |
| `wn16` | `1122.51` |
| `wm32-wn16` | `1145.87` |
| `wm64-wn16` | `1120.58` |
| `wm64-wn32` | `1050.50` |
| `wm16-wn32` | `1155.74` |
| `wm16-wn64` | `1121.48` |

Artifacts:

- `build_logs/agent-workload/e071-vulkan-wm32-wn32-refresh-r3-reposnapshot-b4096-ub1024-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e071-vulkan-wn16-r3-reposnapshot-b4096-ub1024-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e071-rocm-refresh-r3-reposnapshot-b4096-ub1024-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e071-vulkan-wm-wn-new-variants-pp7488.md`
- `build_logs/agent-workload/e071-vulkan-wm16-wn32-r1-reposnapshot-b4096-ub1024-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e071-vulkan-wm32-wn16-r1-reposnapshot-b4096-ub1024-ctx12288-q3ks.diagnostics.md`