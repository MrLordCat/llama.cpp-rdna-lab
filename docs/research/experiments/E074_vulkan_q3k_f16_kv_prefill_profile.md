# E074 Vulkan Q3_K F16 KV Prefill Profile

## Metadata

- Experiment ID: E074
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master after Q4 ROCm fix `172fa02c8`, E071-E073 docs pending
- Target lane: Qwen3.6-27B-Q3_K_S, ctx=12288, b=4096, ub=1024, FlashAttention on, thinking on, no reuse, `--spec-type none`

## Hypothesis

- Statement: On the 12k Q3 prompt-heavy lane, switching KV from q4_0/q4_0 to f16/f16 can significantly improve raw prefill on Vulkan because FlashAttention is the second largest measured block after Q3_K MMQ.
- Mechanism: q4_0 KV saves VRAM and is good for large contexts, but at ctx=12288 the Q3_K_S model fits in 16GB with f16 KV. Wider KV avoids compressed-KV attention overhead and improves prompt eval; the cost is higher VRAM and output-length differences that make wall TPS less comparable.
- Why now: E071-E073 showed tile and Q3_K MMQ shader micro-optimizations did not produce a significant raw prompt win, while perf logger showed FlashAttention remained a large non-MMQ component.

## Math / Theory

- Assumptions: ctx=12288 plus Q3_K_S weights fit under the RX 9070 XT 16GB budget with f16 KV when other GPU apps are quiet; prompt eval, not total completion TPS, is the target metric.
- Expected speedup corridor: +3% to +7% raw prompt vs Vulkan q4_0 KV; potentially smaller but positive vs ROCm with the same f16 KV.
- Failure conditions: VRAM pressure forces partial offload or paging, f16 KV changes generation behavior enough to be undesirable for some workflows, or long-context runs no longer fit.

## Implementation Plan

1. Minimal code surface to change: no backend code; update GUI Q3_K_S speed preset and shared active profile to use f16 KV at ctx=12288.
2. Guard rails: keep notes explicit that q4_0 KV is the fallback when memory is tight; keep long-context claims out of this preset.
3. Rollback path: set `kv_cache` back to q4_0 index `7` if users need the older memory-saving default.

## Benchmark Plan

- Baseline command: E071 Vulkan `wm32-wn32`, q4_0/q4_0 KV, prompt `1165.33 tok/s`; ROCm q4_0/q4_0 prompt `1172.4467 tok/s`.
- Candidate command: same Vulkan lane with f16/f16 KV and `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32`.
- Control command: same ROCm lane with f16/f16 KV.
- Number of runs: 3 for Vulkan f16 and ROCm f16 confirmation.
- Artifacts path: `build_logs/agent-workload/e073-vulkan-wm32-wn32-kvf16-r3-*`, `build_logs/agent-workload/e073-rocm-kvf16-r3-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split

## Result

- Outcome: win for raw prefill profile
- Delta: Vulkan f16 KV reached `1230.7333` prompt tok/s over 3 runs, vs Vulkan q4_0 KV `1165.33` (`+5.6%`), ROCm q4_0 KV `1172.4467` (`+5.0%`), and ROCm f16 KV `1194.22` (`+3.1%`). q8_0 KV was a regression at `1127.86` prompt tok/s. bf16 KV failed on the Vulkan control run.
- Confidence: medium-high for raw prompt at ctx=12288; low for wall TPS because f16 KV changed output length on the benchmark task.
- Recommendation: keep f16 KV in the GUI Q3_K_S 12k speed preset with a clear VRAM caveat. Do not present this as a long-context or wall-TPS universal win.

## Notes

- Vulkan f16 completion length varied (`3`, `35`, `7` tokens) despite no benchmark errors, so aggregate wall TPS is intentionally not used as the primary metric.
- q4_0 KV remains the safer fallback for constrained VRAM or longer contexts.