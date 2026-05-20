# E076 Vulkan 32k Valid Prefill Follow-up

## Metadata

- Experiment ID: E076
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: master, working tree after E075 guard work
- Target lane: Qwen3.6-27B-Q3_K_S, GUI 32k lane, RX 9070 XT, Vulkan

## Hypothesis

- Statement: After rejecting the corrupt E075 `wm32-wn32` shortcut, a correct Vulkan 32k prefill win might still exist in upstream leads, valid coopmat geometry, MMVQ routing, batch/ubatch shape, or KV cache format.
- Mechanism: Q3_K prompt chunks are dominated by Vulkan large matmul / MMQ work; legal warptile retargeting or compressed-KV changes could reduce prefill time without corrupting logits.
- Why now: E075 proved throughput-only benchmarks can be misleading, so the follow-up must search only among correctness-plausible paths.

## Math / Theory

- Assumptions: This lane has about 7673 prompt tokens and 120 generated tokens. Wall TPS is very sensitive to prefill, but decode remains visible because Vulkan decode is strong.
- Expected speedup corridor: A valid tile or route knob would need roughly `+15%` to `+25%` prompt eval to close the gap to the fresh ROCm 32k control (`1155.52 tok/s` prompt, `11.0606 TPS` wall).
- Failure conditions: Any tile shape that violates `BLOCK_SIZE / WARP == (BM / WM) * (BN / WN)` is invalid even if it benchmarks fast. KV formats that improve prompt but hurt decode can still lose wall TPS.

## Implementation Plan

1. Minimal code surface to change: keep GUI at safe `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`; extend backend validation so all manual AMD large-matmul variants are retiled/fallen back instead of silently under-covering coopmat output.
2. Guard rails: compare against same-session safe force-only baseline; use only `--runs 1` gates unless a candidate beats baseline; require real generation smoke before any future promotion.
3. Rollback path: disable any candidate env; if backend guard regresses base, revert the guard patch.

## Benchmark Plan

- Baseline command: `scripts/agent_workload_bench.py --server-bin build-vulkan/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks v2-mini --runs 1 --ctx-size 32768 --batch-size 5120 --ubatch-size 1024 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --gpu-layers 999 --real-context-mode repo-snapshot --real-context-chars 21872 --no-disable-thinking --no-reuse --allow-ctx-above-16k --server-extra "--spec-type ngram-mod" --max-tokens 120 --background-server-policy fail`
- Candidate command: same command with candidate env or batch/KV changes.
- Number of runs: 1 per gate; no 3-run confirmation because no candidate beat baseline.
- Artifacts path: `build_logs/agent-workload/e076-vulkan32k-*.diagnostics.md` and matching server logs.

## Metrics

### External / Local Code Cross-check

- `ggml-org/llama.cpp#23056`: Q3_K/Q6_K 4-byte block loads and 32-bit subtract are already present in `mul_mat_vecq_funcs.glsl`; do not re-test as a new candidate.
- `ggml-org/llama.cpp#22970`: transpose-A work remains mostly Q4_K/Q5_K/Q6_K, already rejected for this Q3_K_S lane in E063.
- `ggml-org/llama.cpp#21024`: broad Vulkan repack PoC has mixed AMD results and no direct Q3_K_S 32k win to port blindly.
- `ggml-org/llama.cpp#23106`: AMD coopmat large `MUL_MAT_ID` was disabled upstream due regressions; not a safe promotion path.

### Same-session Gates

| Gate | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Decision |
| --- | ---: | ---: | ---: | --- |
| Safe force-only base | `9.8493` | `907.02` | `32.59` | baseline |
| `wm128-wn32` | `9.6075` | `870.91` | `32.97` | reject |
| `block128-wm128` | `9.0091` | `792.98` | `33.32` | reject |
| `block128-bn64` | `7.5254` | `624.78` | `33.10` | reject |
| `GGML_VK_DISABLE_MMVQ=1` | `9.3738` | `876.90` | `29.92` | reject |
| `wm128-wn32 + DISABLE_MMVQ` | `9.4646` | `872.35` | `31.27` | reject |
| `b4096/ub1024` | `9.8353` | `905.08` | `32.58` | tie/reject |
| `b4096/ub896` | `9.6864` | `881.63` | `33.11` | reject |
| `b6144/ub1024` | `9.7029` | `883.55` | `32.97` | reject |
| `b5120/ub896` | `9.6107` | `871.46` | `32.98` | reject |
| `q8_0/q8_0` KV | `9.1102` | `865.69` | `28.12` | reject |
| `f16/f16` KV | `8.8361` | `937.74` | `22.40` | reject; prompt up, decode down |
| Post-guard safe force-only base | `9.8389` | `900.12` | `33.08` | no regression |
| Post-guard `wn16` | `4.7196` | `781.57` | `7.71` | reject |

Fresh ROCm 32k control from E075 remains `11.0606 TPS`, prompt `1155.52 tok/s`, decode `28.83 tok/s`.

## Result

- Outcome: no-code/env/shape gates rejected; backend guard kept.
- Delta: best candidate was effectively a tie (`b4096/ub1024`, `9.8353` vs `9.8493`, `-0.14%`) and did not improve prompt eval. `f16` KV improved prompt eval to `937.74 tok/s` but decode fell to `22.40 tok/s`, reducing wall TPS.
- Confidence: high that the remaining 32k Vulkan gap is not from current easy knobs. It is a kernel-level Q3_K prefill problem, likely in large matmul/MMQ/dequant/layout rather than batch/ubatch/KV selection.
- Recommendation: keep GUI Vulkan safe force-only profile. Continue H31 only with deeper source-level profiling and correctness smoke; do not revive E075-style invalid tiles.

## Notes

- The backend now validates/prepares all manual AMD large-matmul variants, not just exact `wm32-wn32`, so variants such as `wn16` cannot silently launch an under-covered coopmat tile.
- The post-guard base result (`9.8389 TPS`) confirms the guard does not materially change the safe profile.
- `ngram-mod` had negligible draft coverage in these gates, so the measurements mostly reflect backend prefill/decode behavior rather than speculative acceptance.