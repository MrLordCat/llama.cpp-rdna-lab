# E075 Vulkan 32k GUI Prefill Runtime Profile

## Metadata

- Experiment ID: E075
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master, working tree after `fda68fb0b`
- Target lane: Qwen3.6-27B-Q3_K_S, GUI autotune 32k lane, RX 9070 XT, Vulkan vs ROCm

## Hypothesis

- Statement: The fresh GUI Vulkan 32k result was slow because GUI autotune launched Vulkan without the E068 `wm32-wn32` runtime profile.
- Mechanism: `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` enables the AMD large matmul path on the proprietary driver, and `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32` reduces Q3_K large-MMQ tile pressure for prompt chunks.
- Why now: User ran matching GUI autotunes and saw Vulkan `8.1139 TPS` vs ROCm `11.0606 TPS` on the same 32k config.

## Math / Theory

- Assumptions: GUI score is generated-token wall TPS, so prefill dominates at 7673 prompt tokens and 120 decode tokens.
- Expected speedup corridor: E068 12k profile suggested a large prompt gain; for 32k a `+30%` to `+50%` Vulkan wall gain was plausible if the same large-MMQ path was active.
- Failure conditions: If 32k used a different Vulkan kernel route or the larger context made attention/KV the dominant bottleneck, the tile env would not recover prefill.

## Benchmark Plan

- Baseline command: same as GUI autotune lane, without Vulkan env overrides.
- Candidate command: same command with `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` and `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32`.
- Number of runs: 1 quick gate plus 3-run confirmation for the winning candidate.
- Artifacts path: `build_logs/agent-workload/e075-vulkan32k-gui-*.diagnostics.md` and matching server logs.

## Metrics

| Run | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Prompt tokens | Decode tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| ROCm GUI fresh control | `11.0606` | `1155.52` | `28.83` | `7673` | `120` |
| Vulkan GUI baseline | `8.1791` | `659.49` | `40.05` | `7673` | `120` |
| Vulkan `wm32-wn32` r1 | `12.1414` | `1121.21` | `40.01` | `7673` | `120` |
| Vulkan `wm32-wn32` r3 | `12.6420` | `1163.96` | `42.22` | `7673` | `120` |
| Vulkan current base retest | `4.1109` | `358.77` | `15.49` | `7673` | `120` |
| Vulkan repaired `wm32-wn32` | `2.2151` | `165.29` | `15.57` | `7673` | `120` |
| Vulkan `wm128-wn32` probe | `4.3572` | `389.01` | `15.46` | `7673` | `120` |

## Result

- Outcome: original performance win rejected; root cause found; exact `wm32-wn32` can be made correct only as a slow profile.
- Delta: Vulkan candidate r3 is `+54.6%` vs Vulkan baseline and `+14.3%` vs fresh ROCm GUI wall TPS. Raw prefill is now effectively tied/slightly ahead of the ROCm r1 control (`1163.96` vs `1155.52 tok/s`, `+0.7%`), and decode remains much faster (`42.22` vs `28.83 tok/s`).
- Correctness follow-up: live `/v1/chat/completions` and `/completion` on the same model/server shape produced all-slash output when `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32` was set (`64/64` reasoning slashes and `32/32` raw completion slashes). Vulkan without the variant, Vulkan with only `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, and ROCm control all produced non-slash text on the same prompts.
- Root cause: the old `wm32-wn32` tile became `[256,128,128,32,32,32,2,16,16,16,64]`. In the coopmat `mul_mm.comp` path, `BM=BN=128`, `WM=WN=32`, and `WARP=64` require `64 * (128/32) * (128/32) = 1024` workgroup invocations, but the profile kept `BLOCK_SIZE=256`. Only 4 subgroups were launched where 16 were needed, leaving most output columns uncomputed. The speedup was therefore invalid work, not just a numerical issue.
- Repair attempt: reducing the honest tile to `BM=BN=64`, `WM=WN=32`, `BLOCK_SIZE=256` fixes the slash-spam smoke (`0/216` slashes on the long raw prompt, short arithmetic returns `4`), but the 32k E075-shape run falls to `2.2151 TPS` and `165.29 prompt tok/s`.
- Nearby valid probe: `wm128-wn32` also passes the same smoke and gives a small same-session gain over current base (`4.3572` vs `4.1109` wall TPS, `389.01` vs `358.77` prompt tok/s), but it does not recover the old rejected `12+ TPS` result or ROCm-level 32k throughput.
- Confidence: high that original `wm32-wn32` must not be restored as a GUI default. The backend now guards/fixes the manual variant so it does not silently produce corrupt text, but GUI should keep only `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` unless a correct profile is confirmed with both speed and generation smoke.
- Recommendation: keep exact `wm32-wn32` rejected for performance promotion; optionally continue from `wm128-wn32`/other valid coopmat geometries as a new H31 follow-up, using correctness smoke before any benchmark claim.

## Notes

- Manual shell runs of the MinGW Vulkan binary require `C:/Strawberry/c/bin` before incompatible MinGW runtime DLLs in `PATH`; GUI launches already worked, but this matters for reproducible terminal A/B.
- `ngram-mod` generated no draft tokens in the r1 gate and only 2 drafts by the end of the r3 run, so the delta is not speculative acceptance noise.
- E076 follow-up clarified that the low `Vulkan current base retest` in this note lacked the safe force-only env. The current honest safe baseline is about `9.85 TPS` / `907 tok/s` prompt with `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` only; valid no-code follow-ups still did not reach the fresh ROCm 32k control.
- Follow-up artifacts: `build_logs/agent-workload/e075-vulkan32k-gui-wm32wn32-fixed-r1-e075shape.diagnostics.md`, `build_logs/agent-workload/e075-vulkan32k-gui-wm128wn32-r1-e075shape.diagnostics.md`, `build_logs/agent-workload/e075-vulkan32k-gui-base-current-r1-e075shape.diagnostics.md`.