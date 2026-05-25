# E228 ROCm cold Q3_K GEMM recenter

## Metadata

- Experiment ID: E228
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 route-body recenter / H43 post-default diagnostic
- Target lane: Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, thinking on, cold/no reuse
- Binary: `build-rocm-vec/bin/llama-server.exe`

## Hypothesis

- Statement: after H43 Q3_K padded storage became the HIP default, the cold-first bottleneck may have shifted enough that the next route should be chosen from fresh point timing rather than old E192 split ratios.
- Mechanism:
  - If `src0`/`src1` staging is still a large share, a storage or activation-route recapture could still have meaningful ceiling.
  - If GEMM dominates, the next +20% cold route must change the GEMM-side body/topology/solution selection or a larger graph route, not another staging cache.
- Why now: E227 showed ngram and Vulkan are not cold-first routes; the next cold work needs point-level evidence on the current default build and driver.

## Math / Theory

- Cold reference: E226 same-task cold-control r3, `7.8890 TPS`, prompt mean `5978.04 ms`, decode `30.45 tok/s`.
- +20% cold target: about `9.47 TPS`.
- Failure conditions:
  - Staging shares below roughly `15%` make a narrow fp16/cache recapture too small for the requested +20% wall gain.
  - If `MUL_MAT/q3_K` remains the majority of traced time, secondary routes such as GDN or FlashAttention are stack items unless they pair with a Q3_K route.

## Benchmark Plan

- Q3_K split trace:
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`
  - `GGML_TRACE_CUBLAS_SPLIT_DETAIL=1`
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING_PRE_SYNC=1`
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`
  - `GGML_TRACE_CUBLAS_Q3K_ROUTE=1`
  - `GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS=1024`
  - one cold `triage_diff` run, `--max-tokens 64`
- Full kernel trace:
  - `--trace-preset kernel-full`
  - one cold `triage_diff` run, `--max-tokens 16`
- rocBLAS log diagnostic:
  - `ROCBLAS_LAYER=2`
  - `ROCBLAS_LOG_BENCH_PATH=build_logs/agent-workload/e228-rocm12k-rocblas-bench.log`
  - one cold `triage_diff` run, `--max-tokens 1`

## Measured Results

### Q3_K cuBLAS split

Trace label: `e228-rocm12k-cold-triage-q3k-splittrace-r1`.

The run completed with `7.0592 TPS`, prompt `1077.01 tok/s`, decode `30.76 tok/s`, errors `0`. This is trace context only; the sync/detail logging is not a speed baseline.

| Scope | Calls | Total ms | src0 convert ms | src1 ms | GEMM ms | GEMM share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all Q3_K rows | `1396` | `4176.768` | `511.880` | `365.241` | `3291.521` | `78.81%` |
| robust rows, excluding `sum_ms >= 20` | `1395` | `3783.195` | `508.206` | `363.521` | `2906.403` | `76.82%` |

Top robust Q3_K shapes:

| Shape `(m,k,n)` | Calls | Total ms | src0 convert ms | src1 ms | GEMM ms | GEMM share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(17408,5120,2048)` | `378` | `1398.159` | `177.540` | `84.195` | `1131.560` | `80.9%` |
| `(5120,17408,2048)` | `189` | `786.054` | `89.144` | `98.414` | `598.457` | `76.1%` |
| `(17408,5120,1345)` | `126` | `342.887` | `58.865` | `21.673` | `262.333` | `76.5%` |
| `(10240,5120,2048)` | `143` | `337.528` | `45.262` | `31.492` | `260.738` | `77.2%` |
| `(6144,5120,2048)` | `144` | `230.931` | `32.537` | `32.763` | `165.594` | `71.7%` |

### Full kernel trace

Trace label: `e228-rocm12k-cold-triage-kernelfull-r1`.

The run completed with `2.0299 TPS`, prompt `1048.55 tok/s`, decode `23.82 tok/s`, errors `0`. This is trace context only.

`MUL_MAT` forward split:

| Route | Time ms | Share of MUL_MAT forward | Rows |
| --- | ---: | ---: | ---: |
| `cublas_backend/q3_K` | `3918.192` | `78.40%` | `1396` |
| `cublas_backend/f32` | `551.298` | `11.03%` | `784` |
| `cublas_backend/q4_K` | `262.057` | `5.24%` | `192` |
| `mul_mat_vec_q_direct/q3_K` | `199.216` | `3.99%` | `639` |

Top `CUDA_NODE` op shares:

| Op/src | Time ms | Share of traced total |
| --- | ---: | ---: |
| `MUL_MAT/q3_K` | `4159.323` | `54.16%` |
| `GATED_DELTA_NET/f32` | `998.759` | `13.01%` |
| `MUL_MAT/f32` | `591.938` | `7.71%` |
| `FLASH_ATTN_EXT/f32` | `340.385` | `4.43%` |
| `MUL_MAT/q4_K` | `282.544` | `3.68%` |

### rocBLAS route log

Diagnostic label: `e228-rocm12k-cold-triage-rocblaslog-r1`.

The rocBLAS bench log produced `2276` command rows and `27` unique command shapes. Top emitted GEMM commands all use f16 inputs, f32 output/compute, algorithm `0`, and `solution_index 0`.

Most common top shape:

```text
rocblas-bench -f gemm_ex --transposeA T --transposeB N -m 17408 -n 2048 -k 5120 --alpha 1 --a_type f16_r --lda 5120 --b_type f16_r --ldb 5120 --beta 0 --c_type f32_r --ldc 17408 --d_type f32_r --ldd 17408 --compute_type f32_r --algo 0 --solution_index 0 --flags 0
```

Header inspection found `rocblas_gemm_ex_get_solutions`, `rocblas_gemm_algo_solution_index`, and `rocblas_gemm_flags_check_solution_index` in the ROCm 7.1 rocBLAS headers. No `rocblas-bench.exe` was present in PATH or under the local ROCm install, so the next cheap route is a local solution-index scout rather than replaying the emitted commands directly.

## Result

- Outcome: diagnostic recenter, no speed claim.
- Delta: no candidate wall delta; traces are intentionally sync-heavy.
- Confidence: high for route direction. Q3_K cuBLAS and full kernel traces agree that cold-first time is still dominated by Q3_K `MUL_MAT`, and the Q3_K split is now mostly GEMM rather than staging.
- Recommendation:
  - Do not spend the next cycle on fp16 staging cache, src1 reuse, or selector-only current MMQ salvage.
  - Next high-ceiling ROCm cold candidate should probe GEMM-side solution/body/topology for the dominant `17408x5120@2048` and `5120x17408@2048` families.
  - Start with a default-off rocBLAS solution-index scout. Only if point timing moves should the runtime path be considered.

## Notes

- Compared with E192, H43/default and the updated driver changed the split enough that stale `src0_convert` ceilings are misleading.
- GDN is the second-largest traced op, but at `13.01%` it cannot alone satisfy the +20% cold target unless paired with the Q3_K core.
- Full kernel trace hints are generic and mention both prefill/decode; the parsed route shares are the source of truth for this decision.

## Artifacts

- `build_logs/agent-workload/e228-rocm12k-cold-triage-q3k-splittrace-r1.diagnostics.md`
- `build_logs/agent-workload/e228-rocm12k-cold-triage-q3k-splittrace-r1.server.log`
- `build_logs/agent-workload/e228-rocm12k-cold-triage-kernelfull-r1.diagnostics.md`
- `build_logs/agent-workload/e228-rocm12k-cold-triage-kernelfull-r1.server.log`
- `build_logs/agent-workload/e228-rocm12k-cold-triage-rocblaslog-r1.diagnostics.md`
- `build_logs/agent-workload/e228-rocm12k-rocblas-bench.log`
