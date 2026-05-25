# E229 ROCm rocBLAS solution-index scout

## Metadata

- Experiment ID: E229
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 GEMM-side route scout
- Target lane: Qwen3.6-27B Q3_K_S cold-first ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on
- Tool: `scripts/research/rocm_rocblas_solution_scout.cpp`

## Hypothesis

- Statement: the dominant Q3_K cuBLAS GEMM shapes may be using a non-optimal rocBLAS default solution on RDNA4, and an alternate `solution_index` could provide a high-ceiling cold-first route.
- Mechanism: E228 showed robust Q3_K split time is `76.82%` GEMM. rocBLAS logs emitted all dominant shapes with `solution_index 0`, while ROCm headers expose explicit solution-index APIs.
- Why now: E227 rejected no-code ngram/Vulkan cold gates, and E228 recentered the bottleneck on GEMM-side Q3_K work.

## Math / Theory

- E228 robust Q3_K split:
  - `(17408,5120,2048)`: `1131.560 ms` GEMM.
  - `(5120,17408,2048)`: `598.457 ms` GEMM.
  - `(17408,5120,1345)`: `262.333 ms` GEMM.
  - `(10240,5120,2048)`: `260.738 ms` GEMM.
  - `(6144,5120,2048)`: `165.594 ms` GEMM.
- Required route scale: a +20% cold wall gain from `7.8890 TPS` needs about `9.47 TPS`; a single GEMM-family tweak needs either a very large local gain on `17408x5120@2048` or a broader multi-shape win.
- Failure condition: if confirmed solution-index wins are only secondary-shape or under a few percent projected wall, do not add a runtime route.

## Implementation Plan

1. Minimal code surface to change:
   - Add a standalone research scout in `scripts/research/rocm_rocblas_solution_scout.cpp`.
   - Do not change `ggml` runtime unless the scout shows enough point-level ceiling.
2. Guard rails:
   - Compile the scout manually to `build_logs/agent-workload/rocm_rocblas_solution_scout.exe`.
   - Run GPU timing sequentially; parallel scout runs are invalid due GPU contention.
   - Treat `--check-solution-index` as validation only, not timing, because the check path can return near-zero timings.
3. Rollback path:
   - The runtime remains unchanged.
   - If the utility becomes stale, remove the standalone scout without touching server paths.

## Benchmark Plan

- Compile:
  - `hipcc -std=c++17 scripts/research/rocm_rocblas_solution_scout.cpp ... -lrocblas`
- Exact GEMM shapes:
  - `17408x5120@2048`
  - `5120x17408@2048`
  - `17408x5120@1345`
  - `10240x5120@2048`
  - `6144x5120@2048`
- Confirm only promising solution indices with sequential `warmup=8`, `iters=20`.
- Check rocBLAS metric/flags:
  - `rocblas_set_performance_metric`: `device`, `cu`
  - GEMM flags: `cu-efficiency`, `fp16-alt`

## Metrics

- isolated GEMM average ms
- relative to default isolated GEMM
- projected E228 trace savings
- runtime wall is not measured because the point ceiling does not justify a runtime patch

## Result

Confirmed sequential shortlist:

| Shape `(m,n,k)` | Default ms | Best confirmed candidate | Candidate ms | Relative | Decision |
| --- | ---: | --- | ---: | ---: | --- |
| `(17408,2048,5120)` | `3.3551` | `60014` | `3.3833` | `1.0084` | reject |
| `(5120,2048,17408)` | `3.1253` | `60017` | `2.5927` | `0.8296` | local win, low wall ceiling |
| `(17408,1345,5120)` | `2.5876` | `60014` | `2.3975` | `0.9265` | small local win |
| `(10240,2048,5120)` | `1.8834` | `59985` | `2.2851` | `1.2133` | reject |
| `(6144,2048,5120)` | `1.1330` | `59996` | `1.1228` | `0.9910` | tie |

Additional gates:

| Gate | Result |
| --- | --- |
| Wide short scan | Useful only as candidate discovery; not evidence because short warmup/iters exaggerated wins. |
| Parallel scout runs | Invalid for timing; GPU contention caused misleading huge wins. |
| `rocblas_set_performance_metric(device/cu)` | Regressed the two dominant main shapes. |
| `rocblas_gemm_flags_use_cu_efficiency` | Regressed the two dominant main shapes. |
| `rocblas_gemm_flags_fp16_alt_impl` | Regressed the two dominant main shapes. |
| `rocblas_gemm_flags_check_solution_index` | Validates solution availability but is not a timing path; observed near-zero timings. |

Projected trace savings from the confirmed wins are too small for the +20% cold target:

- `(5120,17408,2048)` local `0.8296x` on `598.457 ms` GEMM implies about `102 ms` saved.
- `(17408,5120,1345)` local `0.9265x` on `262.333 ms` GEMM implies about `19 ms` saved.
- Combined best-case before runtime overhead is about `121 ms` on a multi-second cold trace, roughly a low single-digit wall ceiling.

## Recommendation

- Reject rocBLAS solution-index override as the next +20% cold route.
- Keep the standalone scout utility for future driver/ROCm upgrades and exact-shape sanity checks.
- Do not add direct rocBLAS runtime plumbing yet; the runtime complexity and beta API exposure are not justified by the confirmed ceiling.
- Continue with a broader structural route. The next candidates should either:
  - change the Q3_K large-prefill body/topology beyond rocBLAS solution selection, or
  - stack the second hotspot (`GATED_DELTA_NET`) only if point timing shows a sizeable local win.

## Notes

- This experiment also updates the workflow rule: do not use `multi_tool_use.parallel` for GPU timing, even for isolated tools. It is fine for file reads, but it corrupts timing evidence.
- The local scout default ms can differ from llama trace ms because it isolates a warmed GEMM with synthetic buffers; only relative confirmed sequential runs are used for decisions.

## Artifacts

- `scripts/research/rocm_rocblas_solution_scout.cpp`
- `build_logs/agent-workload/e229-rocblas-solution-scout-17408x5120n2048-confirm-r1.csv`
- `build_logs/agent-workload/e229-rocblas-solution-scout-5120x17408n2048-confirm-r1.csv`
- `build_logs/agent-workload/e229-rocblas-solution-scout-17408x5120n1345-confirm-r1.csv`
- `build_logs/agent-workload/e229-rocblas-solution-scout-10240x5120n2048-seq-r2.csv`
- `build_logs/agent-workload/e229-rocblas-solution-scout-6144x5120n2048-seq-r2.csv`
- `build_logs/agent-workload/e229-rocblas-flags-cueff-17408x5120n2048-r1.csv`
- `build_logs/agent-workload/e229-rocblas-flags-fp16alt-5120x17408n2048-r1.csv`
