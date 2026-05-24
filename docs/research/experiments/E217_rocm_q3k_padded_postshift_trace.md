# E217 ROCm Q3_K Padded Post-Shift Trace

## Metadata

- Experiment ID: E217
- Date: 2026-05-24
- Owner: Codex
- Hypothesis ID: H42 / H43 boundary
- Target lane: ROCm `build-rocm-vec`, Qwen3.6-27B-Q3_K_S, active repo-snapshot lane

## Hypothesis

- Statement: after the H43 padded-storage opt-in route and E216 safety work, the next actionable bottleneck must be re-identified from point traces before attempting another kernel/body change.
- Mechanism: run the same cold-first lane with padded storage enabled and collect cuBLAS split, MMQ, and MMVQ timing/resource traces. If the large `ncols=2048` cuBLAS Q3_K family still dominates, H42 needs a new body/topology; if a smaller padded MMQ/MMVQ pocket dominates, continue H43.
- Why now: E212 validated the opt-in speed signal, while E214/E215 rejected nearby large-MMQ topology and src1 reuse branches. E216 changed correctness surface only, so this trace is the cheap gate before more code.

## Math / Theory

- Assumptions:
  - no speed claim from trace-heavy runs;
  - trace overhead makes wall TPS diagnostic only;
  - bottleneck classification should use per-route point sums, not aggregate TPS.
- Expected outcome:
  - a ranked point-time map for Q3_K cuBLAS/MMQ/MMVQ under the current opt-in route.
- Failure conditions:
  - server errors or corrupted output;
  - missing trace rows due to wrong env/route contract;
  - using a non-`build-rocm-vec` binary.

## Benchmark Plan

- Command: `scripts/agent_workload_bench.py` with `build-rocm-vec\bin\llama-server.exe`, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FA on, `--no-reuse`, no prime, thinking on, `runs=1`, `max-tokens=64`.
- Env:
  - `GGML_CUDA_Q3K_PADDED_STORAGE=1`
  - `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`
  - `GGML_TRACE_CUBLAS_SPLIT_DETAIL=1`
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING_PRE_SYNC=1`
  - `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`
  - `GGML_TRACE_MMQ_TIMING=1`
  - `GGML_TRACE_MMQ_TIMING_SYNC=1`
  - `GGML_TRACE_MMQ_TIMING_PRE_SYNC=1`
  - `GGML_TRACE_MMQ_RESOURCES=1`
  - `GGML_TRACE_MMVQ_TIMING=1`
  - `GGML_TRACE_MMVQ_TIMING_SYNC=1`
  - `GGML_TRACE_MMVQ_TIMING_PRE_SYNC=1`
  - `GGML_TRACE_MMVQ_RESOURCES=1`
- Artifacts path: `build_logs/agent-workload/e217-rocm-q3k-padded-postshift-*`.

## Metrics

- trace run errors/output sanity
- Q3_K cuBLAS split point sums by `row_diff/ne00/ncols`
- Q3_K MMQ point sums by `ncols_x/ncols_y`
- Q3_K MMVQ point sums by route/shape
- next-route decision

## Result

- Outcome: bottleneck classified. Under the current H43 padded-storage opt-in route, the active prompt-heavy lane is still dominated by large Q3_K cuBLAS GEMM-side prefill, while decode remains dominated by the existing Q3_K MMVQ fused/direct buckets.
- Delta: no speed claim. Trace run `e217-rocm-q3k-padded-postshift-r1` had errors `0`, prompt `1072.48 tok/s`, decode `29.98 tok/s`, and trace-heavy aggregate `6.9945 TPS`. Separate no-graph MMVQ trace `e217-rocm-q3k-padded-postshift-mmvq-nograph-r1` had errors `0`, prompt `1203.05 tok/s`, decode `12.06 tok/s`, and aggregate `3.5800 TPS`; this is diagnostic only because graphs are disabled.
- Confidence: medium-high for bottleneck ranking. cuBLAS split trace produced `1396` Q3_K rows; no-graph MMVQ trace produced `10817` timing rows.
- Recommendation: continue H42 only with a genuinely new large-Q3_K route body/topology. Do not spend more time on current-MMQ selectors, MMQ-X retile, src1 cache, helper-level pairdot, or packed-load tweaks. If no cheap H42 body is ready, continue H43 correctness/default-readiness rather than micro-polishing rejected mechanisms.

## Measured Data

cuBLAS Q3_K split timing, padded opt-in, robust view excluding startup outliers with `sum_ms >= 20`:

| Shape `(row_diff, ne00, ncols)` | Calls | Sum ms | src0 ms | src1 ms | GEMM ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(17408,5120,2048)` | 378 | `1415.111` | `184.753` | `82.721` | `1147.612` |
| `(5120,17408,2048)` | 189 | `790.205` | `89.776` | `96.468` | `603.970` |
| `(17408,5120,1345)` | 126 | `339.125` | `57.756` | `21.479` | `259.882` |
| `(10240,5120,2048)` | 143 | `337.651` | `45.988` | `29.351` | `262.323` |
| `(6144,5120,2048)` | 144 | `227.489` | `33.100` | `28.350` | `166.034` |

Key interpretation:

- `ffn_gate` and `ffn_up` account for all `17408x5120@2048` rows: `189 + 189` calls and `1415.111 ms`.
- In that top bucket, GEMM is `1147.612 / 1415.111 = 81.1%`; source staging is not the primary limiter anymore.
- A safe same-src1 cache or gate/up pairing would mostly save src1 staging (`~82.7 ms` in the top bucket), which is too small for a breakthrough unless the route also changes GEMM/body efficiency.

No-graph MMVQ timing, robust view excluding `total_ms >= 10`:

| Route bucket `(type,ncols_dst,small_k,fusion,ncols_x,grid_x,regs,max_blocks_per_sm)` | Calls | Sum ms |
| --- | ---: | ---: |
| `(q3_K,1,1,1,5120,8704,94,32)` | 1986 | `459.146` |
| `(q3_K,1,1,1,17408,2560,94,32)` | 1986 | `298.079` |
| `(q3_K,1,1,0,5120,5120,91,32)` | 1488 | `167.418` |
| `(q4_K,1,1,0,6144,640,62,8)` | 1488 | `142.401` |
| `(q3_K,1,1,0,5120,3072,91,32)` | 1488 | `142.113` |
| `(q3_K,1,1,0,5120,512,91,32)` | 992 | `71.153` |

Key interpretation:

- Decode remains a Q3_K MMVQ fused/direct body problem.
- The top Q3_K buckets are already at high occupancy; previous helper-level pairdot/load/preload variants regressed through register pressure or failed to move point time enough.
- Any decode continuation needs a structural layout/body change, not another small helper rewrite.

Artifacts:

- `build_logs/agent-workload/e217-rocm-q3k-padded-postshift-r1.server.log`
- `build_logs/agent-workload/e217-rocm-q3k-padded-postshift-r1.diagnostics.md`
- `build_logs/agent-workload/e217-rocm-q3k-padded-postshift-mmvq-nograph-r1.server.log`
- `build_logs/agent-workload/e217-rocm-q3k-padded-postshift-mmvq-nograph-r1.diagnostics.md`

## Notes

- This is a bottleneck map, not a speed candidate.
- Workflow correction: MMVQ point timing did not appear in the normal graph-enabled run; for decode point timing use a separate `GGML_CUDA_DISABLE_GRAPHS=1` diagnostic run. Do not use that run's wall TPS as a speed baseline.
