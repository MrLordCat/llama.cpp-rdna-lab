# E232 ROCm rocBLAS pair/batched GEMM gate

## Metadata

- Experiment ID: E232
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 route-body/topology gate
- Target lane: Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, cold/no reuse
- Binary/tool: standalone `build_logs/agent-workload/rocm_rocblas_pair_scout.exe`

## Hypothesis

- Statement: the dominant FFN gate/up family may benefit from running two same-shape GEMMs as one `rocblas_gemm_batched_ex` call instead of two separate `rocblas_gemm_ex` calls.
- Mechanism: gate/up matmuls often share the same activation matrix `B` and differ only in weight matrix `A`; a pair route could reduce dispatch overhead or allow rocBLAS to choose a better batched topology.
- Why now: E228 shows the hot Q3_K path is GEMM-side dominated, and E229 showed direct solution-index overrides do not help the dominant `17408x5120@2048` shape.

## Math / Theory

- The candidate has to beat two separate GEMMs by a meaningful margin before runtime work is justified.
- If batched GEMM is neutral, the wall ceiling is too low because current individual GEMMs are already large.
- If batched GEMM is worse, do not implement graph-level gate/up pairing in llama.cpp.

## Implementation

- Added `scripts/research/rocm_rocblas_pair_scout.cpp`.
- The scout compares:
  - `separate`: two sequential `rocblas_gemm_ex` calls, `A0 x B -> D0` and `A1 x B -> D1`.
  - `batched`: one `rocblas_gemm_batched_ex` call with `batch_count=2`, device pointer arrays, and shared `B` pointer.
- This is diagnostic-only and not part of the normal build.

## Results

| Shape `(m,n,k)` | Separate pair ms | Batched pair ms | Relative | Decision |
| --- | ---: | ---: | ---: | --- |
| `(17408,2048,5120)` | `6.2697` | `170.0158` | `27.1172x` slower | reject |
| `(17408,1345,5120)` | `4.6582` | `126.8518` | `27.2318x` slower | reject |

## Result

- Outcome: strong regression.
- Delta: batched rocBLAS is roughly `27x` slower than two separate default GEMMs on both tested hot/tail shapes.
- Confidence: high enough to reject this route. The effect is far beyond noise.
- Recommendation:
  - Do not pursue rocBLAS batched gate/up pairing for the ROCm cold lane.
  - Keep the standalone scout for future ROCm/driver checks, but avoid runtime integration.
  - Continue H42 only with a route body that changes real Q3_K storage/topology or a custom kernel; library-level pair batching is not the transfer path.

## Notes

- This closes a plausible but cheap structural route without touching llama.cpp runtime.
- The result also explains why launch-only FFN pairing is unlikely to produce the required +20% cold wall gain: the current hot GEMMs are compute-dominant and rocBLAS batched dispatch falls onto a much worse path.

## Artifacts

- `scripts/research/rocm_rocblas_pair_scout.cpp`
- `build_logs/agent-workload/e232-rocblas-pair-scout-17408x5120n2048-r1.csv`
- `build_logs/agent-workload/e232-rocblas-pair-scout-17408x5120n1345-r1.csv`
