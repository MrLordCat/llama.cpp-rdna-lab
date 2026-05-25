# E242 ROCm rocBLAS Concurrent Pair Gate

## Metadata

- Experiment ID: E242
- Date: 2026-05-25
- Owner: Codex
- Hypothesis ID: H42 route-body/topology gate
- Target lane: cold-first ROCm Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, no reuse, thinking on, `spec=none`
- Binary/tool: standalone `build_logs/agent-workload/rocm_rocblas_pair_scout.exe`

## Hypothesis

- Statement: FFN `gate/up` could run two independent same-shape rocBLAS GEMMs concurrently on separate streams instead of sequentially on one stream.
- Mechanism: `gate` and `up` share the same activation matrix but use different Q3_K weight tensors. If individual GEMMs underfill the GPU, separate rocBLAS handles/streams might overlap useful work without the bad `rocblas_gemm_batched_ex` path from E232 or the tall-output route from E238.
- Why now: E228/E217 show the main `17408x5120@2048` FFN gate/up family is the largest Q3_K prefill bucket. Before writing graph/runtime fusion, the standalone library point must show a clear win on the main bucket.

## Math / Theory

- Assumptions:
  - E217 main `17408x5120@2048` bucket: `1415.111 ms`, GEMM `1147.612 ms`.
  - E217 tail `17408x5120@1345` bucket: `339.125 ms`, GEMM `259.882 ms`.
  - The main bucket is much larger, so a tail-only win is below the cold +20% target and below the complexity threshold.
- Expected speedup corridor:
  - promote only if concurrent streams beat sequential on the main `n=2048` bucket by a clear margin.
  - if main ties/regresses, do not implement runtime graph pairing.
- Failure conditions:
  - main bucket neutral/slower;
  - win appears only on the tail bucket;
  - batched route remains very slow, confirming E232.

## Implementation Plan

1. Minimal code surface to change:
   - extend `scripts/research/rocm_rocblas_pair_scout.cpp` with a diagnostic-only `concurrent_streams` mode using two rocBLAS handles and two HIP streams.
2. Guard rails:
   - standalone only; no llama.cpp runtime change unless the main point gate wins.
   - compare against the same executable's sequential pair timing.
3. Rollback path:
   - keep the scout extension as diagnostic infrastructure; no runtime rollback needed.

## Benchmark Plan

- Compile:
  - `hipcc -std=c++17 scripts\research\rocm_rocblas_pair_scout.cpp -O2 -lrocblas -o build_logs\agent-workload\rocm_rocblas_pair_scout.exe`
- Candidate commands:
  - `build_logs\agent-workload\rocm_rocblas_pair_scout.exe --m 17408 --n 2048 --k 5120 --warmup 8 --iters 20`
  - `build_logs\agent-workload\rocm_rocblas_pair_scout.exe --m 17408 --n 1345 --k 5120 --warmup 8 --iters 20`
- Artifacts path:
  - `build_logs/agent-workload/e242-rocblas-concurrent-pair-*.csv`

## Metrics

- sequential pair average ms
- concurrent-stream pair average ms
- batched pair average ms
- relative to sequential pair

## Result

- Outcome: reject before runtime prototype.
- Delta:
  - main `17408x5120@n2048`: sequential `6.9594 ms`, concurrent `7.0088 ms` (`1.0071x`, slower), batched `153.1834 ms` (`22.0111x`, slower).
  - tail `17408x5120@n1345`: sequential `4.4483 ms`, concurrent `3.8136 ms` (`0.8573x`, faster), batched `114.5618 ms` (`25.7540x`, slower).
- Confidence: high enough to reject runtime work. The dominant main bucket does not improve, and the tail-only win is too small to move the cold wall after graph/runtime integration overhead.
- Recommendation:
  - do not implement concurrent rocBLAS gate/up graph fusion for the current cold lane;
  - keep the scout extension as a diagnostic tool;
  - continue H42 with a real Q3_K body/layout/topology change rather than library scheduling around already-saturated main GEMMs.

## Notes

- `python scripts/research/formula_sanity_checks.py` passed before the gate.
- This closes a different branch from E232 and E238:
  - E232 rejected `rocblas_gemm_batched_ex`;
  - E238 rejected one tall GEMM plus fused output handling;
  - E242 rejects two-stream concurrency because the main bucket is neutral/slower.
