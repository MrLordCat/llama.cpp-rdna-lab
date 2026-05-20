# E104 ROCm Q3_K Persistent Cache Probe

## Metadata

- Experiment ID: E104
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: master @ b6f114650 plus local E104 prototype, reverted after measurement
- Target lane: Qwen3.6-27B-Q3_K_S cold-first ROCm prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: a bounded persistent fp16 cache for repeated Q3_K staging can beat the current Q3_K -> fp16 -> hipBLAS route.
- Mechanism: E103 showed all `349` Q3_K tensor/range keys repeat `8` times; caching a hot subset should remove repeated `src0_convert_ms` without building a fused kernel.
- Why now: this is the lowest-code route after proving reuse, and it tests whether staging reuse is a practical route before investing in a fused RDNA4 kernel.

## Math / Theory

- Assumptions: E103 estimated unlimited cache savings of `2852.549 ms` but `42.002 GiB` fp16 footprint. The best ROI subset was `attn_gate`: about `1445.521 ms` potential saved conversion for `2880 MiB` across `48` keys.
- Expected speedup corridor: if `attn_gate` cache avoided most conversion without residency penalty, expected full-lane improvement was around `+1%` to `+2%`.
- Failure conditions: cache allocation/residency hurts GEMM/prompt eval more than conversion savings, or smaller cache windows miss too much reuse.

## Implementation Plan

1. Minimal code surface to change: prototype an opt-in `GGML_CUDA_Q3K_DEQUANT_CACHE` path inside `ggml_cuda_op_mul_mat_cublas` for Q3_K fp16 staging.
2. Guard rails: default off; pattern filter defaulted to `attn_gate`; max MiB limit; event-based readiness before cross-stream reuse.
3. Rollback path: remove the cache prototype if A/B is not positive.

## Benchmark Plan

- Baseline command: `e104-rocm-q3k-base-notrace-r1`, no trace, no cache.
- Candidate command: `GGML_CUDA_Q3K_DEQUANT_CACHE=1 GGML_CUDA_Q3K_DEQUANT_CACHE_PATTERN=attn_gate`, tested at `MAX_MIB=3072` and `MAX_MIB=480`.
- Number of runs: one-run gates.
- Artifacts path:
  - `build_logs/agent-workload/e104-rocm-q3k-base-notrace-r1.*`
  - `build_logs/agent-workload/e104-rocm-q3k-attngate-cache-trace-r1.*`
  - `build_logs/agent-workload/e104-rocm-q3k-attngate-cache-notrace-r1.*`
  - `build_logs/agent-workload/e104-rocm-q3k-attngate-cache480-notrace-r1.*`

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s
- Q3_K route cache hits and remaining `src0_convert_ms`
- fp16 cache footprint

## Result

- Outcome: regression; code reverted.
- Delta: no-trace baseline `11.74 TPS`; full `attn_gate` cache `9.56 TPS` (`-18.6%`); 480 MiB cache `11.59 TPS` (`-1.3%`). Diagnostic trace confirmed mechanics (`384` eligible calls, `336` hits, `src0_convert_ms` down from `3260.082 ms` to `1914.713 ms`) but wall time still regressed.
- Confidence: high enough to reject persistent fp16 cache for cold-first route work.
- Recommendation: do not keep or promote persistent Q3_K fp16 cache. Cache removes conversion, but fp16 residency/allocator pressure slows prompt eval more than the saved staging time.

## Notes

- Surprises: the full cache lowered traced conversion by about `1.35 s`, yet prompt eval dropped from `1207/1273 tok/s` to `863/936 tok/s`. Even the 480 MiB cache was slightly below baseline (`1185/1260 tok/s` vs `1207/1273 tok/s`).
- Follow-up action: avoid fp16 residency-based routes for this cold-first lane. If reuse is pursued again, it needs a graph-scheduling change with a tiny working set, not a process-wide cache.