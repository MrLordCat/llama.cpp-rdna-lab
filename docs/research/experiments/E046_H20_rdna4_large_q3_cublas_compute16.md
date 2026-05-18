# E046 H20 RDNA4 Large Q3_K cuBLAS Compute16 Route

## Metadata

- Experiment ID: E046
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: forcing `CUBLAS_COMPUTE_16F` for the large RDNA4 quantized cuBLAS route may speed up the current Qwen prefill lane.
- Mechanism: E045 shows large `Q3_K` prefill now routes through `cublas_backend`: quantized weights are dequantized to fp16, activations are converted to fp16, and RDNA4 currently forces fp32 accumulation. The existing `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1` knob switches that branch to fp16 accumulation plus a final fp16-to-fp32 output conversion.
- Why now: broad MMQ force, GDN chunking, GDN fast-exp, and adjacent batch/ubatch sweeps did not beat the `ubatch=2048` baseline. The remaining dominant prefill hotspot is large `MUL_MAT`, mostly `Q3_K`.

## Math / Theory

- Assumptions:
  - E045 `ubatch=2048` prompt trace total: `13969.454 ms`.
  - `MUL_MAT`: `9053.320 ms` (`64.81%` of prompt trace).
  - `Q3_K` share inside `MUL_MAT`: `84.32%`, about `7634 ms` or `54.65%` of prompt trace.
  - Aggregate r3 baseline: `11.6534 TPS`; prompt eval `1197.5567 tok/s`.
- Expected speedup corridor:
  - If only the Q3_K cuBLAS part improves by `5%`, prompt trace improves by about `2.73%`.
  - If only the Q3_K cuBLAS part improves by `10%`, prompt trace improves by about `5.47%`.
  - End-to-end wall gain should be smaller because decode is still present; a useful r1 signal should be roughly `>= +1%` aggregate TPS or a clear prompt-eval gain.
- Failure conditions:
  - Extra `dst_f16 -> dst_f32` conversion costs more than the fp32-accumulation savings.
  - Lower accumulation precision changes output enough to make the setting unsuitable as a default even if it is faster.
  - rocBLAS/hipBLAS chooses an unfavorable fp16 route on RDNA4.

## Implementation Plan

1. Minimal code surface to change: none for the first gate; use existing `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1`.
2. Guard rails: keep the candidate env-gated unless r3 confirms a meaningful speedup and no obvious output/runtime instability appears.
3. Rollback path: unset `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F`; no code rollback needed for the first gate.

## Benchmark Plan

- Baseline command:
  - `python scripts/agent_workload_bench.py --label prefill-e046-cublas16-base-r1 ... --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"`
- Candidate command:
  - same command with `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1`.
- Number of runs:
  - r1 gate, r3 confirmation only if r1 is positive.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e046-cublas16-*.csv`
  - `build_logs/agent-workload/prefill-e046-cublas16-*.server.log`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- output stability smoke via benchmark errors and response length

## Result

- Outcome: negative.
- Baseline: `prefill-e046-cublas16-base-r1 = 11.7908 TPS`, prompt eval `1205.145 tok/s`, prompt mean `6155.665 ms`, decode eval `30.08 tok/s`.
- Candidate: `prefill-e046-cublas16-force16-r1 = 11.4146 TPS`, prompt eval `1145.945 tok/s`, prompt mean `6475.760 ms`, decode eval `29.97 tok/s`.
- Delta: aggregate `-3.19%`; prompt eval `-4.91%`; prompt time `+5.20%` slower.
- Confidence: r1 is enough to reject because the signal is large and aligned with prompt eval.
- Recommendation: reject `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1` for this lane; keep RDNA4 large quantized cuBLAS on the current fp32-accumulation path.

## Notes

- Surprises: the expected fp16-accumulation throughput gain did not appear; the extra fp16 output path and/or rocBLAS kernel choice is slower on this shape.
- Follow-up action: move to split timing of the large `cublas_backend` path before writing another GEMM-route change.
