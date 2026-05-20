# E105 ROCm Q3_K MMQ Route Probe

## Metadata

- Experiment ID: E105
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: master @ b6f114650 plus local E105 prototype, reverted after measurement
- Target lane: Qwen3.6-27B-Q3_K_S cold-first ROCm prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: a narrow route override to the existing direct-quant MMQ path can beat Q3_K fp16 staging for selected hot large-prefill tensors.
- Mechanism: global `GGML_CUDA_FORCE_MMQ_RUNTIME` was previously rejected, but a tensor-pattern/ncols-specific override might avoid the bad parts while removing Q3_K -> fp16 staging for hot shapes.
- Why now: E104 rejected persistent fp16 cache; existing MMQ is the only available direct-quant route before writing a new fused kernel.

## Math / Theory

- Assumptions: target hot shape is `Q3_K 5120x6144 @ ncols=2048` for `attn_gate`; tail chunks are around `ncols=1259/1278`.
- Expected speedup corridor: if MMQ direct route is competitive for selected shapes, it should improve prompt eval without increasing VRAM footprint.
- Failure conditions: existing MMQ is tuned for small/medium batches and loses to hipBLAS at these large ncols; tail-only wins are too small to move wall TPS.

## Implementation Plan

1. Minimal code surface to change: prototype opt-in `GGML_CUDA_Q3K_MMQ_ROUTE` in `ggml_cuda_mul_mat`, with pattern/min/max ncols filters and route trace.
2. Guard rails: default off; no global MMQ forcing; compare `attn_gate` all chunks, `attn_gate` tail chunks, and all-Q3_K tail chunks.
3. Rollback path: remove the override if A/B is not positive.

## Benchmark Plan

- Baseline command: `e104-rocm-q3k-base-notrace-r1`, no trace, default route.
- Candidate commands:
  - `e105-rocm-q3k-attngate-mmqroute-trace-r1`: `pattern=attn_gate`, `ncols>=1024`, with route trace.
  - `e105-rocm-q3k-attngate-mmqroute-tail-notrace-r1`: `pattern=attn_gate`, `1024<=ncols<=1280`.
  - `e105-rocm-q3k-alltail-mmqroute-notrace-r1`: `pattern=*`, `1024<=ncols<=1280`.
- Number of runs: one-run gates.
- Artifacts path:
  - `build_logs/agent-workload/e105-rocm-q3k-attngate-mmqroute-trace-r1.*`
  - `build_logs/agent-workload/e105-rocm-q3k-attngate-mmqroute-tail-notrace-r1.*`
  - `build_logs/agent-workload/e105-rocm-q3k-alltail-mmqroute-notrace-r1.*`

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s
- forced MMQ route count
- route correctness: forced calls must be `mul_mat_q_direct`, not cublas fallback

## Result

- Outcome: regression/tie; code reverted.
- Delta: baseline `11.74 TPS`; `attn_gate` all chunks with route trace `11.54 TPS` and `384` forced routes; `attn_gate` tail-only `11.68 TPS`; all-Q3_K tail-only `11.44 TPS`.
- Confidence: medium. One-run gates are enough to reject because no candidate cleared baseline and the mechanism is an already-known risky route family.
- Recommendation: do not implement shape-specific routing to existing MMQ for large Q3_K prefill. A new route needs a different fused kernel design, not selector tweaks to current MMQ.

## Notes

- Surprises: `attn_gate` tail-only was close to neutral, but still below same-session baseline and too small to justify code complexity. All-Q3_K tail routing was clearly worse.
- Follow-up action: fused Q3_K x F16 should target the `6144x5120@ncols2048`/tail family directly, likely with RDNA4 MFMA tiling and no persistent fp16 staging.