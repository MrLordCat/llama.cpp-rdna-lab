# E192 ROCm Q3_K cuBLAS Split Recapture

## Metadata

- Experiment ID: E192
- Date: 2026-05-23
- Owner: Codex
- Branch/Commit: master after `3080b223c`
- Hypothesis ID: H35 active ROCm Q3_K new-route track, informed by H39 route-chain
- Target lane: L1 ROCm real-context prompt-heavy lane, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: E191 shows the practical real-context lane is now dominated by large-N Q3_K `cublas_backend` prefill routes, so the next high-ceiling branch should refresh H35 split evidence before any fused Q3_K x F16 prototype.
- Mechanism: enable default-off cuBLAS split tracing and Q3_K route tracing to measure current `src0_convert_ms`, `src1_ms`, `gemm_ms`, repeated key counts, and top shape families.
- Why now: E190 rejected MMVQ pair-dot for the decode side; E191 found `cublas_backend q3_K` at `78.70%` of parsed `MUL_MAT forward` time on the real-context trace. This is too large to ignore for wall TPS.

## Math / Theory

- Assumptions:
  - E191 real-context trace: parsed `MUL_MAT forward` `4944.803 ms`; `cublas_backend q3_K` `3891.530 ms` (`78.70%`).
  - E106 historical split trace: Q3_K `src0_convert_ms=3257.251`, `gemm_ms=6107.363`, but stage timing can include queued work because it lacks a pre-stage sync.
- Expected speedup corridor:
  - If current Q3_K conversion/layout is still around `10%` effective wall share, a `20-25%` local route win can produce about `+1.7%..+2.1%` wall.
  - A fused route is only worth coding if it can remove both repeated `src0` staging and enough GEMM/conversion overhead without a persistent fp16 cache.
- Failure conditions:
  - current split trace shows conversion share too small or mostly non-repeated;
  - top shapes differ from the E103/E106 family enough that old H35 plans are stale;
  - trace is contaminated by background server or override env.

## Implementation Plan

1. Minimal code surface to change: none for E192.
2. Guard rails:
   - do not force existing MMQ; E105 rejected that route;
   - do not use persistent fp16 cache; E104 rejected memory/capacity tradeoff;
   - use split trace as structure and budget evidence, not a speed claim.
3. Rollback path: diagnostic-only.

## Benchmark Plan

- Baseline command: E191 clean real-context trace.
- Candidate command: current-tree split trace with `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`, `GGML_TRACE_CUBLAS_SPLIT_DETAIL=1`, `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`, `GGML_TRACE_CUBLAS_Q3K_ROUTE=1`, `GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS=1024`.
- Number of runs: 1 diagnostic trace.
- Artifacts path: `build_logs/agent-workload/e192-rocm-q3k-cublas-split-r1.*`.

## Metrics

- aggregate completion TPS (diagnostic only)
- prompt/decode split
- Q3_K split calls and unique/repeated keys
- `src0_convert_ms`, `src1_ms`, `gemm_ms`, `sum_ms`
- top shape families by ncols/row range

## Result

- Outcome: diagnostic keep; H35 remains the high-ceiling practical wall route.
- Delta: no candidate. Split trace measured Q3_K cuBLAS split total `5213.358 ms`, with `src0_convert_ms=1637.070`, `src1_ms=364.309`, and `gemm_ms=3203.883`.
- Confidence: medium-high for route mix and repeated-key structure; medium for absolute stage attribution because split timing still lacks a pre-stage sync.
- Recommendation: do not repeat persistent fp16 cache, existing MMQ forcing, compute16, hipBLASLt, or nearby ubatch sweeps. The only credible next code branch is a new non-persistent fused/direct Q3_K x F16 route, or a graph-scheduling change that avoids repeated source staging without keeping a large fp16 cache resident.

## Measured Data

Artifact: `build_logs/agent-workload/e192-rocm-q3k-cublas-split-r1.server.log`.

Server summary:

| Metric | Value |
| --- | ---: |
| aggregate completion TPS | `7.3569` |
| prompt eval TPS | `1124.56` |
| decode eval TPS | `31.89` |
| prompt eval ms | `6659.48` |
| decode eval ms | `2006.7` |
| task prompt tokens | `7489` |
| errors | `0` |

Split totals:

| Source type | Calls | Sum | Src0 convert | Src0 total | Src1 total | GEMM | Dst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | `2228` | `6503.024 ms` | `1711.676 ms` | `1719.783 ms` | `411.042 ms` | `4372.180 ms` | `0.000 ms` |
| `q3_K` | `1396` | `5213.358 ms` | `1637.070 ms` | `1645.155 ms` | `364.309 ms` | `3203.883 ms` | `0.000 ms` |
| `q4_K` | `192` | `329.378 ms` | `74.606 ms` | `74.628 ms` | `46.733 ms` | `208.009 ms` | `0.000 ms` |
| `f32` | `640` | `960.288 ms` | `0.000 ms` | `0.000 ms` | `0.000 ms` | `960.288 ms` | `0.000 ms` |

Q3_K split by prompt chunk size:

| ncols | Calls | Sum | Src0 convert | GEMM | Src1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `2048` | `1047` | `4276.212 ms` | `1317.022 ms` | `2660.724 ms` | `290.419 ms` |
| `1345` | `349` | `937.146 ms` | `320.048 ms` | `543.159 ms` | `73.890 ms` |

Top Q3_K shapes:

| Shape | ncols | Calls | Sum | Src0 convert | GEMM | Src1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(17408 x 5120)` | `2048` | `378` | `1425.411 ms` | `233.315 ms` | `1103.096 ms` | `84.193 ms` |
| `(5120 x 17408)` | `2048` | `189` | `901.308 ms` | `219.060 ms` | `583.690 ms` | `98.518 ms` |
| `(6144 x 5120)` | `2048` | `144` | `879.019 ms` | `684.025 ms` | `162.529 ms` | `32.445 ms` |
| `(10240 x 5120)` | `2048` | `144` | `751.862 ms` | `94.183 ms` | `621.434 ms` | `33.090 ms` |
| `(17408 x 5120)` | `1345` | `126` | `346.638 ms` | `68.867 ms` | `255.803 ms` | `21.950 ms` |
| `(5120 x 17408)` | `1345` | `63` | `224.304 ms` | `57.491 ms` | `142.556 ms` | `24.249 ms` |
| `(6144 x 5120)` | `1345` | `48` | `195.478 ms` | `147.851 ms` | `39.023 ms` | `8.600 ms` |

Reuse structure:

| Metric | Value |
| --- | ---: |
| Q3_K route rows | `1396` |
| unique route keys | `698` |
| repeated rows | `1047` |
| keys repeated | `698` |
| max calls per key | `4` |

The repeated-key pattern matches the prompt chunking: three full `ncols=2048` chunks plus one tail `ncols=1345`. For example, `blk.0.attn_gate.weight` is converted four times across the prompt run.

## Interpretation

The practical lane has two different Q3_K problems:

- `attn_gate` / `(6144 x 5120)` is conversion dominated: `831.876 ms` of `1074.497 ms` across full+tail shapes is `src0_convert_ms`.
- FFN gate/up/down shapes `(17408 x 5120)` and `(5120 x 17408)` are GEMM dominated: they account for most total time, and conversion-only fixes cannot move them enough alone.

This explains why E104 full `attn_gate` fp16 cache could reduce conversion but still regress wall: it attacked a real conversion bucket with too much persistent fp16 residency. It also explains why E105 existing-MMQ forcing failed: current direct-quant MMQ is not competitive for these large-N shapes.

The next code branch must be larger than a cache or selector:

1. A non-persistent fused Q3_K x F16 route that avoids source staging and stays competitive with hipBLAS GEMM on `(17408 x 5120)` and `(5120 x 17408)`.
2. Or a graph-scheduling route that processes repeated prompt chunks while a small converted source is live, without holding broad fp16 weights resident across the whole model.

Both are substantial route work. A conversion-only or activation-cache-only patch is below the ceiling unless the follow-up model proves it also reduces GEMM-side cost.

## Notes

- This is the gate before any large fused Q3_K x F16 route design.
- E192 used `max_tokens=64` to keep diagnostic trace time low; do not compare its aggregate TPS to E187/E190 `max_tokens=128` runtime claims.
