# W12: decode-token census (whole-lane node timing)

Date: 2026-08-14
Status: done (pause checkpoint; resume from here)

## Question

D103 and the README budget assumed "FA ~20% of a 49K decode token, ~55%
unknown remainder". Where does the remainder actually go? This census splits
the per-token decode cost across GGML ops using the default-off
`GGML_TRACE_CUDA_NODE_TIMING` infrastructure (D100).

## Method

- Lane: locked 49K lane (`Qwen3.6-27B-Q4_K_M.gguf`, f8_e4m3 KV,
  `ctx=49152,b=8192,ub=1024`, `-dev ROCm1,ROCm0 -sm layer -ts 1,1`,
  `spec=none`, cold/no-reuse, seed 42, `triage_diff`).
- Run: `--trace-preset kernel-full` + `GGML_TRACE_CUDA_NODE_TIMING_SYNC=1`
  + `GGML_TRACE_CUDA_NODE_TIMING_MIN_MS=0.05`, 32 decode tokens
  (`r001-w12-census-r1`).
- Parser: `scripts/research/w12_decode_token_census.py` - nodes repeating
  >=20x are the per-token decode replay; per-node cost = median total_ms.
- Caveat: SYNC mode adds a per-node host-sync overhead that is roughly
  constant per node but not per op, so the shares below are inflated toward
  many-small-node ops (MUL_MAT) and slightly understate few-large-node ops
  (FA). Relative ranking is robust; absolute shares are approximate.
  The traced run itself is slower than production (57.2 ms/token vs ~44 ms
  without traces) - shares, not absolutes, are the result.

## Result (op-level shares, sync-inflated)

| op | nodes/token | med-sum ms | share % |
| --- | ---: | ---: | ---: |
| MUL_MAT | 493 | 685.7 | 62.96 |
| GATED_DELTA_NET | 48 | 147.4 | 13.53 |
| FLASH_ATTN_EXT | 16 | 112.9 | 10.36 |
| GLU | 63 | 30.6 | 2.81 |
| ADD | 174 | 30.6 | 2.81 |
| CONCAT | 48 | 19.3 | 1.77 |
| L2_NORM | 96 | 11.6 | 1.07 |
| CPY | 97 | 11.0 | 1.01 |
| GET_ROWS | 96 | 10.7 | 0.98 |
| UNARY | 96 | 9.9 | 0.90 |
| others (CONT/ROPE/MUL/SET_ROWS/...) | ~180 | ~26 | ~2.4 |

## Findings

1. The "~55% unknown remainder" is MUL_MAT: the dense FFN + attention
   projection weight stream (Q4_K_M, ~17 GB read per token across 2 GPUs).
   Even after correcting the sync inflation, MUL_MAT is >= 50% of a decode
   token; FA is ~10-20%. The weight stream IS the decode bottleneck.
2. GATED_DELTA_NET = 13.5% of the traced token - the Delta-Net chunk
   recurrence runs per token even at 49K (not just prefill). This is a
   second-order target the previous FA-focused program never touched.
3. FA micro-optimizations are capped: the whole FA op is ~10-20% of the
   token, so a candidate must be implausibly strong inside FA to clear the
   `>=3%` whole-lane decode gate (e.g. >= 15-30% FA speedup). This is why
   H77/H79/SR could never have passed even if they had won locally.

## Direction set (resume point for the next RDNA4 session)

Priority order for the next candidates:

1. **Decode MUL_MAT/MMVQ weight-stream work** (W10 shapes: M<=4, K=5120,
   N<=17408; Q4_K_M MMQ/MMVQ kernels). Candidates: Q4_K_M decode kernel
   weight-load layout vs L2/L1 streaming (64 MB Infinity Cache per W09
   logic), K-split/persistent weights, better batch utilization across the
   2-GPU layer split. This is where a >=3% whole-lane win is physically
   possible.
2. GATED_DELTA_NET per-token cost audit (why 13.5% at decode: chunk state
   update path, 48 nodes).
3. FA shelf leftovers (vectorized fp8 KV tile loads, H80 cache-policy hints
   when a toolchain expresses them) - demoted: FA is ~10-20% of the token.

## Artifacts

- `build_logs/agent-workload/r001-w12-census-r1.server.log` (+ jsonl/csv)
- `scripts/research/w12_decode_token_census.py`
- This doc; README progress table updated.
