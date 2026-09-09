# W20: MXFP4 decode measurement on RDNA4 (ROCm 10, 2x RX 9070 XT)

Date: 2026-09-09

Context: R9700-reproduction investigation. Current production Qwen3.8-27B
baseline on this fork is Q4_K_M (qwen35). Question: how much faster can the
2x 9070 XT decode with the same low-precision path the R9700 implementation
uses? First candidate: MXFP4 (already fully wired in this fork - types,
MMVQ, MMQ, convert; `ggml_quantize` exposes it as `MXFP4_MOE`).

## Setup

- Model: `Qwen3.8-27B-Q4_K_M.gguf` -> requantized in-tree:
  `llama-quantize --allow-requantize --tensor-type ".*=MXFP4" ... MXFP4_MOE`
  (the `MXFP4_MOE` ftype maps dense 4-bits by regex; 16 threads).
  Result: 16303.80 MiB -> 13850.58 MiB (5.01 -> 4.25 BPW, -15.2% bytes).
  Caveat: requantized from Q4_K (quality is NOT the target of this
  measurement - only the compute/bandwidth path; a native BF16->MXFP4
  requires the full BF16 GGUF, currently only an incomplete shard exists
  in `models/fp4-dl/Qwen3.8-27B-BF16-00001-of-00002.gguf`).
- Server: build-rocm-linux llama-server, ROCm 10, flash-attn on,
  f8_e4m3 K/V, spec none, ROCm1,ROCm0 -sm layer -ts 1,1, no-warmup, seed 42.
- bench2 L1 (7901 prompt / 128 decode, ctx 16384) and L2
  (30609 prompt / 256 decode, ctx 49152). Same binary, pinned server path
  (`build-rocm-linux/bin/llama-server`; preflight rejects /tmp snapshots
  after system /tmp cleanup).

## Results (A-B-A)

### L1 (7901 / 128)

| run | prefill tok/s | decode tok/s |
| --- | --- | --- |
| A q4k-mxab1 | 1915.98 | 24.4647 |
| B mxfp4-ab1 | 1894.25 | 28.4916 |
| A2 q4k-ab2 | 1909.14 | 24.4558 |

- decode: 28.4916 vs control (24.4602) = **+16.47%**
- prefill: -0.96% (noise); aggregate +8.1% (prefill-dominated)

### L2 (30609 / 256) - production contract

| run | prefill tok/s | decode tok/s |
| --- | --- | --- |
| A q4k-l2a | 1814.87 | 22.3155 |
| B mxfp4-l2b | 1799.47 | 25.7168 |
| A2 q4k-l2a2 | 1811.69 | 22.2934 |

- decode: 25.7168 vs control (22.3044) = **+15.30%**
- prefill: 1799.47 vs 1813.28 = **-0.76% (noise)**
- aggregate: 9.4939 vs 9.0274 = **+5.17%**

## Interpretation

- Decode gain (15.3%) matches the weight-byte reduction (15.2%):
  `4.25/5.01 - 1 = -15.2%`. MXFP4 MMVQ is the *same DP4A architecture* as
  Q4_K (no WMMA on gfx1201; `blackwell_mma_available` false), so the win is
  almost entirely **fewer weight bytes streamed** (plus simpler 1-level
  E8M0 scaling vs Q4_K's dual d/dmin/scales).
- This independently CONFIRMS W19's revised conclusion: decode MMVQ is
  memory-bandwidth limited, not ALU limited (W18: removing 50% DP4A gave
  ~0; W16 prefetch -0.64%; C1b occupancy +1.1% - all exhausted).
- Prefill is NOT helped: MMQ DP4A path has the same instruction density
  and extra quantize/convert overhead; difference within noise.

## MXFP4 + MTP n3 (speculative, UD model with NextN head)

- UD-MXFP4 requant (same CLI; nextn head included, 13.85 GB).
- L1 (7901/128): Q4+MTP 50.132 / MXFP4+MTP 50.993 -> +1.7%
  (acceptance dropped 78% -> 58% - draft head quality from requant).
- **L2 (30609/256) - production contract:**
  | run | decode tok/s | accept |
  | --- | --- | --- |
  | A q4ud-mtp-l2a | 38.9836 | 270/164 |
  | B mxud-mtp-l2b | 48.3167 | 264/166 |
  | A2 q4ud-mtp-l2a2 | 39.9387 | 264/166 |
  -> decode **48.3167 vs 39.461 (control) = +22.44%**; prefill 1727.09 vs
  1713.25 = +0.8% (noise).
- Stack: MXFP4+MTP decode **48.32 = 2.17x** the Q4 spec-none decode (22.31)
  and **1.9x** Q4+MTP. This is the strongest non-R9700-ported number.
- Caveats: acceptance here ~63% and draft weights were also requantized
  (draft quality down); a native BF16->MXFP4 UD will likely raise
  acceptance and speed. Still, the target-verify decode is BW-bound, so the
  MXFP4 gain transfers.

## NVFP4 (4.50 BPW, E4M3 subscale) single measurement

- NVFP4 L1: decode 28.0804 t/s (similar to MXFP4 28.49) BUT
  **prefill 469.296 t/s - ~4x worse** (MMQ DP4A path not optimized for
  NVFP4 on non-Blackwell; the E4M3 subscale lookup is 4x more work).
- Verdict: NVFP4 NOT usable as dense default; MXFP4 wins (simpler single
  E8M0 scale + better MMQ).

## Next candidates

1. **NVFP4** (block 16, E4M3 subscales, VDR 4): maybe similar or better
   quality/speed; file will be a bit larger; quick L1 A/B.
2. **BF16->MXFP4 native** (quality-correct): requires complete BF16 GGUF
   (54 GB), then quantize; needed before any production/quality claim.
3. **MXFP4 + MTP** (spec n3, acceptance ~74% measured on Q4): expected
   effective ~45-46 tok/s (25.7 * 1.74); measure as next spec-lane check.
4. **R9700 / 280 tok/s note**: 280 tok/s on 2x R9700 for 27B Q4/MXFP4
   (~14.3 GB weights) implies ~4.0 TB/s weight traffic - above dual-card
   HBM/GDDR bandwidth (~1.3-1.4 TB/s), so the headline number MUST combine
   speculative decoding / draft model / batch. Non-spec raw ceiling here is
   ~25.7 tok/s (MXFP4) and would be ~30-35 on R9700 clocks; the rest is
   speculative.

## Artifacts

- Model: `models/Qwen3.8-27B-MXFP4-requant.gguf` (13.85 GB)
- Logs: `/tmp/mxfp4_ab_{q4a,mx1,q4a2}.log`, `/tmp/mxfp4_l2{q4a,mx,q4a2}.log`
- CSV: `build_logs/bench/mxfp4-ab/`
