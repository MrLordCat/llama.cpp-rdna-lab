# W24: MXFP4 decode geometry (nwarps=8) measured; MTP draft-precision gate negative

Date: 2026-09-09 (Linux ROCm 10, 2x RX 9070 XT, gfx1201)

Covers the remaining W22-listed levers: "native MXFP4 quality" (lever 1,
in progress in W24b once BF16 is downloaded), "MTP/acceptance" (lever 2,
NEGATIVE here) and "weight-stream geometry" (lever 3, POSITIVE here).

## Lever 2 - MTP acceptance: draft-precision up-quantization does NOT help

Context (W20): UD-Q4_K_M -> UD-MXFP4-requant gives L2+MTP decode 48.3167;
acceptance 166/264 = 62.9%. Hypothesis: the draft path quality is limited by
the MXFP4-quantized NextN head, so up-quantizing the draft path should raise
acceptance and speed.

Tested two in-tree requants from `Qwen3.8-27B-UD-Q4_K_M.gguf`
(`llama-quantize --allow-requantize --tensor-type ... MXFP4_MOE`):

1. **Head-only**: `blk.64.nextn.*=Q6_K` before `.*=MXFP4`
   -> `Qwen3.8-27B-UD-MXFP4-draftq6.gguf` (14,549,539,744 B).
   L2 MTP n3: decode **47.6868** (was 48.3167, -1.3%, noise),
   acceptance **167/263 = 63.5%** (was 62.9%).
2. **Last block + head**: `blk.64.*=Q6_K` then `.*=MXFP4`
   -> `Qwen3.8-27B-UD-MXFP4-lastblockq6.gguf` (+117 MiB).
   L2 MTP n3: decode **46.7774** (-3.2%), acceptance **166/267 = 62.2%**.

Verdict: raising precision of the draft local path (final block + NextN
head) does NOT improve acceptance on this lane; the extra Q6_K bytes slow
decode because the MTP stack is bandwidth-bound (W20). The acceptance gap is
not explained by MXFP4 requant of the draft head. Remaining suspects:
temperature/sampling interaction, draft state, whole-model quality
(lever 1 BF16 native may still move it - untested here).

## Lever 3 - decode weight-stream geometry: MXFP4 ncols=1 nwarps 1 -> 8

`ggml/src/ggml-cuda/mmvq.cu` RDNA4 `calc_nwarps` whitelist already used
`nwarps=8` for simple-vec_dot types (Q4_0..Q6_K, IQ4_NL/XS) at ncols=1 but
**omitted MXFP4** - so MXFP4 decode ran with `nwarps=1` (one warp/CTA).
Change: add `GGML_TYPE_MXFP4` to the RDNA4 ncols=1 `nwarps=8` whitelist.
(`NVFP4` deliberately NOT added until measured.)

A/B (same binary, spec none, f8_e4m3 KV, ROCm1,ROCm0 -sm layer -ts 1,1,
batch 8192 / ubatch 1024, repo-snapshot):

| lane | run | prefill tok/s | decode tok/s |
| --- | --- | --- | --- |
| L1 (8450/128) | before mxfp4-ab1 | 1894.25 | 28.4916 |
| L1 | before mxfp4-l1-trace | 1887.29 | 28.1916 |
| L1 | **after mxfp4-nw8-l1** | 1872.12 | **29.7620** |
| L2 (33865/256) | before mxfp4-l2b | 1799.47 | 25.7168 |
| L2 | before mxfp4-l2-a2 | 1794.80 | 25.2391 |
| L2 | **after mxfp4-nw8-l2** | 1774.96 | 26.2197 |
| L2 | **after mxfp4-nw8-l2a2** | 1772.39 | 26.2850 |

- L1 decode: 29.762 vs 28.492/28.192 avg 28.342 = **+5.01%**.
- L2 decode: 26.252 (avg of 2) vs 25.478 (avg of 2 controls) = **+3.04%**.
- Prefill: L1 -0.9%, L2 -1.2% (within lane noise).

MTP stack (UD-MXFP4-requant, spec mtp n3, L2):
decode **49.195** vs 48.3167 = **+1.8%**; acceptance 169/256 = 66.0%
(was 62.9% - varies, not claimed as a precision effect).

Artifacts: `build_logs/bench/mxfp4-ab/{mxfp4-nw8-l1,mxfp4-nw8-l2,mxfp4-nw8-l2a2,mxud-mtp-nw8-l2,mxud-mtp-draftq6-l2,mxud-mtp-lastblockq6-l2}-*`.

## W24b - Lever 1: native BF16->MXFP4 quality gate (PPL)

- Source: official HF `Qwen/Qwen3.8-27B` (18 safetensors, 55.56 GB) downloaded
  to `models/w24_bf16/`; converted with upstream `convert_hf_to_gguf.py`
  (`--outtype bf16`) -> `Qwen3.8-27B-BF16.gguf` (54.6 GB, 866 tensors, qwen35).
- Native quant: `llama-quantize --tensor-type ".*=MXFP4" ... MXFP4_MOE`
  -> `Qwen3.8-27B-MXFP4-native.gguf` (13,850.58 MiB, 4.25 BPW).
- **imatrix for MXFP4 is NOT implemented**: `quantize_mxfp4()` has
  `GGML_UNUSED(quant_weights)` both in this fork AND in upstream master
  (`ggml/src/ggml-quants.c`). So the quality gate is native BF16->MXFP4
  without imatrix (we would have liked to test imatrix, it cannot help).
- PPL (same 512-chunk contract as W21, ROCm1,ROCm0, flash-attn, no-mmap):
  | model | PPL | delta vs Q4_K |
  | --- | --- | --- |
  | Qwen3.8-27B-Q4_K_M | 6.7802 +/- 0.05045 | - |
  | Qwen3.8-27B-MXFP4-requant (Q4_K->MXFP4) | 7.1937 +/- 0.05471 | +6.10% |
  | **Qwen3.8-27B-MXFP4-native (BF16->MXFP4)** | **7.1391 +/- 0.05414** | **+5.29%** |
- Verdict (quality gate): native BF16->MXFP4 barely improves on requant
  (-0.76% PPL) and is still **+5.29% worse than Q4_K**. The +6.1% worst-case
  was mostly the MXFP4 format itself, not the Q4_K requant cascade. MXFP4
  remains a *speed* choice (+15% decode, W20; +3% nw8, above), not a
  quality-neutral one. Production quality baseline stays Q4_K_M; consider
  MXFP4 only where decode speed dominates quality (e.g., draft/experimental).

## Status

- Lever 3 accepted in tree (default, RDNA4 MXFP4 ncols=1 nwarps=8).
- Lever 2 closed as negative for local draft up-quantization (Q6_K).
- Lever 1 (native BF16->MXFP4 + imatrix) is running: HF
  `Qwen/Qwen3.8-27B` (18 safetensors, ~52.7 GiB) downloaded to
  `models/w24_bf16/`; next: convert BF16 GGUF, `llama-imatrix` on
  `/tmp/qc_corpus.txt`, quantize with `--imatrix`, and re-run the
  512-chunk PPL gate from W21 (7.1937 vs Q4_K 6.7802).
