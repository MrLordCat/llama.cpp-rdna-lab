# W25: MXFP4/NVFP4 hybrid quality trade-offs; MTP acceptance profile; MMQ routing gap

Date: 2026-09-09 (Linux ROCm 10, 2x RX 9070 XT, gfx1201)

Deepening the three levers after W24.

## Lever 1 (quality) - best quality-per-byte found (hybrid and NVFP4-native)

Context: W24 shipped `MXFP4-native` (all MXFP4) = 7.1391 PPL (+5.29% vs Q4_K).
Question: can we reduce the quality cost? Yes - two configs are much better:

Native from `Qwen3.8-27B-BF16.gguf` (55.6 GB safetensors -> 54.6 GB BF16 GGUF
-> `llama-quantize` with per-tensor overrides):

| model | bpw | size | PPL (512 chunks) | delta vs Q4_K | L2 decode | L2 prefill |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-Q4_K_M | 5.01 | 16.3 GiB | 6.7802 ± 0.05045 | - | 22.30 | 1813 |
| Qwen3.8-27B-MXFP4-requant | 4.25 | 13.85 GiB | 7.1937 ± 0.05471 | +6.10% | 25.72 | 1799 |
| Qwen3.8-27B-MXFP4-native | 4.25 | 13.85 GiB | 7.1391 ± 0.05414 | +5.29% | 26.25 | 1780 |
| **Qwen3.8-27B-MXFP4-hybrid-attnq6** | 4.85 | 15.75 GiB | **6.9925 ± 0.05375** | **+3.13%** | 24.26 | 1770 |
| **Qwen3.8-27B-NVFP4-native** | 4.50 | 14.67 GiB | **6.9840 ± 0.05201** | **+3.01%** | 24.81 | 489 (-4x) |

Validated: hybrid PPL ran twice and was bit-identical (6.9925 ± 0.05375);
NVFP4-native also completed cleanly (6.9840 ± 0.05201).

**Hybrid recipe (the best quality-per-decode-speed):**
- `token_embd.weight`, `output.weight`, `blk.*.attn_qkv.weight`,
  `blk.*.attn_q.weight`, `blk.*.attn_k.weight`, `blk.*.attn_v.weight`,
  `blk.*.attn_output.weight` -> `Q6_K`
- everything else -> `MXFP4`
- 4.25 -> 4.85 bpw (+14% bytes), PPL +5.29% -> +3.13% (-2.16% PPL).
- This cuts the PPL penalty nearly in half for +14% weights, at -0.4%
  decode vs Q4_K (24.26 vs 24.46) but +4% vs Q4_K aggregate.

**Verdict:** the "quality vs speed" optimum in-tree is **NVFP4-native
(+3.01% PPL, 4.50 bpw) for quality, and MXFP4-hybrid (+3.13% PPL, 4.85 bpw)
for decode speed** (NVFP4 prefill -4x makes it unusable for prefill-dominant
runs). Both are far better than pure MXFP4 (+5.29%). imatrix still
unavailable for MXFP4/NVFP4 (`quantize_mxfp4` ignores quant_weights).

## Lever 2 (MTP acceptance) - profile measured, no `p_min`/`n_max` win

Exact acceptance profile measured with `--spec-draft-p-min` and
`--verbose` (debug `#acc rate/pos`):

- Baseline n3: `#acc rate/pos = (0.800, 0.600, 0.432)` on L2; mean accept
  length 2.83 tokens per draft cycle with n_max=3. Acceptance falls steeply
  by position (position 3 is only 43%).
- `n_max=2`: acceptance 145/219 (66.2%) but decode **41.574** vs 49.195;
  n3 is the optimum.
- `n_max=4`: acceptance 170/338 (50.3%), decode 44.155; worse than n3 (extra
  positions mostly rejected).
- `p_min=0.5`: acceptance 162/238 (68.1%, slightly better) but decode
  **44.246** vs 49.195 (fewer draft cycles = slower).
- `n_max=3` remains best (0.83 combined decode 49.2).
- Q6_K up-quantization of draft path (W24) also did not help.

Conclusion: the acceptance ceiling (~66%) is a **limit of draft-state
quality / target-vs-draft calibration**, not a tunable knob. Acceptance
could rise with a better draft (higher overall model quality) but the gains
from `p_min`/`n_max` are negative for decode. NVFP4/hybrid MTP untested
here (NVFP4 prefill too slow for the L2 prompt).

## Lever 3 (decode weight-stream) - all found; small_k and MMQ-routing gaps

Confirmed there is nothing left in the findable decode geometry:
- `ggml_cuda_mmvq_is_qwen_hot_type` only includes Q3_K/Q4_K/Q6_K, so
  `small_k` (8 rows/block) does NOT apply to MXFP4. Tested adding MXFP4:
  L2 decode **25.835** vs 26.252 (nw8, no small_k) = **-1.6%** -> REJECTED;
  reverted. MXFP4's lighter vec_dot prefers fresh rows (1/block).
- MMQ routing gap remains: `ggml_cuda_should_use_mmq` RDNA4 switch puts
  MXFP4/NVFP4 into `default: ne11 <= 128` (while Q4_K/Q5_K <= 256,
  Q2/Q3/Q6 <= 192). At prefill ubatch 1024, MXFP4 uses hipBLAS (WMMA-I8
  unused). This is the W23 MMQ-routing item; expected +18-20% prefill when
  routed. No code change yet (needs clean A-B-A).
- The accepted change stays: `calc_nwarps` RDNA4 ncols=1 adds MXFP4
  (`return 8`) - one line, verified +3-5% decode.

Artifacts: `models/Qwen3.8-27B-{MXFP4-hybrid-attnq6}*.gguf`,
`models/Qwen3.8-27B-NVFP4-native.gguf`; `build_logs/bench/mxfp4-ab/{mxfp4-smallk-l2,mxud-mtp-n2-l2,mxud-mtp-n4-l2,mxud-mtp-pmin05-l2,mxud-mtp-n3-verbose,hyb-native-l2,nv-native-l2,nv-native-l2a2}-*`.
