# W30: MTP acceptance on long prompts (L3/64K) - KV precision tail

Date: 2026-09-09 (Linux ROCm 10, 2x RX 9070 XT gfx1201, commit c4be97f30)

Goal: raise MTP draft acceptance for long prompts (especially L3 / 64.3K
synthetic prompt). Focus on acceptance; all clean runs, no background game.

Contract (identical for every run): `ctx=98304`, synthetic 64,287 prompt /
256 output, batch 8192 / ubatch 1024, KV `f8_e4m3 / f8_e4m3`, FlashAttention,
`ROCm1,ROCm0 -sm layer -ts 1,1`, `-ngl 999`, one slot, seed 42, temp 0.2,
top-p 0.9, no warmup, `spec=mtp n3`. Model = Qwen3.8-27B-UD-MXFP4-requant
unless stated.

## Finding: LLAMA_VK_MTP_KV_LAST_F16=16 stabilizes and raises acceptance on L3

The auto hybrid policy uses 12 f16 KV tail layers for f8_e4m3 at `n_ctx >=
98304` (8 otherwise). It left acceptance at ~52%. Raising the explicit f16
tail to **16 layers** produced a fully deterministic result: **171/252 =
67.9% acceptance** in 5 consecutive identical runs, with decode 38.4 tok/s
(+13.6% vs baseline) at prefill 1415 tok/s (-6%).

| Config (L3 MXUD) | runs | draft_n/accepted | Acceptance | Decode TPS | Prefill TPS | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| baseline (auto 12) | 3 | 155/296, 155/297, 155/296 | 52.3% | 33.8 | 1505.5 | stable |
| KV16 | 5 | 171/252 (identical) | **67.9%** | 38.4 | 1415.4 | deterministic |
| KV32 / KV64 | 3 | 171/252 (identical) | 67.9% | 38.4 | 1415.7 | same as KV16 |
| KV12 (explicit) | 2 | 171/249, 159/286 | 68.7% / 55.6% | 40.9 / 35.8 | 1515-1513 | boundary, flaky |
| KV8 | 1 | 161/281 | 57.3% | 39.0 | 1524.5 | below threshold |

Interpretation: the draft head's attention accuracy depends on more than the
last transformer layer; 12 tail layers sit exactly on the unstable boundary
at this context length, 16+ layers fully stabilize it. Memory/performance cost
is the extra f16 KV rows for the last 16 layers (prefill -6% because target
prefill quantizes fewer KV rows; decode unchanged).

## Window / prefill policy did NOT help

Expanding the draft context window is not the lever; it was unstable across
every variant and the full-context prefill (window=0) stayed at baseline:

| Config | run1 | run2 | Notes |
| --- | --- | --- | --- |
| WIN=2048 | 63.7% | - | single |
| WIN=4096 (stride 16384) | 62.6% | - | single |
| WIN=8192 | 66.3% | - | single |
| WIN=16384 | 87.2% | 55.8% | flaky |
| WIN=24576 | 87.2% / 90.7% | 52.5% | flaky |
| WIN=32768 | 65.5% | - | single |
| full ctx (WIN=0) | 53.8% / 58.1% | - | not the lever |
| + host handoff | 63.7/60.3% then 51.0/51.2% | - | not stable |
| + DEFER=0 | 52.5% / 71.9% | - | not stable |

The 87-90% one-off values were timing artifacts (draft prefill flush could
land in a different pipeline phase), not a reproducible gain. KV precision is
the reproducible lever; window size is not.

## L2 (30.6K) and Q4 checks (context/type dependent)

- **L2 MXUD, KV16**: deterministic 161/278 = 57.9%, decode 42.0 - **worse**
  than the L2 production row (auto 8 layers: 66.0%, 49.195). Do not raise the
  tail above 8 for L2. The optimum is context-dependent: 8 at 49K, 16 at 98K.
- **L3 Q4_K_M (UD), KV16/KV32**: flaky (77.7/69.6/72.2 vs 53.8/53.8). Q4 was
  not stabilized by the tail depth (KV16 nor KV32); needs its own
  investigation (different weight layout / draft timing).

## Verdict

- Accepted candidate (MXFP4/UD, L3 98K): `LLAMA_VK_MTP_KV_LAST_F16=16`
  (or 32/64 - identical). Acceptance 52.3% -> 67.9% (+15.6 pp), decode
  33.8 -> 38.4 (+13.6%), prefill 1505 -> 1415 (-6%), aggregate 5.09 -> 4.92
  at 256 outputs (still wins for longer generations).
- Rejected as levers: draft window expansion (WINDOW/STRIDE/CHUNK), host
  handoff, DEFER=0, full-context prefill - all unstable or no gain.
- L2 keeps auto 8 layers; only ctx >= 98K should consider 16 (MXFP4).
- Next steps for Q4 L3: profile the flaky switch, try a different
  stabilization (e.g. forcing synchronous draft decode, or
  `LLAMA_VK_MTP_KV_LAST_F16=16` together with `LLAMA_MTP_DEFER_SPARSE_PREFILL=0`
  and host handoff), or verify per-layer `has_kv` mapping for Qwen3.8.

Artifacts `build_logs/bench/mxfp4-ab/w30l3-*`, `w30l2-mxud-kv16-*`.
