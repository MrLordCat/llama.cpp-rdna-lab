# W28: L3 (98K) format sweep - MXFP4/MTP on long context

Date: 2026-09-09 (Linux ROCm 10, 2x RX 9070 XT gfx1201, commit bf0e70a1f)

First L3 benchmark set on Linux (Windows re-runs still pending). The L1/L2
results were W20-W26 (repo-snapshot), but `repo-snapshot` is capped at ~53K
tokens in `configs/bench/levels.json`, so L3 uses the deterministic
**synthetic** source to reach the real 98K lane.

## Contract

- bench2 `rdna-lab`, `--level 3` -> ctx 98304, prompt_tokens 66560,
  decode_tokens 256; actual prompt = 64,287 tokens (synthetic, seed 20260828).
- `batch 8192 / ubatch 1024`, KV `f8_e4m3 / f8_e4m3`, FlashAttention,
  `ROCm1,ROCm0 -sm layer -ts 1,1`, `-ngl 999`, one slot, cold prompt,
  `--no-warmup`, `seed 42`, temperature 0.2, top-p 0.9.
- Binary: `build-rocm-linux/bin/llama-server` (W24 nwarps=8 + W26 MMQ default).

## Results (clean `r1`)

| Model / format | Spec | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B-Q4_K_M (UD) | none | 1539.60 | 20.80 | 4.735 | - |
| Qwen3.8-27B-Q4_K_M (UD) | MTP n3 | 1329.83 | 33.27 | 4.568 | 64.5% (167/259) |
| Qwen3.8-27B-MXFP4-requant (UD) | none | 1765.49 | 22.60 | 5.362 | - |
| Qwen3.8-27B-MXFP4-requant (UD) | MTP n3 | 1502.55 | 33.68 | 5.081 | 52.4% (155/296) |
| Qwen3.8-27B-MXFP4-requant (dense) | none | 1769.05 | 22.90 | 5.387 | - |
| Qwen3.8-27B-MXFP4-hybrid-attnQ6 | none | 1718.59 | 21.63 | 5.199 | - |
| Qwen3.8-27B-NVFP4-native | none | 473.85 | 21.75 | 1.736 | - |

## Background contamination (Minecraft) - why r1 was rerun

The first pass ran while a Minecraft process was on the GPU. Its records are
kept as `*-r0-mc` and are NOT reference data:

| Run | Prompt TPS (r1 vs r0-mc) | Δ |
| --- | --- | ---: |
| Q4_K_M none | 1539.60 vs 1346.77 | -12.5% |
| Q4_K_M MTP n3 | 1329.83 vs 821.18 | -38.3% |
| MXUD none | 1765.49 vs 1626.06 | -7.9% |
| MXUD MTP n3 | 1502.55 vs 1269.36 | -15.5% |

## Findings

- **W24/W26 wins hold at 98K**: MXFP4-requant none prefill 1765.49 vs Q4_K_M
  1539.60 = **+14.7%**; aggregate 5.362 vs 4.735 = **+13.3%**. MXFP4 decode
  22.60 vs 20.80 = **+8.7%**.
- **MTP is weaker at 98K**: acceptance 52.4% (MXUD) and 64.5% (Q4) vs ~66% and
  ~78% at L2; aggregate MTP 5.081 vs spec-none 5.362 (MXUD) - the additional
  draft verification cost outweighs the decode gain on this prefill-heavy
  output. MTP is still useful for long outputs (>512 tokens).
- **NVFP4-native is not usable** on any prompt-heavy lane (473.9 t/s prefill,
  aggregate 1.736).
- **Hybrid-attnQ6 is neutral at L3** (1718.6 prefill, 21.6 decode) - the
  extra Q6 bytes slightly reduce prefill vs pure MXFP4 (1769.1).

Artifacts: `build_logs/bench/mxfp4-ab/w28l3-{q4k-ud-none,q4k-ud-mtp3,mxud-none,
mxud-mtp3,mx4-none,hyb-none,nvfp4-none}-r1--*` and the 4 `-r0-mc` folders.
