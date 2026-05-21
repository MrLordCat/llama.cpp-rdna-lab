# E117 Driver 5012 Prefill KV Profile

## Metadata

- Experiment ID: E117
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ 054bccb00 plus E115/E116 local notes
- Driver: AMD `32.0.31007.5012`
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, `triage_diff,review_bug`, cold-first, no reuse, `spec=none`, thinking on

## Hypothesis

- Statement: at 12k context, f16 or q8 KV may improve cold-first prefill/decode enough to beat q4 KV if VRAM fit is acceptable.
- Mechanism: compressed KV saves memory, but can add attention/dequant route overhead. At `ctx=12288`, the model may fit with higher precision KV, potentially improving FlashAttention/decode behavior.
- Why now: E116 showed f16 KV helps decode slightly, and previous H34 evidence suggested 12k f16 KV can be a speed profile. The driver update requires a fresh same-lane A/B.

## Math / Theory

- Baseline: E113 q4 cold-first `11.9858 TPS`.
- Expected speedup corridor: `+0.5%` to `+5%` if attention/KV overhead matters; regression if larger KV increases memory pressure.
- Failure conditions: aggregate wall TPS stays below q4, or prompt eval improves but decode/prompt memory pressure offsets it.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: cold-first only; keep no reuse and `spec=none`.
3. Rollback path: no code changes.

## Benchmark Plan

- ROCm f16/f16 KV r1.
- ROCm q8_0/q8_0 KV r1.
- Confirm f16 with r3 only if it beats q4 baseline by more than noise.

## Metrics

- aggregate completion TPS
- prompt eval tok/s
- decode eval tok/s
- VRAM/offload notes if needed

## Result

- Outcome: reject f16/q8 KV as cold-first prompt-heavy replacements for q4 KV.
- Delta: q4 baseline `11.9858 TPS`; f16 KV `11.9028 TPS`; q8 KV `11.6392 TPS`.
- Confidence: medium; one-run gates show no positive signal, so no r3 confirmation is warranted.
- Recommendation: keep q4 KV for the 12k prompt-heavy ROCm default. Use f16 KV only in the decode-only Vulkan route from E116.

## Notes

- f16 KV improved decode eval slightly (`29.705 tok/s` vs q4 cold baseline `28.9183 tok/s`), but prompt eval fell enough that aggregate wall TPS stayed below q4.
- q8 KV regressed both aggregate and decode (`28.305 tok/s`).
- Why this differs from E116: E116 uses short prompts and long generation, so decode dominates. The active prompt-heavy lane still pays the full large-Q3_K prefill cost, and q4 remains the best balance.
- Follow-up action: keep KV-type changes categorized by scenario: q4 for prompt-heavy, f16 for Vulkan decode-heavy.
