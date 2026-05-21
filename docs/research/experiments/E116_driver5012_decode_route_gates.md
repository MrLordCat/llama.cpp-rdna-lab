# E116 Driver 5012 Decode Route Gates

## Metadata

- Experiment ID: E116
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ 054bccb00 plus E115 local notes
- Driver: AMD `32.0.31007.5012`
- Target lane: Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, thinking on

## Hypothesis

- Statement: decode-focused routing may have different best choices than the prompt-heavy cold-first lane.
- Mechanism: with `real-context-mode=off` and larger `max_tokens`, wall time is dominated by token generation. Backend choice, KV type, and speculative acceptance can matter more than Q3_K large-prefill staging.
- Why now: the user asked to search both decode and prefill. E115 covers prefill shape; E116 isolates decode route choices after the driver update.

## Math / Theory

- Assumptions: short prompts make prefill small; decode eval tok/s and wall completion TPS are primary.
- Expected speedup corridor: Vulkan or f16/q8 KV could improve decode if q4 KV/ROCm has decode-side overhead; speculative routes help only if accepted-token bursts appear.
- Failure conditions: prompt overhead dominates unexpectedly, candidate decode eval does not improve, or aggregate wall TPS loses despite decode eval changes.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: label these as decode-focused gates, not prompt-heavy cold-first defaults.
3. Rollback path: no code changes.

## Benchmark Plan

- Use `real-context-mode=off`, `max_tokens=512`, `tasks=quick`, `triage_diff,review_bug`, one-run gates.
- Gate candidates:
  - ROCm q4 KV `spec=none`
  - ROCm f16 KV `spec=none`
  - ROCm q8 KV `spec=none`
  - ROCm q4 KV `ngram-mod 12/16/32`
  - Vulkan q4 KV `spec=none`
- Confirm with r3 only if a route clearly beats the decode-focused control.

## Metrics

- aggregate completion TPS
- decode eval tok/s
- prompt eval ms/tok
- per-task wall TPS
- speculative acceptance stats when enabled

## Result

- Outcome: keep Vulkan f16 as a decode-only route; reject ngram for decode-only and keep q4/q8 below f16.
- Delta: ROCm q4 decode-focused gate `29.1685 TPS`; Vulkan q4 r3 `39.8801 TPS`; Vulkan f16 r3 `40.2753 TPS`.
- Confidence: high for Vulkan decode advantage; medium for f16-over-q4 because the delta is small but r3 confirmed a consistent edge. Correctness was also checked manually on a live server after the gate: the model thinks/answers normally and does not show the slash/symbol corruption seen in the old `wm32-wn32` Vulkan bug.
- Recommendation: use Vulkan + f16 KV as an opt-in decode/long-generation route. Do not promote it as prompt-heavy default because ROCm still wins pp7488/prefill.

## Notes

- One-run gates:
  - ROCm q4 `spec=none`: `29.1685 TPS`, decode eval `29.625 tok/s`
  - ROCm f16 `spec=none`: `29.6355 TPS`
  - ROCm q8 `spec=none`: `29.0405 TPS`
  - ROCm q4 `ngram-mod 12/16/32`: `26.3472 TPS`
  - Vulkan q4 `spec=none`: `39.8095 TPS`
  - Vulkan f16 `spec=none`: `40.0270 TPS`
  - Vulkan q8 `spec=none`: `39.6523 TPS`
  - Vulkan q4 `ngram-mod 12/16/32`: `30.5326 TPS`
- Confirmations:
  - Vulkan q4 r3: `39.8801 TPS`, decode eval `40.8683 tok/s`
  - Vulkan f16 r3: `40.2753 TPS`, decode eval `41.2283 tok/s`
- Live-server sanity follow-up:
  - User manually verified the Vulkan route in a real server/client flow: no corrupted symbol output, no slash spam, thinking/answers remain coherent.
  - Additional direct ROCm server check with `--reasoning-budget 256` produced a normal final `content` answer (`finish_reason=stop`, `436` completion tokens, decode `29.52 tok/s`), confirming that "no final content yet" under unrestricted thinking is a budget/API extraction behavior, not token corruption.
  - Future breakthrough gates should keep this lightweight real-server correctness check: start the actual server, ask a normal prompt, and reject any route that produces repeated punctuation/symbols or broken reasoning, as happened with the old `wm32-wn32` route.
- Why ngram failed in decode-only: without repeated prompt-cache/session structure, `ngram-mod 12/16/32` had very low effective acceptance (`0.000579` on Vulkan q4) and added verify overhead.
- Why Vulkan wins decode but not prefill: decode route is dominated by small-token generation where Vulkan's Q3_K route is faster; prompt-heavy pp7488 remains ROCm-favored (`1159.49` ROCm vs warmed `962.41` Vulkan).
- Follow-up action: keep separate profiles: ROCm for prompt-heavy/cold-first, Vulkan f16 for decode-heavy generation.
