# E189 ROCm Q3_K Fused Pair-Dot Budget Gate

## Metadata

- Experiment ID: E189
- Date: 2026-05-23
- Owner: Codex
- Branch/Commit: master after `8c1610d9e`
- Target lane: L1 ROCm H39, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: the next useful ROCm decode candidate should be a budget-first Q3_K fused route-body change, not another selector/env knob.
- Mechanism: E165 showed that fused Q3_K calls read the same transient `q8_1` activation for the up and gate paths. The naive preload-y implementation reduced some duplicated work in concept, but inflated registers (`84 -> 136`) and lost occupancy. A stricter follow-up would use a Q3_K-only pair-dot helper that streams one `q8_1` lane at a time and accumulates x/gate together, avoiding live `u[]/d8[]` arrays across both dots.
- Why now: Phase 3 of `NEUTRAL_SMALL_PLUS_AUDIT.md` closes the nearby micro routes. E187/E188 lock the current L1 baseline and show active direct/fused Q3_K decode route evidence.

## Math / Theory

- Assumptions:
  - E187 L1 r3 baseline: aggregate `12.4256 TPS`, prompt eval `6069.3967 ms`, decode eval `4197.9733 ms`.
  - Baseline wall share estimate: prefill `59.12%`, decode `40.88%`.
  - Candidate affects decode kernel path only; prefill speedup is assumed `1.0x`.
- Analytic gate:
  - `+2%` decode-kernel speed projects only `12.5260 TPS` (`1.0081x` wall).
  - `+5%` decode-kernel speed projects `12.6723 TPS` (`1.0199x` wall).
- Keep threshold:
  - do not write/keep code for a sub-`+2%` local decode idea;
  - require a credible `>=5%` local decode-kernel mechanism or a stack plan that combines multiple local wins without moving the bottleneck into prefill.
- Failure conditions:
  - resource gate shows a register jump or occupancy drop comparable to E165;
  - fused dominant buckets regress even if a low-share bucket improves;
  - r1 plus fails r3 confirmation.

Analytic commands:

```powershell
python scripts\research\formula_sanity_checks.py
python scripts\research\speedup_model.py --baseline-tps 12.4256 --prefill-share 0.5912 --draft-len 1 --accept-rate 0 --spec-overhead 0 --flash-prefill-speedup 1.0 --decode-kernel-speedup 1.02
python scripts\research\speedup_model.py --baseline-tps 12.4256 --prefill-share 0.5912 --draft-len 1 --accept-rate 0 --spec-overhead 0 --flash-prefill-speedup 1.0 --decode-kernel-speedup 1.05
```

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-cuda/vecdotq.cuh`: add a Q3_K pair-dot helper that computes x/gate against a shared `q8_1` stream without retaining full duplicated arrays.
   - `ggml/src/ggml-cuda/mmvq.cu`: call it only for `type == GGML_TYPE_Q3_K`, `has_fusion`, `use_gate`, `ncols_dst == 1`, and active L1-like small-k route, behind an env gate.
2. Guard rails:
   - env gate default-off;
   - first build/resource run only, no speed claim;
   - reject before r3 if dominant fused buckets increase registers materially or occupancy drops.
3. Rollback path:
   - revert the helper and callsite if resource gate fails or r3 is neutral/regressive.

## Benchmark Plan

- Baseline command: E187 L1 baseline r3 and E188 route evidence.
- Candidate resource command: same L1 with `GGML_TRACE_MMVQ_RESOURCES=1`, `GGML_TRACE_MMVQ_TIMING=1`, `GGML_TRACE_MMVQ_TIMING_SYNC=1`, and candidate env gate enabled.
- Candidate runtime command: same L1 `--runs 1`; only then `--runs 3` if resource gate is clean.
- Artifacts path: `build_logs/agent-workload/e189-rocm-q3k-pairdot-*`.

## Metrics

- aggregate completion TPS (wall)
- decode eval tok/s
- prompt/decode split
- Q3_K fused/direct bucket timings
- regs / occupancy / LDS for dominant fused buckets
- error rate and real-server sanity only if a breakthrough is claimed

## Result

- Outcome: design gate only, no code candidate yet.
- Delta: no measured candidate TPS.
- Confidence: medium for rejecting old E165-style preload; low-medium for the pair-dot variant until resource gate exists.
- Recommendation: ask/choose before coding. This is the most plausible current H39 route-body follow-up, but it must be resource-gated before any full r3 cycle.

## Notes

- This is not E165 repeated. E165 held more live values and paid a large register/occupancy penalty. E189's only reason to exist is to test whether the same y-reuse idea can be expressed as a streaming pair-dot with a strict register budget.
- If the resource gate still raises regs sharply, the next bottleneck is not q8 reloads; move to a different Q3_K topology or to Vulkan-side decode route-body work.
