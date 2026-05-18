# E056 H26 Q3_K Half2 r3 Recheck

## Metadata

- Experiment ID: E056
- Date: 2026-05-18
- Owner: Copilot
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Why Reopen H26

E055 rejected the half2 store prototype because the local split-detail gain was far below the `>=25%` gate for a standalone `>=2%` default-worthy win. That gate was too strict for stackable small improvements: a same-session r3 `~1%` aggregate gain is useful if it is opt-in, reproducible, and can later stack with other independent small wins.

## Updated Acceptance Policy

- Default promotion still needs either a clear larger aggregate gain or a strong modeled local mechanism.
- A small `~0.5-1.5%` aggregate gain is not automatically noise if it survives same-session r3 A/B.
- Small wins may be kept as guarded/experimental knobs when default behavior is unchanged and the maintenance cost is low.
- r1-only positives remain insufficient for keep decisions.

## Candidate

- Code: env-gated fp16-only Q3_K half2 store path in `ggml/src/ggml-cuda/convert.cu`.
- Guard: `GGML_CUDA_Q3K_DEQUANT_HALF2=1`.
- Default path: unchanged.

## Benchmark Plan

1. Build ROCm `llama-server`.
2. Run same-session control r3 with default code path.
3. Run same-session candidate r3 with `GGML_CUDA_Q3K_DEQUANT_HALF2=1`.
4. Keep as opt-in only if candidate is positive on aggregate TPS without serious prompt/decode regression.
5. Revert if r3 does not confirm the E055 r1 signal.

## Results

- Control r3: `prefill-e056-control-r3 = 11.6726 TPS` aggregate by wall time; mean task TPS `11.6739`, median `11.7061`, stdev `0.1344`.
- Candidate r3: `prefill-e056-q3half2-r3 = 11.6375 TPS` aggregate by wall time; mean task TPS `11.6389`, median `11.6845`, stdev `0.1418`.
- Delta: `-0.0351 TPS` (`-0.30%`) aggregate by wall time.
- Decision: reject and revert. The updated policy would keep a reproducible `~1%` r3 win as an opt-in stacking knob, but H26 did not confirm the E055 r1 signal.