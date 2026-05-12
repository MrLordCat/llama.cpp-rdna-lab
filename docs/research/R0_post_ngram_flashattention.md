# R0: Post-ngram / Post-FlashAttention Research Note

Date: 2026-05-12

## Problem Statement

Both ngram-based speculative decoding and Flash Attention are already deployed.
The next gains likely require better adaptation to runtime regime changes rather than one static setting.

## Why Existing Methods Saturate

### 1) ngram speculative decoding saturation

Speculative decode gain depends on acceptance and overhead:

S_spec ~= (1 + a * (D - 1)) / (1 + o)

When acceptance a drops (for example in high-entropy spans), speedup collapses quickly.
With fixed draft length D, mismatch waste can dominate.

### 2) Flash Attention saturation

Flash Attention mostly optimizes memory traffic for prefill.
If prefill share p is reduced, further Flash-only optimization gives diminishing returns:

S_total ~= 1 / (p / S_prefill + (1 - p) / S_decode)

When decode dominates, prefill-only wins become smaller.

## Candidate Discovery Directions

## D1: Entropy-conditioned speculative scheduler

Idea:
- estimate local uncertainty proxy from recent logits (or surrogate signals)
- adapt both ngram size n and draft length D online

Target effect:
- keep high acceptance in stable spans
- reduce wasted verify work in unstable spans

Research questions:
1. Which uncertainty proxy is cheapest and stable enough?
2. Is piecewise policy better than continuous control?
3. How much hysteresis is needed to avoid policy thrashing?

## D2: Hybrid draft router (ngram vs mtp)

Idea:
- route each step to ngram or mtp mode based on confidence and context pattern
- low-overhead branch predictor decides route

Potential novelty:
- dynamic router can beat either static mode on mixed workloads

Risk:
- routing overhead can offset gain unless decision is very cheap

## D3: Verification early-reject bounds

Idea:
- add safe bound checks to reject bad drafts before full expensive path
- only continue full verification for candidates passing the bound

Potential novelty:
- similar spirit to branch-and-bound at token verification level

Risk:
- incorrect bound design can alter exactness or quality

## D4: Decode-local KV layout and fetch policy

Idea:
- tune KV placement and fetch granularity for decode locality
- reduce cache misses and memory transaction overhead in long decode loops

Potential novelty:
- architecture-specific layout policy for RDNA path

Risk:
- complexity in compatibility across kernels and quantized KV types

## D5: Boundary-aware chunk contract

Idea:
- align model-side chunking with runtime batch/ubatch boundaries to avoid pathological transitions
- this is especially relevant if throughput cliffs appear at specific token counts

Potential novelty:
- contract-level optimization between model op and runtime scheduler

Risk:
- model-specific behavior may limit generalization

## Immediate Experimental Sequence

1. Validate D1 with a minimal scheduler prototype and fixed safety guards.
2. Evaluate D5 around known boundary cliffs with controlled sweeps.
3. Test D2 router only after D1 policy signals are proven useful.
4. Run D3 and D4 as deeper kernel/runtime phases.

## Success Criteria

A direction is considered a real win only if:
- it improves wall TPS in the active prompt-heavy lane
- it remains stable across repeated cold-first runs
- it does not increase error/failure rate
- the gain is reproducible and documented with artifacts

## Suggested Artifacts Per Experiment

- benchmark CSV + diagnostics report
- server log with key runtime stats
- one short markdown summary with baseline/candidate delta
- explicit decision: keep, iterate, or revert
