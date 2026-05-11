# P5 Long-Run Server Stability And Throughput

## Objective

Stabilize long-running llama-server throughput in agent-style workloads by reducing overflow-driven cancels, checkpoint churn, and prompt-cache maintenance spikes, while keeping decode quality and output behavior unchanged.

## Scope note

- This is a secondary server long-run lane plan (ctx=65536), not a replacement for the active sub-16k P1 lane.
- This document is planning-only. No runtime code changes are applied here.

## Code study map

- tools/server/server-context.cpp
  - get_available_slot(): LCP/LRU slot selection and prompt cache update path.
  - request size guard in process loop: exceed-context rejection path.
  - checkpoint lifecycle: create_checkpoint(), checkpoint search/restore, invalidation erase loop.
  - prefill progression and near-end prompt batching behavior (frequent batch.n_tokens tail behavior).
- common/common.h
  - server checkpoint controls: n_ctx_checkpoints, checkpoint_every_nt.
- common/arg.cpp
  - CLI wiring for --ctx-checkpoints and --checkpoint-every-n-tokens.
- common/speculative.cpp
  - ngram_mod lifecycle, low-acceptance handling, occupancy/reset behavior.

## What is currently known (from raw long-run log)

Source artifact:

- docs/rocm-deep-plan/raw_logs_llm_server_run.log

Observed run profile:

- ctx=65536, batch=4096, ubatch=192, parallel=1
- cache-type-k/v=q4_0, flash-attn on
- spec-type=ngram-mod (n_min=48, n_match=24, n_max=64)

Long-run facts extracted from the log:

- Context overflow request errors are frequent: 28 occurrences.
- Prompt-cache maintenance latency is highly variable:
  - n=29 updates, min=0.01 ms, avg=441.64 ms, max=1148.90 ms.
- Checkpoint churn is high:
  - restored checkpoints: 76
  - invalidated checkpoints erased: 245
  - checkpoints created: 236
- Forced full prompt re-processing appears (2 occurrences).
- Prompt and decode throughput are highly variable across requests:
  - prompt TPS: n=76, min=10.94, avg=343.76, max=580.42
  - decode TPS: n=76, min=9.82, avg=14.88, max=24.25
- Speculative acceptance is unstable:
  - draft acceptance: n=67, min=0.01562, avg=0.29399, max=0.84375
  - decode bucket summary:
    - acceptance < 0.1: avg decode TPS 12.22
    - 0.1 <= acceptance < 0.3: avg decode TPS 14.88
    - acceptance >= 0.3: avg decode TPS 16.64
- Many prompts finish prefill with batch.n_tokens=192 tail chunks, indicating recurrent tail-shape behavior under this lane.

## Root-cause hypothesis

In this long-run server lane, throughput instability is materially affected by server-level lifecycle behavior (request overflow handling, prompt-cache updates, checkpoint restore/invalidate policy, speculative acceptance swings), not only by GPU kernel-level routing.

## Solution strategy (implementation later)

1. Overflow guardrail before heavy scheduling work: add strict request budget guard rails so requests that cannot fit effective context are rejected or shaped earlier and more deterministically.

1. Checkpoint churn reduction: improve checkpoint restore candidate selection and invalidation behavior to avoid repeated deep fallback and large reprocessing windows.

1. Prompt-cache maintenance stabilization: add policy guards around expensive cache update/save/load paths to reduce latency spikes in hot request flow.

1. Adaptive speculative stability: add low-overhead adaptive controls for ngram-mod under sustained low acceptance streaks to avoid paying speculative overhead when it is not beneficial.

1. Long-run observability surface: add compact counters and periodic summaries for overflow, cache update latency, checkpoint churn, and acceptance bands.

## Planned code changes (not applied yet)

- tools/server/server-context.cpp
  - Add optional preflight context-budget guard path before expensive slot/cache work.
  - Add counters for cache update durations and checkpoint restore/invalidate operations.
  - Add guarded policy extension for checkpoint restore candidate selection under long-run churn.
- common/common.h
  - Add optional knobs for server long-run guard behavior (feature-flagged, default-off).
- common/arg.cpp
  - Add CLI switches for new long-run guard knobs.
- common/speculative.cpp
  - Add optional adaptive ngram-mod guard behavior and summary counters (default-off).

## Validation plan (after implementation)

Long-run server contract for P5:

- model: models/Qwen3.6-27B-Q3_K_S.gguf
- ctx: 65536
- batch/ubatch: 4096/192
- cache: enabled, q4_0/q4_0
- spec: ngram-mod (48/24/64 baseline)

Required metrics per run:

1. Overflow errors count (HTTP 400 exceed-context)
2. Prompt-cache update latency distribution (p50/p95/p99)
3. Checkpoint restore/create/invalidate counts per successful request
4. Forced full prompt re-processing count
5. Prompt TPS and decode TPS distributions on successful requests
6. Speculative acceptance distribution and decode TPS by acceptance bucket

Acceptance criteria:

- Overflow errors from avoidable over-budget requests are reduced by policy (or transformed into earlier deterministic guard behavior).
- Prompt-cache update p95 is reduced versus baseline long-run run.
- Checkpoint invalidation churn per successful request is reduced versus baseline.
- No new crash/hang behavior and no output-integrity regressions.

## Risks

- More aggressive guards can reject requests users previously expected to run.
- Reduced checkpoint density can hurt reuse in some prompt patterns.
- Adaptive speculative controls can regress quality or throughput on specific workloads.
- Additional counters can increase logging overhead if not sampled carefully.

## Rollback criteria

- Any new correctness issue (output mismatch, malformed responses, request handling breakage).
- Any reproducible throughput regression beyond noise on successful requests.
- Any increase in crash/hang or non-deterministic server behavior.

## Open questions

- Should overflow handling be hard-reject only, or optional auto-shaping with explicit budget reserve?
- What checkpoint spacing policy minimizes churn for this hybrid/recurrent long-run pattern?
- Is adaptive speculative gating better keyed by acceptance streak, decode TPS trend, or both?
- Which cache update operations should be deferred out of hot request path without harming reuse quality?
