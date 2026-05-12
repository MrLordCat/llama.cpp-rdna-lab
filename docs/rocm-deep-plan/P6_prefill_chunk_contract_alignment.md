# P6 Prefill Chunk Contract Alignment

## Objective

Increase active-lane prompt-heavy throughput by reducing prefill inefficiency from chunk-contract mismatch between model graph chunking and RDNA4 GDN runtime chunking.

## Implementation status (2026-05-11)

- Implemented in code (P6-A/B/C core hooks):
- `src/models/delta-net-base.cpp`: adaptive non-KDA chunk policy, contract trace, chunk override/policy parsing.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: fused runtime chunk policy alignment (`adaptive/default/override`) + contract trace parity fields.
- `src/llama-batch.cpp`: opt-in planner sync (`LLAMA_UBATCH_SHAPE_CHUNK_HINT_SYNC_DELTA=1`) to consume delta-net contract hints.
- Build validation done:
- CPU build: `cmake --build build-cpu --config Release -j 8` (pass).
- HIP build: `cmake --build build-rocm-vec --config Release --target ggml-hip -j 4` (pass).
- Runtime smoke done:
- `p6-impl-smoke-r2` (adaptive policy): contract trace shows `source=adaptive-policy` and `chunk_size=96` for `n_tokens=192`.
- `p6-impl-sync64-smoke-r1` (explicit override=64): planner trace + runtime trace confirm synced contract (`source=model-override`, `chunk_size=64`, `chunk_tail=0` on `ub192`).
- Promotion note: full `runs=3` gate for active-lane keep/reject is still pending (smoke only).

## Scope note

- Primary target lane: `ctx=12288`, `b/ub=6144/192`, `q4_0/q4_0`, `v2-review`, `repo-snapshot chars=21872`, no-reuse, thinking enabled.
- Secondary safety lane: `ctx=16384` with the same workload contract.
- This document is planning-only. No runtime code changes are applied here.

## Code study map

- `src/models/delta-net-base.cpp`: `build_delta_net_chunking()`, non-KDA default `CS=64`, env override `LLAMA_DELTA_NET_CHUNK_SIZE`.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: `launch_gated_delta_net()`, RDNA4 chunked prefill policy `chunk_size=96/128`, env override `GGML_GDN_CHUNK_SIZE`.
- `src/llama-batch.cpp`: `llama_shape_planned_ubatch()` (`shape-score`, chunk-tail penalties and planner hints).
- `src/llama-context.cpp`: ubatch execution boundaries and prompt-processing flow.

## Evidence gathered in this research pass (2026-05-11)

- Code-level contract mismatch is real and explicit:
- non-KDA model graph chunking uses `CS=64` by default in `delta-net-base.cpp`.
- RDNA4 runtime chunked prefill uses adaptive `chunk_size=96/128` in `gated_delta_net.cu`.
- Active lane is still in `~8.5 TPS` corridor with prefill-dominant behavior (recent `ub192` gates and confirmations).
- `ub194` cliff was fixed by shape planning, but the restored throughput only returns to `ub192` corridor and does not create a new higher plateau.
- `GGML_GDN_CHUNK_SIZE={64,80,96,128}` at `ub192` was already screened and remained in noise.
- `LLAMA_FUSED_GDN_CH=0` (disable chunked prefill) is a no-go due to first prompt-batch hang.
- Wider `ubatch` bands (`>=512`) remain a known prefill regression zone in this lane.

## Root-cause hypothesis

The model graph currently chunks non-KDA DeltaNet with a fixed `CS=64`, while RDNA4 runtime prefill uses `96/128` chunking. This split optimization may create avoidable padding and transition overhead in long incoming prompts, especially near tail boundaries, even when each subsystem is locally tuned.

## Implementation details (execution blueprint)

P6-A: Contract observability first (no behavior changes)

- Add low-overhead trace output in `delta-net-base.cpp` for selected model `CS`, `n_tokens`, `pad`, and `n_chunks`.
- Keep runtime trace in `gated_delta_net.cu` aligned with model trace fields so one benchmark run can correlate both contracts.
- Gate: trace-only patch must be performance-neutral within noise.

P6-B: Adaptive non-KDA model chunk selection

- Add an opt-in policy for non-KDA `CS` selection among `{64, 96, 128}`.
- Scoring priorities: minimal padding, low tiny-tail probability, stable chunk count.
- Keep strict fallback to current `CS=64` on unsupported/ambiguous shapes.

P6-C: Planner coupling to selected chunk contract

- Extend planner-hint path in `llama-batch.cpp` so `shape-score` can avoid local tails hostile to selected model/runtime chunk contract.
- Keep this coupling opt-in and default-off until active-lane gates pass.

P6-D: Promotion path

- Screen with `runs=1` on active lane.
- Confirm with `runs=3` only for promising candidate(s).
- Promote only if correctness and performance gates pass together.

## Planned code changes (not applied yet)

- `src/models/delta-net-base.cpp`: non-KDA adaptive `CS` selector and trace hook.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: optional contract trace fields for per-request chunk correlation.
- `src/llama-batch.cpp`: optional chunk-hint harmonization path for `shape-score`.
- `scripts/agent_workload_bench.py`: no required logic changes; existing diagnostics are sufficient for A/B evidence.

## Theoretical confirmation matrix

| Claim | Status | Evidence |
| --- | --- | --- |
| Model/runtime chunk contract mismatch exists and is measurable | Confirmed | `delta-net-base.cpp` non-KDA `CS=64`; `gated_delta_net.cu` RDNA4 `chunk_size=96/128` |
| Runtime chunk-size override alone can unlock breakthrough on active lane | Rejected | `GGML_GDN_CHUNK_SIZE` sweep at `ub192` stayed in noise |
| Disabling chunked prefill is a safe optimization fallback | Rejected | `LLAMA_FUSED_GDN_CH=0` hang on first prompt batch |
| Contract-aligned adaptive model chunking can improve active-lane prefill | Confirmed in theory | prefill-dominant lane + unresolved model/runtime contract split |
| P6 can be validated without immediate risky kernel surgery | Confirmed | stageable rollout with trace-first then opt-in policy |

## Theoretical verdict (go/no-go)

Verdict: GO for scoped implementation.

Confidence: medium-high.

Rationale:

- The mismatch is concrete in code and not yet directly attacked by prior successful patches.
- Already-tested no-go options (chunk override only, disabling chunked prefill) do not invalidate contract-alignment strategy.
- The plan is implementable with low-risk trace-first staging and clear rollback conditions.

## Validation plan

Gate A: build reliability

- Configure and build pass for default ROCm profile (`build-rocm-vec`) on Windows/HIP 7.1.

Gate B: correctness

- No crash/hang on first prompt batch.
- No benchmark request timeout on active lane.

Gate C: performance

- Primary keep threshold: `>= +10%` aggregate TPS on active lane vs current `ub192` reference.
- Supporting signal: prompt-eval milliseconds reduced (`>= 12%`) with decode metrics not regressing beyond noise.
- Confirmation: `runs=1` for screening, `runs=3` for keep decision.

Gate D: rollback safety

- Any reproducible regression above noise or any hang in active lane blocks promotion.

## Risks

- Adaptive model chunking may alter numerical path and require stricter correctness checks.
- Overfitting to one lane may regress neighboring contexts (`ctx=16384`).
- Added policy complexity can reduce maintainability if not well-instrumented.

## Rollback criteria

- Any reproducible correctness issue.
- Aggregate TPS regression above noise in active lane.
- New instability in prompt processing flow.

## Open questions

- Should model chunk policy be fully automatic or constrained by a small explicit preset table?
- Is chunk-hint coupling best done in planner (`llama-batch`) or in model graph (`delta-net-base`) only?
- Which metric predicts improvements better in this lane: pad ratio, chunk count, or tail class distribution?
