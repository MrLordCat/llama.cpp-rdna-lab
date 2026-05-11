# P1 Prefill Shape Route Blocker

## Objective

Stabilize and speed up prompt-heavy prefill on the active lane by removing shape-dependent slow routes without regressing decode.

## Code study map

- src/llama-batch.cpp
  - llama_shape_planned_ubatch()
  - split_equal() single-sequence planner path
  - env controls: LLAMA_UBATCH_SPLIT_POLICY, LLAMA_UBATCH_SHAPE_PREFERRED, LLAMA_UBATCH_SHAPE_MIN_TAIL, LLAMA_UBATCH_TRACE
- src/llama-context.cpp
  - process_ubatch()
  - scheduler switching logic (TG vs PP)
  - timing controls: LLAMA_UBATCH_TIMING, LLAMA_UBATCH_TIMING_SYNC
- ggml/src/ggml-cuda/gated_delta_net.cu
  - launch_gated_delta_net()
  - RDNA4 chunked prefill selection and chunk size policy
  - env control: GGML_GDN_CHUNK_SIZE
- src/models/delta-net-base.cpp
  - build_delta_net_fused() and keep_intermediates routing

## What is currently known

- ub192 is the local peak on active lane, while ub194 can sharply regress with similar decode throughput.
- The observed regression is prefill-dominant and shape-dependent.
- GDN chunk-size override sweeps around ub192 were effectively noise.
- Disabling fused/chunked GDN is a no-go because it can hang on the first large prompt batch.

## Theoretical validation in current environment

Environment assumptions for this validation:

- OS: Windows 11
- GPU: RDNA4 (gfx1201)
- ROCm/HIP: 7.1 toolchain
- Active lane: Qwen3.6-27B-Q3_K_S, ctx=12288, b=6144, ub=192, q4_0/q4_0, no-reuse

Data points used:

- Baseline (ub192):
  - aggregate TPS: 8.4180
  - prompt_eval_ms: 9828.46
  - decode_eval_ms: 4387.23
- Trace A/B (ub192 vs ub194):
  - ub192: aggregate 8.4661, prompt 9779.83 ms, decode 4358.78 ms
  - ub194: aggregate 6.6646, prompt 13581.42 ms, decode 4361.42 ms
- Shape planner check (ub256 + preferred 192):
  - aggregate 8.4414, prompt 9801.02 ms, decode 4378.53 ms

Observed facts from these numbers:

- Prefill dominates wall time on the active lane:
  - prefill share ~69.1%
  - decode share ~30.9%
- ub194 regression is almost pure prefill regression:
  - prompt time delta (ub194 vs ub192): +38.87%
  - decode time delta (ub194 vs ub192): +0.06%
  - wall time delta (ub194 vs ub192): +26.91%
- Shape-aware planner already proved partial recovery behavior (ub256 path can be pulled back toward ub192-class performance by forcing preferred shape 192).

Quantified target requirement for breakthrough:

- Let total wall proxy be: total_ms = prompt_eval_ms + decode_eval_ms.
- For +10% wall throughput, required prefill reduction is approximately 13.1%.
- Equivalent decode-only requirement would be approximately 29.5% reduction.

Interpretation:

- In this environment, P1 (prefill route control) is mathematically the highest-leverage path.
- A moderate prefill improvement can cross the +10% wall target; decode-only work needs much larger gains to match it.

## Potential verdict

Verdict: real potential is confirmed for P1 in the current environment.

Confidence and expected bands:

- Conservative band: +5% to +9% wall (small route-stability gains).
- Target band: +10% to +16% wall if prefill is reduced by roughly 13% to 20% without decode regressions.
- High upside band (>20%): possible only if pathological shape routes are avoided broadly (ub194-like prefill behavior eliminated across the prompt stream, not just at one boundary).

What is explicitly ruled out by prior evidence:

- Pure chunk_size forcing alone is insufficient (already tested around ub192 and ub512).
- Disabling fused/chunked GDN cannot be used as an optimization path because it is unstable (hang risk).
- Host-side scheduling overhead is not the primary bottleneck on this lane; device prefill kernels/routes remain the main lever.

## Explicit confirmation status (trace synthesis 2026-05-11)

Status:

- P1 is explicitly confirmed as the highest-probability acceleration path for the active lane.
- Final production-speed confirmation is still pending one minimal empirical gate before implementation.

Evidence strength level:

- Causality (prefill route -> wall TPS): strong.
- Transferability to robust production gain: moderate until one more targeted check is passed.

Confirmed now:

- ub192 vs ub194 is a prefill-dominant divergence with near-constant decode:
  - ub192: aggregate 8.4661, prompt_eval_ms 9779.83, decode_eval_ms 4358.78.
  - ub194: aggregate 6.6646, prompt_eval_ms 13581.42, decode_eval_ms 4361.42.
  - Prompt delta is +38.87%, decode delta is +0.06%.
- Trace route split differs exactly at the prefill shape boundary:
  - ub192 recurrent prefill path uses n_tokens=192 with chunk pattern 96 + 96.
  - ub194 recurrent prefill path uses n_tokens=194 with chunk pattern 96 + 96 + 2.
- Flash attention selector is not the differentiator in this pair:
  - both ub192 and ub194 repeatedly select wmma_f16 for the hot prefill path.
- Independent cliff evidence (ub800 fast vs ub832 slow) follows the same shape-route logic:
  - ub800 repeatedly ends prefill with 128 tail 32, while ub832 repeatedly ends with 128 tail 64.
  - corresponding aggregate TPS drops from 9.9074 to 3.6181 with wall dominated by prompt_eval_ms.
- Shape planner control already shows partial recovery behavior without decode harm:
  - baseline ub192 aggregate 8.4180 vs shape-planned ub256 preferred-192 aggregate 8.4414.
  - decode metrics stay effectively unchanged.

Not yet proven:

- Which exact kernel-level mechanism is triggered by the small-tail shape transitions (2 or 64 token tails).
- How broadly the same shape rule generalizes across different prompt distributions in this lane.
- That a patched planner will consistently clear the +10% wall target on cold-first runs.

Minimal next empirical check before implementation:

1. Keep code unchanged and run one strict boundary micro-matrix under the same lane contract:
- ub190, ub192, ub194, ub196 with identical cache/no-reuse and thread settings.
- one run each for screening, then three runs only for the best candidate and baseline.

2. Acceptance to unlock implementation:
- decode_eval_ms variation within noise (<=1%), and
- prompt_eval_ms monotonic/stable improvement around ub194 boundary, and
- aggregate TPS improvement >=5% in the boundary candidate.

If this gate passes, proceed to minimal shape-policy implementation guarded by env flag.

## Pre-implementation theory gates

Before any code change starts, the following theory gates must be satisfied:

1. Route attribution gate
- Ensure candidate shape policy can explain at least one reproducible prefill-only delta with near-constant decode metrics.

2. Stability gate
- No strategy may rely on disabling fused/chunked GDN or any known no-go toggle.

3. Build feasibility gate
- Proposed instrumentation must stay compatible with current ROCm build corridor for this machine.

4. Benefit gate
- Candidate must have a plausible path to >=13% prefill reduction on active lane proxy metrics.

If any gate fails, do not move to implementation for that candidate.

## Root-cause hypothesis

The slowdown is primarily caused by specific prefill shape sequences that trigger less favorable GDN/FATTN kernel behavior, not by decode logic and not by host scheduling overhead.

## Solution strategy (implementation later)

1. Shape planner hardening (runtime side)
- Extend single-sequence planner logic so it optimizes for a stable shape sequence, not just local tail avoidance.
- Introduce deterministic scoring for candidate splits with guardrails against known bad tails.

2. Prefill reserve profile (scheduler side)
- Add a prefill reserve policy that better matches actual planned shapes.
- Avoid over-reserving graph variants that are not used in the active lane.

3. Route visibility improvements
- Keep low-overhead trace summaries for GDN/FATTN selected routes and token histograms.
- Avoid high-volume traces in normal benchmark runs.

## Implemented code changes (2026-05-11)

- src/llama-batch.cpp
  - Added `shape-score` split policy for single-sequence prefill planning.
  - Added deterministic candidate scoring with penalties for:
    - tiny final tails,
    - tiny chunk tails relative to `LLAMA_UBATCH_SHAPE_CHUNK_HINT`,
    - over-fragmentation from excessive chunk count.
  - Preserved legacy `tail-avoid` policy unchanged under its own branch.
  - Added optional env controls:
    - `LLAMA_UBATCH_SHAPE_MIN_STEP`
    - `LLAMA_UBATCH_SHAPE_CHUNK_HINT`
    - `LLAMA_UBATCH_SHAPE_MIN_CHUNK_TAIL`
  - Added trace output for chosen candidate in `shape-score` mode.

- src/PERF_RESEARCH_NOTES.md
  - Updated env and test notes to reflect `shape-score` workflow.

## Gate execution result (2026-05-11)

Lane contract used:

- `tasks=v2-review`, `ctx=12288`, `b=6144`, `q4_0/q4_0`, `spec=none`, no-reuse, `--no-disable-thinking`.

Screening (`runs=1`):

- baseline `ub192`: `8.51 TPS` (`p1-gate-20260511-174248-base-ub192-r1`)
- shape-score `ub190`: `8.43 TPS`
- shape-score `ub192`: `8.54 TPS`
- shape-score `ub194`: `8.53 TPS`
- shape-score `ub196`: `8.54 TPS`
- control baseline `ub194` (policy off): `6.71 TPS` (`p1-gate-20260511-174521-base-ub194-r1`)

Confirmation (`runs=3`):

- baseline `ub194`: `6.83 TPS` (`p1-confirm-20260511-174606-base-ub194-r3`)
- shape-score `ub194`: `8.52 TPS` (`p1-confirm-20260511-174606-shape-ub194-r3`)
- baseline `ub192`: `8.51 TPS` (`p1-confirm-20260511-174606-base-ub192-r3`)

Computed deltas from diagnostics and CSV:

- shape-score `ub194` vs baseline `ub194`:
  - aggregate TPS: `+24.73%`
  - prompt_eval_ms: `-26.42%`
  - decode_eval_ms: `+0.10%` (within noise)
- shape-score `ub194` vs baseline `ub192`:
  - aggregate TPS: `+0.08%`
  - prompt_eval_ms: `-0.11%`
  - decode_eval_ms: `-0.10%`

Verdict:

- P1 gate passed for boundary recovery objective: `ub194` cliff is removed under `shape-score` and restored to `ub192` class throughput without decode regression.

## Next validation steps

Primary benchmark template:

- scripts/agent_workload_bench.py
- tasks=v2-review
- ctx=12288, b=6144, ub=192 reference
- q4_0/q4_0, spec none, no-reuse

Required A/B comparisons:

1. Baseline ub192 vs patched ub192
2. Baseline ub194 vs patched ub194
3. Shape-planned ub256 with preferred 192 vs baseline ub192

Acceptance criteria:

- Minimum +10 percent wall TPS on comparable cold-first run, or clear path to it with stable repeatability.
- No decode regression beyond noise.
- No hangs in first prompt batch.

## Planned research-to-implementation sequence

1. Confirm final theoretical shortlist
- Keep only candidates that pass all pre-implementation theory gates.

2. Implement minimal route-safe variant first
- Start with the smallest shape policy extension that can be toggled by env flag.

3. Validate on active lane only
- Compare against ub192 reference first, then boundary probes (ub194, ub256 preferred-192).

4. Expand only if target band is reached
- If gains stay below the conservative band, rollback and stop branch expansion.

## Risks

- Planner can overfit one prompt shape distribution and regress on others.
- Additional planner logic can increase complexity in split behavior for multi-sequence cases.
- Prefill reserve tuning can accidentally increase graph rebuilds.

## Rollback criteria

- Any hang or deadlock risk in prompt processing.
- Any reproducible regression larger than noise on active lane baseline.
- Any instability in scheduler switching between TG and PP.

## Open questions

- Which exact shape signatures are consistently pathological across multiple prompts?
- Is a static rule enough, or does planner need a small runtime lookup table?
