# P8 RDNA4 FATTN Selector Retargeting

## Objective

Determine whether RDNA4 FlashAttention selector retargeting can provide material speedup on the active lane, and under which shape regimes it is theoretically valid.

This pass upgrades P8 into a falsifiable theory with explicit NO-GO conditions for the current lane.

## Scope

- Primary lane: `ctx=12288`, `b=6144`, no-reuse, repo-snapshot workload.
- Secondary lane: `ctx=16384` for stability checks.

## Code anchors (selector mechanics)

- `ggml/src/ggml-cuda/fattn.cu:711-769`
	- RDNA4 branch for vec/tile/mma selection.
	- Quantized path uses hard threshold on `Q1 * gqa_ratio_eff`.
- `ggml/src/ggml-cuda/fattn.cu:813-831`
	- final `base -> selected` decision and `GGML_TRACE_FATTN_SELECTED` tracing.
- `ggml/src/ggml-cuda/fattn-qwen-reduced.cpp:132-154`
	- reduced mode base rule (`Q1 <= 2 -> vec`, else `wmma`) and force override.

## Evidence corpus (verifiable)

1. Reduced A/B force test (`ub192`):
	 - force-vec: aggregate `2.1306`, prompt `580.15 tok/s`, decode `27.98 tok/s`
	 - force-wmma: aggregate `2.9211`, prompt `823.04 tok/s`, decode `27.51 tok/s`
	 - sources:
		 - `build_logs/agent-workload/nonmtp-fa-reduced-forcevec-ub192-mt32-20260511-r1.diagnostics.md`
		 - `build_logs/agent-workload/nonmtp-fa-reduced-forcewmma-ub192-mt32-20260511-r1.diagnostics.md`

2. New manual run (`ub300`) in default profile:
	 - prompt `826.90`, decode `27.65`, aggregate `8.5199`
	 - source: `build_logs/agent-workload/p1-manual-20260511-202428-shape-ub300-r1.diagnostics.md`

3. New manual run (`ub512`) in default profile:
	 - prompt `336.24`, decode `27.18`, aggregate `4.2357`
	 - source: `build_logs/agent-workload/p1-manual-20260511-202458-shape-ub512-r1.diagnostics.md`

4. Baseline cliff comparison (`ub192` vs `ub194`, base policy):
	 - prompt drops (`826.35 -> 594.10`), decode stays near-flat (`27.63 -> 27.78`)
	 - sources:
		 - `build_logs/agent-workload/p1-gate-20260511-174248-base-ub192-r1.diagnostics.md`
		 - `build_logs/agent-workload/p1-gate-20260511-174521-base-ub194-r1.diagnostics.md`

## What this proves

- For active Qwen shape bands, broad vec forcing is harmful.
- Large regressions are dominated by prompt path while decode stays in a narrow corridor.
- Therefore selector retargeting is not the primary lever for the observed active-lane collapse.

## Formal applicability test

P8 can produce meaningful gain only if all are true:

1. Selector ambiguity exists on high-frequency shape bands.
2. Alternative kernel is faster for those exact bands.
3. Those bands carry enough wall-time share.

If any fails, expected gain is near zero or negative.

In active lane, condition #1 is weak because high `Q1` bands are far from vec/tile boundary thresholds in RDNA4 rules.

## Why current lane is near NO-GO for P8 primary objective

- RDNA4 branch uses small-threshold decisions (`Q1 * gqa_ratio_eff <= 4`) for vec/tile shortcuts.
- Active lane `Q1` bands (`~192`, `300`, `512`) are far above that region.
- Reduced A/B demonstrates forcing vec on these bands hurts prompt throughput severely.

Conclusion: selector tweaks are unlikely to unlock the main active-lane speed target.

## Parameter transfer (other configs)

P8 can still be useful in other regimes, but only conditionally:

- Candidate-positive regimes:
	- workloads with significant low-`Q1` occupancy near selector boundaries;
	- different model shapes where selector ambiguity is frequent.
- Candidate-negative regimes:
	- large-`Q1` dominated prompt-heavy bands where wmma/mma path is already clearly better.

## Verdict

Verdict: CONDITIONAL GO globally, but NO-GO as primary speed track for the current active lane.

Priority placement:

- Primary: P7 (GDN prefill hotpath).
- Secondary: P8 only as targeted side-track with strict band filters.

## Execution blueprint (if pursued)

P8-A: shape histogram

- Build `(Q0, Q1, gqa_ratio_eff, K/V types, K1 bucket)` frequency table from traces.

P8-B: band-limited overrides

- Add opt-in reason-coded overrides only for ambiguous/high-impact bands.

P8-C: reduced smoke, then default confirm

- Reject immediately if reduced A/B does not show prompt uplift.

## Performance gates

- Gate 1 (screen): prompt-eval uplift in target bands must be clearly above noise.
- Gate 2 (promotion): `>= +8%` aggregate TPS on active lane or matched wall-time reduction.
- Gate 3: `runs=3` confirmation before promotion.

## Risks

- Selector overfitting and maintenance complexity.
- Hidden regressions on untested shape bands.
- Compile-set constraints limiting real dispatch options.

## Rollback criteria

- Any reproducible throughput regression above noise.
- Any correctness/runtime stability issue.
