# DFlash Future Workflow

Purpose: define how DFlash work continues after initial integration so performance, stability, and maintenance remain predictable.

## Workflow Modes

## Mode A - Implementation

Use when a phase is not yet complete.

1. follow `IMPLEMENTATION_RUNBOOK.md` strict phase order;
2. keep one active DFlash phase branch;
3. require explicit keep/reject decision for each phase.

## Mode B - Stabilization

Use after full functional path exists.

1. prioritize correctness and crash-free behavior;
2. tune only with lane-matched A/B;
3. avoid broad refactors mixed with tuning changes.

## Mode C - Maintenance

Use after initial DFlash release.

1. monitor upstream divergence and local compatibility;
2. keep docs, presets, and tests synchronized;
3. gate risky improvements behind opt-in flags first.

## Ongoing Weekly Cadence

1. Review new upstream changes touching speculative/runtime/backend surfaces.
2. Re-run DFlash smoke matrix on current local ROCm toolchain.
3. Re-run one cold-first and one repeated-session benchmark lane.
4. Record drift or regressions in `RESULTS_LOG.md` and experiment notes.

## Change Intake Policy

For every new DFlash idea:

1. create/update an experiment note first;
2. classify as one of: correctness, perf, telemetry, UX;
3. pick smallest safe file surface;
4. define rollback signal before coding.

## Testing Matrix (Minimal Ongoing)

1. CLI parse and startup:
   - `spec-type none` unchanged;
   - `spec-type dflash` valid and invalid contract paths.
2. Runtime correctness:
   - deterministic short output smoke;
   - no corruption after accept/rollback cycles.
3. Backend:
   - ROCm configure/build sanity;
   - fallback path when ring hooks are unavailable.
4. GUI:
   - args emitted correctly;
   - non-DFlash defaults preserved.

## Benchmark Workflow (Ongoing)

1. Keep two permanent dashboards in docs:
   - cold-first lane;
   - repeated-session lane.
2. Require lane parity in all reported deltas.
3. Treat repeated-only gain as repeated-only claim; do not relabel as default speedup.

## Documentation Workflow

When any DFlash behavior changes:

1. update `VENDOR_MANIFEST.md` if source anchors changed;
2. update `PHASE_PLAYBOOK.md` or `IMPLEMENTATION_RUNBOOK.md` if process changed;
3. append decision row in `RESULTS_LOG.md`;
4. ensure experiment note links artifacts.

## Handoff Workflow Between Agents

1. active agent writes current phase, branch, and next command in latest DFlash experiment note;
2. incoming agent starts from that note and `IMPLEMENTATION_RUNBOOK.md`;
3. if handoff marker fails, continue execution by explicit branch+phase contract, not by chat-only instruction.

## Emergency Workflow

If severe regression appears:

1. disable DFlash path by default immediately;
2. keep feature behind explicit opt-in switch;
3. revert smallest responsible commit group;
4. file incident-style experiment note with root cause and corrective gate.
