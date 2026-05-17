# E042 Deep Hypothesis Audit: Math, Code Quality, and Research Inputs

## Metadata

- Experiment ID: E042
- Date: 2026-05-17
- Owner: Copilot
- Type: deep process/mathematical upgrade

## Why this was needed

Recent C01 experiments often failed to cross +1% wall because:

1. low-ceiling centers were tested as if they could move wall metrics,
2. activation and coupling risks were discovered too late (after code edits),
3. candidate math gate was not enforced as a hard pre-implementation screen.

## External research inputs used

1. Roofline model principles (work, memory traffic, arithmetic intensity, ceilings/walls).
2. ROCm Compute Profiler documentation pointers (profile/analyze modes and performance-model framing).

Use these as method references, not as direct architecture claims for this repo.

## New quantitative gate

Added tool:

- `scripts/research/required_local_speedup.py`

Model:

- `S_total = 1 / ((1 - s) + s / S_local)`
- Solve for `S_local` given target center wall-share `s` and desired `S_total`.

For current C01 high-share center (`s ~= 0.3901`):

- +1% wall requires `S_local ~= 1.0260`
- +2% wall requires `S_local ~= 1.0529`
- +3% wall requires `S_local ~= 1.0807`
- +5% wall requires `S_local ~= 1.1390`

Interpretation:

- high-share centers can realistically produce >1% wall with moderate local wins,
- low-share centers require unrealistic local gains and should be deprioritized.

## Workflow upgrades (hard rules)

1. Ceiling-first:
   - before code edits, compute required local speedup from measured share;
   - reject if required local gain is implausible for that center.

2. Activation-first:
   - prove shape/route presence in trace before implementation.

3. Coupling-first:
   - if target bucket wins but pre_sync/sync or neighbor center regresses, treat as non-keep.

4. Comparability-first:
   - baseline/candidate signature must match (E040 audit gate).

## Expected outcome

This does not guarantee immediate wins, but it reduces low-value experiments and increases keep-rate by filtering non-viable hypotheses before coding.
