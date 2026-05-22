# DFlash Branch and Commit Plan

Purpose: keep DFlash rollout reversible and easy to review while parallel ROCm work is active.

## Branch Layout

1. `feature/dflash-phase0-cli-skeleton`
2. `feature/dflash-phase1-runtime-cpu-safe`
3. `feature/dflash-phase2-backend-hooks`
4. `feature/dflash-phase3-server-adaptive`
5. `feature/dflash-phase4-gui`

Each phase branch rebases on current master after previous phase merge.

## Commit Slicing Rules

1. One concern per commit.
2. Separate runtime logic from telemetry/logging commits.
3. Separate docs commits from code commits.
4. Keep fallback-path changes in isolated commit for fast revert.

## Suggested Commit Sequence

## Phase 0

1. enum and parser support for `spec-type dflash`.
2. fail-closed state creation and contract checks.
3. docs and help text.

## Phase 1

1. add `src/models/dflash_draft.cpp` and compile integration.
2. context/cparams/graph plumbing for capture metadata.
3. API surface updates in `include/llama.h`.
4. smoke checks and docs.

## Phase 2

1. backend proc hooks registration.
2. ring/interleave implementation.
3. reduced verifier argmax path (gated).
4. ROCm lane validation artifacts.

## Phase 3

1. adaptive draft-depth controller.
2. profile categories and acceptance hist.
3. guardrails for fallback safety.

## Phase 4

1. GUI controls.
2. preset and UX labels.
3. docs update.

## Revert Strategy

1. if phase fails gates, revert whole phase branch or single concern commit.
2. never leave partial adaptive policy enabled by default.
3. preserve non-DFlash behavior parity as hard requirement.

## Handoff Checklist

Before handing to implementation agent:

1. phase objective stated in PR body.
2. exact lane commands listed.
3. expected artifacts and pass/fail criteria listed.
4. links to `VENDOR_MANIFEST.md` and source provenance included.
