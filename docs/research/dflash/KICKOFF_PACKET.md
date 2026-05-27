# DFlash Kickoff Packet

Purpose: one-screen startup contract for the next implementation agent.

## Mission

Implement Phase 0 first, then continue full DFlash rollout phase-by-phase.

Primary references:

1. `docs/research/dflash/IMPLEMENTATION_RUNBOOK.md`
2. `docs/research/dflash/PHASE_PLAYBOOK.md`
3. `docs/research/dflash/BRANCH_AND_COMMIT_PLAN.md`
4. `docs/research/dflash/VENDOR_MANIFEST.md`

## First Branch and Scope

1. Create branch: `feature/dflash-phase0-cli-skeleton`.
2. Edit scope only:
   - `common/arg.cpp`
   - `common/speculative.h`
   - `common/speculative.cpp`
3. Do not mix unrelated ROCm micro-optimization edits.

## Phase 0 Deliverables

1. `--spec-type dflash` is parsed and selectable.
2. DFlash contract validation exists and is fail-closed.
3. Clear startup error when drafter/target contract is invalid.
4. Existing speculative modes remain behaviorally unchanged.

## Minimal Command Sequence

1. `git status --short --branch`
2. `cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release`
3. `python -m py_compile run.py gui/main_window.py gui/server_tab.py gui/benchmark_tab.py gui/build_tab.py gui/build_manager.py gui/dependency_checker.py gui/hardware_detector.py`
4. `git diff --check`

## Commit Slices (Phase 0)

1. enum/CLI parse slice.
2. fail-closed contract slice.
3. docs/help slice.

## Evidence Contract

1. Add one experiment note entry for phase completion or rejection.
2. Append one row to `docs/research/RESULTS_LOG.md`.
3. Link artifacts in `build_logs/agent-workload/` when applicable.

## Immediate Stop Conditions

1. non-DFlash mode behavior changes.
2. ambiguous contract errors.
3. inability to revert a single failing slice cleanly.

If any stop condition hits, revert the smallest failing slice and re-run checks.
