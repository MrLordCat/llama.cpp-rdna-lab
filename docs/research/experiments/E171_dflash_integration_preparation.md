# E171 - DFlash Integration Preparation

## AGENT SWITCH INSTRUCTION (LIVE HANDOFF)

Instruction for another agent: switch now to DFlash implementation track and execute `docs/research/dflash/PHASE_PLAYBOOK.md` from Phase 0. Do not continue ROCm micro-optimization work in this handoff window.

## Metadata

- Experiment ID: E171
- Date: 2026-05-22
- Owner: Copilot (prep pass)
- Branch/Commit: local working tree
- Target lane: documentation and implementation planning only

## Hypothesis

- Statement: A staged DFlash port from BeeLlama can be integrated into this fork without destabilizing active ROCm performance work if we separate phases and gates.
- Mechanism: Integrate DFlash first as a safe speculative mode skeleton, then add backend hooks and adaptive server logic.
- Why now: Active ROCm tuning is ongoing in parallel; preparation can de-risk the upcoming merge window.

## Math / Theory

- Assumptions:
  - DFlash gains are strongest on repeated/decode-heavy structured outputs.
  - cold-first prompt-heavy lane may not gain and can regress.
- Expected speedup corridor:
  - no claim in this preparation step.
- Failure conditions:
  - introducing DFlash breaks non-DFlash speculative modes or server stability.

## Implementation Plan

1. Minimal code surface to change:
   - `common/speculative.*`, `common/arg.cpp`, `src/models/dflash_draft.cpp` (new), context/graph glue, server verify loop.
2. Guard rails:
   - DFlash off by default;
   - explicit `--spec-type dflash` opt-in;
   - profile/debug toggles default-off.
3. Rollback path:
   - keep DFlash in isolated commits by phase (CLI/state, runtime graph, backend hooks, server adaptive).

## Benchmark Plan

- Baseline command: existing lane controls (`spec-type none`) per `PERF_WORKSPACE.md`.
- Candidate command: same lane plus `--spec-type dflash` once functional.
- Number of runs:
  - `--runs 1` for gate;
  - `--runs 3` for confirmation of promising deltas.
- Artifacts path:
  - `build_logs/agent-workload/`.

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate/correctness
- draft acceptance and coverage stats
- prefill/decode split

## Result

- Outcome: planning milestone complete.
- Delta: no TPS claim.
- Confidence: high for scope mapping; medium for effort estimate until phase-0 code lands.
- Recommendation: proceed with Phase-0 implementation skeleton after active ROCm tuning batch completes.

## Notes

- Main finding: DFlash is a large subsystem port, not a micro-optimization.
- Follow-up action: use `docs/research/DFLASH_IMPLEMENTATION_PREP.md` plus phase playbooks in `docs/research/dflash/` as execution checklist.
