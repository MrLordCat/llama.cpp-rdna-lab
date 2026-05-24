# E205 ROCm Q3_K padded storage P0

## Metadata

- Experiment ID: E205
- Date: 2026-05-24
- Owner: Copilot
- Branch/Commit: master after E204
- Target lane: ROCm Q3_K backend storage correctness-first branch

## Hypothesis

- Statement: Vulkan-like Q3_K layout gains can only transfer to ROCm through a backend-private padded storage contract, not through storage-blind helper rewrites.
- Mechanism: if ROCm owns a padded Q3_K storage format end-to-end, later MMVQ/MMQ/dequant routes can use more vector-friendly loads without violating the current `110-byte` raw GGUF block assumptions.
- Why now: E199 and E200 already rejected the helper-only shortcut and mapped the required blast radius. The remaining valid path is correctness-first.

## Math / Theory

- Assumptions:
  - E199/E200 are the current gates: no speed claim until `test-backend-ops` Q3_K `MUL_MAT` passes on padded storage.
- Expected speedup corridor:
  - none at P0; this is a correctness branch.
- Failure conditions:
  - partial storage changes create silent offset/view/copy corruption;
  - the branch is benchmarked before basic correctness smoke passes.

## Implementation Plan

1. Minimal code surface to change:
   - none yet in P0; this note opens the correctness-first branch only.
2. Guard rails:
   - env-gated only;
   - no server benchmark before backend correctness passes.
3. Rollback path:
   - if the storage branch becomes too broad without a clean P1/P2 split, stop and re-scope before editing code.

## Benchmark Plan

- Baseline command:
  - current-tree `test-backend-ops` / Q3_K `MUL_MAT` smoke once the branch exists.
- Candidate command:
  - pending P1 implementation.
- Number of runs:
  - pending.
- Artifacts path:
  - `build_logs/agent-workload/e205-rocm-q3k-padded-storage-*`

## Metrics

- Q3_K `MUL_MAT` correctness pass/fail
- offset/view/copy correctness
- no TPS claim at P0

## Result

- Outcome: pending
- Delta: pending
- Confidence: pending
- Recommendation: pending

## Notes

- Surprises: pending
- Follow-up action:
  - start P1 only after the direct-route branch either lands or stalls;
  - keep this route separate from H42, because one is compute-route work and the other is storage-contract work.