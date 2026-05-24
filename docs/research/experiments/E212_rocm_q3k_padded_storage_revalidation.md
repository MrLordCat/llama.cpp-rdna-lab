# E212 ROCm Q3_K Padded Storage Revalidation

## Metadata

- Experiment ID: E212
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after `7b3327cab`
- Target lane: H43 opt-in revalidation on ROCm `build-rocm-vec`, Qwen3.6-27B-Q3_K_S, active repo-snapshot lane

## Hypothesis

- Statement: after E209/E210 safety patches and the reverted E211 probe, the known E201-P2a padded-storage/MMQ opt-in route should still produce a real wall signal on the active prompt-heavy lane.
- Mechanism: no code change; compare same-build default control against `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` on the cold-first repo-snapshot lane.
- Why now: the user asked whether the last speedup is in code and default. It is in code but still opt-in; before any further default-readiness work, confirm the opt-in route still behaves on the real server lane after the safety commits.

## Math / Theory

- Assumptions:
  - E201-P2a active prompt-heavy r1 showed `11.8483 -> 12.0795 TPS` (`+1.95%`);
  - E209/E210 should not change single-GPU non-split speed;
  - E211 was reverted and should not affect runtime.
- Expected speedup corridor:
  - around `+1%..+2%` r1 on the active lane if the previous signal still holds.
- Failure conditions:
  - candidate output errors/corruption;
  - candidate loses to same-build control;
  - no speed claim if only one side is run or lane contract differs.

## Benchmark Plan

- Control: active repo-snapshot lane, no padded env.
- Candidate: same command with `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`.
- Number of runs: r1 only unless the signal is borderline and worth confirming.
- Artifacts path: `build_logs/agent-workload/e212-rocm-q3k-padded-reval-*`.

## Metrics

- aggregate completion TPS
- prompt eval TPS
- decode eval TPS
- errors/output sanity

## Result

- Outcome: validated opt-in win, but smaller than the original E201-P2a active r1 signal when confirmed with same-build r3.
- Delta:
  - r1 control vs candidate: `11.7304 -> 11.9473 TPS` (`+1.85%` wall), prompt `1193.60 -> 1216.77 tok/s`, decode `30.16 -> 30.70 tok/s`, errors `0`.
  - r3 control vs candidate: `12.0989 -> 12.1813 TPS` (`+0.68%` wall), median task TPS `12.2719 -> 12.3539` (`+0.67%`), prompt `1258.0533 -> 1259.2167 tok/s`, decode `30.1600 -> 30.6267 tok/s`, errors `0`.
- Confidence: medium. The r3 comparison keeps the sign positive with matching lane contract, but most of the r3 gain is decode-side while prompt is effectively neutral.
- Recommendation: keep `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` as a real opt-in route. Do not make it default yet; E209/E210 reduced async/split risk, but default-readiness still needs broader tensor movement/view/MoE coverage and a policy decision for split buffers.

## Notes

- This is not a new route. It is a guard against accidentally carrying stale speed assumptions after safety work.
- The user's "is the last speedup still in code?" answer is yes: it is committed and working as opt-in. The "is it default?" answer is no: default remains raw Q3_K storage until the remaining safety/correctness surface is closed.

## Artifacts

- `build_logs/agent-workload/e212-rocm-q3k-padded-reval-control-r1.csv`
- `build_logs/agent-workload/e212-rocm-q3k-padded-reval-candidate-r1.csv`
- `build_logs/agent-workload/e212-rocm-q3k-padded-reval-control-r3.csv`
- `build_logs/agent-workload/e212-rocm-q3k-padded-reval-candidate-r3.csv`
- `build_logs/agent-workload/e212-rocm-q3k-padded-reval-control-r3.diagnostics.md`
- `build_logs/agent-workload/e212-rocm-q3k-padded-reval-candidate-r3.diagnostics.md`
