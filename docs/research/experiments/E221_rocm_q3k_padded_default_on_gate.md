# E221 ROCm Q3_K padded default-on gate

## Metadata

- Experiment ID: E221
- Date: 2026-05-24
- Owner: Copilot
- Hypothesis ID: H43
- Target lane: ROCm Q3_K padded-storage default policy gate

## Hypothesis

- Statement: after E219/E220 hardening, H43 might be ready for default-on on HIP with env opt-out (`...=0`).
- Mechanism: switch padded storage and padded MMQ gates to default enabled on HIP, then run no-env backend correctness smoke.
- Why now: user asked to continue H43 autonomously toward full readiness.

## Math / Theory

- Assumptions:
  - if storage contract is default-ready, no-env `test-backend-ops` Q3_K `MUL_MAT`/`MUL_MAT_ID` should match CPU reference;
  - this gate is correctness-first; speed is secondary.
- Failure conditions:
  - NaN or high error in canonical Q3_K no-env smoke.

## Benchmark Plan

- Build: `cmake --build build-rocm-vec --target llama-server test-backend-ops -j 8`
- No-env smoke:
  - `test-backend-ops test -b ROCm0 -o MUL_MAT,MUL_MAT_ID -p "type_a=q3_K|type=q3_K|type_src=q3_K|type_dst=q3_K"`

## Result

- Outcome: reject default-on and revert policy patch.
- Delta:
  - no-env smoke failed hard: multiple `MUL_MAT` and `MUL_MAT_ID` rows showed NaN / large error; only `3/13` passed.
  - after reverting default-on policy, opt-in smoke returned to green: `13/13` pass with `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`.
- Confidence: high for rejection of current default-on attempt.
- Recommendation: keep H43 default-off policy for now. Next work must identify why no-env default path diverges (allocation/route ordering or mixed raw/padded assumptions outside opt-in contract) before another default promotion attempt.

## Measured Data

Default-on attempt (no env):

- `build_logs/agent-workload/e221-rocm-q3k-defaulton-broad-smoke.txt`
- `MUL_MAT` failures include NaN and error near 1.0.
- `MUL_MAT_ID` includes NaN failure.
- Final: `3/13` passed, backend FAIL.

After rollback to opt-in policy:

- `build_logs/agent-workload/e221b-rocm-q3k-after-revert-broad-smoke.txt`
- Final: `13/13` passed, backend OK.

## Notes

- This gate prevented shipping an unsafe default.
- E219/E220 safety hardening remains valid and kept; only default policy attempt was reverted.
