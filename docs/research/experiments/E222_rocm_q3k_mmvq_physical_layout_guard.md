# E222 ROCm Q3_K MMVQ physical-layout guard

## Metadata

- Experiment ID: E222
- Date: 2026-05-24
- Owner: Copilot
- Hypothesis ID: H43
- Target lane: ROCm Q3_K padded storage correctness hardening

## Hypothesis

- Statement: MMVQ Q3_K padded detection must be based on physical allocation layout, not env-presence, to avoid route divergence.
- Mechanism: remove direct `GGML_CUDA_Q3K_PADDED_STORAGE` requirement from `ggml_cuda_mmvq_q3k_padded_storage_tensor()` and keep strict physical-allocation match check.
- Why now: E221 showed default-on attempt can break correctness if one path still keys on env while others key on physical layout.

## Math / Theory

- Assumptions:
  - if tensor allocation size equals padded allocation contract, MMVQ must use padded interpretation regardless of env source;
  - env is a policy input, but kernel data interpretation must be derived from physical layout.
- Failure conditions:
  - no-env and opt-in broad Q3_K backend smokes regress.

## Benchmark Plan

- Build: `cmake --build build-rocm-vec --target test-backend-ops llama-server -j 8`
- No-env broad smoke:
  - `test-backend-ops test -b ROCm0 -o MUL_MAT,MUL_MAT_ID -p "type_a=q3_K|type=q3_K|type_src=q3_K|type_dst=q3_K"`
- Opt-in broad smoke:
  - same command with `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`

## Result

- Outcome: keep.
- Delta:
  - no-env broad smoke: `13/13` pass.
  - opt-in broad smoke: `13/13` pass.
- Confidence: medium-high (correctness hardening only, no performance claim).
- Recommendation: keep this change as prerequisite for any future H43 default-policy attempt.

## Measured Data

- `build_logs/agent-workload/e222-rocm-q3k-noenv-broad-smoke.txt`
- `build_logs/agent-workload/e222-rocm-q3k-optin-broad-smoke.txt`

## Notes

- This change does not re-enable default-on policy by itself.
- It removes one concrete env/physical mismatch class discovered during E221 rollback analysis.
