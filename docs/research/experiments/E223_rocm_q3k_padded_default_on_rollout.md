# E223 ROCm Q3_K padded default-on rollout

## Metadata

- Experiment ID: E223
- Date: 2026-05-24
- Owner: Copilot
- Hypothesis ID: H43
- Target lane: ROCm Q3_K padded storage default policy

## Hypothesis

- Statement: after E220-E222 predicate hardening, HIP can safely run Q3_K padded storage as default-on with explicit opt-out.
- Mechanism:
  - set HIP policy to default enabled in `ggml_cuda_q3k_padded_storage_enabled()` and `ggml_cuda_q3k_padded_storage_mmq_enabled()`;
  - keep explicit opt-out via `GGML_CUDA_Q3K_PADDED_STORAGE=0` and `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=0`;
  - align MMQ policy (`mmq.cu`) with the same HIP default/opt-out semantics;
  - keep MMVQ on physical-layout detection from E222.

## Build / Correctness Gates

- Build:
  - `cmake --build build-rocm-vec --target test-backend-ops llama-server -j 8`
- Broad Q3_K backend smokes (`MUL_MAT`,`MUL_MAT_ID`):
  - no-env (new default): `13/13` pass;
  - explicit off (`...=0`): `13/13` pass;
  - explicit on (`...=1`): `13/13` pass.

## Control A/B Bench Plan

- A/B lane 1 (primary prompt-heavy):
  - model `Qwen3.6-27B-Q3_K_S.gguf`
  - `ctx=12288`, `b=4096`, `ub=1024`, q4/q4 KV, FA on
  - `tasks=quick`, `task_ids=triage_diff`, `max_tokens=64`
  - `real-context-mode=repo-snapshot`, thinking on, no reuse, no prime, `spec=none`, `runs=1`
- A/B lane 2 (32k control lane):
  - `ctx=32768`, `b=5120`, `ub=1024`, q4/q4 KV, FA on
  - `tasks=v2-mini`, `task_ids=v2_write_function`, `max_tokens=120`
  - `real-context-mode=repo-snapshot`, thinking on, no reuse, `runs=1`

## Measured Results

- 12k lane:
  - control explicit-off: `7.20 TPS`
  - candidate default/no-env: `7.25 TPS`
  - delta: `+0.69%`
- 32k lane:
  - control explicit-off: `11.03 TPS`
  - candidate default/no-env: `11.07 TPS`
  - delta: `+0.36%`

## Decision

- Keep and promote default-on policy for HIP Q3_K padded storage/MMQ with explicit env opt-out.
- Rationale:
  - correctness gates are green in default, explicit-off, and explicit-on modes;
  - measured A/B is neutral-to-small-plus in both required lanes;
  - prior E221 failure mode (env/layout divergence) is closed by E222 + this policy alignment.

## Artifacts

- Correctness:
  - `build_logs/agent-workload/e223-rocm-q3k-noenv-broad-smoke.txt`
  - `build_logs/agent-workload/e223-rocm-q3k-explicit-off-broad-smoke.txt`
  - `build_logs/agent-workload/e223-rocm-q3k-explicit-on-broad-smoke.txt`
- 12k A/B:
  - `build_logs/agent-workload/e223-rocm12k-defaultoff-control-r1.diagnostics.md`
  - `build_logs/agent-workload/e223-rocm12k-defaulton-candidate-r1.diagnostics.md`
- 32k A/B:
  - `build_logs/agent-workload/e223-rocm32k-defaultoff-control-r1.diagnostics.md`
  - `build_logs/agent-workload/e223-rocm32k-defaulton-candidate-r1.diagnostics.md`
