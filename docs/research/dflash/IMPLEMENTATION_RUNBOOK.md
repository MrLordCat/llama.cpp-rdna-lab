# DFlash End-to-End Implementation Runbook

Status: planning and execution guide.
Intent: run the full DFlash rollout from first code edit to stable default policy.

## 0. Startup Packet (Before Any Code)

1. Confirm source baseline:
   - Bee repo: `../beellama.cpp`
   - commit pin: `c6dfa39e36` (update if implementation starts from another commit)
2. Confirm local target docs are current:
   - `docs/research/DFLASH_IMPLEMENTATION_PREP.md`
   - `docs/research/dflash/VENDOR_MANIFEST.md`
   - `docs/research/dflash/PHASE_PLAYBOOK.md`
   - `docs/research/dflash/BRANCH_AND_COMMIT_PLAN.md`
3. Lock lane contract for all DFlash performance claims:
   - cold-first lane and repeated lane must stay split;
   - no mixed comparisons across different tasks/tokens/reuse settings.

## 1. Implementation Order (Hard Sequence)

## Phase 0 - Spec and contract shell

Goal: add DFlash mode safely, no backend dependence.

Primary file order:

1. `common/arg.cpp`
2. `common/speculative.h`
3. `common/speculative.cpp`

Bee anchors to port carefully:

1. `COMMON_SPECULATIVE_TYPE_DFLASH` parsing and dispatch.
2. contract/fail-closed checks.
3. explicit debug/profile env guards default-off.

Definition of done:

1. `--spec-type dflash` accepted by CLI/server.
2. invalid drafter/target contract fails clearly.
3. non-DFlash modes behavior unchanged.

## Phase 1 - Functional runtime path (CPU-safe fallback first)

Goal: functional DFlash with correctness first.

Primary file order:

1. `src/models/dflash_draft.cpp` (new)
2. `src/llama-cparams.h`
3. `src/llama-graph.h`
4. `src/llama-graph.cpp`
5. `src/llama-context.h`
6. `src/llama-context.cpp`
7. `src/llama-arch.cpp`
8. `src/llama-model.cpp`
9. `include/llama.h`

Bee anchors to port carefully:

1. hidden capture contract and layer-id validation.
2. ring bookkeeping and prefill/verify offsets in cparams.
3. discard/rollback behavior on unsafe capture state.

Definition of done:

1. functional draft/verify/accept cycle with safe fallback.
2. deterministic smoke does not corrupt outputs.
3. graph reuse path remains valid when DFlash is off.

## Phase 2 - Backend hooks and ROCm path

Goal: enable performant ring path via ggml-cuda/HIP.

Primary file order:

1. `ggml/src/ggml-cuda/ggml-cuda.cu`
2. `ggml/src/ggml-cuda/cross-ring-interleave.cu`
3. `ggml/src/ggml-cuda/argmax.cu`
4. `ggml/src/ggml-cuda/CMakeLists.txt` (if wiring needed)

Bee anchors to port carefully:

1. cross-ring GPU init/set/free hooks.
2. device-to-device tensor set helper hooks.
3. reduced verifier path as explicit opt-in gate.

Definition of done:

1. hook discovery works on local ROCm build.
2. missing hook path falls back safely.
3. no crash across repeated runs and reuse toggles.

## Phase 3 - Server adaptive control

Goal: optional adaptive draft-depth and telemetry hardening.

Primary file order:

1. `tools/server/server-context.cpp`
2. `tools/server/server-adaptive-dm.h` (new/ported helper if needed)

Bee anchors to port carefully:

1. adaptive policy state transitions.
2. acceptance/coverage counters and profile categories.
3. deterministic rollback path under mismatch.

Definition of done:

1. adaptive mode predictable on/off behavior.
2. diagnostics are informative and default-off.
3. non-adaptive default behavior unchanged.

## Phase 4 - GUI integration

Goal: expose DFlash safely without changing existing defaults.

Primary file order:

1. `gui/server_tab.py`, `gui/benchmark_tab.py`, `gui/build_tab.py`
2. `gui/model_presets.json` (if presets are added)
3. GUI docs touched by UX changes

Definition of done:

1. GUI emits correct CLI args.
2. warnings for compatibility and lane intent are explicit.
3. non-DFlash presets remain unchanged.

## 2. Verification Contract Per Phase

Run after each phase:

1. `python -m py_compile run.py gui/main_window.py gui/server_tab.py gui/benchmark_tab.py gui/build_tab.py gui/build_manager.py gui/dependency_checker.py gui/hardware_detector.py`
2. `git diff --check`
3. CPU configure/build sanity if touched paths require it:
   - `cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release`
   - `cmake --build build-cpu --config Release -j`
4. ROCm configure sanity for backend-touching phases:
   - `cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release`

## 3. Benchmark Contract

DFlash performance reporting must include two separate headlines:

1. cold-first baseline/candidate;
2. repeated-session baseline/candidate.

Rules:

1. no cross-lane comparisons.
2. same tasks, token limits, context and reuse mode for A/B.
3. `--runs 1` for gate, `--runs 3` only for final confirmation.
4. keep thinking mode consistent.

## 4. Evidence and Logging Contract

For each phase completion or rejection:

1. update one experiment note under `docs/research/experiments/`;
2. append a row to `docs/research/RESULTS_LOG.md`;
3. store diagnostics in `build_logs/agent-workload/`;
4. include source provenance (Bee path + commit) in PR description.

## 5. Revert and Incident Playbook

Immediate rollback triggers:

1. non-DFlash mode regression;
2. output corruption in deterministic smoke;
3. graph reuse instability;
4. repeated crash on ROCm path.

Rollback policy:

1. revert smallest failing phase commit slice first;
2. keep fail-closed behavior always active during rollback;
3. do not keep partially working adaptive policy by default.

## 6. Release Gate for First Public DFlash Cut

Ship only if all pass:

1. functional CLI path with clear contract errors;
2. ROCm functional stability for chosen model lane;
3. reproducible repeated-session benefit or neutral outcome with opt-in default;
4. no cold-first default policy replacement;
5. docs and workflow updated.
