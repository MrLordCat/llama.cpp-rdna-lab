# DFlash Phase Playbook (No-Code Preparation)

Status: planning only.
Goal: make implementation start deterministic once active ROCm work is free.

## Phase 0 - Spec Skeleton and Fail-Closed Contract

## Scope

- add DFlash mode in speculative type and CLI surface;
- add strict validation path with clear error messages;
- no backend ring hooks yet.

## Target files

- `common/arg.cpp`
- `common/speculative.h`
- `common/speculative.cpp`
- optional docs updates in `docs/speculative.md`

## Deliverables

1. `--spec-type dflash` recognized.
2. required DFlash draft model parameters parsed.
3. fail-closed startup path when contract is invalid.

## Must-not-break checks

1. existing modes (`none`, `draft`, `mtp`, `ngram-*`) behave unchanged.
2. server starts in non-DFlash modes exactly as before.

## Gate commands

1. quick compile targets used by active workflow.
2. `python -m py_compile` for GUI files remains clean.
3. `git diff --check` clean.

## Exit criteria

- DFlash mode is visible in CLI help and route selection.
- no runtime calls to missing backend hooks.

## Phase 1 - Functional DFlash Runtime Path (CPU-safe first)

## Scope

- introduce DFlash drafter graph path;
- introduce target hidden capture plumbing and context params;
- keep GPU ring optional/off if unavailable.

## Target files

- `src/models/dflash_draft.cpp` (new)
- `src/llama-context.h`
- `src/llama-context.cpp`
- `src/llama-cparams.h`
- `src/llama-graph.h`
- `src/llama-graph.cpp`
- `src/llama-arch.cpp`
- `src/llama-model.cpp`
- `include/llama.h`

## Deliverables

1. DFlash draft cycle executes with CPU-safe hidden cross path.
2. deterministic correctness smoke passes on single-slot runs.
3. rollback/accept logic is stable and fail-closed.

## Correctness gates

1. same seed/temp outputs are well-formed (no symbol spam/corruption).
2. abort paths are explicit on invalid model contract.
3. no graph reuse corruption with DFlash toggles.

## Exit criteria

- functional DFlash path without mandatory CUDA ring dependency.

## Phase 2 - Backend Hooks (CUDA/HIP path used by ROCm build)

## Scope

- add cross-ring and d2d helper hooks to backend registry;
- wire reduced verifier path only behind explicit toggles;
- keep diagnostics default-off.

## Target files

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/cross-ring-interleave.cu`
- `ggml/src/ggml-cuda/argmax.cu`
- required build glue in `ggml/src/ggml-cuda/CMakeLists.txt` if needed

## Deliverables

1. hook discovery from runtime succeeds on supported backend.
2. fallback to CPU-safe path if hook is missing.
3. no regression for non-DFlash workloads.

## Perf-gate policy

1. cold-first lane and repeated lane reported separately.
2. no speed claim without lane-matched A/B.

## Exit criteria

- stable DFlash execution on local ROCm build path.

## Phase 3 - Server Adaptive Control and Hardened Telemetry

## Scope

- adaptive n-draft controller and profile categories;
- acceptance histogram and context-bucket diagnostics;
- keep controls minimal and avoid dead knobs.

## Target files

- `tools/server/server-context.cpp`
- optional helper: `tools/server/server-adaptive-dm.h`

## Deliverables

1. adaptive policy can be enabled/disabled predictably.
2. cycle metrics expose enough data for tuning.
3. no hidden fallback behavior on verifier mismatch.

## Exit criteria

- adaptive mode is stable and does not regress defaults when off.

## Phase 4 - GUI Integration and Presets

## Scope

- expose DFlash mode and core args in GUI;
- keep defaults unchanged for non-DFlash users;
- add explicit warnings about lane intent.

## Target files

- `gui/llama_gui.py` and/or modular GUI tab files
- `gui/model_presets.json` if preset support is added

## Deliverables

1. GUI emits correct server args.
2. preset labels clearly separate cold-first vs repeated/session routes.

## Exit criteria

- GUI launch parity with CLI for DFlash mode.

## Cross-Phase Safety Rules

1. no broad copy from Bee repo; keep surgical ports.
2. each phase ends with docs updates and explicit keep/reject status.
3. negative probes reverted unless intentionally env-gated.
4. no mixed-lane TPS claims.
