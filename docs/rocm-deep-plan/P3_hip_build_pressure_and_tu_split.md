# P3 HIP Build Pressure And TU Split

## Objective

Establish a reliable build workflow for aggressive kernel experiments so valid hypotheses can be tested quickly and safely.

## Code study map

- ggml/CMakeLists.txt
  - options: GGML_HIP_ROCWMMA_FATTN, GGML_HIP_QWEN_FA_REDUCED
- ggml/src/ggml-hip/CMakeLists.txt
  - source filtering when GGML_HIP_QWEN_FA_REDUCED is enabled
  - reduced FATTN source set and host-only reduced dispatcher wiring
  - compiler and language setup for HIP on Windows
- ggml/src/ggml-cuda/fattn.cu
  - kernel selector and reduced filter compatibility
- ggml/src/ggml-cuda/gated_delta_net.cu and mmvq.cu
  - known high-pressure edit targets

## What is currently known

- build-rocm-fa-reduced exists as an experiment corridor to bypass heavy FATTN template pressure.
- Reduced mode is valid for A/B smoke testing but is not itself a speedup.
- Fresh edits in fattn/gdn/mmvq areas can trigger link-stage failures on this machine.

## Root-cause hypothesis

Large HIP translation units and broad template instantiation surfaces create unstable link pressure; experiment-friendly source partitioning and staged build modes are needed.

## Solution strategy (implementation later)

1. Staged build modes
- Smoke mode: reduced source matrix for hypothesis checks.
- Final mode: normal source matrix for production-grade validation.
- Optional MMVQ-isolated mode: focused build for decode path experiments.

2. Translation-unit boundaries
- Keep selector/dispatcher logic in light files.
- Move heavy template instantiations into narrow, dedicated units.

3. Build reproducibility contract
- Standardize compile flags and command templates per mode.
- Keep explicit guardrails for ROCm Windows toolchain behavior.

## Planned code changes (not applied yet)

- ggml/src/ggml-hip/CMakeLists.txt
  - Add optional experiment profile for MMVQ-isolated builds.
  - Keep existing reduced FATTN path and clarify source bundles.
- ggml/CMakeLists.txt
  - Add explicit option grouping for experiment profiles.
- potentially new CMake include fragment
  - centralize source bundle definitions for experiment modes.

## Validation plan (after implementation)

Build-only gates:

1. Configure and build each mode with deterministic flags.
2. Verify llama-server target links successfully in each mode.
3. Verify no source drift between smoke and final mode unless intended.

Runtime gates:

1. Smoke mode used only for route hypothesis checks.
2. Final claims only from normal ROCm build runs.

Acceptance criteria:

- Repeated clean builds for experiment targets without link-stage failure.
- No accidental runtime behavior drift between mode boundaries.

## Risks

- Build matrix complexity can increase maintenance overhead.
- Mode drift can cause false conclusions if not documented strictly.

## Rollback criteria

- Any mode introduces hidden behavior changes in default build.
- Build scripts become harder to operate than current baseline flow.

## Open questions

- Which exact source subsets maximize build stability while preserving representative routing behavior?
- Should mode selection be exposed in GUI build tooling later or remain CLI-only during research?
