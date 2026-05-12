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

## Evidence gathered in this research pass (2026-05-11)

Source-matrix pressure (HIP CMake assembly, default `GGML_CUDA_FA_ALL_QUANTS=OFF`):

- Normal profile compiles `137` HIP `.cu` translation units.
- Reduced profile compiles `103` HIP `.cu` translation units.
- Delta: `-34` units in reduced profile.

Where the `-34` comes from:

- Reduced profile excludes direct heavy files: `fattn.cu`, `fattn-tile.cu`.
- Reduced profile skips template groups: `fattn-tile*.cu` (`11`) and `fattn-mma*.cu` (`21`).

Dry-run fanout evidence (`cmake --build ... -- -d explain -n`):

| Scenario | Normal profile (`build-rocm-vec`) | Reduced profile (`build-rocm-fa-reduced`) | Conclusion |
| --- | --- | --- | --- |
| touch `fattn.cu` | Rebuild `fattn.cu.obj` + relink chain to `llama-server` (`7` steps) | `ninja: no work to do.` | Reduced corridor fully removes FATTN edit fanout. |
| touch `mmvq.cu` | Rebuild `mmvq.cu.obj` + relink chain to `llama-server` (`7` steps) | Rebuild `mmvq.cu.obj` + relink chain to `llama-server` (`7` steps) | MMVQ pressure is unchanged by reduced corridor. |

Artifacts:

- `build_logs/agent-workload/p3-dryrun-normal-fattn.txt`
- `build_logs/agent-workload/p3-dryrun-reduced-fattn.txt`
- `build_logs/agent-workload/p3-dryrun-normal-mmvq.txt`
- `build_logs/agent-workload/p3-dryrun-reduced-mmvq.txt`

Related historical evidence:

- `BENCHMARKS.md` and P2 dossier already recorded repeated `amdgcn-link ... signal` failures around pre-split MMVQ iteration and later unblock after P2 split.

## Root-cause hypothesis

Large HIP translation units and broad template instantiation surfaces create unstable link pressure; experiment-friendly source partitioning and staged build modes are needed.

Current confidence after this pass: **high** for FATTN-side build pressure; **high** that MMVQ still needs a dedicated corridor.

## Theoretical leverage estimate

P3 does not primarily target direct TPS gain. It targets faster and safer hypothesis iteration.

- Research cycle model: `T_cycle = T_build + T_bench + T_analysis`.
- For FATTN-side edits, reduced profile can collapse `T_build` fanout to near-zero (no-op dry-run in current evidence).
- For MMVQ-side edits, current reduced profile gives no fanout relief; this is the remaining leverage opportunity for P3 implementation.

Practical implication:

- P3 can materially increase engineering throughput (more valid A/B cycles per day) even when single-run TPS is unchanged.

## Implemented strategy (2026-05-11)

1. Staged build modes

- Smoke mode: reduced source matrix for hypothesis checks.
- Final mode: normal source matrix for production-grade validation.
- Optional MMVQ-isolated mode: focused build for decode path experiments.

1. Translation-unit boundaries

- Keep selector/dispatcher logic in light files.
- Move heavy template instantiations into narrow, dedicated units.

1. Build reproducibility contract

- Standardize compile flags and command templates per mode.
- Keep explicit guardrails for ROCm Windows toolchain behavior.

## Theoretical confirmation matrix

| Claim | Status | Evidence |
| --- | --- | --- |
| Reduced mode is useful as build-pressure corridor for FATTN experiments | Confirmed | `ggml/src/ggml-hip/CMakeLists.txt`, `build_logs/agent-workload/p3-dryrun-normal-fattn.txt`, `build_logs/agent-workload/p3-dryrun-reduced-fattn.txt` |
| Reduced mode alone solves MMVQ build-pressure | Rejected | `build_logs/agent-workload/p3-dryrun-normal-mmvq.txt`, `build_logs/agent-workload/p3-dryrun-reduced-mmvq.txt` |
| P3 can improve end-to-end experimentation throughput | Confirmed in theory | fanout matrix above + P2 historical linker-pressure records in `BENCHMARKS.md` |
| P3 guarantees direct wall TPS uplift by itself | Rejected | P3 is an enabler phase, not a kernel-speed phase |

## Theoretical verdict (go/no-go)

**Verdict: GO for scoped implementation.**

Rationale:

- The phase has clear theoretical confirmation for its intended value (build/iteration throughput).
- The remaining unsolved hotspot is explicit and narrow (MMVQ-side pressure in both profiles).

Minimum implementation candidates for P3:

1. Centralize HIP source bundle composition in one place (reduce mode drift risk).
2. Add an optional MMVQ-focused experiment profile (decode-path corridor) that preserves runtime guardrails from P4.
3. Keep default profile behavior unchanged.

Implementation status: **completed for build-system scope**.

## Implemented code changes

- ggml/src/ggml-hip/CMakeLists.txt
  - Added fail-fast guard on Windows: HIP configure now errors early when `CMAKE_CXX_COMPILER` is not ROCm `clang++`/`hipcc`.
  - Switched HIP source assembly to centralized source-bundle collection function.
  - Added profile-aware compile definitions for no-FA mode (`GGML_CUDA_NO_FA`, `GGML_HIP_MMVQ_FOCUSED_PROFILE`).
- ggml/CMakeLists.txt
  - Added explicit profile selector option `GGML_HIP_EXPERIMENT_PROFILE` (`default`, `qwen-fa-reduced`, `mmvq-focused`, alias `mmvq-isolated`).
- ggml/src/ggml-hip/hip-source-bundles.cmake
  - New centralized HIP source-bundle module with profile routing and guarded FA template inclusion.
- ggml/src/ggml-cuda/fattn-qwen-reduced.cpp
  - Added no-FA-safe stubs/guards so MMVQ-focused profile links cleanly while FA is disabled.

## Validation results (build gates, 2026-05-11)

Build-only gates:

1. Configure gate passed for all modes:
   - `default` (`build-rocm-vec`)
   - `qwen-fa-reduced` (`build-rocm-fa-reduced`)
   - `mmvq-focused` (`build-rocm-mmvq-focused`)
2. Build/link gate passed for all modes:
   - `cmake --build ... --target llama-server -j 4` finished with `Linking CXX executable bin\\llama-server.exe` in all three profiles.
3. Toolchain guard gate passed:
   - Intentional bad configure (GNU/Strawberry C++) now fails early with clear error message instead of late HIP compile failure.

Artifacts:

- `build_logs/agent-workload/p3-implementation-build-gates.txt`
- `build_logs/agent-workload/p3-guard-bad-config.txt`

Runtime/perf gate check (2026-05-11):

- Active lane contract (`v2-review`, `repo-snapshot chars=21872`, `ctx=12288`, `b/ub=6144/192`, no-reuse), `default` (`build-rocm-vec`): passed, aggregate `8.54 TPS`.
- Active lane contract (`v2-review`, `repo-snapshot chars=21872`, `ctx=12288`, `b/ub=6144/192`, no-reuse), `qwen-fa-reduced` (`build-rocm-fa-reduced`): passed, aggregate `8.55 TPS`.
- Active lane contract (`v2-review`, `repo-snapshot chars=21872`, `ctx=12288`, `b/ub=6144/192`, no-reuse), `mmvq-focused` (`build-rocm-mmvq-focused`): request timed out (`TimeoutError('timed out')`), server log stalls at prompt progress `6144/8030`.
- Short decode-biased sanity lane (`tasks=quick`, no real-context, same ctx/b/ub, `max_tokens=64`), `default`: aggregate `26.57 TPS`.
- Short decode-biased sanity lane (`tasks=quick`, no real-context, same ctx/b/ub, `max_tokens=64`), `qwen-fa-reduced`: aggregate `26.59 TPS`.
- Short decode-biased sanity lane (`tasks=quick`, no real-context, same ctx/b/ub, `max_tokens=64`), `mmvq-focused`: aggregate `17.56 TPS` (substantial regression).
- Additional guard fact: with `--flash-attn off` and KV `q4_0/q4_0`, context init fails as expected: `V cache quantization requires flash_attn`.

Runtime verdict for P3 profiles:

- `default` and `qwen-fa-reduced` remain valid for active-lane performance measurements.
- `mmvq-focused` is **not** valid for active prompt-heavy lane and is retained only as a build-pressure / narrow debug profile.

Acceptance criteria:

- Repeated clean builds for experiment targets without link-stage failure. ✅
- No accidental runtime behavior drift between mode boundaries. ⚠️

Final P3 closure decision:

- P3 is closed for its primary objective (HIP build-pressure workflow and TU/source-matrix control).
- Runtime promotion of `mmvq-focused` is explicitly rejected based on measured regressions/timeouts.

## Risks

- Build matrix complexity can increase maintenance overhead.
- Mode drift can cause false conclusions if not documented strictly.

## Risk of mode drift and guardrails

- Any non-default mode must be explicitly labeled in benchmark artifacts.
- Final performance claims remain bound to normal ROCm profile only.
- If mode boundaries become ambiguous, P3 implementation must pause until source-bundle ownership is centralized.

## Rollback criteria

- Any mode introduces hidden behavior changes in default build.
- Build scripts become harder to operate than current baseline flow.

## Open questions

- Which exact source subsets maximize build stability while preserving representative routing behavior?
- Should mode selection be exposed in GUI build tooling later or remain CLI-only during research?
