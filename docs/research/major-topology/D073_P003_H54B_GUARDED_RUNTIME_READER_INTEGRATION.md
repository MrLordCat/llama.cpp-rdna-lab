# D073 - P003 H54-B guarded runtime reader integration

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: runtime reader integrated (guarded, fail-closed)

## Scope

Integrate a minimal runtime-sidecar reader in core C++ load path under strict
opt-in env gate, with fail-closed behavior and no default-path changes.

## What Was Implemented

Code integration:

- `src/llama-model-loader.cpp`

Runtime contract:

- enabled only when `LLAMA_Q4_METACOMP_ENABLE=1`;
- sidecar path read from `LLAMA_Q4_METACOMP_SIDECAR`;
- if gate is off, no runtime work is performed;
- parse/load failures do not throw and fall back to legacy path.

Reader behavior (current MVP):

- reads sidecar payload;
- extracts tensor names from `selected` section;
- validates names against loaded model `weights_map`;
- validates tensor type is Q4 family (`Q4_0`, `Q4_1`, `Q4_K`);
- logs summary counters (`selected/unique/validated_q4/missing/non_q4/duplicates`).

Fail-closed policy:

- missing env sidecar path -> warn and fallback;
- sidecar parse/open error -> warn and fallback;
- empty selected set -> warn and fallback;
- per-tensor missing or non-Q4 rows are counted and marked for fallback.

## Validation

- incremental build validation passed:
  - `cmake --build build-vulkan --config Release --target llama-server`
- IDE diagnostics on modified file: no errors.

## Decision

Keep as guarded runtime-reader integration step.

- default behavior remains unchanged;
- sidecar path is now visible in runtime logs under explicit env opt-in;
- next stage is wiring validated sidecar rows into an actual Q4 runtime transform
  path, followed by controlled A/B and quality checks.

## Artifacts

- `src/llama-model-loader.cpp`
- `docs/research/major-topology/D073_P003_H54B_GUARDED_RUNTIME_READER_INTEGRATION.md`
