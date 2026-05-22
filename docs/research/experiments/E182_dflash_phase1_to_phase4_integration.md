# E182 - DFlash Phase 1-4 Integration Pass

## Metadata

- Experiment ID: E182
- Date: 2026-05-22
- Owner: Copilot
- Branch/Commit: local working tree (no commit yet)
- Target lane: implementation integration (no TPS claim)

## Scope

Deliver a working DFlash path across runtime/server/backend/GUI with fail-safe behavior:

1. Runtime (Phase 1): DFlash now runs through draft-backed speculative state instead of hard fail-closed.
2. Backend hooks (Phase 2): CUDA/HIP backend exposes optional DFlash helper proc hooks.
3. Server control (Phase 3): adaptive DFlash draft-depth controller with acceptance thresholds.
4. GUI surface (Phase 4): DFlash mode selectable in server/benchmark UI and emits DFlash flags.

## Key Changes

### Runtime / Common

- Added `common_params_speculative_dflash` config block and wiring in `common/common.h`.
- Added CLI flags in `common/arg.cpp`:
  - `--spec-dflash-n-min`
  - `--spec-dflash-n-max`
  - `--spec-dflash-adaptive`
  - `--spec-dflash-accept-low`
  - `--spec-dflash-accept-high`
- Updated `common/speculative.cpp`:
  - DFlash contract: requires draft context/model.
  - DFlash implementation path uses existing draft state with DFlash-specific min/max depth.
  - Removed phase-0 hard fail-closed for DFlash runtime.

### Backend Hooks

- Added DFlash helper API declarations in `ggml/include/ggml-cuda.h`.
- Implemented helper hooks in `ggml/src/ggml-cuda/ggml-cuda.cu` and exported via backend proc-address:
  - `ggml_backend_cuda_dflash_cross_ring_supported`
  - `ggml_backend_cuda_dflash_d2d_copy`

### Core API / Server

- Added `llama_dflash_backend_hooks_available()` API in `include/llama.h` and `src/llama-context.cpp`.
- `tools/server/server-context.cpp` now:
  - fail-closes launch when `--spec-type dflash` is used without draft model;
  - reports whether backend helper hooks are available;
  - applies adaptive per-slot DFlash draft depth by acceptance ratio.

### GUI

- Added DFlash mode to:
  - `gui/llama_gui.py`
  - `gui/server_tab.py`
  - `gui/benchmark_tab.py`
  - `gui/build_tab.py` mode resolution order
- GUI now emits `--spec-type dflash` and core `--spec-dflash-*` args.

## Validation

1. Build gate:
   - `cmake --build build-rocm-vec --target llama-server --config Release -j 8`
2. Python syntax gate:
   - `python -m py_compile gui/llama_gui.py gui/server_tab.py gui/benchmark_tab.py gui/build_tab.py`
3. CLI surface gate:
   - `llama-server --help` includes `dflash` and `--spec-dflash-*` flags.
4. Diff hygiene:
   - `git diff --check`

## Result

- Outcome: Keep.
- Delta: no TPS claim (integration pass).
- Recommendation: run lane-matched cold-first and repeated/session A/B before any speed claims.
