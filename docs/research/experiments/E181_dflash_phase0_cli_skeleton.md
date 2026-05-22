# E181 - DFlash Phase 0 CLI/Skeleton

## Metadata

- Experiment ID: E181
- Date: 2026-05-22
- Owner: Copilot
- Branch/Commit: local working tree after `caa451655`
- Target lane: implementation gate (no TPS claim)

## Hypothesis

- Statement: We can safely expose `--spec-type dflash` now, while keeping runtime fail-closed until DFlash execution path is implemented.
- Mechanism: Add enum/CLI/string-map wiring and a hard startup error in speculative init for the DFlash route.
- Why now: This unblocks phased implementation without pretending runtime/backend support already exists.

## Implementation

1. Added `COMMON_SPECULATIVE_TYPE_DFLASH` in `common/common.h`.
2. Added `dflash` CLI option handling in `common/arg.cpp`.
3. Added `dflash` name maps and `to_str` in `common/speculative.cpp`.
4. Added explicit fail-closed return path in `common_speculative_init()` when `params.type == dflash`.

## Validation

1. Build gate:
   - `cmake --build build-rocm-vec --target llama-server --config Release -j 8`
2. CLI visibility gate:
   - `build-rocm-vec/bin/llama-server.exe --help` shows `--spec-type ... dflash ...`
3. Repo hygiene gates:
   - `python -m py_compile gui/llama_gui.py gui/build_manager.py gui/dependency_checker.py gui/hardware_detector.py`
   - `git diff --check`

## Result

- Outcome: Keep.
- Delta: no TPS claim (contract/safety phase only).
- Recommendation: proceed to Phase 1 runtime path (`dflash_draft` and context/graph plumbing), preserving fail-closed behavior until hooks are complete.

## Notes

- This phase intentionally does not add backend ring hooks or GPU verifier shortcuts.
- Existing speculative modes remain unchanged by design.