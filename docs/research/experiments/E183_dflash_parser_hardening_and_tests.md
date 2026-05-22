# E183 - DFlash Parser Hardening and Unit-Test Coverage

## Metadata

- Experiment ID: E183
- Date: 2026-05-22
- Owner: Copilot
- Branch/Commit: local working tree (no commit yet)
- Target lane: implementation hardening (no TPS claim)

## Scope

Harden DFlash argument parsing and add direct unit coverage for DFlash parser contracts:

1. Add cross-field validation for DFlash depth bounds.
2. Add cross-field validation for DFlash adaptive acceptance thresholds.
3. Extend `test-arg-parser` with DFlash positive/negative parse cases.

## Key Changes

### Parser Validation

- Updated `common/arg.cpp` post-parse validation in `common_params_parse()`:
  - reject `--spec-dflash-n-min > --spec-dflash-n-max`;
  - reject `--spec-dflash-accept-low > --spec-dflash-accept-high`.

This avoids silent normalization or ambiguous runtime behavior when the controller limits are contradictory.

### Unit Tests

- Updated `tests/test-arg-parser.cpp`:
  - invalid parse case: `n-min > n-max` (must fail);
  - invalid parse case: `accept-low > accept-high` (must fail);
  - valid parse case with full `--spec-dflash-*` set (must succeed and match expected values).

## Validation

1. Build test target:
   - `cmake --build build-rocm-vec --target test-arg-parser --config Release -j 8`
2. Run test binary:
   - `build-rocm-vec/bin/test-arg-parser.exe`
3. Diff hygiene:
   - `git diff --check -- common/arg.cpp tests/test-arg-parser.cpp`

## Result

- Outcome: Keep.
- Delta: no TPS claim (parser/test hardening pass).
- Recommendation: proceed to DFlash runtime correctness tests that exercise actual token generation and acceptance-loop behavior under `--spec-type dflash`.
