# E300: ROCm Q3_K Four-Warp Recheck

Date: 2026-07-14

## Hypothesis

E289 reduced the hot direct Q3_K MMVQ kernel to 54 registers and the fused
gate/up kernel to 62 registers, both with modeled full occupancy.  This could
have reopened the formerly expensive four-wave launch geometry.

## Validation

The RDNA4 Q3_K `calc_nwarps` value was changed from 2 to 4.  The ROCm backend
built successfully and focused Q3_K `MUL_MAT` coverage passed 11/11 shapes.

Matched game-loaded short decode runs gave:

| Variant | Decode TPS | Prompt TPS |
| --- | ---: | ---: |
| 2 waves, control before | 27.36 | 420.67 |
| 4 waves | 25.41 | 406.68 |
| 2 waves, control after | 27.76 | 409.37 |

Four waves are about 7-9% slower than the surrounding controls.  With register
pressure no longer limiting occupancy, processing four rows per block reduces
grid parallelism and loses on this decode shape.

## Decision

Reject and restore two waves.  The production source and `ggml-hip.dll` were
rebuilt with `nwarps=2`.  Do not repeat this geometry unless a future kernel
changes row granularity or the occupancy model.

Primary artifacts:

- `e300-lol-rocm-q3-nwarps4-r1.*`;
- `e299-lol-rocm-ppsync-control-r1.*`;
- `e300-lol-rocm-q3-nwarps2-control-r2.*`.

