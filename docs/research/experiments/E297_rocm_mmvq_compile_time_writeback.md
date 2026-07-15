# E297: ROCm MMVQ Compile-Time Writeback

Date: 2026-07-14

## Hypothesis

Upstream commit `683f0c72e` moved MMVQ result writeback inside the compile-time
unrolled row loop. This avoids indexing local accumulators with `threadIdx.x`
and removes runtime bias branches. On HIP this could keep Q3_K accumulators in
registers and reduce writeback instructions.

The change was ported onto this fork's Q3_K padded-storage and fused pair-dot
paths without changing arithmetic.

## Correctness and Resources

- ROCm build passed.
- Focused Q3_K `MUL_MAT` coverage passed 11/11 shapes for `N=1..9`.
- Hot direct Q3_K remained at 54 registers and 100% modeled occupancy.
- Hot fused Q3_K remained at 62 registers and 100% modeled occupancy.

The unchanged resource profile shows that clang 21 already eliminated the main
dynamic-local-memory risk in the current hot specialization.

## Game-Loaded A/B

The 21,634-prompt-token / 64-output dual-GPU sequence was candidate, control,
candidate, control:

| Variant | Decode TPS | Prompt TPS |
| --- | ---: | ---: |
| Candidate 1 | 23.89 | 1,563.78 |
| Control 1 | 22.87 | 1,520.39 |
| Candidate 2 | 23.59 | 1,535.96 |
| Control 2 | 23.62 | 1,606.61 |

Candidate mean decode was 23.74 versus 23.25 for control, nominally +2.1%, but
prompt throughput moved in the opposite direction and exposed substantial game
load drift.

A decode-heavy 159-prompt-token / 256-output check was even less stable:

| Variant | Decode TPS |
| --- | ---: |
| Control before | 25.33 |
| Candidate | 26.61 |
| Control after | 29.72 |

The control swing is much larger than the proposed optimization.

## Decision

Reject for now and restore the original MMVQ writeback. The candidate did not
change kernel resources and no gain survived the load-stability gate. Revisit
only in a clean A/B if another upstream change depends on the same structure;
do not claim the nominal game-loaded delta as a speedup.

Primary artifacts:

- `e297-lol-rocm-mmvq-compile-index-resources4k-mt4-r1.*`;
- `e297-lol-rocm-compile-index-candidate24k-mt64-r1.*`;
- `e297-lol-rocm-compile-index-control24k-mt64-r1.*`;
- `e297-lol-rocm-compile-index-candidate24k-mt64-r2.*`;
- `e297-lol-rocm-compile-index-control24k-mt64-r2.*`;
- `e297-lol-rocm-compile-index-control4k-mt256-r1.*`;
- `e297-lol-rocm-compile-index-candidate4k-mt256-r1.*`;
- `e297-lol-rocm-compile-index-control4k-mt256-r2.*`.
