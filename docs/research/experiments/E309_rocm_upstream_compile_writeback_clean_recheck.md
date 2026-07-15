# E309: ROCm Upstream Compile-Time Writeback Clean Recheck

Date: 2026-07-14

## Reason for Recheck

Upstream commit `683f0c72e` moved MMVQ result writeback into the compile-time
unrolled row loop. E297 tested the same structure while a game was active; the
candidate was nominally faster in one sequence, but control drift was larger
than the claimed gain. The experiment therefore needed a clean-system recheck.

## Validation

The upstream writeback was ported over the fork's padded Q3_K and fused
pair-dot routes. Focused Q3_K `MUL_MAT` coverage passed all 11 shapes. Kernel
resources were unchanged: direct Q3_K remained at 54 registers and fused Q3_K
at 62 registers, both with 100% modeled occupancy.

## Clean Measurements

The matched lane used an 8,604-token prompt, 256 generated tokens, dual ROCm,
16K context, batch/ubatch 8192/1024, q8 KV, and no speculative decoding. No
game or other intentional GPU workload was active.

| Route | Prompt TPS | Decode TPS | Aggregate TPS |
| --- | ---: | ---: | ---: |
| Control A1 | 1742.73 | 27.92 | 18.11 |
| Control A2 | 1737.18 | 27.57 | 17.94 |
| Upstream B1 | 1726.33 | 26.72 | 17.53 |
| Upstream B2 | 1736.14 | 28.41 | 18.29 |
| Control A3, after revert | 1747.16 | 28.56 | 18.39 |
| Control mean | 1742.36 | 28.02 | 18.15 |
| Upstream mean | 1731.24 | 27.57 | 17.91 |

The clean candidate changed prompt throughput by `-0.64%`, decode by `-1.61%`,
and aggregate throughput by `-1.30%`.

## Decision

Reject and restore the production writeback. Clang 21 already keeps the hot
Q3_K specialization at the same register and occupancy level, and the upstream
source structure does not improve this HIP target. E297's nominal game-loaded
gain was drift, not a missing production optimization.

Artifacts use the prefix `e309-clean-rocm-compile-writeback-*`.
