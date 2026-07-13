# D083 P003 Vulkan Q3 NITER2 12k Gate

## Status

Rejected and removed from runtime source.

## Hypothesis

Compute two adjacent N tiles per invocation to reuse Q3 A-side dequantization
and reduce repeated K-loop setup for the two dominant `N=1024` Q3_K shapes.

## Result

- The candidate pipeline was selected for both dominant shapes.
- Resource usage rose from `82` to `106` VGPR, with `31,744 B` LDS and zero
  scratch.
- Prompt speed was `1131.21 tok/s`, well below the `1821.13 tok/s` control.
- The extra accumulator state reduced occupancy more than A-side reuse saved.

## Decision

Do not reopen multi-N accumulation in the current `mul_mm.comp` body. A future
Q3 route must reduce dequant/dot work without adding a second full accumulator
set.
