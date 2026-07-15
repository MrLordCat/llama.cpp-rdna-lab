# E288: Upstream RDNA4 MMQ Probe

Date: 2026-07-14

## Scope

Fresh upstream commit `ec0dbef81` was built in an isolated worktree with the
same ROCm 7.1 / gfx1201 toolchain. The goal was to test whether the newer
upstream MMQ refactor already solved the local ROCm gap and should be ported as
a block.

## Results

| Single-GPU lane | Local fork | Fresh upstream | Upstream delta |
| --- | ---: | ---: | ---: |
| 207-token prompt | 749.74 prompt / 32.84 decode | 508.22 / 31.88 | -32.2% / -2.9% |
| 7,923-token prompt | 1,048.28 prompt / 29.64 decode | 924.67 / 29.34 | -11.8% / -1.0% |

The long upstream result was stable across three requests (`919.95`, `927.43`,
and `926.64 prompt tok/s`; `29.33-29.34 decode tok/s`). Its newer aggregate
MMQ code is therefore not a drop-in performance fix for this Windows RDNA4
fork.

## Result

Do not port the full upstream MMQ refactor. Continue reviewing isolated
changes, but require a local same-shape A/B before adoption. The local fused
Q3_K route and Windows-specific tuning are materially faster on this hardware.

Primary artifacts:

- `e288-upstream-ec0-rocm1-short-none-r1.*`;
- `e288-upstream-ec0-rocm1-12k-none-r1r3.*`.
