# E305: ROCm Q3_K Pair-Dot Recheck After E289

Date: 2026-07-14

## Hypothesis

After E289 changed the Q3_K arithmetic and register profile, the production
pair-dot route might no longer be the best choice for dual-GPU decode. A
clean short-lane A/B compared pair-dot enabled with the existing
`GGML_MMVQ_Q3K_DISABLE_PAIRDOT=1` rollback.

The first artifact labels still contain `lol`, but the game had already been
closed for about 20 minutes. These are clean-system measurements.

## Measurements

The first alternating sequence always launched pair-dot first:

| Sequence | Pair-dot on | Pair-dot off |
| --- | ---: | ---: |
| 1 | 26.93 tok/s | 28.46 tok/s |
| 2 | 27.60 tok/s | 28.45 tok/s |

This initially suggested a 4.36% win from disabling pair-dot. A temporary
RDNA4 default-off build was then tested with an enable rollback. The apparent
result reversed when pair-dot was always the second launch:

| Sequence | Default off | Rollback on |
| --- | ---: | ---: |
| 1 | 27.52 tok/s | 28.78 tok/s |
| 2 | 27.57 tok/s | 28.24 tok/s |
| 3, reverse-order continuation | 29.50 tok/s | 29.75 tok/s |

Resource traces confirmed that both routes really executed: the separate-dot
kernel used 75 registers and the pair-dot kernel used 62, with 100% modeled
occupancy for both. The strong dependence on launch order and the rising
throughput across the sequence identify warmup or clock drift, not a robust
pair-dot regression.

## Decision

Reject the RDNA4 default-off change and restore production pair-dot. The final
reverse-order comparison slightly favors pair-dot, and the complete series
does not establish a repeatable win for disabling it. The source was reverted
and `build-rocm-full` was rebuilt with the original default-on behavior.

Relevant artifacts use the prefixes:

- `e305-lol-rocm-dual-pairdot-*`
- `e305-clean-rocm-dual-pairdot-*`

