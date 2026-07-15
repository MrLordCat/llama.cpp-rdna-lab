# E304: ROCm Q3_K U32 Load Recheck

Date: 2026-07-14

## Hypothesis

E211 rejected aligned 32-bit Q3_K loads because they raised the hot MMVQ
kernel from 94 to 97 registers and reduced occupancy. E289 later removed the
packed saturating-subtract register cliff, so the same mechanism deserved a
new isolated resource and timing gate.

## Scout

`rocm_q3k_scale_layout_scout.cpp` gained a `u32_block_loads` variant. It reads
the padded `qs` and `hmask` payload through aligned `get_int_b4` operations but
otherwise uses the current E289 arithmetic. The hot `17408 x 5120` shape was
measured for 80 rounds with 100 timed launches per sample.

| Variant | Kernel time | Registers | Shared memory |
| --- | ---: | ---: | ---: |
| Current byte loads | 0.12756 ms | 63 | 256 B |
| Aligned U32 loads | 0.12732 ms | 70 | 256 B |

The outputs matched exactly. The U32 route was only 1.00185x faster while
adding seven registers. That is effectively a tie at this measurement scale
and leaves less resource margin for the full fused production kernel.

## Decision

Reject the U32 load rewrite for production. E289 removed E211's occupancy
failure, but it did not reveal a useful load-side speedup. Keep the scout
variant as a reproducible gate and retain the current byte-load route.

