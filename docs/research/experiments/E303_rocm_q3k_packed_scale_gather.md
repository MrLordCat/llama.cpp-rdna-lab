# E303: ROCm Q3_K Packed Scale Gather

Date: 2026-07-14

## Hypothesis

After E289 removed the expensive saturating packed subtract, unpacking Q3_K's
six-bit scales could represent a larger fraction of the fused pair-dot body.
A packed32 candidate loaded the existing 12-byte scale payload, gathered four
low/high scale components with RDNA `perm`, and produced four signed scales in
one packed word. It did not change the model format or memory footprint.

## Scout

The existing `rocm_q3k_scale_layout_scout.cpp` was extended with the packed
candidate and three-way rotating launch order. It used the hot fused shape
(`17408 x 5120`), 60 rounds, and 100 timed launches per sample on the free GPU.

| Variant | Kernel time | Registers | Shared memory |
| --- | ---: | ---: | ---: |
| Current packed scale decode | 0.12684 ms | 63 | 256 B |
| Packed32 gather | 0.12945 ms | 49 | 256 B |
| Expanded 116-byte scale layout | 0.13036 ms | 57 | 256 B |

Both alternatives matched the control output exactly. The packed32 gather was
only 0.9798x as fast despite lowering registers by 14, while the expanded
layout reproduced E298's roughly 2.7% regression.

## Decision

Reject the packed scale gather for production. The hot fused kernel is not
controlled by occupancy or scale unpack alone; extra packed selection
instructions outweigh the lower live register count. Keep the scout extension
as a reproducible gate and do not repeat scale-only rewrites without new ISA
evidence.

