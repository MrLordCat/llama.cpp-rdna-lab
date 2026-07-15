# E298: ROCm Q3_K Expanded Scale Layout Gate

Date: 2026-07-14

## Hypothesis

The hot Q3_K MMVQ path repeatedly unpacks 12 scale bytes into 16 signed
values.  A narrow storage experiment kept the existing `hmask` and `qs`
payloads but expanded only the scale field, growing a block from 112 to 116
bytes.  This was intended to lower decode instructions and register pressure
without repeating the much larger 160-byte layout rejected by D027.

## Scout

`scripts/research/rocm_q3k_scale_layout_scout.cpp` compares the current packed
block with the 16-byte scale representation at the fused gate/up shape
(`17408 x 5120`, Q8_1 input).  Both variants run interleaved in one process and
their output is checked exactly.

On the free GPU, 100 outer rounds with 100 timed launches each produced:

| Layout | Kernel time | Registers | Shared memory |
| --- | ---: | ---: | ---: |
| Current 112-byte | 0.12748 ms | 63 | 256 B |
| Expanded 116-byte | 0.13101 ms | 57 | 256 B |

The candidate was `0.9731x` as fast, despite reducing registers.  Both kernels
already model full occupancy, while the candidate adds 3.57% block traffic.
An apparent `1.172x` result on the game-loaded GPU did not reproduce on the
free GPU and is treated as contention noise.

## Decision

Reject the expanded-scale production layout.  Lower register count alone does
not pay for the larger Q3_K stream on this RDNA4 lane.  Keep the standalone
scout as a reproducible gate for any future layout redesign.

