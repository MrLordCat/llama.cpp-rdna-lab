# E299: ROCm Dual-GPU Decode Route Refresh

Date: 2026-07-14

## Purpose

Refresh the post-E289 dual-GPU route and test whether a redundant pipeline
scheduler synchronization is a material part of the remaining ROCm decode
gap.

## Node Trace

A two-graph trace used a 161-token prompt and per-node synchronization.  The
captured graphs have `ne02=161`, so they are the two device halves of prefill,
not one-token decode graphs.  The trace is diagnostic only: synchronizing
after every node inflates absolute times and disables normal overlap.

| Device | Timed groups | Distorted total | MUL_MAT | Flash Attention | GDN |
| --- | ---: | ---: | ---: | ---: | ---: |
| ROCm1 | 951 | 328.763 ms | 288 / 222.446 ms | 8 / 17.129 ms | 25 / 16.994 ms |
| ROCm0 | 891 | 203.823 ms | 272 / 128.161 ms | 8 / 14.280 ms | 23 / 10.330 ms |

The model route remains dominated by matrix multiplication and contains eight
full-attention layers per model half.  ROCm1 was also visibly slower while the
game occupied that device.  This trace must not be used as either a clean
device comparison or a decode timing profile.

## Pipeline Synchronization Gate

The server output read already synchronizes token results.  Disabling the TG
pipeline scheduler path with `LLAMA_MTP_PIPELINE_PARALLEL=0` tested whether the
following scheduler sync still carried work.

| Variant | Decode TPS | Prompt TPS |
| --- | ---: | ---: |
| Control | 27.36 | 420.67 |
| Pipeline path disabled | 27.03 | 483.42 |

Decode did not improve.  The later synchronization is effectively empty on
this route; prompt movement is background-load variance on the tiny prompt.

## Decision

Reject pipeline-sync removal as a decode optimization.  Keep the production
scheduler behavior and focus on the dominant MMVQ path plus long-KV vector
Flash Attention.

Primary artifacts:

- `e299-lol-rocm-decode-node-trace-short.server.log`;
- `e299-lol-rocm-ppsync-control-r1.*`;
- `e299-lol-rocm-ppsync-off-r1.*`.
