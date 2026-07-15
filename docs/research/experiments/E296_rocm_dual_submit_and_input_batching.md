# E296: ROCm Dual Submit and Input Batching

Date: 2026-07-14

## Goal

Measure the remaining dual-GPU per-token overhead after E294, then test whether
combining scheduler input copies can reduce the ROCm layer-split decode tax.
All wall runs in this experiment were made while League of Legends was active,
so they are diagnostic and are not clean performance baselines.

## Split Topology

The Qwen3.6 dual-GPU decode graph contains a CPU split followed by two large GPU
splits:

- ROCm1: 1,986 nodes and 9 inputs;
- ROCm0: 1,862 nodes and 10 inputs;
- the ROCm0 split receives two inter-GPU tensors together with CPU inputs.

The two inter-GPU payloads are small: about 20 KiB for `l_out` and about 43 KiB
for the converted long-context mask. Windows HIP 7.1 cannot access either RX
9070 XT peer directly, so both transfers use pinned host staging.

## Measurements

An environment-gated scheduler prototype batched compatible copies by source
backend. The two GPU-to-GPU entries completed in one staged operation:

| Phase | Time |
| --- | ---: |
| D2H | 0.288 ms |
| H2D | 0.685 ms |
| Total | 0.973 ms |

Batching six host inputs per GPU showed about 0.39 ms for ROCm1 and 0.84 ms for
ROCm0 in one point trace. Forced-sync split totals suggested only about 0.3 ms
of possible point savings, while complete game-loaded A/B runs were neutral or
negative and drifted more than the candidate effect.

HIP graph host timing isolated a larger fixed cost:

| Device split | Nodes | Total host submit | `hipGraphLaunch` path |
| --- | ---: | ---: | ---: |
| ROCm1 | 1,986 | 1.015 ms | 0.917 ms |
| ROCm0 | 1,862 | 0.720 ms | 0.632 ms |

Compatibility scanning costs only 0.035-0.041 ms and graph-key/property checks
are negligible. `hipGraphUpload` cannot remove the steady launch cost, and HIP
7.1 documents graph instantiate flags as unsupported.

## Decision

Reject and remove the `tensor_copy_batch` backend interface and both ROCm batch
implementations. They added scheduler and ABI complexity without a reproducible
wall gain. Keep the split-input and graph-host timing traces for future topology
work, but cache their environment gates once per process so normal decode does
not repeatedly call `getenv`.

The next useful dual-GPU route must reduce the number of graph submissions or
change the split topology. Repacking the same small host-staged copies is not a
large enough lever.

Primary artifacts:

- `e296-lol-rocm-batchstage-subset-candidate24k-mt64-r1.*`;
- `e296-lol-rocm-batchstage-subset-control24k-mt64-r1.*`;
- `e296-lol-rocm-hostinputs-sync24k-mt64-r1.*`;
- `e296-lol-rocm-graph-host-timing4k-mt4-r1.*`.
