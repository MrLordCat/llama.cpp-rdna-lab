# E347 Vulkan 49K Q3/FA Candidate Gates

## Scope

This experiment tested three source-level Vulkan candidates before reopening the
long-prompt optimization program:

- nearby Q3_K low-tile split-K geometry;
- the ROCm E289 packed biased-byte subtract transplanted into Vulkan q3quad;
- the D081 q8/q8 FlashAttention two-query-tile dataflow.

The lane used `Qwen3.6-27B-Q3_K_S_mtp.gguf`, Vulkan, `ctx=49152`,
`b8192/ub1024`, q8/q8 KV, FlashAttention, MTP `n_max=2`, one cold
`triage_diff` task, no reuse, no warmup, and roughly 32.1k measured prompt
tokens.

## Q3_K Low-Tile Split-K

The current automatic low-tile route was compared with forced split-K 3:

| Variant | Prompt tok/s | Decode tok/s | Aggregate TPS |
| --- | ---: | ---: | ---: |
| automatic control | 1483.00 | 42.23 | 5.1686 |
| forced split-K 3 | 1453.43 | 42.22 | 5.0814 |

Prompt throughput changed by `-1.99%`. Decode was neutral. The existing
automatic policy remains preferable.

## Vulkan Packed Q3_K Subtract

The E289 ROCm packed subtract was applied only to the Vulkan q3quad dequant
helper and validated through a Vulkan rebuild plus `spirv-val` on the affected
Q3 shaders.

| Variant | Prompt tok/s | Decode tok/s | Aggregate TPS |
| --- | ---: | ---: | ---: |
| adjacent automatic control | 1483.00 | 42.23 | 5.1686 |
| packed q3quad subtract | 1454.97 | 42.27 | 5.0865 |

Prompt throughput changed by `-1.89%`; the small decode difference is noise.
The backend-specific ROCm win does not transfer to this Vulkan shader body.

## D081 Two-Query FlashAttention

### Hardware Gate

The Windows Vulkan driver exposes `maxComputeSharedMemorySize=32768` on both
RX 9070 XT devices. The original `Br32/Bc64` design needs about 42-43 KiB
even with compact V staging, so the host route correctly rejected it and the
resource smoke fell back to scalar FA. That run is not a candidate speed result.

A compact `Br32/Bc32,row_split=8,WG=512` redesign preserved a 32x32 score
tile and shared each K/V tile across two 16-row query halves. The driver
accepted it as coopmat1 for q8/q8 `N=1024`, `KV=1k..32k`:

| Resource | Result |
| --- | ---: |
| VGPR | 65 |
| SGPR | 79 |
| LDS | 32,256 B |
| Scratch | 0 B |

### Deterministic A/B

Both orders used `temperature=0`, the same prompt, server seed, binary and MTP
settings. All four saved responses were identical (55 characters), and each
MTP run accepted 9 of 10 drafted tokens.

| Order | Control prompt tok/s | Candidate prompt tok/s | Delta |
| --- | ---: | ---: | ---: |
| control -> candidate | 1417.16 | 1428.40 | +0.79% |
| candidate -> control | 1443.11 | 1403.92 | -2.72% |
| two-run center | 1430.13 | 1416.16 | -0.98% |

The short 16-token decode figures are not used as an MTP speed claim. The prompt
result is decisively below the required 1.3x local-FA gate and does not justify a
long confirmation.

## Decision

Reject and remove all three prototypes. Keep the current automatic Q3_K
low-tile policy and default coopmat1 q8 FA geometry. Do not reopen exact
two-query FA by nearby `Br/Bc/row_split` tuning on this Windows driver: the
32 KiB Vulkan LDS limit and measured occupancy/dataflow cost close that family.

The next Vulkan route needs a different mechanism, such as attention
sparsity/KV compression with measured attention-mass evidence, or a Q3_K body
change that reduces matrix work rather than helper arithmetic.

## Evidence

- `build_logs/agent-workload/vulkan-mtp49k-lowtile-auto-r1.*`
- `build_logs/agent-workload/vulkan-mtp49k-lowtile3-r1.*`
- `build_logs/agent-workload/vulkan-mtp49k-q3packed-r1.*`
- `build_logs/agent-workload/vulkan-mtp49k-fa-twoquery-resource-r1.*`
- `build_logs/agent-workload/vulkan-mtp49k-fa-twoquery-bc32-resource-r1.*`
- `build_logs/agent-workload/vulkan-mtp49k-fa-bc32-{control,candidate}-r{1,2}.*`
- `docs/research/major-topology/D081_P003_VULKAN_Q8_FA_TWO_QUERY_TILE_GATE.md`
