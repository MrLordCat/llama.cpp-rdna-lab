# D089 Q4_K_M Primary Baseline Promotion

Date: 2026-07-20

Status: policy and preset promotion. No new speed claim is introduced; the
decision uses the measured E332-E345 Q4_K_M lanes.

## Decision

`models/Qwen3.6-27B-Q4_K_M.gguf` is now the primary production and performance
baseline for this fork. It is the best current quality/capability point that is
demonstrably usable on the reference two-card RX 9070 XT machine.

Q3_K_S remains supported as a secondary headroom, vision, maximum-context, and
historical Q3-kernel research lane. Existing Q3 measurements are not rewritten
or compared directly against Q4 measurements.

## Primary Lane Contract

The safe repeatable Q4 baseline is:

- backend: ROCm, `-dev ROCm1,ROCm0 -sm layer -ts 1,1`;
- model: `Qwen3.6-27B-Q4_K_M.gguf`, including NextN tensors;
- context: `49152`, actual reference prompt `29561` tokens;
- batch: `8192`, ubatch: `1024`, one server slot;
- KV: `q8_0/q8_0`, Flash Attention, full GPU offload;
- cold prompt, no cache reuse, no warmup, seed 42;
- adjacent `spec=none` control for every MTP measurement;
- production agent profile: MTP n3 when generated answers are long enough to
  amortize its small prompt cost.

Measured current rows:

| Mode | Prompt tok/s | Decode tok/s | Aggregate TPS | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| none | 1778.59 | 21.98 | 5.6829 | - |
| MTP n3 | 1731.71 | 39.58 | 6.2802 | 74.36% |

MTP costs 2.64% prompt throughput, improves decode by 80.11%, and improves the
128-output-token request by 10.51% end to end.

## Extended Context Policy

The validated 98K production lane uses the one-copy ROCm scheduler:

| Mode | Prompt / output | Prompt tok/s | Decode tok/s | Shared peak |
| --- | ---: | ---: | ---: | ---: |
| none | 59045 / 64 | 1493.21 | 19.15 | 3.204 GiB |
| MTP n3 | 59045 / 64 | 1435.97 | 35.44 | 3.261 GiB |

`ctx=131072` remains a residency stress lane rather than the default Q4
profile. Vulkan has a healthy recorded Q4 row at 1051.67 prompt tok/s. ROCm
requires memory-aware placement at that size and needs a fresh post-E337/E338
adjacent baseline before a new production claim.

## KV Policy

Q8 remains the primary Q4_K_M KV cache. Alternative KV formats require a Q4
long-agent quality/perplexity gate and a matched performance gate before they
can replace q8 in the primary preset.

## Repository Effects

- Future generic performance work starts from Q4_K_M and the 49K ROCm lane.
- Q3-specific D/P programs keep their original lane contracts and are labeled
  secondary or historical instead of being silently reused as Q4 evidence.
- The GUI's first exact-model preset is the Q4_K_M safe 49K MTP n3 profile.
- Benchmark history remains model-scoped; no Q3 and Q4 TPS rows are compared as
  if they were the same lane.
