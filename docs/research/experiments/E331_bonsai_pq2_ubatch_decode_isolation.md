# E331 Bonsai PQ2 ubatch/decode isolation

## Metadata

- Date: 2026-07-15
- Model: `Ternary-Bonsai-27B-PQ2_0.gguf`
- Backend: ROCm/HIP, `build-rocm-full`
- Devices: `ROCm1,ROCm0 -sm layer -ts 1,1`
- Context: 49,152
- Prompt/output: 32,085 / 128 tokens
- Batch: 8192
- KV: q8_0/q8_0
- Speculative decoding: none

## Question

Does increasing prefill ubatch from 128 to 1024 cause the observed long-context
decode regression, or is the GUI result confounded by device placement?

## Controlled result

| UBatch | Prompt TPS | Decode TPS | Aggregate TPS |
| ---: | ---: | ---: | ---: |
| 128 | 1067.99 | 36.35 | 3.81 |
| 1024 | 1819.10 | 36.59 | 6.04 |

At ubatch 1024, prompt throughput improves by 70.33%, decode improves by 0.66%,
and aggregate throughput improves by about 58.5%. The controlled pair therefore
does not reproduce a decode penalty from the larger ubatch.

Both server logs report `tg_max_seq_tokens = 1` and release the inactive PP
scheduler after prefill. The existing dual-scheduler path successfully keeps
the single-token decode graph independent from the prefill ubatch allocation.

## GUI confound

The preceding GUI autotune used backend-default device order, which initialized
`ROCm0,ROCm1`, and reported 33.97 decode TPS. The controlled ubatch-1024 run used
the rig's recommended `ROCm1,ROCm0` order and reported 36.59 decode TPS. GPU0 is
the display/system device, so making it the leading/output side can reduce and
destabilize measured decode performance.

The benchmark GUI now defaults new configurations to the measured recommended
device order and performs a one-time migration from the old `Auto` default.
Manual Auto and all diagnostic orders remain available.

## Decision

- Keep `ubatch=1024` for Bonsai long-prompt workloads.
- Keep the separate TG scheduler; no backend ubatch workaround is needed.
- Use explicit `ROCm1,ROCm0` device order for comparable dual-GPU benchmarks.
- Treat a future decode regression as real only after prompt, context, device
  order, output placement, and generation length are held constant.

## Artifacts

- `build_logs/agent-workload/e331-bonsai-pq2-rocm10-long32k-ub128-r1.*`
- `build_logs/agent-workload/e331-bonsai-pq2-rocm10-long32k-ub1024-r1.*`
- `build_logs/agent-workload/gui-autotune-Ternary-Bonsai-27B-PQ2_0-20260715-181422-cfg01.server.log`
