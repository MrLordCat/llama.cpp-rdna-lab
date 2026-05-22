# E177 ROCm RDNA4 Q4_K MMVQ Nwarps1 Probe

## Metadata

- Experiment ID: E177
- Date: 2026-05-22
- Owner: Codex
- Hypothesis ID: H39
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: RDNA4 `Q4_K`, `ncols_dst=1` MMVQ might be faster with `nwarps=1` instead of the current RDNA4 `nwarps=8` policy.
- Mechanism: Fresh upstream PR `ggml-org/llama.cpp#23528` reports that RDNA3 `Q4_K` MMVQ improves when `Q4_K` falls through to `nwarps=1`, with correctness preserved. Our active model is Q3_K_S, but the decode trace still has a measurable `Q4_K` MMVQ bucket from SSM output projection.
- Risk: The external evidence is RDNA3/gfx1100 and Q4_K-heavy models, not RX 9070 XT/gfx1201 and Q3_K_S. H39 `Q4_K` is only a secondary route, so even a strong local win has a limited wall ceiling.

## Analytic Gate

Fresh same-session control:

- `e177-h39-rocm-current-control-r1`: aggregate `29.5387 TPS`, decode eval `31.97 tok/s`.
- `e177-h39-rocm-current-route-r1` trace: `Q4_K` `MUL_MAT forward` is `118.948 ms` / `4.08%` of decode node time; parsed `Q4_K` MMVQ bucket is `79.060 ms` / `8.66%` of parsed MMVQ.

Required local speedup:

- Against the node-share proxy (`4.08%`): local `1.1389x` for `+0.5%` wall, `1.3204x` for `+1%`, `1.9252x` for `+2%`.
- Against the parsed-MMVQ proxy (`8.66%`): local `1.0610x` for `+0.5%` wall, `1.1291x` for `+1%`, `1.2927x` for `+2%`.

This is a cheap external-signal probe, not a parity-closing route by itself.

## Method

Temporary code:

- remove `GGML_TYPE_Q4_K` from the RDNA4 `ncols_dst == 1` `return 8` group in `ggml/src/ggml-cuda/mmvq.cu`;
- this makes RDNA4 `Q4_K` use the existing default `nwarps=1` for one-token MMVQ;
- leave Q3_K, Q6_K, Q8_0, IQ4, and non-RDNA4 policies unchanged.

Validation plan:

1. Build `build-rocm-vec`.
2. Run H39 `triage_diff`, `runs=1`, `max_tokens=128`.
3. Run short resource trace with `GGML_TRACE_MMVQ_RESOURCES=1`.
4. Keep only if wall is not below same-session control and the `Q4_K` bucket improves enough to justify a follow-up confirmation.

## Results

Speed gate:

| Run | Aggregate TPS | Decode eval | Prompt eval | Errors |
| --- | ---: | ---: | ---: | ---: |
| Same-session control `e177-h39-rocm-current-control-r1` | `29.5387` | `31.97 tok/s` | `511.67 tok/s` | `0` |
| Candidate `e177-h39-rocm-q4k-nwarps1-r1` | `29.2799` | `31.69 tok/s` | `508.33 tok/s` | `0` |

Delta:

- aggregate wall: `-0.88%`;
- decode eval: `-0.88%`.

Resource trace:

| Bucket | Control | Candidate | Interpretation |
| --- | ---: | ---: | --- |
| `Q4_K ncols_x=6144` | `79.060 ms`, avg `0.1098 ms`, regs `62`, block `(32,8,1)` | `78.250 ms`, avg `0.1087 ms`, regs `25`, block `(32,1,1)` | local `~1.01x`, below even the `+0.5%` wall requirement |
| Parsed MMVQ total | `912.742 ms` | `974.824 ms` | trace run overall worse; unchanged Q3_K buckets also timed slower in this trace |

The route activated correctly (`block.y=8 -> 1`, regs `62 -> 25`), but the
Q4_K bucket barely moved. The external RDNA3 Q4_K result does not transfer to
the local RDNA4/Qwen3.6-27B-Q3_K_S H39 lane because Q4_K is a secondary bucket
and its local speedup is only about `1%`.

## Decision

- Reject and revert.
- Keep current RDNA4 `Q4_K` `nwarps=8` policy.
- Do not use upstream PR `#23528` as direct evidence for RDNA4/Q3_K_S H39. It remains useful external context for RDNA3/Q4_K-heavy validation, but the local activation gate failed.

## Artifacts

- `build_logs/agent-workload/e177-h39-rocm-current-control-r1.diagnostics.md`
- `build_logs/agent-workload/e177-h39-rocm-current-route-r1.server.log`
- `build_logs/agent-workload/e177-h39-rocm-q4k-nwarps1-r1.diagnostics.md`
- `build_logs/agent-workload/e177-h39-rocm-q4k-nwarps1-trace-r1.server.log`
