# E310: ROCm Direct F16 KQ-Mask Input

Date: 2026-07-14

## Hypothesis

The attention mask is created as F32 host input and cast to F16 for flash
attention. Creating the host input directly as F16 could remove the GPU CAST,
halve mask upload bytes, and potentially avoid a host-staged mask transfer at
the dual-GPU boundary.

An opt-in `LLAMA_F16_KQ_MASK_INPUT=1` path generalized KV-mask filling to F16.
The produced values are identical to the existing F32-to-F16 conversion, and a
deterministic eight-token response matched the control exactly.

## Trace Correction

Current `GGML_TRACE_CUDA_HOST_STAGE` output showed that both the candidate and
the production control already transfer only `l_out-32` between the GPUs. The
converted attention-mask transfer observed in E294 has disappeared after later
scheduler work. Therefore this candidate only removes the CAST and changes the
host mask representation; it does not remove a current cross-device copy.

## Clean A/B/B/A

The matched lane used an 8,604-token prompt, 256 generated tokens, dual ROCm,
16K context, batch/ubatch 8192/1024, q8 KV, and no speculative decoding. No
game or other intentional GPU workload was active.

| Route | Prompt TPS | Decode TPS | Aggregate TPS |
| --- | ---: | ---: | ---: |
| F32 control A1 | 1705.23 | 28.40 | 18.17 |
| F16 input B1 | 1694.44 | 27.91 | 17.93 |
| F16 input B2 | 1363.50 | 25.59 | 15.65 |
| F32 control A2 | 1703.86 | 28.36 | 18.14 |
| F32 control mean | 1704.55 | 28.38 | 18.16 |
| F16 input mean | 1528.97 | 26.75 | 16.79 |

The direct F16 input reduced prompt throughput by `10.30%`, decode by `5.75%`,
and aggregate throughput by `7.54%`.

## Root Cause and Decision

Reject and remove the prototype. Long-prompt masks contain many elements, and
converting every written value to F16 on the CPU is substantially more costly
than filling F32 and letting the GPU cast it. More importantly, the current
scheduler already avoids the cross-device mask copy that motivated the idea.
Keep the production F32 host input plus GPU CAST.

Artifacts use the prefix `e310-clean-rocm-*mask-*`.
