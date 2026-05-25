# E248 ROCm cuBLAS Adjacent src1 F16 Reuse

## Metadata

- Experiment ID: E248
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H35 / H42 side route
- Target lane: ROCm cold-first Qwen3.6-27B-Q3_K_S, `ctx=12288`,
  `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, `spec=none`, no
  reuse, no prime, thinking on

## Hypothesis

Adjacent cuBLAS Q3_K matmuls often use the same f32 activation (`src1`). The
baseline converts that activation to fp16 separately for each matmul. A tiny
adjacent-only fp16 cache can reuse the conversion for the next matching cuBLAS
call without storing fp16 weights or extending activation lifetime across prompt
chunks.

This is intentionally narrower than the rejected E215 long-gap cache idea:
reuse is allowed only for the immediately adjacent matching call on the same
device/stream, with matching `src1` tensor, data pointer, device pointer,
`ne10`, and `ncols`.

## Implementation

- Added opt-in env gate: `GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE=1`.
- Optional threshold: `GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE_MIN_NCOLS`,
  default `1024`.
- Added one persistent fp16 activation buffer per CUDA/HIP backend context and
  device. The buffer grows as needed and is freed with the context.
- The cache is only used on `curr_stream_no == 0`; adjacent-hit epoch checks
  prevent long-gap reuse across prompt chunks where data pointers can be reused
  with changed contents.

## Correctness And Activation

- Real-server traced smoke completed successfully.
- Trace counted `429` adjacent hits on the prompt smoke (`ncols=2048`):
  `ffn_up` 189, `z` 144, `Vcur` 48, `Kcur` 48.
- No output errors were reported by the benchmark harness in candidate runs.

## Results

Single-run exploration, same lane:

| Run | Control TPS | Candidate TPS | Delta | Notes |
| --- | ---: | ---: | ---: | --- |
| r1 | `7.6354` | `7.7234` | `+1.15%` | candidate `MIN_NCOLS=2048` |
| r2 | `7.6047` | `7.6207` | `+0.21%` | candidate `MIN_NCOLS=2048` |
| min1 r1 | `7.6047` | `7.7005` | `+1.26%` | candidate `MIN_NCOLS=1`; decode was neutral |

Borderline 3-run confirmation, same server/run protocol:

| Run Set | Aggregate TPS | Median TPS | Prompt ms mean | Decode ms mean |
| --- | ---: | ---: | ---: | ---: |
| control final3 | `7.8981` | `8.02` | `5988.23` | `2076.79` |
| src1 reuse min1 final3 | `7.9608` | `8.17` | `5948.21` | `2058.56` |

Confirmed delta: aggregate `+0.79%`, median about `+1.87%`, prompt mean
`-40.02 ms`, decode mean `-18.23 ms`.

## Decision

- Keep as an opt-in positive micro-route, not default.
- This is the first runtime wall-positive result in the E245-E248 branch, but
  the margin is small and should not be promoted globally without broader lanes
  and correctness coverage.
- E246 streaming Q3_K pipeline remains rejected for runtime promotion for now:
  the safe synchronous prototype regressed, and the async prototype timed out in
  the real server path despite standalone point wins.

## Artifacts

- `build_logs/agent-workload/e248-rocm12k-control-final3.diagnostics.json`
- `build_logs/agent-workload/e248-rocm12k-src1-reuse-min1-final3.diagnostics.json`
- `build_logs/agent-workload/e248-rocm12k-src1-reuse-trace-r1.server.log`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/common.cuh`