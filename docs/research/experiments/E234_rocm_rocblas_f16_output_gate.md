# E234 ROCm rocBLAS f16-output gate

## Metadata

- Experiment ID: E234
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 route-body/output gate
- Target lane: Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, cold/no reuse
- Tool/binary: standalone `rocm_rocblas_output_scout.exe`; temporary runtime probe in `build-rocm-vec/bin/llama-server.exe`

## Hypothesis

- Statement: current Q3_K cublas fallback may be partially output-bandwidth bound because it writes F32 GEMM results. F16 output with F32 compute may be faster.
- Mechanism: run rocBLAS with f16 inputs, f32 compute, f16 output, then convert back to f32 for current graph compatibility. If this is locally faster, a future route might keep gate/up intermediates in f16 and avoid convert-back entirely.
- Why now: E228 recentered the route on GEMM-side work; output size is large for the dominant `ncols=2048` buckets.

## Implementation

- Added `scripts/research/rocm_rocblas_output_scout.cpp`.
- The standalone scout measures:
  - current f32 output GEMM;
  - f16 output GEMM only;
  - f16-to-f32 convert only;
  - f16 output GEMM plus convert.
- A temporary env-gated runtime route was tested for Q3_K hot shapes and then reverted:
  - `GGML_EXPERIMENTAL_Q3K_CUBLAS_F16_OUT=1`
  - exact filters for the narrow wall test: `row_diff=10240`, `ne00=5120`, `ncols=2048`.

## Standalone Scout

| Shape `(m,n,k)` | F32 output ms | F16 output only ms | Convert only ms | F16 output + convert ms | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `(17408,2048,5120)` r3 | `3.1398` | `3.0404` | `0.3727` | `2.9367` | local signal, needs runtime check |
| `(5120,2048,17408)` r2 | `3.0235` | `3.0965` | `0.0732` | `2.6180` | local signal, needs runtime check |
| `(10240,2048,5120)` r1 | `1.9483` | `1.8437` | `0.2272` | `2.0425` | reject with convert |
| `(6144,2048,5120)` r1 | `1.2530` | `1.1593` | `0.1402` | `1.3701` | reject with convert |

## Runtime Point Results

Broad hot-shape runtime probe with split timing:

| Shape | Control sum ms | Candidate sum ms | Main cause | Decision |
| --- | ---: | ---: | --- | --- |
| `17408x5120@2048` | `1417.51` | `1599.70` | `dst_ms=194.51` dominates small GEMM gain | reject |
| `5120x17408@2048` | `792.36` | `833.44` | convert-back dominates | reject |
| `10240x5120@2048` | `813.75` | `773.41` | GEMM `731.03 -> 638.90`, `dst_ms=49.66` | narrow local win |
| `6144x5120@2048` | `231.62` | `262.65` | convert-back dominates | reject |

## Wall Results

| Label | Route | Aggregate TPS | Prompt mean ms | Decode tok/s mean | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `e234-rocm12k-q3k-f16out-wall-control-r1` | default | `7.89` | see diagnostics | see diagnostics | control |
| `e234-rocm12k-q3k-f16out-wall-10240-r1` | exact `10240x5120@2048` f16-output+convert | `7.63` | see diagnostics | see diagnostics | regression |

## Result

- Outcome: rejected for current cold-first route.
- Delta:
  - Broad runtime point worsened the largest buckets because convert-back cost exceeded GEMM savings.
  - Narrow `10240x5120@2048` point win did not convert to wall: `7.89 -> 7.63 TPS`.
- Confidence: medium-high for rejection of the current-compatible f16-output+convert route.
- Recommendation:
  - Do not keep the runtime env route; it was reverted.
  - Keep the standalone scout as a future diagnostic.
  - Only revisit f16 output if a larger graph route consumes f16 gate/up intermediates directly and has a correctness plan; f16-output plus convert-back is not a speed route.

## Artifacts

- `scripts/research/rocm_rocblas_output_scout.cpp`
- `build_logs/agent-workload/e234-rocblas-output-scout-17408x5120n2048-r3.csv`
- `build_logs/agent-workload/e234-rocblas-output-scout-5120x17408n2048-r2.csv`
- `build_logs/agent-workload/e234-rocblas-output-scout-10240x5120n2048-r1.csv`
- `build_logs/agent-workload/e234-rocblas-output-scout-6144x5120n2048-r1.csv`
- `build_logs/agent-workload/e234-rocm12k-q3k-f16out-point-control-r1.server.log`
- `build_logs/agent-workload/e234-rocm12k-q3k-f16out-point-candidate-r1.server.log`
- `build_logs/agent-workload/e234-rocm12k-q3k-f16out-wall-control-r1.diagnostics.md`
- `build_logs/agent-workload/e234-rocm12k-q3k-f16out-wall-10240-r1.diagnostics.md`
