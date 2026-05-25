# E233 ROCm rocBLAS layout contract gate

## Metadata

- Experiment ID: E233
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 route-body/layout gate
- Target lane: Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, cold/no reuse
- Tool: standalone `build_logs/agent-workload/rocm_rocblas_transpose_scout.exe`

## Hypothesis

- Statement: current Q3_K cublas fallback may be using a less favorable rocBLAS contract by staging weights as `k x m` and calling GEMM with `transA=T`.
- Mechanism: if a hypothetical fp16 staging layout stores weights as `m x k`, rocBLAS can run the same logical GEMM as `transA=N`; that may select a faster kernel even before considering dequant/staging cost.
- Why now: E228/E229 show the current cold bottleneck is GEMM-side Q3_K, while solution-index overrides did not help the dominant shape.

## Implementation

- Added `scripts/research/rocm_rocblas_transpose_scout.cpp`.
- The scout compares:
  - current contract: `A` stored as column-major `k x m`, `transA=T`, `lda=k`;
  - hypothetical pretransposed contract: `A` stored as column-major `m x k`, `transA=N`, `lda=m`.
- No llama.cpp runtime code was changed.

## Results

| Shape `(m,n,k)` | Current `A^T*B` ms | Pretransposed `A*B` ms | Relative | Decision |
| --- | ---: | ---: | ---: | --- |
| `(17408,2048,5120)` | `3.6408` | `3.7694` | `1.0353x` | reject |
| `(5120,2048,17408)` | `3.2421` | `3.7714` | `1.1632x` | reject |
| `(10240,2048,5120)` | `1.9843` | `2.2676` | `1.1428x` | reject |
| `(6144,2048,5120)` | `1.1570` | `1.3113` | `1.1334x` | reject |

## Result

- Outcome: regression.
- Delta: pretransposed BLAS contract is `3.5%` to `16.3%` slower across tested hot shapes, before paying any transposed dequant/staging cost.
- Confidence: high enough to reject this route.
- Recommendation:
  - Do not implement transposed Q3_K fp16 staging for rocBLAS.
  - Keep the current `transA=T` contract.
  - Continue H42 with routes that change real body/output/fusion behavior, not A-layout for the same rocBLAS GEMM.

## Artifacts

- `scripts/research/rocm_rocblas_transpose_scout.cpp`
- `build_logs/agent-workload/e233-rocblas-transpose-scout-17408x5120n2048-r1.csv`
- `build_logs/agent-workload/e233-rocblas-transpose-scout-5120x17408n2048-r1.csv`
- `build_logs/agent-workload/e233-rocblas-transpose-scout-10240x5120n2048-r1.csv`
- `build_logs/agent-workload/e233-rocblas-transpose-scout-6144x5120n2048-r1.csv`
