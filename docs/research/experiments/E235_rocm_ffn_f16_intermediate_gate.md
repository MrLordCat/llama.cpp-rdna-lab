# E235 ROCm FFN f16-intermediate gate

## Metadata

- Experiment ID: E235
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 graph/body gate
- Target lane: Qwen3.6-27B Q3_K_S FFN prefill family, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, cold/no reuse
- Tool: standalone `build_logs/agent-workload/rocm_ffn_f16_intermediate_scout.exe`

## Hypothesis

- Statement: E234 showed f16 GEMM output can be locally faster, but convert-back kills the current-compatible route. A larger FFN route might keep up/gate/GLU intermediates in f16 and feed down-projection directly, avoiding convert-back.
- Mechanism:
  - current-style scout route: up GEMM f32 output + gate GEMM f32 output + f32 GLU + f32-to-f16 conversion for down GEMM input + down GEMM f32 output;
  - candidate route: up GEMM f16 output + gate GEMM f16 output + f16 GLU + down GEMM f32 output.
- Why now: this is the smallest standalone gate for a true graph/body route, and it tests whether f16 intermediates are worth the correctness risk before touching llama.cpp graph execution.

## Implementation

- Added `scripts/research/rocm_ffn_f16_intermediate_scout.cpp`.
- The scout uses rocBLAS for the three GEMMs and custom HIP kernels for GLU and conversion.
- No runtime code was changed.

## Results

| Shape | Current-style ms | F16-intermediate ms | Relative | Decision |
| --- | ---: | ---: | ---: | --- |
| `n=2048,k_in=5120,k_ff=17408,m_out=5120` | `9.7029` | `10.8515` | `1.1184x` slower | reject main chunk |
| `n=1382,k_in=5120,k_ff=17408,m_out=5120` | `7.9951` | `6.7055` | `0.8387x` | tail-only local signal |

## Result

- Outcome: reject as the next cold-first route.
- Delta:
  - main `n=2048` chunk regresses by `11.84%`;
  - tail chunk has a standalone local win, but the tail-only share is too small and the implementation would require graph-level f16 intermediate correctness work.
- Confidence: high for rejecting broad/main-chunk FFN f16 intermediate route; medium for the tail-only observation because it was a standalone scout.
- Recommendation:
  - Do not implement broad FFN f16 intermediate fusion.
  - Do not start a correctness-risky runtime branch for the tail-only signal unless a later route stack needs tail-specific work.
  - Continue with other structural hotspots; if f16 intermediates are revisited, require a correctness plan plus a tail-only route ceiling model first.

## Notes

- This explains why E234 could show isolated f16-output GEMM signal yet fail wall: once the surrounding GLU/down path is included, the main prompt chunk loses.
- Tail-only gains are interesting but below the current +20% target as a solo route.

## Artifacts

- `scripts/research/rocm_ffn_f16_intermediate_scout.cpp`
- `build_logs/agent-workload/e235-ffn-f16-intermediate-scout-n2048-r1.csv`
- `build_logs/agent-workload/e235-ffn-f16-intermediate-scout-n1382-r1.csv`
