# E236 ROCm secondary f32 route gate

## Metadata

- Experiment ID: E236
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 secondary-route ceiling check
- Target lane: Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, thinking on, cold/no reuse
- Source artifact: `build_logs/agent-workload/e228-rocm12k-cold-triage-kernelfull-r1.server.log`

## Hypothesis

- Statement: after rejecting several Q3_K body/library gates, the next bottleneck might be a secondary ROCm route such as `MUL_MAT/f32`, where a narrower route swap could stack with Q3_K work.
- Mechanism: aggregate the existing kernel-full trace by actual `GGML_TRACE_CUDA_MUL_MAT_ROUTE` shape and remove startup outliers before deciding whether to implement a f32 selector/body probe.
- Why now: E228 showed `MUL_MAT/f32` as `7.71%` of traced total, large enough to inspect but too small to trust without shape-level breakdown.

## Method

- Parsed `GGML_TRACE_CUDA_MUL_MAT_ROUTE` and adjacent `GGML_TRACE_CUDA_NODE_TIMING` rows from E228.
- Reported:
  - all rows;
  - robust rows with `total_ms < 10` to remove obvious first-use/enqueue outliers.
- No runtime code was changed.

## Results

Route-level aggregate:

| Route/source | Calls | All total ms | Robust total ms | Notes |
| --- | ---: | ---: | ---: | --- |
| `cublas_backend/q3_K` | `1396` | `3918.192` | `3533.209` | still dominant route |
| `cublas_backend/f32` | `784` | `551.298` | `443.826` | secondary ceiling only |
| `cublas_backend/q4_K` | `192` | `262.057` | `262.057` | small share |
| `mul_mat_vec_q_direct/q3_K` | `639` | `199.216` | `169.336` | decode/tail direct route |
| `mul_mat_vec_f_direct/f32` | `336` | `40.640` | `19.615` | mostly tiny decode-side rows |

Top f32 route shapes:

| Route/shape | Calls | All total ms | Robust total ms | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `cublas_backend (5120,48) x (5120,2048) -> (48,2048)` | `288` | `169.737` | `169.737` | SSM alpha/beta prefill family |
| `cublas_backend (256,256) x (256,48) -> (256,48)` | `16` | `109.581` | `2.109` | first-use/enqueue outlier dominates all total |
| `cublas_backend (256,256) x (256,49152) -> (256,49152)` | `48` | `100.821` | `100.821` | attention/reshape-side f32 batch |
| `cublas_backend (5120,48) x (5120,1382) -> (48,1382)` | `96` | `45.373` | `45.373` | tail SSM alpha/beta family |
| `cublas_backend (64,64) x (64,196608) -> (64,196608)` | `48` | `40.507` | `40.507` | smaller f32 attention/state family |

Selector/code check:

- `mmvf` only supports broad f32 AMD use for very small `ne11` when fp32 MMA is available (`ne11 <= 3`), so it cannot cover the `ncols=2048/1382` SSM prefill family.
- `mmf` rejects the same broad f32 shapes because current policy caps normal `src1_ncols > 16`, and the SSM row count `48` is not aligned to the default RDNA rows-per-block gate.
- Forcing these shapes into existing direct f32 kernels would therefore be another selector-only route, not a new body, with a low ceiling and high regression risk.

## Result

- Outcome: reject as the next cold-first route.
- Delta: no wall candidate. Even deleting the robust `cublas_backend/f32` bucket entirely would be below the +20% target; a realistic local f32 route win would be a low-single-digit wall stack item.
- Confidence: high for rejecting f32-first work because the shape breakdown is regular and the selector constraints explain why current direct paths are not active.
- Recommendation:
  - Do not spend the next implementation cycle on broad f32 route forcing.
  - Keep f32 SSM/attention routes as later stack candidates only after a real Q3_K body/layout shift creates a new bottleneck.
  - Continue H42/H43 work on Q3_K body/layout/topology, where the robust traced share is still dominant.

## Notes

- The apparent `107.472 ms` f32 node was a single first-use/enqueue outlier; robust accounting drops that shape from `109.581 ms` to `2.109 ms`.
- This is another bottleneck triage step, not a speed claim.

## Artifacts

- `build_logs/agent-workload/e228-rocm12k-cold-triage-kernelfull-r1.server.log`
- `docs/research/experiments/E228_rocm_cold_q3k_gemm_recenter.md`
