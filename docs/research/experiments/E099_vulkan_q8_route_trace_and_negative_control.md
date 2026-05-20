# E099 Vulkan Q8 Route Trace And Negative Control

## Metadata

- Experiment ID: E099
- Date: 2026-05-20
- Owner: Copilot
- Type: route tracing + upstream-diff check + negative runtime probe
- Hypothesis: H31
- Target lane: Vulkan Q3_K prompt-heavy prefill, AMD proprietary driver, KHR coopmat route

## Hypothesis

Upstream Vulkan integer-dot work (`#22693`, `#23005`) might expose a faster Q8_1 route for quantized prefill if the active Q3_K x F32 route can quantize `src1` and use `matmul_q3_k_q8_1_l`.

## Evidence

Added opt-in `GGML_VK_MATMUL_ROUTE_TRACE=1` to `ggml_vk_mul_mat_q_f16`. This logs unique route keys including pipeline name, src types, shape, aligned flag, integer-dot state, Q8 candidate availability, and dequant flags.

Normal active route remained:

- `matmul_q3_k_f32_f16acc_aligned_l`
- `src0=q3_K`, `src1=f32`
- KHR coopmat `mul_mm.comp`, not MMQ-only route

Under route trace, preconditions for quantizing `src1` were initially true (`integer_dot=1`, `y_contiguous=1`, `quantize_y_initial=1`), but the KHR coopmat branch had no non-empty Q8_1 matmul pipeline for Q3_K (`q8_candidate_empty=1`, `q8_mmp_found=0`).

Temporary KHR-coopmat Q8_1 pipeline creation was then tested and rejected:

| Probe | Route | Resource stats | Result |
| --- | --- | --- | --- |
| forced Q8_1 route | `matmul_q3_k_q8_1_l` | `143 VGPR / 43 SGPR / 28672 B LDS / scratch 0` | pp256 `225.08`, strongly negative |
| no-force large control | non-large default | not promoted | pp7488 `467.18`, strongly negative |

The Q8 route switched successfully, so the negative result is a real route measurement, not a fallback miss. The extra Q8 quantization and integer-dot shader are not competitive on this RX 9070 XT / AMD proprietary driver for active Q3_K prefill.

## Decision

- Rejected Q8_1 route as an acceleration path for H31.
- Removed the temporary KHR-coopmat Q8_1 pipeline creation and `GGML_VK_FORCE_INTEGER_DOT_PRODUCT` knob.
- Kept `GGML_VK_MATMUL_ROUTE_TRACE=1` because it is diagnostic-only and default-off.
- Keep upstream `#22693` / `#23005` as useful context, but do not assume their Q8/integer-dot work is a local speed win for Q3_K prompt-heavy Vulkan.

## Artifacts

- `build_logs/agent-workload/e099-vulkan-matmul-route-trace-p256.txt`
- `build_logs/agent-workload/e099-force-intdot-route-trace-p256-v3.txt`
- `build_logs/agent-workload/e099-q3k-q8-pipeline-create-stats-p256.txt`
- `build_logs/agent-workload/e099-force-intdot-q8-route-p256.txt`
- `build_logs/agent-workload/e099-no-force-large-q3k-pp7488.txt`