# E264 Vulkan 12k FFN F16 src1 Gate

## Metadata

- Experiment ID: E264
- Date: 2026-05-26
- Owner: Copilot
- Branch/Commit: local dirty tree after E260; source prototype reverted after measurement
- Target lane: Vulkan, Qwen3.6-27B-Q3_K_S, `ctx=12288`, cold/no-reuse/no-prime, thinking on, `quick:triage_diff`, `max_tokens=64`, `spec=none`

## Hypothesis

- Statement: instead of changing Q3_K weight layout, cast selected FFN activations (`src1` for Q3_K matmuls) to F16 so Vulkan can use the existing `pipeline_dequant_mul_mat_mat_f16` route and halve B-side traffic.
- Mechanism: gate/up FFN matmuls share the same normalized activation input, and down FFN has a large `k=17408` activation matrix. If B-side bandwidth is meaningful, an F16 `src1` route could offset the cast cost.
- Why now: E258 showed Q3_K weight-layout changes can improve prompt but hurt decode, while E260 closed broad no-code f16/queue/batch pivots. This was a source-level graph topology gate that did not touch shader code or defaults.

## Prototype

- Temporary env knob: `GGML_EXPERIMENTAL_VK_FFN_F16_SRC1`.
- Modes tested:
  - unset: paired control.
  - `1`: cast Q3_K FFN gate/up input to F16.
  - `down`: cast Q3_K FFN down input to F16.
- The prototype was removed after measurement because both candidates regressed.

## Result

| Label | Mode | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `e264-vulkan12k-ffn-f16src1-control-r1` | unset | 7.1320 | 1016.23 | 40.89 | paired control |
| `e264-vulkan12k-ffn-f16src1-gate-r1` | gate/up | 5.9323 | 817.31 | 40.34 | reject |
| `e264-vulkan12k-ffn-f16src1-down-r1` | down only | 6.8000 | 961.87 | 40.37 | reject |

- Delta vs paired control: gate/up `-16.82%` wall and `-19.57%` prompt; down-only `-4.65%` wall and `-5.35%` prompt.
- Confidence: high enough to revert immediately; both candidate paths lost prompt throughput clearly while decode stayed near the control.
- Recommendation: do not pursue graph-level F16 `src1` casts for Vulkan Q3_K FFN. If a future route wants F16 activations, it must avoid per-layer cast traffic and prove reuse/fusion in the same shader or graph schedule.

## Notes

- This is not the same as E259 f16 KV or E260 `GGML_VK_DISABLE_F16=1`; E264 targeted the activation input type of Q3_K FFN matmuls.
- The negative result suggests the conversion/global-temp cost and/or F16 pipeline tradeoff exceeds B-side traffic savings on this lane.
- Source was reverted to default behavior after the A/B.
