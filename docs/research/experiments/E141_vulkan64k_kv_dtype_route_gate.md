# E141 Vulkan 64k KV Dtype Route Gate

## Metadata

- Experiment ID: E141
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E140 (`a2f9a7112`)
- Hypothesis ID: H38 / H05
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, FlashAttention on, no reuse

## Hypothesis

- Statement: the q4/q4 KV cache may be capping the Vulkan long-context FlashAttention route because `flash_attn_cm1.comp` dequantizes K/V inside each long-KV pass.
- Mechanism: replacing q4/q4 KV with wider KV types is a route-level upper-bound test for removing q4 dequant from the active coopmat1 FA path. If f16/q8 is not meaningfully faster, a backend-private q4->f16 FA cache/dequant route is unlikely to pay for its VRAM and sync cost.
- Why now: E129-E132 rejected simple FA tile/staging toggles and E138 rejected split-K. Before writing a new FA route, measure whether the q4 dequant branch is actually the dominant local blocker.

## Math / Theory

- Assumptions: E134 route ceiling uses Vulkan best `1.3406 TPS`, ROCm comparison `1.5545 TPS`, and FA parsed share `0.4160`.
- Required local FA speedup to close the full gap alone: about `1.494x`.
- A q4-removal upper bound below `~1.15x` local FA is not enough alone and weak as a stack component unless Q3_K also improves.
- Failure conditions: f16/q8 KV does not fit at 64k, wider KV increases memory bandwidth enough to offset lower dequant ALU, or the pp gate is not representative of the full real-server 64k lane.

## Implementation Plan

1. Minimal code surface to change: none for the first gate; use existing KV dtype routes.
2. Guard rails: keep baseline q4/q4 command unchanged, clear override env, and keep `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`.
3. Rollback path: no runtime code change.

## Benchmark Plan

- Baseline command: `llama-bench` pp7488 with q4/q4 KV, FlashAttention on, `b8192/ub1024`, `-mmp 0`.
- Candidate commands: same command with f16/f16 and q8_0/q8_0 KV where supported.
- If a candidate is promising and fits, run one 64k real-server max-token-1 check.
- Artifacts path: `build_logs/agent-workload/e141-*`

## Metrics

- prompt eval tok/s
- decode tok/s where available
- route trace for FA pipeline state
- fit/OOM status for 64k real-server

## Result

- Outcome: reject as a primary 64k acceleration route.
- Delta:
  - q4/q4 pp7488: `970.03 tok/s`;
  - f16/f16 pp7488: `996.00 tok/s` (`+2.68%` total pp);
  - q8_0/q8_0 pp7488: `940.03 tok/s` (`-3.09%` total pp);
  - f16/f16 64k real-server fit: failed before ready; projected Vulkan device use `16183 MiB` vs `15221 MiB` free, need to reduce by `1986 MiB`.
- Confidence: high for rejecting f16/q8 KV as a near-term 64k route; medium for the inferred FA local ceiling because pp7488 is not the full 64k wall.
- Recommendation: keep q4/q4 KV for H38. Do not implement a backend-private q4->f16 FA cache unless a future design also solves VRAM residency and proves a much larger local FA win than the measured f16 upper bound.

## Notes

- This is not a proposed default away from q4 cache yet. It is an upper-bound screen for whether a larger FA route is worth implementing.
- The f16 pp gate implies only about `1.067x` local FA speedup if the E134 FA share (`0.416`) is used as a rough Amdahl proxy, far below the `1.494x` local speedup needed for FA alone to close the ROCm 64k wall.
- The f16 64k fit failure came from server startup, not output correctness: the model never became ready because the memory fitter could not satisfy the requested context with full GPU layers.
- This also explains why E120 preferred Vulkan q4 over f16 for practical long-answer sessions: f16 is at best a diagnostic upper-bound route, while q4 preserves memory headroom.

## Artifacts

- `build_logs/agent-workload/e141-vulkan-kv-q4q4-pp7488.log`
- `build_logs/agent-workload/e141-vulkan-kv-f16f16-pp7488.log`
- `build_logs/agent-workload/e141-vulkan-kv-q8q8-pp7488.log`
- `build_logs/agent-workload/e141-vulkan64k-f16kv-ctx64k.server.log`
- `build_logs/agent-workload/e141-vulkan64k-f16kv-server.log`
