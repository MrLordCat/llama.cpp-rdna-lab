# E256 ROCm 27B UBatch Fine Gate

## Metadata

- Experiment ID: E256
- Date: 2026-05-25
- Owner: Copilot
- Target lane: dense `Qwen3.6-27B-Q3_K_S.gguf` cold-first 12k repo-snapshot lane, no reuse, no prime, thinking on.

## Hypothesis

- Statement: after E253/E254 current no-env controls landed below the earlier E248 final3 control, a small intermediate `ubatch` screen may reveal a better residency point than the current `ubatch=2048` without touching code.
- Mechanism: prior gates rejected `ubatch=3072` and `4096`, while `1024` is slower; intermediate values below/near `2048` were not the main H42 route but may recover some current run-to-run/residency loss.
- Risk: this is likely a low-ceiling no-code retune and must not distract from H42 kernel/body work if it does not beat the same-lane best.

## Benchmark Plan

- Control reference: latest clean same-shape r1 `e254-rocm12k-clean-control-r1 = 7.5254 TPS`.
- Test `batch=6144` with `ubatch=1536`, `1792`, and `2560`.
- Keep `ctx=12288`, KV `q4_0/q4_0`, FlashAttention on, `--spec-type none`, no reuse, no prime, thinking on.

## Result

- Clean reference: `e254-rocm12k-clean-control-r1 = 7.5254 TPS` at `ubatch=2048`.
- `ubatch=1536`: `e256-rocm12k-ub1536-r1 = 7.47 TPS`.
- `ubatch=1792`: `e256-rocm12k-ub1792-r1 = 7.46 TPS`.
- `ubatch=2560`: `e256-rocm12k-ub2560-r1 = 7.32 TPS`.

## Decision

- Reject. Intermediate `ubatch` values do not recover the historical E248 control level and do not move toward the dense 27B cold `10 TPS` target.
- Continue H42 as a route-body/kernel problem rather than another batch/ubatch retune.

## Artifacts

- `build_logs/agent-workload/e254-rocm12k-clean-control-r1.diagnostics.md`
- `build_logs/agent-workload/e256-rocm12k-ub1536-r1.diagnostics.md`
- `build_logs/agent-workload/e256-rocm12k-ub1792-r1.diagnostics.md`
- `build_logs/agent-workload/e256-rocm12k-ub2560-r1.diagnostics.md`
