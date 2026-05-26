# E255 A3B Cold 10 TPS Practical Profile

## Metadata

- Experiment ID: E255
- Date: 2026-05-25
- Owner: Copilot
- Target lane: practical Qwen3.6 cold-first 12k repo-snapshot lane, no reuse, no prime, thinking on.
- Model: `models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`

## Hypothesis

- Statement: the local Qwen3.6 A3B MoE model may clear the user-visible cold `10 TPS` target on the same 12k agent lane, even though it is not the strict 27B-Q3 kernel target.
- Mechanism: MoE active parameters are much lower than dense 27B, so prefill/decode work should be lower at the same context and sampler contract.
- Guard rail: do not compare this as a speedup over `Qwen3.6-27B-Q3_K_S.gguf`; it is a practical model/profile result.

## Benchmark Plan

- Server: `build-rocm-vec/bin/llama-server.exe`
- Shape: `ctx=12288`, `batch=6144`, `ubatch=2048`, `ngl=999`, KV `q4_0/q4_0`, FlashAttention on.
- Runtime: `--spec-type none`, `--no-reuse`, `--no-v2-prime-pass`, `--no-disable-thinking`, `triage_diff`, `max_tokens=64`.
- Confirm with r3 if r1 clears `10 TPS`.

## Result

- r1: `e255-a3b12k-cold-control-r1` completed at `19.64 TPS`.
- r3: `e255-a3b12k-cold-control-r3` completed at aggregate `22.14 TPS`, mean task `22.29 TPS`, median `23.55 TPS`, stdev `1.7975`, errors `0`.

## Decision

- Keep as a practical cold `>10 TPS` local Qwen3.6 profile.
- Add a first-match GUI model preset for `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` using the confirmed 12k cold shape.
- Continue the separate dense `Qwen3.6-27B-Q3_K_S.gguf` H42 kernel/body target; E255 does not close that kernel goal.

## Artifacts

- `build_logs/agent-workload/e255-a3b12k-cold-control-r1.diagnostics.md`
- `build_logs/agent-workload/e255-a3b12k-cold-control-r3.diagnostics.md`
- `build_logs/agent-workload/e255-a3b12k-cold-control-r3.server.log`
