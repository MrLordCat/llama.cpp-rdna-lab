# E224 ROCm post-H43 baseline and batch gate

## Metadata

- Experiment ID: E224
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 / H43 follow-up
- Target lane: ROCm Qwen3.6-27B Q3_K_S, `ctx=12288`, q4/q4 KV, FA on, no reuse, no prime, thinking on

## Hypothesis

- Statement: after H43 default-on and the driver refresh, the active 12k ROCm lane may have moved to a different best batch/ubatch shape.
- Mechanism: if the prior batch boundary or ROCm vbuffer residency was the limiter, larger `-b`/`-ub` shapes could improve prompt wall without kernel changes.
- Failure condition: if prompt/decode split and aggregate TPS do not move, batch shape is not the next speed route.

## Benchmark Plan

- Binary: `build-rocm-vec/bin/llama-server.exe`
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- Common args:
  - `--tasks quick --task-ids triage_diff --runs 1`
  - `--ctx-size 12288`
  - `--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --gpu-layers 999`
  - `--max-tokens 64 --real-context-mode repo-snapshot`
  - `--no-reuse --no-v2-prime-pass --no-disable-thinking`
  - `--server-extra "--spec-type none"`

## Measured Results

| Label | Batch / ubatch | Aggregate TPS | Prompt eval tok/s | Decode tok/s | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `e224-rocm12k-default-baseline-r1` | `4096 / 1024` | `7.2046` | `1106.83` | `30.70` | post-H43 compatibility baseline |
| `e224-rocm12k-active6144-baseline-r1` | `6144 / 2048` | `7.5575` | `1180.65` | `30.59` | current active baseline |
| `e224-rocm12k-b8192-ub2048-r1` | `8192 / 2048` | `~7.65` | near-tie | near-tie | tiny r1 noise only |
| `e224-rocm12k-b8192-ub4096-r1` | `8192 / 4096` | failed | n/a | n/a | hard-timeout cliff |
| `e224-rocm12k-b8192-ub4096-singlechunk-r1` | `8192 / 4096` + `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` | failed | n/a | n/a | negative control also timed out |
| `e224-rocm12k-b12288-ub2048-r1` | `12288 / 2048` | `~7.57` | near-tie | near-tie | outer batch boundary not limiter |

## Result

- Outcome: tie / reject as speed route.
- Current active baseline for follow-up: `7.5575 TPS` on `b=6144,ub=2048`.
- +20% target from this lane: about `9.07 TPS`.
- Interpretation:
  - larger outer batch does not materially improve wall time;
  - `ub=4096` is still a real cliff and single-chunk does not rescue it;
  - the next +20% attempt needs a structural route, speculative/session path, Vulkan decode path, or a new Q3_K body, not nearby batch retuning.

## Artifacts

- `build_logs/agent-workload/e224-rocm-q3k-defaulton-smoke-after-format.txt`
- `build_logs/agent-workload/e224-rocm12k-default-baseline-r1.diagnostics.md`
- `build_logs/agent-workload/e224-rocm12k-active6144-baseline-r1.diagnostics.md`
- `build_logs/agent-workload/e224-rocm12k-b8192-ub2048-r1.diagnostics.md`
- `build_logs/agent-workload/e224-rocm12k-b8192-ub4096-r1.diagnostics.md`
- `build_logs/agent-workload/e224-rocm12k-b8192-ub4096-singlechunk-r1.diagnostics.md`
- `build_logs/agent-workload/e224-rocm12k-b12288-ub2048-r1.diagnostics.md`
