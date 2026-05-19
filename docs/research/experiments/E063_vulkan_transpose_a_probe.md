# E063 Vulkan transpose-A probe

## Metadata

- Experiment ID: E063
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master @ 8c1195ab4 plus local Vulkan prototype
- Target lane: RX 9070 XT, Windows Vulkan proprietary driver, `build-vulkan`, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `b=4096`, `ub=512`, `q4_0/q4_0`, thinking on, no reuse

## Hypothesis

Upstream `ggml-org/llama.cpp#22970` transposes K-quant A tensors on upload and reported RDNA4 prefill gains on Q4_K/Q6_K models. The test here checked whether that idea helps the active Q3_K_S prompt-heavy lane.

## Implementation

- Applied upstream `#22970` locally.
- Changed the upstream default-on behavior to an opt-in guard: `GGML_VK_TRANSPOSE_A=1`.
- Rebuilt `llama-server` and `llama-bench` successfully.
- Reverted the source prototype after the negative result; artifacts remain for traceability.

Important limitation: the patch creates transposed pipelines for Q4_K/Q5_K/Q6_K, not Q3_K. The active model is Q3_K_S, so most large dense weights are not directly covered.

## Results

`llama-bench`, `pp4096/pp8192`, `b=4096`, `ub=512`, `fa=1`, `q4_0/q4_0`, runs=1:

| Vulkan mode | pp4096 tok/s | pp8192 tok/s |
| --- | ---: | ---: |
| transpose opt-off | `618.02` | `609.26` |
| `GGML_VK_TRANSPOSE_A=1` | `602.95` | `596.41` |
| `GGML_VK_TRANSPOSE_A=1 GGML_VK_DISABLE_MMVQ=1` | `655.83` | `602.32` |

Full prompt-heavy workload (`triage_diff`, repo-snapshot, 7489 prompt tokens, 64 generated):

| Candidate | Wall TPS | Prompt eval TPS | Decode eval TPS |
| --- | ---: | ---: | ---: |
| E062 best Vulkan `GGML_VK_DISABLE_MMVQ=1` | `4.7172` | `639.81` | `35.15` |
| E063 transpose-A + DISABLE_MMVQ | `4.3765` | `588.05` | `34.81` |

## Decision

Reject for the current Q3_K_S prompt-heavy lane and revert the prototype. The Q4/Q5/Q6 transpose path is not the bottleneck for this target; continue with Q3_K/Q6_K alignment/repack and AMD large-tile probes.

## Artifacts

- `build_logs/agent-workload/e063-vulkan-transposea-optoff-llamabench-pp4096-8192.md`
- `build_logs/agent-workload/e063-vulkan-transposea-on-llamabench-pp4096-8192.md`
- `build_logs/agent-workload/e063-vulkan-transposea-disablemmvq-llamabench-pp4096-8192.md`
- `build_logs/agent-workload/e063-vulkan-transposea-disablemmvq-ctx12288-q3ks.diagnostics.md`
