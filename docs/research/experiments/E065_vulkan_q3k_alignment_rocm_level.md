# E065 Vulkan Q3_K alignment with fresh ROCm control

## Metadata

- Experiment ID: E065
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master @ 8c1195ab4 plus local Vulkan prototypes
- Target lane: RX 9070 XT, Windows Vulkan proprietary driver, `build-vulkan`, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `q4_0/q4_0`, thinking on, no reuse

## Hypothesis

After E064, Vulkan was still slightly below the E061 ROCm reference because Q3_K layout and selector behavior remained suboptimal. Upstream `ggml-org/llama.cpp#22951` pads Q3_K/Q6_K to 32-bit alignment, adds Vulkan-specific device type sizing/offsets, and re-enables the Q3_K/Q6_K MMVQ/block-load path. This should combine with the E064 large-tile path and close the historical E061 gap.

## Implementation

- Applied upstream `#22951` locally.
- Kept E064's `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` guard for AMD proprietary large tiles.
- Rebuilt `llama-server` and `llama-bench` successfully.
- Smoke-tested `llama-server.exe --version`.

Key source effects:

- `ggml_vk_device_type_size()` pads `Q3_K` and `Q6_K` by 2 bytes for Vulkan device storage.
- `vk_tensor_view_offset()` maps host GGML offsets to Vulkan padded device offsets.
- Q3_K/Q6_K MMVQ selector restrictions are removed, allowing the new aligned path.
- Shader block layout and `mul_mat_vecq` loads are adjusted for the padded Q3_K/Q6_K representation.

## Results

`llama-bench`, `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, Qwen3.6-27B-Q3_K_S, runs=1:

| Mode | b | ub | pp4096 | pp7488 | pp8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| + `GGML_VK_DISABLE_MMVQ=1` | 4096 | 512 | `880.06` | `888.77` | `886.24` |
| + `GGML_VK_DISABLE_MMVQ=1` | 4096 | 1024 | `926.11` | `881.70` | `872.15` |
| default MMVQ | 4096 | 512 | `826.85` | `880.16` | `881.79` |
| default MMVQ | 4096 | 1024 | `948.47` | `876.83` | `861.61` |
| default MMVQ | 8192 | 1024 | `940.31` | `865.52` | `866.45` |

Prompt-heavy workload (`triage_diff`, repo-snapshot, 7489 prompt tokens, 64 generated):

| Config | Runs | Wall TPS | Prompt eval TPS | Decode eval TPS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| ROCm E061 baseline, `b4096/ub512` | 1 | `6.3327` | `960.26` | `28.32` | historical reference |
| ROCm same-session control, `b4096/ub1024` | 3 | `7.3868` aggregate / `7.49` median | `1173.2367` | `28.62` | fair current target |
| E064 large tile + DISABLE_MMVQ, `b4096/ub512` | 3 | `6.18` | n/a | n/a | near miss |
| E065 large tile + default MMVQ, `b4096/ub512` | 1 | `6.25` | n/a | n/a | similar to E064 |
| E065 large tile + default MMVQ, `b4096/ub1024` | 1 | `6.32` | n/a | n/a | around old E061 baseline |
| E065 large tile + default MMVQ, `b8192/ub1024` | 1 | `6.32` | n/a | n/a | around old E061 baseline |
| E065 large tile + default MMVQ, `b4096/ub1024` | 3 | `6.4180` aggregate / `6.38` median | `897.63` | `40.35` | `+1.35%` vs E061; `-13.1%` vs fresh ROCm r3 |

The confirmed E065 candidate beats the historical E061 ROCm wall by about `+1.35%` on aggregate TPS (`6.4180` vs `6.3327`). That comparison was useful for iteration, but it is not the fair current claim. A same-session ROCm control at the same `b4096/ub1024` shape reached `7.3868` aggregate / `7.49` median, so E065 is still `-13.1%` wall vs the current ROCm target. Vulkan decode remains much faster (`40.35` vs `28.62`, about `+41%`), but prompt eval is lower (`897.63` vs `1173.2367`, about `-23.5%`), so ROCm still wins the active cold prompt-heavy lane.

## Decision

Keep the E065 Vulkan candidate as an opt-in RDNA4/Vulkan improvement, but do not call it ROCm-level against the fresh control. It is the first local Vulkan result to exceed the old E061 ROCm reference and it substantially improves Vulkan versus E064, but the active target remains the same-session ROCm `b4096/ub1024` r3 control. The recommended E065 test profile is:

```text
GGML_VK_FORCE_AMD_LARGE_MATMUL=1
ctx=12288, b=4096, ub=1024, q4_0/q4_0, flash_attn=on, spec=none, no reuse
```

Do not claim a universal default. The large-tile piece is still env-gated because AMD proprietary defaults were disabled upstream for cross-device regression risk. Next validation should focus on additional Vulkan prefill work, because decode is already ahead while prompt eval is the remaining gap.

## Artifacts

- `build_logs/agent-workload/e065-vulkan-q3k-align-large-disablemmvq-llamabench.md`
- `build_logs/agent-workload/e065-vulkan-q3k-align-large-mmvq-default-llamabench.md`
- `build_logs/agent-workload/e065-vulkan-q3k-align-large-disablemmvq-b4096-ub512-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e065-vulkan-q3k-align-large-mmvq-default-b4096-ub512-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e065-vulkan-q3k-align-large-mmvq-default-b4096-ub1024-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e065-vulkan-q3k-align-large-mmvq-default-b8192-ub1024-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e065-vulkan-q3k-align-large-mmvq-default-b4096-ub1024-ctx12288-q3ks-r3.diagnostics.md`
- `build_logs/agent-workload/e065-rocm-control-b4096-ub1024-ctx12288-q3ks-r3.diagnostics.md`
