# E064 Vulkan AMD large matmul probe

## Metadata

- Experiment ID: E064
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master @ 8c1195ab4 plus local Vulkan prototype
- Target lane: RX 9070 XT, Windows Vulkan proprietary driver, `build-vulkan`, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `q4_0/q4_0`, thinking on, no reuse

## Hypothesis

Vulkan on Windows AMD proprietary driver is losing prompt-heavy prefill partly because `ggml-vulkan.cpp` disables large matmul tiles for AMD proprietary drivers even when cooperative matrix support is available. RDNA4 with AMD driver `26.3.1` may benefit from the same large-tile path used on non-proprietary AMD drivers.

## Implementation

Added an opt-in environment guard:

```text
GGML_VK_FORCE_AMD_LARGE_MATMUL=1
```

When set, AMD proprietary Vulkan devices with cooperative matrix support can use the large matmul tile path and AMD tuned `l_warptile` values. Default behavior remains unchanged when the variable is absent.

## Results

`llama-bench`, Qwen3.6-27B-Q3_K_S, `b=4096`, `ub=512`, `fa=1`, `q4_0/q4_0`, runs=1:

| Vulkan mode | pp4096 tok/s | pp8192 tok/s |
| --- | ---: | ---: |
| default large tile off | `617.89` | `611.37` |
| `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` | `758.79` | `859.66` |
| `GGML_VK_FORCE_AMD_LARGE_MATMUL=1 GGML_VK_DISABLE_MMVQ=1` | `907.45` | `835.29` |

Prompt-heavy workload (`triage_diff`, repo-snapshot, 7489 prompt tokens, 64 generated):

| Config | Wall TPS | Prompt eval TPS | Decode eval TPS | Notes |
| --- | ---: | ---: | ---: | --- |
| E062 best Vulkan `GGML_VK_DISABLE_MMVQ=1`, `b4096/ub512` | `4.7172` | `639.81` | `35.15` | previous Vulkan best |
| large tile only, `b4096/ub512` | `5.6963` | `786.43` | `38.30` | strong prefill gain |
| large tile + DISABLE_MMVQ, `b4096/ub512` | `6.2619` | `885.69` | `37.10` | near ROCm in r1 |
| large tile + DISABLE_MMVQ, `b4096/ub512`, runs=3 | `6.18` | n/a | n/a | stable but slightly below ROCm |
| large tile + DISABLE_MMVQ, `b8192/ub1024` | `6.2795` | `883.42` | `38.23` | best E064 r1 shape |

Reference ROCm E061 prompt-heavy baseline: wall `6.3327`, prompt eval `960.26`, decode eval `28.32`.

## Decision

Keep the opt-in env knob as a confirmed Vulkan acceleration probe. On this RDNA4/Windows driver it moves Vulkan from far below ROCm to within roughly one percent of the ROCm wall baseline. Do not make it an unconditional default yet; upstream disabled the proprietary-driver large path for regression risk, so broader device validation is required.

## Artifacts

- `build_logs/agent-workload/e064-vulkan-force-amd-large-llamabench-pp4096-8192.md`
- `build_logs/agent-workload/e064-vulkan-force-amd-large-disablemmvq-llamabench-pp4096-8192.md`
- `build_logs/agent-workload/e064-vulkan-force-amd-large-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e064-vulkan-force-amd-large-disablemmvq-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e064-vulkan-force-amd-large-disablemmvq-ctx12288-q3ks-r3.diagnostics.md`
- `build_logs/agent-workload/e064-vulkan-large-disablemmvq-b8192-ub1024-ctx12288-q3ks.diagnostics.md`
