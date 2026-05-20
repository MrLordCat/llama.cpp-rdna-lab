# E068 Vulkan AMD Large Matmul Tile Tuning

## Metadata

- Experiment ID: E068
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: local working tree after E067 packed32 rollback
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, ctx=12288, b=4096, ub=1024, q4_0/q4_0 KV, FlashAttention on, thinking on, no reuse, `--spec-type none`

## Hypothesis

- Statement: The AMD proprietary Vulkan Q3_K prompt path may need a different large matmul tile shape than the E064 opt-in default.
- Mechanism: E064 showed that allowing the large cooperative-matrix matmul tile on AMD is the biggest Vulkan prefill lever so far. E067 profiling shows Q3_K large `MUL_MAT` dominates the lane, so small specialization-constant changes to the large tile can shift occupancy/register pressure and improve prompt throughput.
- Why now: Shader-load changes regressed, and no-code fusion/int-dot probes did not beat E065 reliably.

## Math / Theory

- Assumptions: The hot Q3_K shapes (`m=17408/5120/10240`, `n=1024`) are tile/occupancy sensitive on RDNA4.
- Expected speedup corridor: +2-10% prompt eval for a good tile; larger gains are unlikely without a new kernel path.
- Failure conditions: E064 tile is already near the sweet spot; alternative workgroup shapes reduce occupancy or increase register pressure.

## Implementation Plan

1. Minimal code surface to change: add an experimental `GGML_VK_AMD_LARGE_MATMUL_VARIANT` selector around the AMD large matmul tile constants.
2. Guard rails: active only when `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`; defaults unchanged.
3. Rollback path: remove the selector if no variant beats E065.

## Benchmark Plan

- Baseline command: E065 Vulkan `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` pp7488 and active lane.
- Candidate command: same plus `GGML_VK_AMD_LARGE_MATMUL_VARIANT=<variant>`.
- Number of runs: pp7488 `-r 1` gate, active lane `--runs 1`, `--runs 3` only if promising.
- Artifacts path: `build_logs/agent-workload/e068-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- pp7488 llama-bench prompt throughput

## Result

- Outcome: superseded/rejected by later validation
- Delta: best confirmed variant `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32` reached `7.6446` aggregate TPS / `7.58` median over 3 runs, vs E065 Vulkan `6.4180` aggregate (`+19.1%`) and same-session ROCm control `7.3868` aggregate (`+3.5%`). Prompt eval improved to `1110.09 tok/s` but remains below ROCm `1173.24`; decode stayed much faster than ROCm (`40.40` vs `28.62`).
- Confidence: downgraded. Later H31 validation found this route was corrupt/undercovered and should not be used as a profile.
- Recommendation: historical only. Do not use `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32` as an RDNA4/Vulkan opt-in baseline.

## Notes

- Surprises: reducing the large tile's warp-column shape (`WN=16` or `WM=32/WN=32`) helped far more than block size changes. `wn16` and `wm32-wn32` both reached about `7.41-7.42 TPS` in single-run full-lane probes; `wm32-wn32` confirmed best at r3.
- Follow-up correction: E075/E078 invalidated this profile; keep only as a cautionary example that tile variants need static layout and active-route validation.

## Key Measurements

| Config | Runs | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| ROCm same-session control | 3 | `7.3868` | `1173.2367` | `28.62` | current fair target |
| Vulkan E065 large + Q3_K align | 3 | `6.4180` | `897.63` | `40.35` | previous best |
| Vulkan E068 `wn16` | 1 | `7.4109` | `1068.58` | `40.32` | reached ROCm level in r1 |
| Vulkan E068 `wm32-wn32` | 1 | `7.4193` | `1069.97` | `40.36` | best r1 |
| Vulkan E068 `wm32-wn32` | 3 | `7.6446` | `1110.0867` | `40.40` | confirmed best |

Important pp7488 gates:

| Variant | pp7488 tok/s |
| --- | ---: |
| restored E065 default | `875.25` |
| `block128` | `900.32` |
| `wn32` | `981.28` |
| `wn16` | `1039.53` |
| `wm32-wn32` | `1035.80` |

Artifacts:

- `build_logs/agent-workload/e068-vulkan-large-tile-variant-pp7488.md`
- `build_logs/agent-workload/e068-vulkan-large-tile-variant2-pp7488.md`
- `build_logs/agent-workload/e068-vulkan-wn-centered-variant-pp7488.md`
- `build_logs/agent-workload/e068-vulkan-large-wm32-wn32-b4096-ub1024-ctx12288-q3ks-r3.diagnostics.md`