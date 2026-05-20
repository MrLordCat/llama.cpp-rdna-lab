# E100 Vulkan Current Controls, Store Path, And Shape Gates

## Metadata

- Experiment ID: E100
- Date: 2026-05-20
- Owner: Copilot
- Type: same-session control refresh + shader/store/shape negative gates
- Hypothesis: H31
- Target lane: Qwen3.6-27B-Q3_K_S Vulkan 32k diagnostic lane, `ctx=32768`, `b=5120`, `ub=1024`, `q4_0/q4_0`, FlashAttention on, spec none, no reuse, thinking on

## Control Refresh

Same-session controls after cleaning the temporary Q8 route experiment:

| Backend/config | Aggregate TPS | Prompt eval TPS | Decode eval TPS | Notes |
| --- | ---: | ---: | ---: | --- |
| Vulkan safe large path | `10.5230` | `993.94` | `32.93` | `e100-vulkan32k-store-baseline-r1` |
| ROCm `build-rocm-vec` | `10.8879` | `1132.44` | `28.49` | `e100-rocm32k-current-control-r1` |

Current Vulkan gap on this 32k lane is `-3.35%` aggregate and `-12.2%` prompt eval, while Vulkan decode remains faster by about `+15.6%`.

## Store Path Probe

The active aligned coopmat shader still compiled the unaligned full-tile store branch. A temporary specialization removed that branch for `ALIGNED` variants and made the C++ aligned flag include `stride_d % 4 == 0`.

Static effect:

- `OpCooperativeMatrixStoreKHR`: `3 -> 2`
- `OpControlBarrier`: `6 -> 4`

Measured result:

| Config | pp7488 | Workload aggregate | Decision |
| --- | ---: | ---: | --- |
| aligned-store candidate r3 | `986.31 ± 1.85` | `10.4974` | reject |
| same-session original store baseline | not needed | `10.5230` | keep original |

The cleaner SPIR-V did not translate to workload speed. Reverted the store specialization.

## Shape Gates

No-code Vulkan shape gates did not improve the 32k lane:

| Shape | Result | Decision |
| --- | ---: | --- |
| `b=4096,ub=1024` | `10.16` aggregate | reject |
| `b=6144,ub=1024` | `10.30` aggregate | reject |
| `b=5120,ub=512` | `10.06` aggregate | reject |
| `b=5120,ub=1536` | hard timeout | reject |

The pp-only ubatch scan showed `ub=1536` slightly above `ub=1024`, but real workload timed out, so `ub=1024` remains the safe shape.

## Decision

- Keep original store path.
- Keep 32k Vulkan diagnostic shape at `b=5120,ub=1024`.
- Current prompt gap remains code-level prefill work, not a nearby batch/ubatch/store branch.

## Artifacts

- `build_logs/agent-workload/e100-vulkan32k-store-baseline-r1.diagnostics.md`
- `build_logs/agent-workload/e100-rocm32k-current-control-r1.diagnostics.md`
- `build_logs/agent-workload/e100-aligned-store-q3k-pp7488-r3.txt`
- `build_logs/agent-workload/e100-vulkan-pp7488-ub-scan.txt`
- `build_logs/agent-workload/e100-vulkan32k-ub1536-r1.diagnostics.md`