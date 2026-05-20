# E102 Vulkan AMD Auto Large Matmul Default

## Metadata

- Experiment ID: E102
- Date: 2026-05-20
- Owner: Copilot
- Type: default behavior improvement + rollback gate
- Hypothesis: H31
- Target lane: Vulkan Q3_K prompt-heavy prefill, AMD proprietary driver, RX 9070 XT

## Hypothesis

The local fast Vulkan Q3_K route required `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`. On the local AMD proprietary coopmat device this should be the default instead of an agent-only env requirement, while preserving an explicit rollback switch.

## Change

Auto-enable AMD large matmul when all of these are true:

- `vendor_id == VK_VENDOR_ID_AMD`
- `driver_id == vk::DriverId::eAmdProprietary`
- `architecture == AMD_RDNA3` (the local RDNA4 path is classified here by existing backend detection)
- KHR cooperative matrix support is present
- device is not UMA

Kept explicit controls:

- `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`: force large path on other cases for experiments
- `GGML_VK_DISABLE_AMD_LARGE_MATMUL=1`: disable the new auto-default for rollback/negative control

## Validation

No-env pp gate after code cleanup:

| Config | Route/resource | pp7488 tok/s |
| --- | --- | ---: |
| no env, auto-large default | `matmul_q3_k_f32_f16acc_aligned_l`, `113 VGPR / 45 SGPR / 20480 B LDS / scratch 0` | `983.48` |
| disable auto-large | slow non-large route | `708.19` |

No-env 32k workload sanity:

| Config | Aggregate TPS | Notes |
| --- | ---: | --- |
| `e102-vulkan32k-auto-large-noenv-r1` | `10.46` | same fast path without force env |

This is not a new algorithmic speedup over the prior forced-env fast path, but it is a real default/readiness improvement: future local Vulkan runs now pick the fast prefill route automatically on the target device.

## Decision

Keep the auto-large default and rollback env. Future performance comparisons on this workstation should not require `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`; use `GGML_VK_DISABLE_AMD_LARGE_MATMUL=1` only as a negative control or rollback.

## Artifacts

- `build_logs/agent-workload/e102-vulkan-auto-large-noenv-pp7488-r1.txt`
- `build_logs/agent-workload/e102-vulkan-auto-large-disable-control-pp7488-r1.txt`
- `build_logs/agent-workload/e102-vulkan-auto-large-clean-pp7488-r1.txt`
- `build_logs/agent-workload/e102-vulkan32k-auto-large-noenv-r1.diagnostics.md`