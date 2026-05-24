# E199 ROCm Q3_K Padded Layout Gate

## Metadata

- Experiment ID: E199
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after E198 rollback
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: part of Vulkan's Q3_K decode advantage comes from its backend-private 112-byte Q3_K device layout, which enables packed32-style loads and cleaner shader code. ROCm currently uses the GGUF host `block_q3_K` stride of `110` bytes directly.
- Mechanism: if ROCm stored Q3_K tensors in backend-private 112-byte blocks, Q3_K kernels could use a Vulkan-like aligned layout instead of mixed 110-byte block pointer arithmetic.
- Why now: E197 rejected wave64/row-warp reduction-only transfer, and E198 rejected graph-local q8 activation caching. The remaining high-ceiling decode route has to reduce real Q3_K dot/dequant work or change the weight layout/route structure.

## Math / Theory

- Assumptions:
  - E196 clean decode comparator is ROCm r3 `31.9233 TPS` / decode `32.3833 tok/s` vs Vulkan r3 `40.8007 TPS` / decode `41.795 tok/s`;
  - E196 steady forward split puts Q3_K near `59.5%` of measured forward MUL_MAT time in the optimistic parsed view;
  - a conservative synchronized wall proxy treats the directly addressable Q3_K share as `32.0%`;
  - Vulkan pads Q3_K/Q6_K device blocks by `+2` bytes, while ROCm CUDA/HIP buffer set/get currently copies raw host bytes and all `block_q3_K *` arithmetic assumes `110` bytes.
- Expected speedup corridor:

| Share label | Share | Target wall | Required local |
| --- | ---: | ---: | ---: |
| optimistic parsed Q3_K/MMVQ share | `0.595` | `1.020x` | `1.0341x` |
| optimistic parsed Q3_K/MMVQ share | `0.595` | `1.050x` | `1.0870x` |
| optimistic parsed Q3_K/MMVQ share | `0.595` | `1.100x` | `1.1803x` |
| optimistic parsed Q3_K/MMVQ share | `0.595` | `1.278x` | `1.5763x` |
| conservative sync wall share proxy | `0.320` | `1.020x` | `1.0653x` |
| conservative sync wall share proxy | `0.320` | `1.050x` | `1.1748x` |
| conservative sync wall share proxy | `0.320` | `1.100x` | `1.3968x` |
| conservative sync wall share proxy | `0.320` | `1.278x` | `3.1228x` |

- Memory overhead if ROCm replaces storage with padded blocks:

| Tensor group | Tensor count | Current Q3_K bytes | Padded 112-byte bytes | Delta |
| --- | ---: | ---: | ---: | ---: |
| FFN all Q3_K | `192` | `6.85 GiB` | `6.97 GiB` | `127.50 MiB` (`1.818%`) |
| FFN gate/up Q3_K | `128` | `4.57 GiB` | `4.65 GiB` | `85.00 MiB` (`1.818%`) |
| FFN down Q3_K | `64` | `2.28 GiB` | `2.32 GiB` | `42.50 MiB` (`1.818%`) |
| all Q3_K | `353` | `9.64 GiB` | `9.81 GiB` | `179.47 MiB` (`1.818%`) |

- Failure conditions:
  - transient per-node repack is needed before every hot matvec route;
  - padded copy is duplicated beside the existing ROCm weight storage;
  - only `vecdotq.cuh` load helpers change while buffer stride remains `110` bytes;
  - correctness cannot be proven before real-server testing, because every Q3_K kernel must agree on the same device stride.

## Implementation Plan

1. Minimal code surface to change:
   - no runtime code change in E199; this is an analytic/design gate;
   - future E2xx storage branch would need a CUDA/HIP backend device type-size layer, padded set/get/set_2d/get_2d, view-offset translation, Q3_K padded device structs/helpers, and an audit of all Q3_K kernels.
2. Guard rails:
   - reject transient repack and duplicate persistent padded copy before code;
   - do not attempt a local 32-bit load rewrite on `110`-byte blocks;
   - future implementation must start with a storage correctness smoke/micro test before `llama-server`;
   - future candidate must be env-gated until Q3_K MMVQ/MMQ/dequant/get_rows paths pass correctness and route traces.
3. Rollback path:
   - no code rollback needed for E199;
   - future storage branch must keep a single guard that returns CUDA/HIP to raw GGUF `110`-byte Q3_K storage.

## Benchmark Plan

- Baseline command: E196 clean ROCm r3 reference `31.9233 TPS`, decode `32.3833 tok/s`.
- Candidate command: none in E199; route rejected/deferred before runtime code.
- Number of runs: no runtime run.
- Artifacts path: `build_logs/agent-workload/e199-rocm-q3k-padded-layout-gate.md`.

## Metrics

- projected memory delta
- required local speedup from Amdahl gate
- route blast radius
- correctness scope before any future speed claim

## Result

- Outcome: design gate / no runtime speed claim.
- Delta:
  - replacing all Q3_K storage with padded 112-byte blocks would add only `179.47 MiB` (`1.818%`) for this model;
  - duplicating padded Q3_K beside current storage is not the same cost; it effectively adds another multi-GiB Q3_K model copy and is rejected for the 16 GiB lane;
  - Vulkan parity still needs `1.5763x` local under the optimistic `59.5%` Q3_K share, or `3.1228x` under the conservative `32.0%` share.
- Confidence: medium-high for rejecting quick paths, medium for the future full-storage route. The Vulkan/ROCm storage-contract difference is concrete, but the speed payoff of a full ROCm padded layout still needs kernel/resource proof.
- Recommendation: reject quick padded-layout variants. Defer the only plausible transfer, backend-private ROCm Q3_K padded storage, as a large E2xx design with correctness smoke tests and full Q3_K kernel audit.

## Notes

- Surprise: the memory overhead of replacing Q3_K storage is small (`1.818%`), so the blocker is not padding size. The blocker is that ROCm currently lacks Vulkan's device type-size abstraction and every Q3_K user assumes the host block stride.
- Why this matters: a local `vecdotq.cuh` packed32 patch would not reproduce Vulkan's route. Every other block starts at a `2 mod 4` offset under `110`-byte stride, so unaligned 32-bit loads become irregular, and the dot/dequant body still remains.
- Follow-up action: either draft a full ROCm padded-storage E2xx design, or choose another structural Q3_K route that reduces dot/dequant work without changing the global tensor storage contract.
