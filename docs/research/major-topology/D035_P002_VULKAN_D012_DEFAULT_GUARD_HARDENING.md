# D035 P002 Vulkan D012 Default Guard Hardening

Date: 2026-05-27

Status: implemented, built, and single-run validated as a default-stability
recovery. Not an accepted speed gain over the D012 `2.0013 TPS` r3 baseline.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, `max_tokens=16`, cold-first, no reuse, no v2 prime, thinking on.
- Launch still uses `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`; D035 promotes the D012 route pieces and the narrow residency guard into guarded source defaults, not a no-env graphics-queue policy change.

## Trigger

Fresh same-lane D012 server controls reproduced the 130k slow pocket instead of
the accepted D012 corridor. The representative clean control
`p002-resume-d012-control-r1` used the active `b512/ub256` lane and fell to
`0.3582 TPS`, prompt `185.11 tok/s`, decode `41.37 tok/s`, with the full
`2304.00 MiB` KV buffer resident on `Vulkan0` and `2` graph splits.

D034 showed that moving about `576 MiB` of KV into `Vulkan_Host` is enough to
recover prompt eval, but broad host-KV placement remained diagnostic because it
paid decode back and did not beat D012. D035 turns only the narrow recovered
case into a guarded default for the Qwen35-like 130k q4_0/q4_0 shape.

## Implementation

Guarded Vulkan defaults now cover the old D012 opt-in route pieces:

- AMD proprietary coopmat devices default the large matmul variant to `bn256`
  when eligible. Opt out with `GGML_VK_DISABLE_AMD_BN256_DEFAULT` or override
  with `GGML_VK_AMD_LARGE_MATMUL_VARIANT`.
- Q3_K quad dequant defaults on for the measured aligned Q3_K/F32 shapes. Opt
  out with `GGML_VK_DISABLE_Q3K_QUAD_DEQUANT` or set
  `GGML_VK_Q3K_QUAD_DEQUANT=0`.
- AMD proprietary Q3_K low-tile candidates default split-K by `n`: `3` for
  `n < 512`, `2` for `n >= 512`. Opt out with
  `GGML_VK_DISABLE_QK_LOW_TILE_DEFAULT`; manual
  `GGML_VK_QK_LOW_TILE_SPLIT_K` still wins.

The KV cache allocator now has a narrow Vulkan host-KV residency guard:

- Auto condition: offload enabled, `kv_size >= 131072`, `n_layer=64`,
  `n_embd=5120`, `16` active KV layers, and K/V cache types both `q4_0`.
- Auto action: place the last `4/16` filtered KV layers on the device host
  buffer type, moving `576.00 MiB` of KV out of `Vulkan0`.
- Manual override: `LLAMA_VK_KV_HOST_LAYERS=N`.
- Auto opt-out: `LLAMA_DISABLE_VK_KV_HOST_AUTO`.

## Results

| Variant | Label | B/UB | TPS | Prompt tok/s | Decode tok/s | Residency | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Clean slow-pocket control | `p002-resume-d012-control-r1` | `512/256` | `0.3582` | `185.11` | `41.37` | `Vulkan0 KV 2304.00 MiB`, splits `2` | reproduced failure |
| D035 default guard | `vscode-vulkan130k-defaultguard-b512-ub256-r2` | `512/256` | `1.8736` | `1014.61` | `37.59` | `Vulkan0 KV 1728.00 MiB`, `Vulkan_Host KV 576.00 MiB`, splits `10` | keep as stability guard |
| D035 ub512 before split guard | `vscode-vulkan130k-defaultguard-b512-ub512-r1` | `512/512` | `0.3155` | `162.99` | `37.17` | host-KV guard active | rejects lowtile3 for `n=512` |
| D035 ub512 with lowtile2 | `vscode-vulkan130k-defaultguard-b512-ub512-lowtile2-r1` | `512/512` | `1.9045` | `1078.00` | `20.94` | host-KV guard active, splits `10` | prefill recovers but decode tax too high |

The kept `b512/ub256` result is a recovery from `0.3582 -> 1.8736 TPS`
(`+423%` wall TPS) on the fresh slow-pocket control, with prompt eval recovering
`185.11 -> 1014.61 tok/s`. It remains below the accepted D012 r3 baseline
`2.0013 TPS` and below its decode corridor (`37.59` vs `42.72 tok/s`), so it is
not a new speed baseline or a `2.4 TPS` route.

## Decision

Keep D035 as a practical default-stability hardening layer for the active 130k
Vulkan lane. It makes the D012 route pieces fail-closed by default on the local
AMD proprietary coopmat device and avoids the severe full-server residency slow
pocket with a narrow Qwen35-like host-KV guard.

Do not use the `0.3582 TPS` control as a future speed baseline. The accepted
speed comparator remains D012 `2.0013 TPS` r3 unless a new candidate runs a
matching confirmation. For final promotion language, run a 3-run confirmation of
the D035 default guard and, if needed, an opt-out negative control with
`LLAMA_DISABLE_VK_KV_HOST_AUTO=1`.

Canonical artifacts are in `build_logs/agent-workload/`:

- `p002-resume-d012-control-r1.diagnostics.md`
- `vscode-vulkan130k-defaultguard-b512-ub256-r2.diagnostics.md`
- `vscode-vulkan130k-defaultguard-b512-ub512-r1.diagnostics.md`
- `vscode-vulkan130k-defaultguard-b512-ub512-lowtile2-r1.diagnostics.md`