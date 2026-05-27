# D036 P002 Vulkan Direct Host-KV Decode Recovery

Date: 2026-05-27

Status: implemented, built, and 3-run validated as a default decode recovery.
This restores `40+ tok/s` decode on the fresh default-stability corridor, but it
is not a new speed baseline over the accepted D012 `2.0013 TPS` r3 result.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, `max_tokens=16`, cold-first, no reuse, no v2 prime, thinking on.
- Launch still uses `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`.

## Trigger

D035 recovered the fresh 130k prompt slow pocket but paid decode back. Its kept
default guard result was `1.8736 TPS`, prompt `1014.61 tok/s`, decode
`37.59 tok/s`, with `576.00 MiB` KV on `Vulkan_Host`, `10` graph splits, and a
large TG compute buffer. The user asked to restore decode to `40+ tok/s` while
keeping the prompt/residency recovery.

A no-host negative control proved the decode ceiling was still present
(`41.44 tok/s`) but fell back into the prompt slow pocket (`0.3552 TPS`, prompt
`183.51 tok/s`). The problem was therefore the host-KV scheduler/copy tax, not
the decode kernels themselves.

## Implementation

The Vulkan backend now exposes a separate pinned host buffer type:

- `ggml_backend_vk_host_direct_buffer_type()` returns `Vulkan_Host_Direct`.
- The device reports support for `Vulkan_Host_Direct`, but not for ordinary
  `Vulkan_Host`, so model tensors in the normal host buffer do not become direct
  Vulkan operands.
- `ggml_vk_tensor_subbuffer()` resolves pinned host tensors through
  `ggml_vk_host_get()`.
- `ggml_vk_build_graph()` overlap tracking now resolves host-pinned tensor
  storage before comparing unsynchronized buffer ranges. This avoids treating a
  CPU host buffer context as a Vulkan buffer context.

The Qwen35-like Vulkan host-KV guard from D035 now defaults to a smaller direct
placement:

- Auto condition stays narrow: offload enabled, `kv_size >= 131072`,
  `n_layer=64`, `n_embd=5120`, `16` active KV layers, and K/V cache types both
  `q4_0`.
- Auto action: place the last `3/16` filtered KV layers on
  `Vulkan_Host_Direct`, moving `432.00 MiB` of KV out of `Vulkan0`.
- Manual override: `LLAMA_VK_KV_HOST_LAYERS=N`.
- Direct override: `LLAMA_VK_KV_HOST_DIRECT=0/1`; auto uses direct by default,
  manual host placement only becomes direct when this is set to `1`.
- Placement diagnostics: `LLAMA_VK_KV_HOST_POSITION=first`.
- K/V split diagnostics: `LLAMA_VK_KV_HOST_MODE=k|v|kv|none`.
- Auto opt-out: `LLAMA_DISABLE_VK_KV_HOST_AUTO`.

## Failed Probes

| Probe | Result | Decision |
| --- | ---: | --- |
| Broad `GGML_VK_DIRECT_HOST_BUFFER=1` support for ordinary `Vulkan_Host` | startup assert at `ggml-vulkan.cpp:8159`, `d_Qx != nullptr` | reject: regular host model weights must not become direct operands |
| Separate `Vulkan_Host_Direct` before graph-overlap fix | access violation in `ggml_vk_build_graph()` overlap tracking | fixed by resolving host-pinned storage before overlap checks |
| Direct host-KV first4 | `1.9378 TPS`, prompt `1048.18`, decode `39.63` | close but below 40 tok/s |
| Direct host-KV last4 | `1.9440 TPS`, prompt `1051.77`, decode `39.65` | close but below 40 tok/s |
| Direct host-KV last2 | `0.3510 TPS`, prompt `181.35`, decode `40.73` | rejects: not enough residency relief, prompt slow pocket returns |

## Results

| Variant | Label | B/UB | TPS | Prompt tok/s | Decode tok/s | Residency | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| D035 default guard | `vscode-vulkan130k-defaultguard-b512-ub256-r2` | `512/256` | `1.8736` | `1014.61` | `37.59` | `Vulkan0 KV 1728.00 MiB`, `Vulkan_Host KV 576.00 MiB`, splits `10` | old default recovery |
| D036 direct host-KV last3 r1 | `d036-vulkan130k-directkv-last3-b512-ub256-r1` | `512/256` | `1.9487` | `1053.78` | `40.17` | `Vulkan0 KV 1872.00 MiB`, `Vulkan_Host_Direct KV 432.00 MiB`, splits `2` | candidate |
| D036 default direct host-KV last3 r1 | `d036-vulkan130k-default-directkv-last3-b512-ub256-r1` | `512/256` | `1.9512` | `1055.38` | `40.19` | `Vulkan0 KV 1872.00 MiB`, `Vulkan_Host_Direct KV 432.00 MiB`, splits `2` | default smoke |
| D036 default direct host-KV last3 r3 | `d036-vulkan130k-default-directkv-last3-b512-ub256-r3` | `512/256` | `1.9410` | `1049.28` | `40.2033` | `Vulkan0 KV 1872.00 MiB`, `Vulkan_Host_Direct KV 432.00 MiB`, splits `2` | keep |

D036 improves the kept D035 default guard from `1.8736 -> 1.9410 TPS`
(`+3.60%`), prompt `1014.61 -> 1049.28 tok/s`, and decode
`37.59 -> 40.2033 tok/s`. The 3-run confirmation held decode above the user
target with min/mean/max `40.14/40.2033/40.24 tok/s`.

Compared with the accepted D012 speed baseline, D036 remains below wall TPS
(`1.9410` vs `2.0013`) and below the old D012 decode corridor (`40.2033` vs
`42.7233`). It is therefore a default-stability/decode-recovery fix, not a new
Vulkan speed baseline or a `2.4 TPS` route.

## Decision

Keep D036 as the default D035 follow-up for the Qwen35-like 130k Vulkan guard.
It preserves prompt recovery, reduces graph splits/TG compute pressure from the
ordinary host-KV path, and restores default decode to `40+ tok/s` without using
the slow-pocket `0.36 TPS` control as a speed baseline.

The accepted speed comparator remains D012 `2.0013 TPS` r3. The next speed route
still needs true Q3_K compute-body/compressed-dot work or another topology that
beats/ties D012 on the same cold lane.

Canonical artifacts are in `build_logs/agent-workload/`:

- `d036-vulkan130k-default-directkv-last3-b512-ub256-r3.diagnostics.md`
- `d036-vulkan130k-default-directkv-last3-b512-ub256-r3.server.log`
- `d036-vulkan130k-directkv-last3-b512-ub256-r1.diagnostics.md`
- `d036-vulkan130k-directkv-last2-b512-ub256-r1.diagnostics.md`
- `d036-vulkan130k-directkv-last4-b512-ub256-r1.diagnostics.md`
- `d036-vulkan130k-directkv2-kvhost4-first-b512-ub256-r1.diagnostics.md`