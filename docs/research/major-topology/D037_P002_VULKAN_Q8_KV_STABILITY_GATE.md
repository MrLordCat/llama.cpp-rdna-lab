# D037 P002 Vulkan Q8 KV Stability Gate

Date: 2026-05-27

Status: implemented, built, and smoke validated as an opt-in q8/q8 stability
profile. This is not a default speed route. The kept default for the active
Vulkan 130k lane remains the D036 q4_0/q4_0 direct host-KV last3 guard.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, cold-first, no
  reuse, no v2 prime, thinking on.
- Full speed probes used `max_tokens=16`; route/warning smokes used
  `max_tokens=1` only where noted.
- Launch still uses `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`.

## Trigger

The user asked to try q8 KV for better long agent-work stability and to study
how the speed loss could be recovered. D036 already restored the q4/q4 default
decode corridor above `40 tok/s`, but q4 KV can be a quality/stability tradeoff
for long agent contexts.

The test question was therefore narrower than a normal speed experiment:

- Can q8/q8 fit at `ctx=131072` on RX 9070 XT 16 GB?
- If it fits, can it stay near the D036/D012 speed corridor?
- Can mixed q4/q8 or q8/q4 recover speed while improving one side of KV?
- What should be exposed as default, opt-in, or warning-only behavior?

## Implementation

`src/llama-kv-cache.cpp` now separates the narrow Qwen35-like 130k Vulkan shape
predicate from the q4/q4 default action:

- q4/q4 default remains unchanged: last `3/16` KV layers move to
  `Vulkan_Host_Direct` automatically, unless `LLAMA_DISABLE_VK_KV_HOST_AUTO` is
  set.
- q8/q8 gets a new explicit opt-in: `LLAMA_VK_KV_HOST_AUTO_Q8=1` moves the last
  `8/16` KV layers to `Vulkan_Host_Direct`, unless the auto guard is disabled.
- Manual controls from D036 still apply: `LLAMA_VK_KV_HOST_LAYERS`,
  `LLAMA_VK_KV_HOST_DIRECT`, `LLAMA_VK_KV_HOST_POSITION`, and
  `LLAMA_VK_KV_HOST_MODE`.
- Mixed K/V cache types on Vulkan now emit a warning from the KV allocator:
  mixed q4/q8 can force Flash Attention fallback without coopmat2 mixed-KV
  support and can create large graph splits.
- The Vulkan support check also warns once when mixed K/V Flash Attention is
  rejected because the device lacks the coopmat2 path.

## Results

| Variant | Label | B/UB | TPS | Prompt tok/s | Decode tok/s | Residency / route | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| D036 q4/q4 default r3 | `d036-vulkan130k-default-directkv-last3-b512-ub256-r3` | `512/256` | `1.9410` | `1049.28` | `40.2033` | `Vulkan0 KV 1872.00 MiB`, `Vulkan_Host_Direct KV 432.00 MiB`, splits `2` | kept default |
| D037 q4/q4 postpatch smoke | `d037-vulkan130k-q4default-postpatch-smoke-b512-ub256-r1` | `512/256` | `1.9480` | `1054.28` | `40.15` | same last3 direct host-KV default, splits `2` | confirms q4 default unchanged |
| q8/q8 no host | `d037-vulkan130k-q8kv-control-b512-ub256-r1` | `512/256` | startup fail | n/a | n/a | projected `16164 MiB` vs `15221 MiB`, need `1967 MiB` reduction | rejects full-device q8/q8 |
| q8/q8 direct host-KV last8 | `d037-vulkan130k-q8kv-directkv-last8-b512-ub256-r1` | `512/256` | `0.3630` | `187.94` | `34.36` | `Vulkan0 KV 2176.00 MiB`, `Vulkan_Host_Direct KV 2176.00 MiB`, splits `2` | fits but too slow for default |
| q8/q8 K-only host all16 | `d037-vulkan130k-q8kv-directkv-konly16-b512-ub256-r1` | `512/256` | `0.3601` | `186.40` | `34.74` | K in `Vulkan_Host_Direct`, V on `Vulkan0`, splits `2` | no speed recovery |
| q8/q8 V-only host all16 | `d037-vulkan130k-q8kv-directkv-vonly16-b512-ub256-r1` | `512/256` | `0.3557` | `184.82` | `25.42` | V in `Vulkan_Host_Direct`, K on `Vulkan0`, splits `2` | V-side q8 host path is worse for decode |
| q8/q8 route smoke | `d037-vulkan130k-q8kv-directkv-last8-route-b512-ub256-r1` | `512/256` | max1 only | `187.56` | n/a | FA route `flash_attn_f32_f16_aligned_f32accq8_0`, `path=coopmat1`, `k=q8_0`, `v=q8_0`, splits `2` | route identified, still slow |
| q8/q8 auto-q8 smoke | `d037-vulkan130k-q8kv-autoq8-last8-smoke-b512-ub256-r1` | `512/256` | max1 only | `184.42` | n/a | selected last `8/16`, direct `(q8 opt-in)`, `Vulkan0 KV 2176.00 MiB`, `Vulkan_Host_Direct KV 2176.00 MiB`, splits `2` | keep opt-in selector |
| q4/q8 mixed no-host | `d037-vulkan130k-kq4-vq8-control-b512-ub256-r1` | `512/256` | timeout | partial | n/a | projected `15140 MiB`, `Vulkan0 KV 3328.00 MiB`, `Vulkan_Host TG 208.79 MiB`, splits `34` | reject mixed path |
| q8/q4 mixed no-host | `d037-vulkan130k-kq8-vq4-control-b512-ub256-r1` | `512/256` | timeout | partial | n/a | projected `15140 MiB`, `Vulkan0 KV 3328.00 MiB`, `Vulkan_Host TG 208.79 MiB`, splits `34` | reject mixed path |
| q4/q8 mixed direct last3 | `d037-vulkan130k-kq4-vq8-directkv-last3-b512-ub256-r1` | `512/256` | timeout | partial | n/a | projected `14516 MiB`, `Vulkan_Host_Direct KV 624.00 MiB`, still splits `34` | host relief does not fix fallback |
| q4/q8 mixed no-FA | `d037-vulkan130k-kq4-vq8-nofa-control-b512-ub256-r1` | `512/256` | startup fail | n/a | n/a | `V cache quantization requires flash_attn` | no-FA is not a workaround |
| q4/q8 mixed warning smoke | `d037-vulkan130k-kq4-vq8-mixedwarn-postpatch2-b512-ub256-r1` | `512/256` | timeout | partial | n/a | warning emitted, `Vulkan_Host TG 208.79 MiB`, splits `34` | warning works |

## Route Explanation

Full q8/q8 is internally coherent on this device: the route trace uses the
same-type coopmat1 Flash Attention shader (`k=q8_0`, `v=q8_0`) and keeps graph
splits at `2` when enough KV is moved to `Vulkan_Host_Direct`. The problem is
not graph fragmentation; the problem is that prompt eval collapses to about
`185-188 tok/s`, versus the q4/q4 D036 corridor at about `1049-1054 tok/s`.

Mixed q4/q8 and q8/q4 are worse on the current RDNA Vulkan path. The generated
same-type scalar/coopmat1 Flash Attention shaders use one K/V data type family;
the mixed `FaTypeK/FaTypeV` path is only in the coopmat2 mixed shader family.
This device does not take that route, so mixed K/V falls back, creates `34`
graph splits, allocates a large `Vulkan_Host TG compute buffer`, and times out
even when direct host-KV reduces projected device memory.

Disabling Flash Attention is not a workaround for mixed K/V because quantized V
cache requires Flash Attention on this model/lane.

## Decision

Keep D036 q4/q4 direct host-KV last3 as the only default 130k Vulkan profile.
It preserves the fast prompt path and keeps decode above the user-requested
`40 tok/s` threshold.

Keep D037 q8/q8 only as an explicit stability/offline profile:

```bash
LLAMA_VK_KV_HOST_AUTO_Q8=1
```

This selector chooses last `8/16` direct host-KV layers for the narrow
Qwen35-like 130k Vulkan shape. It is useful for testing whether q8 KV improves
agent stability/quality, but it is too slow for the default speed profile.

Reject mixed q4/q8 and q8/q4 on this Vulkan/RDNA lane until there is a real
mixed-KV Flash Attention path for the active device or another mechanism that
keeps graph splits near the q4/q4/q8/q8 same-type corridor. The new warning is
kept so users see the reason instead of only observing a timeout.

The accepted speed comparator remains D012 `2.0013 TPS` r3, and the current
guarded default checkpoint remains D036 `1.9410 TPS` r3. D037 does not change
the next speed requirement: a future win still needs true Q3_K compute-body or
compressed-dot work, not KV type mixing.

Canonical artifacts are in `build_logs/agent-workload/`:

- `d037-vulkan130k-q4default-postpatch-smoke-b512-ub256-r1.diagnostics.md`
- `d037-vulkan130k-q8kv-control-b512-ub256-r1.server.log`
- `d037-vulkan130k-q8kv-directkv-last8-b512-ub256-r1.diagnostics.md`
- `d037-vulkan130k-q8kv-directkv-konly16-b512-ub256-r1.diagnostics.md`
- `d037-vulkan130k-q8kv-directkv-vonly16-b512-ub256-r1.diagnostics.md`
- `d037-vulkan130k-q8kv-directkv-last8-route-b512-ub256-r1.server.log`
- `d037-vulkan130k-q8kv-autoq8-last8-smoke-b512-ub256-r1.server.log`
- `d037-vulkan130k-kq4-vq8-control-b512-ub256-r1.server.log`
- `d037-vulkan130k-kq8-vq4-control-b512-ub256-r1.server.log`
- `d037-vulkan130k-kq4-vq8-directkv-last3-b512-ub256-r1.server.log`
- `d037-vulkan130k-kq4-vq8-nofa-control-b512-ub256-r1.server.log`
- `d037-vulkan130k-kq4-vq8-mixedwarn-postpatch2-b512-ub256-r1.server.log`