# D034 P002 Vulkan 130k Residency Recheck

Date: 2026-05-27

Status: closed as diagnostic evidence; code prototypes reverted.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract baseline: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, `max_tokens=16`, cold-first, no reuse, no v2 prime, thinking on.
- Current accepted baseline remains D012 `d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3`: `2.0013 TPS`, prompt `1053.1067 tok/s`, decode `42.7233 tok/s`.

## Trigger

Fresh same-lane D012 controls in this session fell back into the old full-server
130k residency slow pocket rather than reproducing the accepted D012 corridor.
Representative controls stayed near `0.36-0.39 TPS` with prompt around
`190 tok/s`, even while route traces still selected the q3quad/split-K path.

Direct `llama-bench` did not reproduce the failure, so this is not a broad
Q3_K shader/build regression:

- D012 direct `pp4096` with graphics queue: `1066.39 tok/s`, `tg1 40.12 tok/s`.
- Same shape at `ctx=65536` full server: `d034-vulkan-64k-d012-shape-residency-probe-r1`, `1.9212 TPS`, prompt `1037.12 tok/s`, decode `41.04 tok/s`.
- Full `ctx=131072` server stayed slow: `d034-vulkan-130k-q3-fulltile-store-r1`, `0.3897 TPS` class after a small shader prototype.

Conclusion: the current failure is the full `ctx=131072` server residency/
paging footprint, not the raw Q3_K matmul route by itself.

## Probes

| Probe | Best label | Result | Decision |
| --- | --- | ---: | --- |
| Memory priority feature enable, pageable local memory, explicit `setMemoryPriorityEXT` | `d034-vulkan-130k-setmempriority-th4096-r1` | still `~0.37 TPS` | reverted; not a recovery route |
| Full KV in backend host memory | `d034-vulkan-130k-kv-host-gpu-dev-r1` | `1.7369 TPS`, prompt `952.29`, decode `28.21` | diagnostic only; decode tax too high |
| V-only backend-host KV | `d034-vulkan-130k-k-dev-v-host-r1` | `1.8052 TPS`, prompt `983.20`, decode `32.47` | diagnostic only |
| Q3_K q3quad aligned full-tile store body | direct `llama-bench pp4096` | `1066.39 -> 1085.72 tok/s` (`~+1.8%` point) | local micro gain but below 2.4 gate; reverted |
| Full-tile store at `ctx=65536` | `d034-vulkan-64k-fulltile-store-r1` | `1.9640 TPS` vs prior `1.9212` probe | historical/diagnostic only |
| V-host + full-tile + `ub512` | `d034-vulkan-130k-v-host-fulltile-ub512-r1` | `1.9312 TPS`, prompt `1056.05`, decode `32.71` | below D012; not promoted |
| Partial K+V host, last layers, full-tile + `ub512` | `d034-vulkan-130k-kvhost14-fulltile-lowtile2-ub512-r1` | `1.9826 TPS`, prompt `1078.72`, decode `36.98` | best recovery, still below D012 and below 2.4 |

`V:16` and `V:24` partial-host checks showed a sharp residency threshold:
`V:16` timed out at the task hard limit, while `V:24` still only reached
`0.8581 TPS`. Around `576 MiB` of KV moved to `Vulkan_Host` was enough to
recover prompt eval, but the host-KV decode tax prevented a new accepted speed
baseline.

## Decision

D034 is closed as diagnostic evidence and all D034 code prototypes were
reverted. It produced a useful causal map but no accepted speed gain over the
D012 baseline:

- Do not compare future candidates against the `0.37 TPS` slow-pocket controls.
- Backend-host KV placement is a residency recovery diagnostic, not a speed
  route while decode drops from the D012 `42.72 tok/s` band to about
  `36-37 tok/s` or lower.
- Q3_K full-tile shared-stage removal is a small point/local win, but the
  prebuild gate projected only about `+1.4%` total from a `+2%` local estimate,
  far below the D028/D030 target.
- The next Vulkan `2.4 TPS` candidate still needs a true Q3_K compute body or
  compressed-dot route that changes the Q3 work itself, or a stronger lifetime
  design that recovers residency without paying decode back.

Canonical artifacts are in `build_logs/agent-workload/` under the `d034-*`
labels above.