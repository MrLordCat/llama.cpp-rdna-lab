# D003 P002 Vulkan Ubatch Cliff Gate

Status: closed as a speed route; keep `GGML_VK_ENABLE_MEMORY_PRIORITY=1` as a
diagnostic/recovery knob, not a default 2 TPS route.

## Intent

P002 Vulkan `ubatch=256` is the current 130k cold quick winner, but nearby
larger ubatches fell into a severe prompt cliff. D003 tested whether this was a
simple Q3_K route-selection issue, a tail-N artifact, or a memory/residency
placement issue that could unlock a larger-ubatch route.

## Lane

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072`, `batch=512`, `q4_0/q4_0`, FlashAttention on,
  `--spec-type none --no-mmap`.
- Workload: `quick:triage_diff`, repo-snapshot real context,
  `real-context-chars=24576`, no reuse, no v2 prime, thinking on.
- Vulkan lane requires `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`.
- Current speed anchor: `p002-vulkan-ub256-current-r1`, `1.6654 TPS`,
  prompt `866.47 tok/s`, decode `43.42 tok/s`.

## Baseline Cliff Evidence

Diagnostic traces used `max_tokens=1` for prompt-route timing. They are not
completion TPS claims.

| Variant | Prompt tok/s | Parsed total | Q3_K share | Key observation |
| --- | ---: | ---: | ---: | --- |
| `p002-vulkan-ub256-routepack-current-r1` | `795.48` | `8893.65 ms` | `80.50%` | active fast route, `n=256` hot shapes |
| `p002-vulkan-ub320-routepack-r1` | `202.60` | `27798.81 ms` | `76.81%` | severe cliff, `n=320` plus `n=192` chunks |
| `p002-vulkan-ub384-routepack-r1` | `211.55` | `25893.19 ms` | `75.20%` | severe cliff, `n=384` plus `n=128` chunks |

The `ub320` memory fit still left `1047 >= 1024 MiB` and all `65/65` layers
were offloaded, so the first-order fit path did not explain the drop. `ub384`
missed the free-memory target by only `33 MiB`, but `ub320` already proved the
cliff exists even when the target is met.

## Negative Route Probes

Two source probes were tested and then reverted.

| Probe | Local idea | Result | Decision |
| --- | --- | ---: | --- |
| `GGML_VK_Q3K_NGT256_MEDIUM=1` | Force Q3_K/F32 coopmat1 `n > 256` from large to medium pipeline | `141.33 tok/s`, parsed total `44946.80 ms`, Q3_K `38461.68 ms` on `ub384` | reject |
| `GGML_VK_Q3K_SPLIT_N256=1` | Split logical `n > 256` Q3_K matmul into multiple `n <= 256` dispatches | `204.42 tok/s`, parsed total `27367.18 ms`, Q3_K `20808.84 ms` on `ub384` | reject |

Conclusion: the `ub>=320` cliff is not fixed by choosing the existing medium
pipeline and is not a simple tail/split-N issue. The current `mul_mm.comp`
topology remains hostile to larger logical N in this 130k residency lane.

## Memory Placement Gates

The allocator/residency branch produced one useful causal signal but no speed
route.

| Route | Label | Prompt tok/s | Full TPS | Decision |
| --- | --- | ---: | ---: | --- |
| `ub256` control | `d003-vulkan-ub256-gfxq-control-r1` | `869.70` | n/a | prompt control |
| `ub256` no host-visible VRAM | `d003-vulkan-ub256-no-hostvisible-r1` | `871.30` | n/a | tie |
| `ub256` memory priority full | `d003-vulkan-ub256-memory-priority-full-r1` | `866.47` | `1.6646` | wall tie |
| `ub320` control | `d003-vulkan-ub320-gfxq-control-r1` | `174.21` | n/a | cliff |
| `ub320` no host-visible VRAM | `d003-vulkan-ub320-no-hostvisible-r1` | `261.36` | n/a | partial recovery only |
| `ub320` memory priority | `d003-vulkan-ub320-memory-priority-r1` | `808.73` | n/a | recovery, not above `ub256` |
| `ub320` memory priority + no host-visible VRAM | `d003-vulkan-ub320-memory-priority-nohostvisible-r1` | `804.56` | n/a | no stack win |
| `ub320` memory priority full | `d003-vulkan-ub320-memory-priority-full-r1` | `807.40` | `1.5562` | below `ub256` |
| `ub288` memory priority | `d003-vulkan-ub288-memory-priority-r1` | `802.00` | n/a | below `ub256` |
| `ub384` memory priority | `d003-vulkan-ub384-memory-priority-r1` | `199.99` | n/a | still cliff |

`GGML_VK_MEMORY_LOGGER=1` confirmed that `ub320` mainly increases the compute
buffers (`Vulkan0 compute 228.27 -> 285.28 MiB`, host compute
`138.27 -> 172.78 MiB`) and does not show an obvious system-memory fallback.
The logger itself slows prompt evaluation heavily, so it is diagnostic only.

## Decision

Reject D003 as a route to 2 TPS. `VK_EXT_memory_priority` is a strong recovery
signal for `ub320`, but the recovered wall result (`1.5562 TPS`) is still below
the current `ub256` cold lane (`1.6654 TPS`), and the same knob on `ub256` ties
the baseline (`1.6646 TPS`). Do not promote memory priority or larger ubatch by
default for the active 130k route.

Useful workflow correction: any future Vulkan 130k probe must set
`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`; otherwise it measures a different lane and
can produce false negatives.

Next route direction should return to a structural Q3_K/FFN or FA body/layout
change. The larger-ubatch cliff can be revisited only if the candidate changes
the shader body or allocation lifetime enough to make recovered `ub320+`
strictly faster than `ub256`, not merely less broken.
