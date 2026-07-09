# E272 Vulkan Dual Layer 130k Baseline

## Metadata

- Experiment ID: E272
- Date: 2026-07-09
- Owner: Codex
- Target lane: Vulkan single-vs-dual baseline, no MTP
- Driver: AMD display `32.0.23033.1002` (`2026-03-09`)

## Setup

Model and lane:

- `models/Qwen3.6-27B-Q3_K_S.gguf`
- `ctx=131072`
- `real-context-mode repo-snapshot`
- `real-context-chars=152000`
- observed prompt tokens: `56371`
- `max_tokens=64`
- `batch=512`, `ubatch=256`
- KV: `q4_0/q4_0`
- FlashAttention on
- `--spec-type none`
- `--no-mmap`
- cold-first: `--no-reuse --no-v2-prime-pass`
- thinking enabled

Vulkan route env:

- `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`
- `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn256`
- `GGML_VK_QK_LOW_TILE_SPLIT_K=3`
- `GGML_VK_Q3K_QUAD_DEQUANT=1`

Single GPU1 was isolated via `GGML_VK_VISIBLE_DEVICES=1`; inside the process it
appears as `Vulkan0`.

## Results

| Variant | Label | Wall s | Aggregate TPS | Prompt tok/s | Decode tok/s | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Single physical GPU0 | `vulkan-single-gpu0-130k-big-c152k-mt64-none-r1` | `95.9958` | `0.6667` | `639.61` | `8.29` | completed |
| Single physical GPU1 | `vulkan-single-gpu1-130k-big-c152k-mt64-none-r1` | `122.5133` | `0.5224` | `506.70` | `5.74` | completed |
| Dual layer | `vulkan-dual-layer-130k-big-c152k-mt64-none-r2` | `115.4604` | `0.5543` | `504.91` | `17.45` | completed |

## Memory / Residency

Device topology notes:

- Vulkan/HIP device 0 maps to Windows `PCI bus 11`.
- Vulkan/HIP device 1 maps to Windows `PCI bus 6`.
- HIP reports `hostNativeAtomicSupported=1` for device 0 and
  `hostNativeAtomicSupported=0` for device 1.
- The local motherboard, Gigabyte B550 Gaming X V2, is not symmetric for two
  graphics cards: the primary CPU slot is PCIe 4.0 x16, while the second
  x16-sized chipset slot (`PCIEX2`) is PCIe 3.0 x2 according to the official
  Gigabyte specification.

Single GPU0 startup:

- `Vulkan0 model buffer`: `11434.19 MiB`
- `Vulkan_Host model buffer`: `521.00 MiB`
- `Vulkan0 KV buffer`: `1872.00 MiB`
- `Vulkan_Host_Direct KV buffer`: `432.00 MiB`
- `Vulkan0 compute buffer`: `228.27 MiB`
- `Vulkan_Host compute buffer`: `138.27 MiB`
- graph splits: `2`, scheduler copies: `1`

Single GPU0 final memory breakdown:

- `Vulkan0`: `13462 MiB` self (`11434` model + `2021` context + `7` compute)
- `Host`: `953 MiB` (`520` model + `432` context + `0` compute)

Dual layer startup:

- `Vulkan0 model buffer`: `5380.54 MiB`
- `Vulkan1 model buffer`: `6053.65 MiB`
- `Vulkan_Host model buffer`: `521.00 MiB`
- `Vulkan0 KV buffer`: `1152.00 MiB`
- `Vulkan1 KV buffer`: `720.00 MiB`
- `Vulkan_Host_Direct KV buffer`: `432.00 MiB`
- `Vulkan0 compute buffer`: `912.08 MiB`
- `Vulkan1 compute buffer`: `626.91 MiB`
- `Vulkan_Host compute buffer`: `523.10 MiB`
- graph splits: `9`, scheduler copies: `4`

Dual layer final memory breakdown:

- `Vulkan0`: `6617 MiB` self (`5380` model + `1229` context + `7` compute)
- `Vulkan1`: `6855 MiB` self (`6053` model + `791` context + `9` compute)
- `Host`: `953 MiB` (`520` model + `432` context + `0` compute)

The observed Windows shared-memory value around `~1.1 GiB` matches the expected
host-resident model/KV guard scale. This is not a large uncontrolled system-RAM
spill. The explicit host-resident pieces are about `953 MiB` final, with a
higher temporary reserve-time host compute allocation on dual.

## Decision

Dual Vulkan is stable on the rollback driver, but `-sm layer -ts 1/1` is not a
speed baseline improvement versus the best single GPU. It improves decode
(`8.29 -> 17.45 tok/s`) but loses prompt eval (`639.61 -> 504.91 tok/s`), and
the 56k-token prompt dominates wall time. Net result versus best single GPU0:

- aggregate TPS: `0.6667 -> 0.5543` (`-16.9%`)
- prompt eval: `639.61 -> 504.91 tok/s` (`-21.1%`)
- decode eval: `8.29 -> 17.45 tok/s` (`+110.5%`)

Use physical GPU0 single as the no-MTP Vulkan 130k big-prompt baseline for this
driver. Dual layer is useful as a decode/residency diagnostic, but not as the
default for prompt-heavy agent work until the split/scheduler overhead is
reduced or a different split placement improves prefill.
