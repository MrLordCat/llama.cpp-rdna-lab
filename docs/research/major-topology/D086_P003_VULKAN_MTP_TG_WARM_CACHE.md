# D086: Vulkan MTP TG warm-cache gate

Date: 2026-07-13

## Problem

Vulkan MTP had normal draft acceptance but decode did not beat `--spec-type none` in the GUI long-prompt lane. The failure was strongest for the autotune default of 16 generated tokens.

The old 48k trace measured:

- first 4-row target verify: about 2.45 s on a cold shape;
- later 4-row verifies: about 45-70 ms;
- a late first-use 3-row verify: about 2.35 s;
- MTP decode: 14-18 tok/s while the paired baseline was 26-28 tok/s.

Acceptance was not the blocker. Representative `n_max=3` runs accepted 58-77% of drafts.

## Root cause

There were two independent Vulkan cold-path costs.

1. Vulkan MMVQ pipelines are specialized by row count. First use of verify widths 1 through `n_max + 1` compiled and initialized separate pipelines during generation.
2. Windowed NextN prompt processing correctly used a cheap non-NextN PP graph for most of the prompt, but toggling NextN rebuilt both PP and TG schedulers. Even after retaining the warmed TG scheduler, the first verify changed `sched_reserve_pp_outputs` from 1 to all rows. That PP-only setting forced another full reserve and discarded the restored TG graph immediately before use.

There was also a separate steady-state `n_max=4` problem. On Windows/AMD, a safety guard routes F32 x F32 batched mat-vec with five or more columns away from the fast shader because AMDVLK crashes while compiling that specialization. The generic fallback made every 5-row verify take 125-140 ms. Enabling the unsafe shader reproduced an access violation before server readiness, so removing the guard is not acceptable.

The decisive debug sequence was:

```text
retained warmed NextN TG scheduler
restored warmed NextN TG scheduler
sched_reserve: reserve took 291.54 ms
spec phase=target rows=4 decode_return=478.000 ms
```

The verify batch was routed to TG, so rebuilding PP for its output width was unnecessary.

## Accepted implementation

- At server startup, Vulkan MTP evaluates target verify widths `1..n_max+1` and clears model memory/perf counters afterward. This moves row-specialized pipeline creation outside request timing.
- The target context can retain its warmed NextN TG scheduler while the main prompt uses the non-NextN PP topology, then restore it for the final prompt window and generation.
- A batch routed to the TG scheduler no longer changes `sched_reserve_pp_outputs`; TG output width cannot invalidate the PP reservation.
- The Windows/AMD F32 safety path now splits aligned, unfused widths 5 through 8 into safe batched mat-vec dispatches of `4 + remainder`. Unsupported or misaligned cases retain the generic fallback. This keeps the driver-crash guard while avoiding its steady-state performance penalty.
- `LLAMA_VK_MTP_VERIFY_WARMUP=0` remains an opt-out and also disables the server-side TG retention path.

The target PP path remains non-NextN for most of the prompt. This avoids the severe prompt regression of keeping the unmasked NextN graph active for every ubatch.

## Rejected approach

Keeping the unmasked NextN graph topology active throughout prompt processing made each 1024-token PP ubatch spend about 429-486 ms in graph allocation. Prompt throughput fell from about 1715 tok/s to 1011-1074 tok/s. Do not retry that design without an independent PP allocator cache or a masked topology proven by trace.

## Validation

Hardware load note: League of Legends was active, so all conclusions use adjacent paired runs.

48k context, `b=8192`, `ub=1024`, Q8 KV, dual Vulkan layer split `5,6`, `n_max=3`:

| Output | Mode | Prompt tok/s | Decode tok/s | Relative decode |
| ---: | --- | ---: | ---: | ---: |
| 16 | none | 1196.87 | 23.12 | 1.00x |
| 16 | MTP | 1239.62 | 32.42 | 1.40x |
| 128 | none | 1231.94 | 24.68 | 1.00x |
| 128 | MTP | 1185.33 | 35.01 | 1.42x |
| 64 | none | 1457.36 | 33.70 | 1.00x |
| 64 | MTP, `n_max=4` | 1408.49 | 43.52 | 1.29x |

The 128-token MTP run accepted 81 of 137 drafts (59.12%). Its final first-use 3-row verify took 48.9 ms instead of the previous 2.35 s.

Before the Windows/AMD split, the same 48k/64 `n_max=4` lane produced 16.63 tok/s. After the split it produced 43.52 tok/s, with steady 5-row target verifies at 49-53 ms. The deterministic response preview was identical before and after the routing change.

The residual first verify is still slower than steady state (roughly 157 ms versus 54-80 ms at 48k), but it no longer erases speculative gain even in the 16-token GUI lane.

The GUI-selected `build-vulkan` directory was regenerated after the repository moved from `C:` to `D:` and rebuilt with Vulkan plus Strawberry OpenSSL 3.6.1. Its final 12k `n_max=4` smoke produced 1560.31 prompt tok/s and 39.81 decode tok/s.

## Artifacts

- `p003-vulkan48k-lol-mtp-n3-tgcache2-max16-r1.server.log`
- `p003-vulkan48k-lol-none-tgcache2-max16-control-r1.server.log`
- `p003-vulkan48k-lol-mtp-n3-tgcache2-max128-r1.server.log`
- `p003-vulkan48k-lol-none-tgcache2-max128-control-r1.server.log`
- `p003-vulkan12k-lol-mtp-n3-tgcache2-debug-r1.server.log`
- `p003-vulkan48k-lol-mtp-n4-tgcache2-max64-r1.server.log`
- `p003-vulkan48k-lol-mtp-n4-split41-max64-r1.server.log`
- `p003-vulkan48k-lol-none-split41-max64-control-r1.server.log`
- `p003-gui-build-vulkan12k-mtp-n4-final-smoke-r1.server.log`
