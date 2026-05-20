# E093 Vulkan Warptile Static Scout

## Metadata

- Experiment ID: E093
- Date: 2026-05-20
- Owner: Copilot
- Type: workflow/tooling + H31 correction
- Target lane: H31 Vulkan Q3_K prefill research

## Problem

E091 found a small positive `wn48` result, but it was still a no-code env variant and the gain was not large enough to trust without proving that the requested warptile was actually valid and active. The next step needed a no-build validator for warptile geometry before more tile experiments.

## Tool

Added `scripts/research/vulkan_warptile_static_scout.py`.

It mirrors `ggml_vk_matmul_prepare_variant_warptile()` layout checks and reports static proxies:

- valid/invalid layout;
- runtime-effective route (`variant` or `base-fallback`);
- whether the effective prepared variant is identical to base;
- prepared block size;
- `BMxBN`, `WMxWN`;
- `BK`, representative K-block count, and barrier-round proxy;
- Q3 stride18 shared-memory footprint;
- workgroup counts for representative Q3_K prefill shapes;
- full-K B reload proxy and A pair-dequant proxy;
- measured notes from E075/E085/E091.

Smoke artifact:

- `build_logs/agent-workload/e093-vulkan-warptile-static-scout.md`

## Finding

The static scout marks `wn48` and `wn96` invalid for the current `BN=128` layout because `128 % 48 != 0` and `128 % 96 != 0`. It now models those as `runtime_effective=base-fallback`, matching the backend restore-to-base path in `ggml_vk_matmul_prepare_variant_warptile()`. That means E091's `wn48` measurement must not be promoted as an opt-in profile unless a separate backend log/proof shows a different active `BN` or a valid prepared tile.

The updated scout also marks `block128` and `wn64` as `same_as_base=yes` after backend preparation. Treat those as measurement-noise checks, not new candidates, unless a backend log proves a hidden route difference.

The scout now includes static-only BK probes. `BK=64` halves K-block/barrier rounds (`160 -> 80`) but keeps full-K B reload and A pair-dequant proxy unchanged while increasing Q3 shared memory from `18432 B` to `34816 B`. `BK=16` lowers shared memory but doubles K-block/barrier rounds. BK-depth work therefore needs pipeline/resource proof before any build.

Accepted H31 baseline should therefore remain E086 source-only:

- fixed pp7488: `961.82 tok/s`
- workload r1: `6.6277`, prompt eval `934.8 tok/s`

E091 remains a useful warning that tile env measurements can look positive while static layout validation is suspect.

## Decision

Keep the static scout and downgrade E091 `wn48` from opt-in profile to `needs-layout-validation`. Future tile work must pass static scout before any benchmark claim.