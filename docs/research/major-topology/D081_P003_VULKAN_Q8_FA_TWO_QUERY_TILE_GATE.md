# D081 P003 Vulkan Q8 FA Two-Query-Tile Gate

## Status

- Design/compile gate only; runtime source prototype is not yet authorized.
- Target route: q8/q8 coopmat1 FlashAttention at `N=1024`, `KV=1k..56k`, head size 256.
- Baseline: D080 `1350.01 prompt tok/s` cold-first.

## Evidence

- D079 parsed FlashAttention share: 46.4%.
- Active route: `Br=16`, `Bc=64`, `D_split=8`, `row_split=4`, `split_k=1`.
- q8 FA pipeline: 98 VGPR, 76 SGPR, 26,112 B LDS, zero scratch.
- The shader stages each q8 K and V tile once per 16 query rows. At long KV, exact attention is dominated by repeatedly reading the same K/V for adjacent query tiles.

## Candidate Mechanism

Process two independent 16-row query tiles in one workgroup while sharing each K/V tile:

- logical output tile: 32 query rows;
- retain 16x16 cooperative-matrix operations;
- map subgroups across query-tile and key/value-tile dimensions;
- keep four live rows per thread instead of the rejected generic Br32 path's eight;
- avoid split-K and any extra global reduction dispatch.

The intended effect is up to 2x lower K/V traffic per query token while keeping exact attention semantics.

## Static Resource Gate

Estimated shared memory:

- direct `row_split=8` layout: about 50-51 KiB;
- compact V-staging layout with four effective V groups: about 42-43 KiB;
- device limit: 64 KiB.

Risks:

- 512-thread workgroup plus about 100 VGPR can reduce occupancy to one workgroup;
- naive Br32 previously reached 133 VGPR and regressed;
- query-group offsets must be applied consistently to Q, score, P, PV, and output stores;
- extra barriers can erase the KV-read reduction.

Compile gate:

1. Variant must remain below 120 VGPR, below 56 KiB LDS, and use zero scratch.
2. SPIR-V must retain cooperative-matrix KQ/PV operations.
3. No scalar FA fallback is accepted.
4. Variant is enabled only by an env knob and only for q8/q8, HSK/HSV=256, N>=32.

Runtime gate:

1. Deterministic output comparison against default on a small prompt.
2. Point/short FA timing must improve at least 1.3x before a long benchmark.
3. Promote to the 56k lane only if local FA improvement projects at least one third of the remaining target gap.

## Stack Ceiling

FA alone cannot reach 2000:

- 1.5x local FA projects roughly 1.18x total, about 1595 tok/s.
- 1.8x local FA projects roughly 1.26x total, about 1700 tok/s.
- 2.0x local FA projects roughly 1.30x total, about 1755 tok/s.

At 1.8x FA, the remaining Q3_K center still needs about 1.34x local speedup. D081 is therefore the first half of a required FA+Q3 stack, not a standalone target claim.

## Rejection Conditions

- Compile resource gate fails.
- Any output mismatch beyond expected floating-point tolerance.
- Shader falls back to scalar or adds split/reduce dispatches.
- Short local FA gain below 1.3x.
- Runtime or WDDM residency instability.
