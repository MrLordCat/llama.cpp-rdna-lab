# E273 Dual Backend 130k Big-Prompt Delta

## Metadata

- Experiment ID: E273
- Date: 2026-07-09
- Owner: Codex
- Target lane: ROCm dual vs Vulkan dual, no MTP
- Driver: AMD display `32.0.23033.1002` (`2026-03-09`)

## Setup

Common lane:

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- `ctx=131072`
- `real-context-mode repo-snapshot`
- `real-context-chars=152000`
- observed prompt tokens: `56371`
- `max_tokens=64`
- KV: `q4_0/q4_0`
- FlashAttention on
- `--spec-type none`
- cold-first: `--no-reuse --no-v2-prime-pass`
- thinking enabled

Backend-specific shape:

- Vulkan dual: `b512/ub256`, `-dev Vulkan0,Vulkan1 -sm layer -ts 1/1 --no-mmap`
- ROCm dual: `b512/ub128`, `-dev ROCm1,ROCm0 -sm layer -ts 1,1`

## Results

| Backend | Label | Wall s | Aggregate TPS | Prompt tok/s | Decode tok/s | Prompt ms | Decode ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ROCm dual | `rocm-dual-layer-130k-big-c152k-mt64-none-r1` | `156.5195` | `0.4089` | `370.05` | `15.76` | `152331.58` | `4060.68` |
| Vulkan dual | `vulkan-dual-layer-130k-big-c152k-mt64-none-r2` | `115.4604` | `0.5543` | `504.91` | `17.45` | `111645.84` | `3667.08` |

Vulkan dual vs ROCm dual on the same prompt scale:

- aggregate TPS: `+35.6%`
- prompt eval: `+36.4%`
- decode eval: `+10.7%`
- wall time: `-26.2%`

## Residency

ROCm dual final memory breakdown:

- `ROCm1`: `6617 MiB` self (`5380` model + `1229` context + `7` compute)
- `ROCm0`: `7274 MiB` self (`6044` model + `1223` context + `6` compute)
- `Host`: `521 MiB`
- graph splits: `3`
- scheduler copies: `4`

Vulkan dual final memory breakdown:

- `Vulkan0`: `6617 MiB` self (`5380` model + `1229` context + `7` compute)
- `Vulkan1`: `6855 MiB` self (`6053` model + `791` context + `9` compute)
- `Host`: `953 MiB`
- graph splits: `9`
- scheduler copies: `4`

## Decision

For dual-GPU no-MTP `ctx=131072` big-prompt work, Vulkan is currently faster
than ROCm on this machine despite the chipset-limited second GPU. The advantage
is mostly prompt-side (`504.91` vs `370.05 tok/s`), while decode is closer
(`17.45` vs `15.76 tok/s`).

This does not overturn the earlier ROCm short decode/MTP work. It means the
current practical no-MTP long-prompt dual baseline favors Vulkan, while ROCm
remains the active MTP backend because the MTP path is developed and measured
there.
