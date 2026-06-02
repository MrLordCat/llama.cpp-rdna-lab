# CPU Performance Research — P004

Date: 2026-05-30  
Owner: Copilot/perf workspace  
Branch: `research/cpu-perf`

## Hardware Context

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 5800X3D, 8 cores / 16 threads, 3.4 GHz base |
| L3 Cache | 96 MB 3D V-Cache (key differentiator) |
| RAM | 64 GB DDR4 |
| OS | Windows 11 Pro build 26200 |

## Objective

Explore and implement optimisations for llama.cpp CPU backend on the 5800X3D,
leveraging the unique 96MB 3D V-Cache. The large L3 can fit substantial model
working sets that would otherwise go to RAM, potentially reducing the memory
wall for matrix-vector and attention operations.

## Constraints

- No quality regression — output must remain identical
- CPU backend only (no GPU offload)
- Focus on decode speed (matvec-dominated) for interactive use
- Small benchmark (`--real-context-chars 24576`, `max_tokens=16`) for fast iteration

## Folder Structure

```
docs/research/cpu-perf/
├── README.md           # This file (program overview)
├── P004_PLAN.md        # Detailed research plan
├── D078_*.md           # Design notes
├── experiments/        # Narrow measured ledger entries
└── scouts/             # Analysis scripts and tooling
```

## Active Lanes

| Lane | Model | ctx | batch/ubatch | threads | Purpose |
|---|---|---|---|---|---|
| Quick smoke | Qwen3.6-27B-Q3_K_S | 4096 | 512/512 | 16 | Fast iteration, route smoke |
| Practical decode | Qwen3.6-27B-Q3_K_S | 4096 | 512/512 | 8 | Realistic single-user decode |

## Research Directions

1. **Thread affinity + 3D V-Cache aware scheduling**: pin compute threads to
   cores sharing the same CCX to maximise L3 hit rate
2. **matvec kernel improvements**: Q3_K/Q4_K matvec dominates decode; study
   AVX2 codegen, register blocking, prefetch
3. **Multi-threaded matvec**: current matvec may be single-threaded; explore
   parallel reduction strategies
4. **KV cache in L3**: at small ctx, KV cache fits in 96MB L3; ensure no
   accidental eviction
5. **Memory layout / repacking**: Q3_K block order may cause cache line splits
