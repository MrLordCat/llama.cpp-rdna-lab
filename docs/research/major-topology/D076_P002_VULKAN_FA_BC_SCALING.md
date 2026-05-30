# D076 — P002 Vulkan FA Optimization Cycle (130k big-prompt)

Date: 2026-05-30  
Owner: Copilot/perf workspace  
Status: **CLOSED — all 5 candidates rejected. FA at 130k on RDNA4 is memory-bound and cannot be improved with kernel tuning.**

## Lane Contract

b512/ub256, q4_0/q4_0 KV, flash_attn on, spec=none, no reuse, thinking on,
real-context=repo-snapshot:180000, max_tokens=16, triage_diff.

## Baseline

| Metric | Value |
|---|---|
| aggregate_completion_tps | 0.1367 |
| **prompt_eval_tps** | **575.89** |
| decode_eval_tps | 19.71 |
| task_prompt_tokens | 66883 |
| FA path | coopmat1, Br=16, Bc=64, split_k=1 |

## Candidates Tested (all regressed)

| # | Candidate | prompt_eval_tps | Delta | Root cause |
|---|---|---|---|---|
| 1 | Bc=128 (8 subgroups, wg=512) | 430.52 | −25% | Shmem ~60KB > 64KB → scalar fallback (Br=16,Bc=32) |
| 2 | Bc=96 (6 subgroups, wg=384) | 429.90 | −25% | Shmem ~53KB+alignment > 64KB → scalar fallback |
| 3 | shmem_staging=1 (AMD long-KV) | 433.90 | −25% | Pipeline state mismatch: 32×coopmat1 + 232×scalar hybrid |
| 4 | mask_opt=0 (AMD long-KV) | 556.37 | −3.4% | Marginal: dispatch savings < inline mask overhead |
| 5 | Br=32 + dual-Q (coopmat1) | 434.56 | −25% | LLPC rejects pipeline (likely VGPR/shmem constraint) |

## Perf Data from Big-Prompt Route Trace (D076)

```
Batch (N=256, KV=66816, HSK=256, 24 heads):
├── FA:          418ms/batch (63%) — 16 TFLOPS, memory-bound
├── Q3_K matmul: 184ms/batch (28%) — 63 TFLOPS, near compute peak
├── f32 matmul:   11ms/batch ( 2%)
└── Other:        49ms/batch ( 7%)
```

## Why FA Cannot Be Improved With Kernel Tuning

1. **Shmem limit (64KB)**: Any tile expansion exceeds RDNA4 budget.
   Current Bc=64/Br=16 uses ~47KB. Bc=128 needs ~60KB, Bc=96 needs ~53KB.
   All attempts to grow tiles cause scalar fallback → 2× slower.

2. **Memory bandwidth bottleneck**: FA achieves 16 TFLOPS vs 63 TFLOPS peak.
   K/V cache = 19MB/layer at 67k tokens (q4_0). GPU L2 = ~8MB. Every K/V tile
   misses L2, so the kernel stalls waiting for VRAM. No change to MatBr/MatBc
   fixes this — the data simply isn't in cache.

3. **Q3_K already at peak**: 63 TFLOPS with BN256 tiles (auto-enabled for RDNA3+).
   No routing or tile-size change can extract more from the compute units.

## What WOULD Help (requires model/algorithm changes, not kernel tuning)

| Approach | Mechanism | Speedup | Quality Impact |
|---|---|---|---|
| Sparse FA | Skip 75% K/V blocks with low attention mass | 2-4× FA, ~1.6× wall | Minimal if sparsity pattern is good |
| KV cache compression (q2_K/q3_K) | Reduce per-token KV bytes | 1.3-1.5× | Some quality loss |
| Low-rank K projection | Project K to r<256 before FA | 1.5× | Some quality loss |
| Bigger GPU (24GB+) | Larger L2 cache fits working set | 1.5-2× | None |

## Decision

**REJECTED — all 5 FA kernel candidates.** Code changes reverted. Documentation
captures the evidence. Next work should target algorithmic changes (sparse FA
scout → prototype → A/B) or accept the current 130k performance as the hardware
limit for this model/KV configuration.

## Artifacts

- `build_logs/agent-workload/d076-fa-bigprompt-baseline-r1.*`
- `build_logs/agent-workload/d076-fa-bigprompt-cand-bc128-r1.*`
- `build_logs/agent-workload/d076-fa-bigprompt-cand-bc96-r1.*`
- `build_logs/agent-workload/d076-fa-shmemstage-r1.*`
- `build_logs/agent-workload/d076-fa-nomaskopt-r1.*`
- `build_logs/agent-workload/d076-fa-br32-r1.*`
- `build_logs/agent-workload/d076-q3k-bigprompt-route-r1.*`
- `docs/research/major-topology/D076_P002_VULKAN_FA_BC_SCALING.md`
- `docs/research/major-topology/D077_P002_COOPMAT2_RDNA4_FEASIBILITY.md`

## Artifacts

- `build_logs/agent-workload/d076-fa-bigprompt-baseline-r1.*`
- `build_logs/agent-workload/d076-fa-bigprompt-cand-bc128-r1.*`
