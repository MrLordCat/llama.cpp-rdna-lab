# D077 — Coopmat2 on RDNA4: Software Emulation Feasibility

Date: 2026-05-30  
Owner: Copilot/perf workspace  
Status: analytical gate

## Question

Can we implement our own adaptation of Vulkan coopmat2 (workgroup-scope cooperative
matrix) on RDNA4 using existing GLSL primitives, to unlock wider FA tiles despite
the 64KB shmem limit?

## What coopmat2 provides (NVIDIA-only)

| Feature | coopmat1 (KHR, available on RDNA4) | coopmat2 (NV, NOT on RDNA4) |
|---|---|---|
| Scope | Subgroup | Workgroup |
| Max tile | 16×16×16 (fixed) | 32×32×16 (flexible, up to 512) |
| Block loads | No | Yes (single instruction) |
| Reductions | No | Yes (sum/max across subgroups) |
| Tensor addressing | Manual | Built-in |
| Element-wise ops | No | Yes (fused with matmul) |

## Can we emulate workgroup-scope on RDNA4?

**Technically: yes.** We can build a workgroup-scope tiled matmul using existing
GLSL primitives:

```glsl
// Conceptual: workgroup-scope 32x32x16 matmul on RDNA4
// Strategy: 4 subgroups, each owns a 16x16 sub-tile of the 32x32 output

// Step 1: Load 32x16 A into shmem (all subgroups cooperate)
// Step 2: Load 16x32 B into shmem (all subgroups cooperate)
// Step 3: Each subgroup does its local coopmat 16x16x16
// Step 4: No cross-subgroup communication needed
//         (each subgroup writes to distinct output region)
```

This is essentially what `mul_mm.comp` already does for Q3_K matmuls — the
workgroup cooperatively loads data into shmem, then each subgroup computes its
partition independently.

## Would this help FlashAttention on RDNA4?

**No, for three reasons.**

### 1. FA subgroup decomposition is already perfectly parallel

In the current FA coopmat1 shader, the two core operations are:

```
S = Q @ K^T  [Br=16, Bc=64]
  → 4 subgroups, each computes [16, 16] sub-tile
  → No cross-subgroup communication needed
  → Coopmat2 would NOT change this

O += P @ V  [Br=16, HSV=256]
  → 4 subgroups, each handles HSV/4 columns
  → No cross-subgroup communication needed
  → Coopmat2 would NOT change this
```

Making the matmul "workgroup-scope" adds zero benefit when subgroups already work
on independent output regions. The parallelism is embarrassingly parallel —
coopmat1 already exploits this perfectly.

### 2. The FA bottleneck is memory bandwidth, not compute

From the D076 perf trace:
```
FA: 16 TFLOPS (25% of GPU peak)
Q3_K matmul: 63 TFLOPS (near GPU peak)
```

FA achieves only 16 TFLOPS because the compute units are idle waiting for K/V
data from VRAM. Each FA kernel iteration:
- Loads 64 K values (256 bytes after q4_0→f16 dequant) per Bc tile
- Loads 64 V values (256 bytes)
- Computes ~2K FLOPs per value
- Arithmetic intensity: ~4 FLOP/byte → memory-bound

Widening the matmul instruction (coopmat2) addresses the computation side, which
is **already not the bottleneck**.

### 3. Wider tiles don't fit in shmem anyway

| Bc | K/V shmem (kvsh) | Total shmem | Fits 64KB? |
|---|---|---|---|
| 64 (current) | 33,792 B | ~47,300 B | ✅ |
| 128 | 67,584 B | ~81,100 B | ❌ |
| 96 | 50,688 B | ~64,200 B | ❌ (tight) |

Even if coopmat2 were magically available, Bc=128 still wouldn't fit in 64KB
shared memory. The bottleneck is the K/V tile, not the matmul instruction width.

## Where coopmat2 WOULD help

Coopmat2 would be transformative for **Q3_K matmuls**, not FA:

```
Current: 16×16×16 coopmat1 × 4 subgroups = 64×64 output tile
With coopmat2: 32×32×16 × workgroup = 128×128 output tile
→ 4× fewer K-loop iterations for same output
→ 4× fewer global memory reads of Q3_K weights
→ Potentially 1.5-2× speedup for Q3_K matmul
```

But Q3_K is already near compute peak (63 TFLOPS), so even a 2× matmul speedup
would give only ~1.14× wall improvement (28% share × 2× local = 1.14× wall).

## What CAN we do instead?

The KV loop in FA does ~1047 iterations of:
1. Load K tile from VRAM → dequant q4_0 → f16
2. S = Q @ K^T (coopmat, 256/16 = 16 sub-iterations)
3. Softmax + rescale
4. Load V tile from VRAM → dequant q4_0 → f16
5. O += P @ V (coopmat, 256/64 = 4 sub-iterations)

Three research directions that attack the MEMORY problem, not the compute:

### Direction A: K-cache compression (mathematical)
Store K in a lower-precision or factored format. Load compressed K (e.g., 4-bit
instead of 8-bit after q4_0), dequant on-the-fly with a cheaper decoder. The
q4_0 dequant is: `d × (nibble − 8)`. If we drop to 3-bit or 2-bit with a custom
codebook, we reduce K bandwidth proportionally.

### Direction B: Attention sparsity (algorithmic)
Not all 1047 K/V blocks are equally important for each Q block. Use an
approximate top-K selection:
- Pass 1: Q @ K_reduced^T with K compressed to 64/256 dimensions
- Pass 2: Full attention on top-256 blocks
- Expected speedup: 2-4× for FA (75% K/V skipped)

### Direction C: DeltaNet-like reformulation (architectural)
Qwen3.6 has 16 FA layers and 48 DeltaNet layers. The DeltaNet layers handle
long-range dependencies via SSM (O(n) complexity vs O(n²) for attention). If the
FA layers could be replaced or augmented with a similar mechanism, the FA
bottleneck disappears entirely. This is model-level, not kernel-level.

## Verdict

| Question | Answer |
|---|---|
| Can we implement coopmat2 on RDNA4? | Technically yes (software emulation), but it doesn't solve the FA bottleneck |
| Does coopmat2 help FA? | No — FA is memory-bound, not compute-bound; wider matmul doesn't help memory |
| What's the real path forward? | Algorithmic changes: sparse FA, K compression, or architectural modifications |

## Next Steps

Immediate: open a scout script to measure attention-score sparsity on Qwen3.6
layers (how many K/V blocks contribute >90% of attention mass). This determines
whether sparse FA is viable. Use `scripts/research/attention_sparsity_scout.py`.

## References

- D076: FA Bc scaling rejection (shmem limit)
- D032: Q3+FA stack gate (FA can't carry 2.4 TPS target)
- E138: forced split-K rejection (sync overhead)
- Qwen3.6 architecture: 64 layers, full_attention_interval=4, HSK=HSV=256
