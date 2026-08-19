# D131: Vulkan fp8 K/V — state review and remaining work

Date: 2026-08-19

Status: open (review complete, candidates gated for the next research
branch). Sibling of D130; consolidates D095/D096/D097 + the post-audit
code state on master.

## 1. How K and V live today (Vulkan, Qwen3.6/3.8-27B)

Both caches are `GGML_TYPE_F8_E4M3`, 1.0 B/value, no block scale
(`ggml.c`: blck_size=1, is_quantized=false). Qwen GQA: 4 KV heads x 256,
so K and V are equal in size (1024 values/token/layer, 128 KiB/token
across 64 layers).

| stage | where | what happens |
|---|---|---|
| write | `copy_to_quant.comp` (`f32_to_fp8_e4m3`) | bit-twiddle E4M3 encoder, ~12 instrs (D096-L; software Log2 variant rejected) |
| decode read | scalar FA (`flash_attn.comp`), Qwen GQA remap N=6 | `dequantize4` per element; f8 forces `shmem_staging=1` (f8->LDS f16->dot) |
| prefill read | default = **preconvert** (`vk_dispatch.inc` `fa_preconvert`) | whole-KV f8->f16 pass, then pure-f16 coopmat1 FA |
| prefill read | `GGML_VK_FA_F8_DIRECT` probe | canonical cm1 with `DATA_A_F8_E4M3` loads, no preconvert |
| prefill read | `GGML_VK_FA_F8_NATIVE` opt-in | `fp8_fa_cm1.spvasm`, fp8 coopmat S stage only (P4-class; D130: rejected) |
| hybrid | `LLAMA_VK_MTP_KV_LAST_F16=N` | last N layers K+V stay f16; default N=8 (ctx<98K), N=12 (ctx>=98K) with MTP |

Mixed K/V (e.g. K=f8, V=f16) exists only on the scalar path; coopmat2
mixed types are NVIDIA-only.

## 2. Scoreboard vs q8_0 (all from fork measurements)

| metric | q8_0 | f8 | note |
|---|---|---:|---|
| bytes/value | 1.125 | 1.0 | f8 reads 11% less KV |
| attention-logit MSE | 0.00010289 | 0.00307644 | f8 is 29.9x worse (R6 scout) |
| prefill 49K (pt/s) | 1445.81 | 1445.58 | parity (preconvert default) |
| decode 49K (t/s) | 27.08 | 25.56 | f8 -5.6% |
| MTP acceptance 98K, no tail | 72.6% | 60.9% | D097 |
| MTP acceptance 98K, with tail | 90.61% (q8+M6 bridge) | 73.79% (f8+N12) | tail costs memory |
| KV mem 98K | 4680 MiB (q8+M6) | 5376 MiB (f8+N12) | **f8 ends up heavier** |

So today fp8 KV on Vulkan is a memory option that does not save memory
once its quality tail is paid for, and it loses 5.6% decode. The open
items below attack exactly those two deficits.

## 3. Candidate work items

### C1 — R9 K-only block scale (main candidate)

K stored as 256 E4M3 + 1 f16 scale per block (258/256 = 1.0078 B/value);
V stays raw f8. K-scale multiplies the score column AFTER Q*K and before
softmax (factored out of P*V), so decode pays 1 FMUL per (q,k) pair
instead of 64 per-element scale muls — cheaper AND more accurate.
Offline gate already passed (D096 R9): logit MSE -20.5%, metadata
+0.78%. Runtime does not exist yet (no `F8_K_SCALE` code anywhere).

R8 explains why this must be K-only: a V scale differs per key row inside
the Bc reduction and cannot be applied after the coopmat product; the
symmetric K/V format is closed.

Scope (from the D096 R9 design): encoder max-reduce over 256 + scale
write; two-buffer K lifecycle (data + scale satellite tensor through
SET_ROWS/copy/MTP window); FA decode `Sf *= scale[j]`; P5-class prefill
column multiply; descriptor set 7->8 across all FA paths.

Gate: MTP acceptance at least q8-hybrid level at equal-or-better memory,
decode above the current f8 25.56 baseline. Secondary gate: can the
f16 tail N shrink (target: drop the 5376-vs-4680 MiB deficit).

### C2 — clean A/B of `GGML_VK_FA_F8_DIRECT` vs preconvert — CLOSED 2026-08-19

Measured on the 49K lane (Qwen3.8-27B-Q4_K_M, f8 K/V, spec=none, dual
Vulkan, b8192/ub1024, interleaved A-B-A-B, route-verified via
`GGML_VK_FA_ROUTE_TRACE`): the direct runs really took
`path=coopmat1, f8_direct=1, preconvert=0, shmem_staging=0`.

| config | prompt t/s | decode t/s |
|---|---:|---:|
| preconvert (control x2, spread 0.09%) | 1553.95 / 1555.40 | 24.73 / 24.58 |
| f8_direct (x2) | 1186.03 / 1191.43 | 24.52 / 24.55 |

**-23.5% prefill, decode neutral.** Per-element f8->f16 conversion inside
the cm1 A-stage costs more than converting the KV once. Preconvert stays
the default; the direct route joins D130 in the closed set. Artifacts:
`d131-c2-preconv-r{1,2}`, `d131-c2-direct-r{1,2}`.

### C3 — V-only / per-layer hybrid policy (medium)

K=f8, V=f16 gives full V quality at 1.5 B/value (vs 2.0 f16, 1.0 f8).
D096 D5 route exists but is forced-scalar (slow); needs a coopmat1 mixed
path or preconvert of K only. The other end of the policy is the existing
N tail: N=2/3 mid-points never measured. R8 data says V precision is the
real cost of f8, so C3 and C1 are complements, not alternatives.

### C4 — KV-share-based route policy (medium)

At 49K KV is ~27% of decode bytes, at 98K ~43% (D105 budget). The
f8-decode -5.6% vs q8 at 49K may invert at 98K; no clean 98K spec=none
decode pair exists post-audit. If it inverts, the lane-specific default
(policy by ctx) becomes worth building.

## 4. Fences (do not reopen without new evidence)

- power-of-two block scale (R6): grid-phase invariant, MSE unchanged.
- BFP8 int8+exp (R7): precision passes but int8 accumulators are closed.
- symmetric K/V general-scale E4M3 (R8): V scale not factorable from P*V.
- native fp8 coopmat prefill (D130): preconvert wins, -7.6% at 49K.
- direct f8 cm1 prefill (C2 above): preconvert wins, -23.5% at 49K.
- cooperative decode (D4.2): 13.28 vs 25.45 t/s scalar; needs KV-batched
  fragment design, not query-batched.
- direct scalar f8 decode without staging (D095 R1): wall-time tie.
- MMQ / integer dot: `V_DOT2_F32_F16` is CDNA; no int-dot on RDNA.
- coopmat shapes: 16x16x16 only.

## 5. Next branch shape

`research/vulkan-fp8-kv`:
1. [x] C2 probe — CLOSED, preconvert wins by 23.5%.
2. C1 R9 implementation (the real work) with the D096 R9 design as spec,
   acceptance + memory + decode gates against q8-hybrid.
3. C4 clean 98K decode pair to decide lane policy.
4. C3 only if R9 shows V is the remaining precision cost.

All harness lanes use the canonical b8192/ub1024 batch/ubatch sizes
(2026-08-19 decision); the D129/D104-R3 runs at b512/ub128 are exceptions,
not the template.

## 6. References

- `D095_Q4KM_VULKAN_FP8_MTP_POLISH.md` (R1-R9 verdicts, scout method)
- `D096_ROADMAP.md` (D4.3 R9 design; D1-D4 history)
- `D097_Q4KM_VULKAN_FP8_LONG_ACCEPTANCE.md` (98K acceptance/policy)
- `D130_VULKAN_FP8_NATIVE_PREFILL_REAUDIT.md` (native prefill rejection)
- `D105_VULKAN_DECODE_BANDWIDTH_CEILING.md` (decode byte budget)
