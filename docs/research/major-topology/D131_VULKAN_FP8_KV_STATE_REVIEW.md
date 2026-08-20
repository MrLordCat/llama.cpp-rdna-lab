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

### C1 — R9 K-only block scale (main candidate) — IMPLEMENTED, validating

K stored as 256 E4M3 + 1 f16 scale per block (258/256 = 1.0078 B/value);
V stays raw f8. K-scale multiplies the score column AFTER Q*K and before
softmax (factored out of P*V), so decode pays 1 FMUL per (q,k) pair
instead of 64 per-element scale muls — cheaper AND more accurate.
Offline gate passed (D096 R9): logit MSE -20.5%, metadata +0.78%.

Implementation (branch `research/vulkan-fp8-kv`, opt-in
`LLAMA_VK_F8_K_SCALE`):

- encoder: `ggml_pool_2d(MAX, 256)` + repeat-broadcast + `ggml_div`
  before the existing f32->f8 set_rows; the same pool result is stored
  into a per-layer f16 satellite cache via a second set_rows
  (`cpy_k` / `cpy_k_scale` in llama-kv-cache.cpp);
- storage: `cache_k_scale_l%d` 3d [n_blk, kv_size, n_stream] f16,
  windowed views, stream copy and state save/load mirrored;
- graph: `ggml_flash_attn_ext_set_k_scale()` attaches the windowed scale
  view to the FA node (new src[5], upstream f16_extra_data pattern);
- Vulkan FA decode: binding 7 + `K_SCALE_BIT` in `mask_n_head_log2`
  (push constants were full at 128 B); the scalar kernel multiplies the
  finished score column, scale index `kv_col * 4 + ik2` (block == head
  for Qwen; `ik2 = iq2/rk2` — NOT `iq2/gqa_ratio`, which is wrong on the
  non-GQA prefill);
- Vulkan prefill preconvert: f8 dequant shader gains a 3rd binding and
  folds the scale into the f16 K copy (`p.stride_a` = has-scale flag);
  the scheduler materializes the permuted K as a CONTIGUOUS tensor
  `[d=256][token][kv-head]` (flat `i = d + tok*256 + hk*256*n_kv`), so
  the scale index is `tok=(i/256)%p.M, hk=i/(256*p.M)` (the earlier view
  `[d][kv-head][token]` assumption was wrong and cost ~+4.7% ppl);
  `GGML_VK_FA_F8_DIRECT` is disabled while scales are present;
- all FA pipelines now use a uniform 8 descriptors (the scalar shader's
  extra binding 7); split_k needs no reduce change because the scale is
  applied inside the main kernel before the partial sums.

First smoke (12K, f8 K/V, spec=none, dual Vulkan) after all four
index/layout fixes: decode 27.87 t/s vs 27.78 control, coherent output,
prompt 1630.95 vs 1638.39.

Numeric confirmation (wikitext-2 perplexity, dual Vulkan, b512/ub512,
`-ctk f8_e4m3 -ctv f8_e4m3`): R9 vs plain-f8 control on the same binary.

| chunks | control (plain f8) | R9 (scale) |
|---|---:|---:|
| 16 | 45.93 ± 0.77 | 44.60 ± 0.74 |
| 72 (full) | 48.833 ± 0.393 | 48.440 ± 0.389 |

R9 is never worse than plain f8; the block scale removes underflow /
overflow and is mildly better (up to ~2.9% on the 16-chunk head). This
closes the perplexity half of the numeric gate.

Greedy decode logits (first token of a 14-token prompt, greedy, top-8
logprobs via llama-server): R9's top-4 set is `{young, little, group,
very}`, the same set as the f16-KV reference (`{group, young, little,
very}`), while plain f8 diverges to `{young, princess, girl, king}`. So
the K scale keeps the decode distribution closer to f16 than raw f8 does;
decode parity (vs the f16 reference, not bit-exact vs plain f8) is
confirmed. Remaining gates: MTP acceptance, decode t/s, memory vs
q8-hybrid.

Memory gate (analytic, 16 full-attention layers x 1024 K + 1024 V per
token, q8_0=1.125 B/val, f8=1.0, f16=2.0, R9 K = f8 + 4x f16 scale per
block; MTP tail = last N layers f16, N=8 below 98K / N=12 at 98K):

| ctx | q8 (no tail) | f8-R9 (no tail) | q8-hybrid | f8-R9-hybrid |
|---|---:|---:|---:|---:|
| 49K | 1728 MiB | 1542 MiB | 2400 MiB | 2307 MiB |
| 98K | 3456 MiB | 3084 MiB | 5472 MiB | 5379 MiB |

The K scale costs only ~6 MiB (49K) / ~12 MiB (98K) = +0.4% over raw f8,
so R9 keeps the ~11% KV savings vs q8 in the no-tail lane; the MTP f16
tail dominates both. Memory gate PASSED.

Decode + acceptance gates (49K lane, dual Vulkan, b8192/ub1024,
`--real-context-chars 147456` -> ~30.8K tokens, Qwen3.8-27B-Q4_K_M,
`agent_workload_bench.py`, q8-hybrid vs f8-R9 on the fixed commit):

| gate | q8-hybrid | f8-R9 | delta |
|---|---:|---:|---:|
| decode t/s (spec=none) | 23.45 | 22.48 | -4.1% |
| decode t/s (draft-mtp n=2) | 48.83 | 47.88 | -1.9% |
| MTP acceptance (n=2, 128 tok) | 81.3% | 74.9% | -6.4 pt |
| KV memory 49K (no tail) | 1728 MiB | 1542 MiB | -10.8% |

Artifacts: `d131-r9-decode-{q8,r9}-r1`, `d131-r9-mtp128-{q8,r9}-r1`.
R9 does not close the f8 acceptance gap vs q8-hybrid: the K scale fixes
K underflow, but V stays raw f8 and V precision (not K) is the dominant
acceptance cost (R8). So R9 remains a memory option, same verdict as the
wider f8 line: ~11% less KV, ~4% slower spec=none decode, ~6 pt lower
MTP acceptance. Gates vs q8-hybrid: memory PASS, decode/acceptance do NOT
pass.

R8 explains why this must be K-only: a V scale differs per key row inside
the Bc reduction and cannot be applied after the coopmat product; the
symmetric K/V format is closed.

### R10 — where fp8 quality is actually lost (diagnostic, 2026-08-20)

Greedy top-logprobs of the first token isolate K vs V (14-token prompt):

| config | group | young | little | very |
|---|---:|---:|---:|---:|
| f16 K+V (reference) | -0.72 | -1.46 | -1.84 | -3.43 |
| K=f16, V=f8 | -0.74 | -1.44 | -1.83 | -3.36 |
| K=f8-R9, V=f16 | -1.01 | -0.93 | -2.03 | -3.58 |
| K=f8-R9, V=f8 | -2.17 | -0.93 | -1.69 | -2.47 |

`K=f16,V=f8` ~= f16, so V=f8 is nearly harmless and **K dominates the
residual error**; swapping V to f16 barely moves the distribution, so the
remaining gap is the e4m3 3-bit mantissa (6.25% relative error), not a
scale-addressable artifact. A synthetic V-quantization sweep agrees:
per-head V scale adds only +0.25 dB SNR (25.54 -> 25.79), and per-64-block
+0.74 dB — both too small to close an acceptance gate.

Mixed K/V on Vulkan is closed by hardware: the device reports `mixed K/V
Flash Attention requires coopmat2` and falls back to a non-Vulkan (CPU)
backend, so a `K=f16,V=f8` hybrid cannot run its prefill on Vulkan.

Conclusion: e4m3 (3-bit mantissa) is fundamentally coarser than q8_0
(8-bit integer, ~0.4%); R9 removed the underflow outlier but not the 6%
mantissa floor, and the only quality-parity lever (f16 K) needs coopmat2.
So fp8 quality parity with q8 is not reachable in e4m3 on this device; the
remaining actionable axis is decode efficiency (the -4..-5% scalar-decode
gap), i.e. D1-style f8->f16 dot work.

### MTP window path audit — PASSED (2026-08-20)

Checked every R9 scale touchpoint along the speculative path:

- graph wiring: `build_attn` for kv/k/iswa paths attach `get_k_scale`;
  the NextN drafter cross-attends through `build_attn(k)` (llama-graph.cpp
  `build_attn(k)` -> `build_attn_mha(..., get_k_scale, ...)`), so draft
  tokens read the scaled K. Only encoder-decoder cross (VLM) passes
  `nullptr`, which is correct (encoder K/V are f16);
- `k_idxs` are absolute `strm[s]*kv_size + idx` (set_input_k_idxs), and
  `cpy_k_scale` flattens to `[n_blk, kv_size*n_stream]` before set_rows, so
  the write side is stream-correct;
- state save/load and stream copy mirror the K ranges for the scale
  satellite (llama-kv-cache.cpp state_write/read, update);
- K-shift aborts when the scale satellite is present (Qwen never shifts).

One latent bug found and fixed: `get_k_scale` offset was
`row_size(n_blk)*sinfo.s0` (one row per stream) instead of
`row_size(n_blk)*kv_size*sinfo.s0` (a full stream, matching `get_k`/`get_v`).
It only mattered for multi-stream (s0>0); single-stream MTP (s0=0) is
unaffected, which is why the gates above were already correct. Smoke after
the fix: coherent output, 29.49 t/s.

Not yet done: f16-tail N shrink test.

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
