# D095: Q4_K_M Vulkan FP8 MTP polish

Status: R1-R9 prebuild program complete; runtime K-scale sidecar is deferred to
a separate default-off prototype after the current stable benchmark refresh.

## Objective

Make the `f8_e4m3` KV route at least as fast as `f16` for the production
Qwen3.6-27B-Q4_K_M Vulkan MTP lane without reducing output quality or draft
acceptance. Preserve `f8_e4m3`'s KV-memory advantage and keep every candidate
behind a narrow rollback gate until an adjacent A/B passes.

## Accepted baseline

Adjacent 2026-08-10 runs, `ctx=49152,b=512/ub=256`, 8601-token prompt,
`draft-mtp,n=2`, `Vulkan1,Vulkan0`, no reuse:

| KV route | Prompt tok/s | Decode tok/s | Draft acceptance |
|---|---:|---:|---:|
| f8 hybrid, last 8 layers f16 | 1618.7 | 51.7 | about 75% |
| f16 | 1673.5 | 58.1 | about 90% |

Artifacts: `d096-ab-mtp-f8hyb-vs-f16` and `d096-ab-f16-mtp-ctl`.
The adjacent spec-none control is already at parity (`30.4` f8 vs `29.5`
f16 decode tok/s), so no scalar-kernel change may be promoted from an
unbracketed short-prompt result.

## Route evidence

`d096-r-route-f8mtp-control.server.log` proves the long-KV verify/decode graph
contains these routes at `KV=8704`, `HSK=HSV=256`, `N=6`:

- f8 scalar: `Br=8,Bc=32,shmem_staging=1,split_k=31`;
- f8 preconvert/coopmat1: `Br=16,Bc=64,shmem_staging=0,split_k=10`;
- f16 coopmat1: `Br=16,Bc=64,shmem_staging=0,split_k=28`.

The first probe targets only the confirmed f8 scalar staging cost. It does not
claim that scalar FP8 arithmetic is the main MTP gap; most of the current gap
tracks draft acceptance.

## R1: direct FP8 scalar reads

Add `GGML_VK_FA_F8_SCALAR_DIRECT=1` to force `shmem_staging=0` only for the
scalar `f8_e4m3` FlashAttention tuning state. The shader already has the
direct `dequantize4` path when staging is disabled, so the probe changes route
selection rather than tensor layout or numerical format.

Expected benefit: remove the full K/V f8-to-f16 LDS staging pass and trade it
for register-local dequantization. Primary risk: repeated dequantization and
global reads can be slower, or register pressure can reduce occupancy.

## Validation contract

1. Build `build-vulkan/bin/llama-server.exe` with Ninja.
2. Route trace must show the f8 scalar row with `shmem_staging=0`; f16 and P5
   routes must remain unchanged.
3. Run adjacent control/candidate on the accepted 12K MTP lane with at least
   128 output tokens; record prompt, decode, acceptance and errors.
4. Promote only if decode improves outside the local noise band without a
   prompt or acceptance regression. Otherwise remove the code probe.
5. Finish with output sanity, no surviving server, and `git diff --check`.

## R1 result: rejected and removed

Route activation passed: the long-KV scalar f8 row changed from
`shmem_staging=1` to `shmem_staging=0`. The adjacent 128-token MTP pair was a
strict tie:

| | Control | R1 direct | Delta |
|---|---:|---:|---:|
| Prompt tok/s | 1613.08 | 1612.88 | -0.01% |
| Decode tok/s | 53.40 | 53.43 | +0.06% |
| Acceptance | 78/97 | 78/97 | unchanged |

Artifacts: `d095-r1-control-postbuild-r1` and
`d095-r1-f8-scalar-direct-r1`. The code gate was removed; f8 scalar LDS
staging is not the current wall-time limiter.

## R2: precision placement at fixed memory

K-only or V-only f16 layers are not a valid RX 9070 XT candidate: mixed K/V
FlashAttention is implemented only by coopmat2, while this device uses
coopmat1. The Vulkan registry would reject the FA node and split the graph to
another backend.

R2 therefore keeps matching K/V types and the same count of f16 layers, but
adds a default-off placement selector:

- `last`: current behavior and unchanged default;
- `first`: earliest N attention-KV layers;
- `interleaved`: N evenly distributed layers, including the final layer.

The hypothesis is that quantization error from early attention layers is
amplified by later recurrent/attention blocks. Interleaving may improve draft
acceptance at the same 2304 MiB N=8 cache size. Unknown values must fail back
to `last`; no mode may create mixed K/V tensors.

The preliminary no-code N=12 probe is rejected: prompt `1623.18 tok/s`, decode
`48.07 tok/s`, acceptance `72/110` (65.5%), and KV `2688 MiB`, versus the
adjacent N=8 control `1613.08/53.40`, `78/97` (80.4%). More f16 layers are not
monotonically better on a single deterministic generation.

R2 placement bracket is also rejected and the selector was removed:

| N=8 placement | Decode tok/s | Acceptance |
|---|---:|---:|
| last A | 52.96 | 78/97 (80.4%) |
| interleaved | 48.19 | 73/106 (68.9%) |
| first | 45.14 | 68/117 (58.1%) |
| last D | 50.85 | 78/97 (80.4%) |

The closing control shows thermal throughput drift, but acceptance is exactly
stable. Late attention-KV precision is materially more valuable than early or
evenly distributed precision for this model. Keep the contiguous `last`
policy. Artifacts: `d095-r2-bracket-{a-last,b-interleaved,c-first,d-last}`.

## R3: raw-FP8 small-N coopmat1

Specialized gfx1201 offline ISA for the observed routes changes the next
candidate. The actual scalar f8 route uses 119 VGPR, 21,504 bytes LDS and a
28,748-byte ISA body with heavy bit extraction/conversion. The f16 coopmat1
route uses 100 VGPR, 17,920 bytes LDS, a 14,636-byte ISA body and 32 WMMA
instructions. The scalar f8 route has no packed-dot/WMMA body.

R3 must not preconvert the entire long KV cache on every autoregressive token.
Instead, `GGML_VK_FA_F8_SMALL_CM1=1` enables the existing DATA_A_F8_E4M3
coopmat1 shader for the `neq1==1` GQA route. It reads raw f8 K/V, dequantizes
only the current tiles into shared f16, and uses f16 WMMA. The opt-in also
disables full-tile f8 LDS staging in the coopmat1 tuning state so the pipeline
fits the RX 9070 XT shared-memory limit.

Activation invariant at long KV: the previous `path=scalar,k=f8_e4m3,N=6`
row must become `path=coopmat1` with `f8_direct=1`, `preconvert=0` and
`shmem_staging=0`. P5 prefill and f16 layers must remain unchanged. A failure
to create the pipeline or any output error rejects and removes R3.

R3 activation passed exactly, but the adjacent MTP pair was negative:
`1595.87/52.22` prompt/decode for control versus `1589.00/51.99` for R3
(-0.43% prompt, -0.44% decode). Acceptance was identical at `78/97`. The code
probe was removed. WMMA replacing the scalar f8 row is not material at wall
level for this mixed eight-f8/eight-f16 graph.

An unbiased E4M3 round-to-nearest-even encoder was also screened
analytically. It changes only exact halfway FP32 inputs; identical seeded
Gaussian samples produced effectively identical encodings/MSE. Do not add a
shader variant for this negligible ceiling.

## R4: robust hybrid-depth sweep

Two-task (`triage_diff,review_bug`) 128-token runs make the N=8 result more
stable and reduce the apparent f16 gap:

| Last f16 KV layers | KV MiB | Prompt tok/s | Decode tok/s | Acceptance |
|---:|---:|---:|---:|---:|
| 8 | 2304 | 1605.79 | 55.49 | 159/186 (85.5%) |
| 9 | 2400 | 1603.92 | 51.28 | 151/204 (74.0%) |
| 10 | 2496 | 1611.88 | 49.80 | 148/209 (70.8%) |
| 16 | 3072 | 1627.67 | 58.81 | 163/182 (89.6%) |

N=8 remains the hybrid optimum. N=9/10 are non-monotonic regressions; do not
change the default. The robust N=8-to-N=16 decode gap is now 5.6%, with 25%
less KV memory, rather than the earlier single-task 11% estimate.

## R5: MTP depth at fixed N=8

With aggregate N=8 acceptance at 85.5%, a deeper draft may amortize verify
overhead better than the current `--spec-draft-n-max 2`. Bracket n=2/n=3/n=4
on the same two tasks, then repeat n=2. This is a runtime-profile experiment;
it does not change source or the default unless both tasks improve and the
closing control validates the bracket.

The four-run bracket rejects deeper drafts:

| Run | Draft max | Prompt tok/s | Decode tok/s | Acceptance |
|---|---:|---:|---:|---:|
| A | 2 | 1605.31 | 55.62 | 159/186 (85.5%) |
| B | 3 | 1608.93 | 45.13 | 151/308 (49.0%) |
| C | 4 | 1603.55 | 48.81 | 177/302 (58.6%) |
| D | 2 | 1602.49 | 54.84 | 159/186 (85.5%) |

Prompt evaluation is flat, while n=3/n=4 create many more rejected draft
tokens and lose 18.3%/11.6% decode against the n=2 bracket mean (`55.23`).
Keep `--spec-draft-n-max 2`; this model/precision lane is acceptance-bound,
not verify-depth-bound. Artifacts: `d095-r5-{a-n2,b-n3,c-n4,d-n2}`.

## R1-R5 verdict

- R1 direct scalar f8 reads: wall-time tie, removed.
- R2 alternate f16-layer placement and N=9/10/12: worse acceptance, removed or
  rejected. Contiguous last-eight remains the memory/quality optimum.
- R3 raw-f8 small-N coopmat1: route activated, wall time slightly worse,
  removed.
- R4 robust depth sweep: N=8 is 5.6% behind full-f16 decode while using 25%
  less KV memory; the earlier single-task 11% estimate was pessimistic.
- R5 deeper MTP: n=2 is decisively best.

No retained performance probe remains from R1-R5. Two correctness fixes are
kept: `LLAMA_VK_MTP_KV_LAST_F16=0` now really disables the hybrid cache, and
the hybrid-cache log passes a stable string to the variadic logger.

## R6: block-scaled FP8 precision gate

The remaining decode gap follows draft acceptance, so the next candidate must
improve the information stored by the early FP8 KV layers without returning
them to f16. A promising representation is MX-style block-scaled E4M3: one
power-of-two scale per 32 values plus 32 E4M3 payload bytes. Its storage is
`33/32 = 1.03125` bytes/value, still 48.4% below f16, and a power-of-two scale
can be folded into exponent adjustment instead of adding a general multiply.

Do not add a GGML type or shader path before this prebuild gate passes:

1. Capture representative K and V blocks from early attention layers on both
   benchmark prompts without changing normal execution.
2. Compare raw E4M3, block-scaled E4M3 (block 16/32/64), q8_0 and f16 offline:
   MSE, cosine error, saturation, subnormal/zero rate and attention-logit error.
3. Require block-32 to reduce raw-E4M3 attention-logit error materially while
   keeping its scale metadata below 4% of payload. If it does not, close R6
   analytically.
4. If it passes, implement a default-off format/prototype and first validate
   output and draft acceptance. Promotion requires aggregate acceptance at
   least the current 85.5%, decode above 55.49 tok/s, and no P5 prefill loss.

This gate targets the actual bottleneck. More scalar/coopmat route toggles are
fenced off unless a new profile shows FlashAttention wall share increased.

### R6 result: rejected before format work

The diagnostic-only `llama-kv-precision-scout` captured complete post-RoPE K/V
for full-attention layers 3/7/11 on both canonical prompts (`8570` and `8551`
tokens). Q was captured only for the final chunk. Every layer has exact
prompt/K/V token-count agreement.

Weighted attention-logit MSE across all six task/layer pairs:

| Format | Bytes/value | Weighted logit MSE | vs raw E4M3 |
|---|---:|---:|---:|
| raw E4M3 | 1.00000 | 0.0030764424 | baseline |
| block E4M3, B16 | 1.06250 | 0.0030764148 | -0.0009% |
| block E4M3, B32 | 1.03125 | 0.0030764146 | -0.0009% |
| block E4M3, B64 | 1.015625 | 0.0030764146 | -0.0009% |
| q8_0 | 1.06250 | 0.0001028896 | -96.66% |
| f16 | 2.00000 | 0.0000001875 | -99.994% |

Power-of-two block scaling does not add mantissa precision. For normal E4M3
values it only shifts the exponent, so the relative quantization grid and its
dominant error are invariant. It reduces rare raw-E4M3 zeros, but zero/subnormal
error is not material to the observed tensor or logit MSE. R6 fails the 25%
gate by four orders of magnitude. Do not add a block-scaled E4M3 GGML type or
shader path.

Artifacts: `build_logs/agent-workload/d095-r6-kv-precision.{csv,md}` and
per-task prompt/log files. The standalone scout is retained because it
provides the required real-KV precision gate for the next format candidate.

## R7: block-floating int8 precision gate

The same capture identifies a stronger storage-level candidate. Existing
q8_0 cuts logit MSE by 96.66% at only 6.25% more bytes than raw E4M3, but the
current Vulkan q8 route preconverts the full KV and loses 43-45% P5 prompt
throughput at 49K/98K. Test a block-floating int8 representation before any
runtime work: 32 signed int8 values plus one signed power-of-two scale exponent
(`33/32 = 1.03125` bytes/value).

Admission gate:

1. Add only the offline BFP8 reconstruction to the retained scout and rerun
   the same two prompts/layers.
2. Require at least 75% weighted logit-MSE reduction versus raw E4M3 and no
   task/layer regression. Otherwise close R7 analytically.
3. A passing precision result authorizes a design/resource note only. A
   runtime prototype must prove raw tiled dequantization without the existing
   q8 whole-KV preconvert; otherwise D096-H already predicts a large prefill
   regression.
4. Promotion thresholds remain acceptance >=85.5%, decode >55.49 tok/s and no
   P5 prompt loss, with 1.03125 bytes/value storage.

### R7 result: precision passes, runtime admission fails

The repeated six-pair capture is complete and deterministic. BFP8 P2/B32
produces weighted attention-logit MSE `0.0002514458`, a 91.83% reduction from
raw E4M3. The worst task/layer ratio is `0.0939`, so all six pairs improve by
at least 10.6x. The precision gate passes at `1.03125` bytes/value.

That result does not justify a new runtime format. Existing q8_0 is 2.44x more
accurate (`0.0001028896`) for only another 3.03% storage, and its exact runtime
family is already closed on this GPU:

- D094 cycle 5 measured raw int8/scalar and int8-coopmat paths 22-25% slower;
  the 16x16 int32 accumulator uses about 64 VGPR/lane and reduces occupancy.
- D094 cycle 6 then adopted whole-KV q8->f16 preconvert: the 131K control fell
  from `54.1s` to `47.8s` (-11.7%). Therefore the existing raw q8 cm1 path is
  slower than preconvert even before adding a new format.
- BFP8 would still require tile dequantization or the rejected int8 accumulator,
  plus a new cross-backend GGML type. It has no work-volume/resource mechanism
  that invalidates the q8 result.

R7 is closed after the transfer/resource gate. Keep its offline method as
evidence; do not add a BFP8 type or shader. Artifact:
`build_logs/agent-workload/d095-r7-bfp8-precision.{csv,md}`.

## R8: general-scale E4M3 upper bound

R6 used a power-of-two scale, which is exponent-shift invariant. Test the
remaining E4M3 precision upper bound with one general f16 scale per block:
`scale = max(abs(block))/240`, followed by E4M3 encode/decode. B16/B32/B64
cost 1.125/1.0625/1.03125 bytes/value. A general scale changes mantissa-grid
phase and may reduce block error, but it also requires a multiply and partial
scale application in a future WMMA route.

Admission gate:

1. Add only offline general-scale B16/B32/B64 methods to the retained scout.
2. Require at least 25% weighted logit-MSE reduction versus raw E4M3 and no
   task/layer regression. Close analytically if B16 cannot pass; larger blocks
   cannot offer a stronger scale fit.
3. A precision pass still needs a static WMMA plan that applies one scale per
   K=16 partial accumulation and V output block without full-KV preconvert.
4. No GGML type or runtime shader before both gates pass.

### R8 result: precision passes, P*V factorization fails

All general-scale sizes improve every captured pair:

| Format | Bytes/value | Weighted logit MSE | Reduction |
|---|---:|---:|---:|
| general-scale E4M3 B16 | 1.1250 | 0.0016826834 | 45.30% |
| general-scale E4M3 B32 | 1.0625 | 0.0019487026 | 36.66% |
| general-scale E4M3 B64 | 1.03125 | 0.0021820721 | 29.07% |

B16 passes the >=25% upper-bound gate; its worst task/layer ratio is `0.5872`.
The corrected report allows its declared 12.5% metadata and marks precision
PASS.

Runtime feasibility fails in the V phase. A K-block scale can multiply each
K=16 partial Q*K score before accumulation. A V scale, however, differs for
each key/value row inside the Bc reduction of P*V. It cannot be applied after
the cooperative-matrix product. Applying it before the product would require
a separately scaled/quantized P tile for every V output block; the alternative
is the full f16 V dequant path already superseded by P5. Both erase the intended
work-volume win. Do not add the symmetric K/V block-scaled format.

Artifact: `build_logs/agent-workload/d095-r8-gse4m3-precision.{csv,md}`.

## R9: per-token K scale (factorable upper bound)

Test the remaining factorable variant: one f16 scale for the complete
256-value K vector of each `(token, kv_head)`, with V left as raw E4M3. The K
representation costs `1 + 2/256 = 1.0078125` bytes/value; averaged across K
and unchanged V, KV payload overhead is only `0.390625%`.

This scale is factorable: Q*K runs on the normalized E4M3 payload, then the
completed score column is multiplied by that key token's scale before softmax.
No scale enters P*V.

Admission gate:

1. Add offline general-scale B256 and use only its K attention-logit result.
2. Require >=15% weighted logit-MSE reduction and no task/layer regression.
3. A pass authorizes a sidecar/resource design only; it must account for one
   f16 scale load and score multiply per `(query,key)` plus SET_ROWS/copy/view
   lifecycle. A fail closes scale-based E4M3 work.
4. V remains byte-identical P5 raw E4M3 in every case.

R9 passes the offline gate:

- weighted raw-E4M3 logit MSE: `0.003076`;
- K-only general-scale B256 logit MSE: `0.002445` (`-20.52%`);
- worst task/layer candidate/raw ratio: `0.8737` (no local regression);
- K metadata: `2/256 = 0.78125%`; average K+V KV overhead: `0.390625%`.

Artifact: `build_logs/agent-workload/d095-r9-konly-gse4m3-precision.{csv,md}`.

This is a precision result, not a runtime win. A runtime implementation needs a
separate scale sidecar for every quantized `(layer, token, kv_head)`, including
SET_ROWS writes, cache copies, sequence moves, views and rollback-safe graph
selection. The Q*K shader also pays one f16 scale load and one score multiply
per key, though GQA can reuse that scale across the six query heads mapped to a
KV head. Those lifecycle and resource changes are intentionally not mixed into
the stable P5 binary used for the full Q4 Vulkan benchmark refresh.

D097 subsequently changed only the hybrid precision policy at 98K: FP8+MTP
now uses last-12 f16 to recover acceptance. The P5 shader and D095 R1-R9
runtime conclusions are unchanged; q8 and FP8 contexts below 98K remain N=8.

Decision: close D095's offline polish program with R9 PASS. The next source
candidate is a separately owned, default-off K-scale sidecar prototype; raw V
must remain P5 E4M3 and the prototype must first beat the current N=8
`1605.79/55.49` prompt/decode and `85.5%` acceptance bracket before any default
or README speed claim.

## Stable Q4 Vulkan refresh

After closing R9, the unchanged P5 binary was measured against q8 on the full
documentation matrix: contexts 12,288/49,152/98,304, `b8192/ub1024`, matched
repo-snapshot prompts, 128 output tokens, spec-none and MTP n2. All 12 runs
completed without critical server errors or surviving processes.

Spec-none FP8 prompt throughput improves by `+7.1%/+10.4%/+12.6%`; aggregate
TPS improves by `+3.0%/+6.8%/+9.3%`. At 12K/49K MTP, FP8 aggregate improves
by `+4.6%/+1.5%`. At 98K MTP it is a wall tie (`2.9494` vs `2.9407`) but loses
acceptance (`51.61%` vs `68.87%`) and decode (`32.40` vs `37.79`), so q8 is
retained for generation-heavy 98K use. Full rows and KV memory are in
`Q4_K_M_RESULTS.md`; artifacts use
`d095-refresh-vk-q4km-{12k,49k,98k}-{q8,f8}-{none,mtp2}-r1`.