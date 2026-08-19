# D130: Vulkan fp8 native prefill re-audit

Date: 2026-08-19

Status: closed — H0 confirmed; prefill re-audit done, the wider fp8 K/V
review (quality + speed vs q8_0) continues as its own work item.

## Objective

Decide whether the Vulkan fp8 direction was closed on contaminated evidence,
and if so what remains worth building on RDNA4 (gfx1201, RX 9070 XT).

## Why this is being reopened

The D096 P5 route (raw f8 V, native fp8 S and P*V) measured against q8_0:

| lane | P5 prompt | q8 prompt | delta |
|---|---:|---:|---:|
| 12K | 1633.7 | 1432.5 | +14.0% |
| 49K | - | - | +43% |
| 98K | - | - | +45.5% |

with KV 96 MiB smaller at ctx 49152.

Two days later the bench runner was found to inherit `GGML_VK_FA_F8_NATIVE=1`
and `GGML_VK_FA_F8_NATIVE_DECODE=1` without recording them, so a set of
series measured a route they did not think they were measuring. The decode
conclusion was correctly retracted. The direction was then closed
(W11 batch 1, 2026-08-14) with the rationale "scalar f8 + prefill preconvert
is the memory-only route", and `GGML_VK_FA_F8_P2..P5` plus the transform
machinery were deleted.

The gap: the post-audit prefill parity number (default f8 1445.58 versus q8
1445.81 pt/s) is the **preconvert** route. The native kernel and the
preconvert route were not re-compared against each other in one clean
session after the audit. The hand-written `fp8_fa_cm1.spvasm` kernel and
`GGML_VK_FA_F8_NATIVE` both survived the cleanup, so the comparison is still
runnable.

## Hypothesis

H1: the native fp8 prefill kernel still beats the f16-preconvert default at
long context, and the +43/45% class of result was discarded on evidence that
only invalidated the decode claim.

H0 (null): preconvert wins or ties, the closure was correct, and the fp8
headroom on Vulkan lies entirely in the two unbuilt items (R9 block scale
for K, KV-parallel decode geometry).

## Method

Env differs per process, so there is no in-run control available the way
there is for a two-model `llama-bench` invocation. Established fork practice
applies instead: alternating same-binary invocations (A-B-A-B), accept only
if both controls agree within the noise band and both candidate runs land on
the same side.

- binary: one build, `build-vulkan`
- model: `models/Qwen3.6-27B-Q4_K_M.gguf`
- KV: `-ctk f8_e4m3 -ctv f8_e4m3`, flash attention on
- A = default (preconvert), B = `GGML_VK_FA_F8_NATIVE=1`
- reject the run if control spread exceeds ~1%

## Results (measured 2026-08-19, llama-bench, f8 K/V, dual RX 9070 XT)

D-N-D-N-D sandwiches, controls must agree before a native number is read.

| ctx | default (preconvert) pt/s | native pt/s | delta |
|---:|---:|---:|---:|
| 16K | 1430.31 / 1429.75 (spread 0.04%) | 1416.43 / 1422.91 | ~-0.7% |
| 49K | 1220.22 / 1218.58 / 1221.67 (spread 0.25%) | 1137.09 / 1118.69 | **-7.6%** |

**H1 rejected.** The native kernel loses, and the loss grows with KV.
Header of `fp8_fa_cm1.spvasm` explains why: the surviving kernel is the
P4-class binary (fp8 only in the S stage; V stage is f16 coopmat with
per-tile f8->f16 staging). Its staging cost repeats for every query tile,
while the default preconvert pays f8->f16 once per KV. The deleted P5
(fp8 P*V) is the route that carried the +14/43/45% numbers, and it was
not re-measurable after the 2026-08-14 cleanup.

Conclusion for prefill: preconvert is the correct default; restoring the
native path has no measured case. The fp8 headroom that remains is
quality-side (R9 block scale for K) and decode-side, not this kernel.

## Fences

- MMQ / integer dot: closed statically. `V_DOT2_F32_F16` is CDNA;
  `VK_KHR_shader_integer_dot_product` is not exposed on RDNA AMD. fp8 math on
  this device is reachable only through coopmat/WMMA.
- Cooperative matrix shapes on gfx1201: `16x16x16` only (f16/f32, f16/f16,
  int8, fp8, bf16). No freedom in fragment geometry.
- Native cooperative **decode** is closed for the current geometry (13.28
  versus 25.45 t/s scalar at 49K): `Br=16` against grouped-query `N=6` leaves
  most of the fragment idle. Reopening needs a design that batches KV rather
  than queries into the fragment.
