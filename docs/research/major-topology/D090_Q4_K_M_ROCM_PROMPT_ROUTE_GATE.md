# D090 Q4_K_M ROCm Prompt Route Gate

Date: 2026-07-20

Status: closed. Fresh idle brackets reject Candidates A, C and D. All temporary
source/build/task gates are removed and the production ROCm binary is rebuilt.

## Goal

Improve cold prompt evaluation for the primary Qwen3.6-27B Q4_K_M model on
the reference dual-RX 9070 XT machine. Select the next code change from an
exact-lane route trace and measured waste signal rather than transferring a
proxy-model result.

## Locked Lane

- Backend/build: ROCm, `build-rocm-full/bin/llama-server.exe`.
- Model: `models/Qwen3.6-27B-Q4_K_M.gguf`.
- Devices: `-dev ROCm1,ROCm0 -sm layer -ts 1,1`; default output placement on
  the last device remains unchanged.
- Context: `49152`; reference prompt about `29561` tokens.
- Batch/ubatch: `8192/1024`; one slot.
- KV: `q8_0/q8_0`; Flash Attention on.
- Cold/no-reuse/no-prime/no-warmup; thinking on.
- Speculation: `none` for all prompt-kernel controls and candidates.
- Workload: `quick:triage_diff`, repo snapshot `96000` characters, seed 42.
- Iteration: adjacent `r1`; `r3` only for a promising or borderline candidate.

The D089 reference is `1778.59 prompt tok/s` and `21.98 decode tok/s`. A fresh
adjacent control is required before a new claim because the worktree and
background load may have changed.

## Evidence Ladder

1. Confirm that no `llama-server` process is active.
2. Run an uninstrumented adjacent control on the locked lane.
3. Run a diagnostic exact-model trace with node, route, MMQ and Flash
   Attention timing enabled. Trace throughput is diagnostic only.
4. Rank prompt-only cost by operation, quant type, route and shape.
5. Require both a top-share bottleneck and a measured waste signal before
   editing source.
6. Estimate wall upside with Amdahl's law. Skip candidates whose plausible
   gain is below the approximately 3% cold-order swing unless the harness
   noise is reduced first.
7. Run a focused correctness/route gate, then paired lane A/B.

## Existing Fence

- Keep E344's Q4_K/Q5_K `y128/w8` geometry. Narrower `mmq_x` values lost.
- Keep Q6_K prefill on hipBLAS; forced MMQ was substantially slower.
- Keep E345's Q6_K decode `small_k=false` policy. It is not a prompt route.
- Keep Q5_K decode at `small_k=false,nwarps=8`; row batching and `nwarps=4`
  lost.
- Do not use MTP, TKV4, KV type changes, or layer-split sweeps as prompt-kernel
  candidates in this gate.
- Do not enable unsafe peer copy. The production layer split and existing safe
  host-staged fallback remain the transport baseline.

## Candidate Admission

A code prototype is admitted only when the exact Q4_K_M trace identifies one
of these with target-closing evidence:

- Q4_K/Q5_K MMQ: a high-share shape plus resource or work-imbalance waste not
  already covered by E344;
- Q6_K hipBLAS path: a high-share prompt bucket plus a library/packing or
  batching mechanism that preserves the efficient large-GEMM route;
- q8 Flash Attention: a high-share long-KV bucket plus redundant staging,
  excess scratch, or avoidable work-volume evidence;
- graph/device boundary: a measured prompt share materially larger than the
  earlier approximately 1-1.5% layer-boundary copy ceiling.

## Exact-Model Trace

The `kernel-full` diagnostic on the locked lane recorded `29696` prompt tokens.
Its throughput is not a comparator because tracing reduced prompt evaluation to
`650.36 tok/s`; only device-time shares are used here.

- `MUL_MAT`: `26419.42 ms`.
- `FLASH_ATTN_EXT`: `6267.02 ms` (about `15.4%` of summed major node time).
- `GATED_DELTA_NET`: `5053.96 ms`.
- Q4_K MMQ alone: `15645.62 ms`, but E344 already closed the nearby geometry
  family with `y128/w8` as the winner.
- Q6_K prompt work remained on hipBLAS; E345 measured forced MMQ substantially
  slower.

The q8 chunked-WMMA Flash Attention path uses a fixed `4096`-token staging
window. Across the 16 full-attention layers and a roughly 30k prompt, that
causes about `2048` internal chunk passes. An `8192` window reduces this to
about `1152`, roughly 44% fewer q8-to-f16 conversion, launch and online-combine
iterations. It adds about 16 MiB of bounded K/V scratch per active device.

If the avoided iterations reduce local Flash Attention time by 20-25%, Amdahl's
law predicts about 3.2-4.0% prompt-wall improvement. This clears the prototype
threshold but not the keep threshold by itself.

## Candidate A: Bounded Q8 Chunk Override

Add `GGML_ROCM_FATTN_Q8_CHUNK_SIZE` as a process-start diagnostic override for
the existing RDNA4 q8 chunked-WMMA path. Requirements:

- default remains `4096` when the variable is absent or invalid;
- accepted values are multiples of 256 in `[1024,16384]`;
- allocation, dispatch loop and route diagnostics use the same latched value;
- first measured candidate is `8192`; `16384` is tried only if 8192 is
  promising or exposes a clear size trend;
- no KV type, graph topology, device split or peer-copy behavior changes.

### Candidate A Result

The route/allocation smoke confirmed `chunk=8192`, unchanged q8 KV residency of
`816 MiB` per GPU, and a bounded Flash Attention allocation of `80.38 MiB`
versus about `64.38 MiB` at 4096. The smoke's throughput is invalid because
`GGML_TRACE_FATTN_ALLOC_SIZE` emitted 1952 synchronous log records.

The clean adjacent A/B on one rebuilt binary was negative:

| Variant | Prompt tok/s | Decode tok/s | Prompt delta |
| --- | ---: | ---: | ---: |
| `8192` candidate | `1495.96` | `7.42` | `-1.32%` |
| `4096` post-control | `1516.00` | `7.40` | reference |

Absolute decode and prompt rates in this pair were below the earlier D090
controls, and external GPU load was not recorded. The measured `-1.32%` was
therefore retained only as a provisional negative until the idle repeat below.

## Rollback And Decision

Any negative source probe must be removed before closing D090 unless it is a
documented default-off diagnostic. Kept changes require focused correctness,
an exact-lane adjacent A/B, a rebuilt ROCm server, canonical benchmark history,
and final `git diff --check`.

The next candidate requires measured timing inside the chunked path; launch
count alone is not sufficient evidence.

## Diagnostic B: Q8 Chunk Phase Timing

Add a HIP-only, default-off diagnostic behind
`GGML_TRACE_FATTN_Q8_CHUNK_TIMING=1`. It uses stream events to report separate
q8 K conversion, q8 V conversion, WMMA Flash Attention, and online-combine time
for each chunked Flash Attention node. The diagnostic synchronizes phases and
is therefore never a throughput comparator. It is admissible only for locating
the next optimization target and has zero event or synchronization work when
disabled.

The keep decision is based on phase share, not traced wall throughput.

### Diagnostic B Result

The trace covered `464` chunked Flash Attention nodes and `1920` internal chunk
passes. Summed synchronized event time was:

| Phase | Time | Share of chunked FA |
| --- | ---: | ---: |
| q8 K to f16 | `837.32 ms` | `3.24%` |
| q8 V to f16 | `799.51 ms` | `3.09%` |
| WMMA Flash Attention | `20130.27 ms` | `77.77%` |
| online combine | `4115.78 ms` | `15.90%` |

HIP returned a small negative elapsed value for 22 of 3840 conversion samples;
the conversion percentages are therefore approximate. They are still too small
to change the decision: both conversions together have a theoretical ceiling
near 1% of total prompt time, while eliminating combine entirely is only about
2.5%. The WMMA body is the only phase with a target-closing ceiling.

## Candidate C: D256 32-Column WMMA Gate

The exact route is `D=256`, `prec=10`, `q_rows=1024`, `selected_cols=16`.
The half-accumulator dispatch already instantiates D256 with 32 columns, but the
float-accumulator switch had that case commented out. Enable only this existing
template case so `GGML_FATTN_WMMA_FORCE_COLS_PER_BLOCK=32` can test it. The
default selector remains at 16 columns. A 25% local WMMA-body win would imply
about 3% prompt-wall upside; anything materially smaller is rejected.

### Candidate C Result

The route smoke confirmed `forced_cols=32,selected_cols=32` for all 464 D256
Flash Attention calls, with no runtime error and graceful server cleanup.
`29696` prompt tokens took `163789.22 ms`, only `181.31 tok/s`. Later runs
with confirmed Sovereign load reproduced the same approximately 178-246 tok/s
corridor, so the original result could not remain a clean rejection without an
idle load record. The compile-gated template was retained only for the idle
repeat below.

## Diagnostic C: Exact Q4_K/Q5_K MMQ Resources

`GGML_TRACE_MMQ_PATH=1` plus `GGML_TRACE_MMQ_RESOURCES=1` was run on the
locked Q4_K_M lane without synchronized kernel timing. Resource collection and
5344 route records pushed the request past the 45-second diagnostic hard
timeout after 16384 prompt tokens, so its throughput is invalid. The target
resource records were already complete and identical across repeated hot
dispatches:

| Type | Tile | Threads | Registers | LDS | LDS share | Active blocks/CU | Occupancy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | `128x128`, 8 waves | 256 | 236 | 57856 B | 88.28% | 1 | 12.50% |
| Q5_K | `128x128`, 8 waves | 256 | 240 | 57856 B | 88.28% | 1 | 12.50% |

This does not reopen the narrower-tile family: E344 measured every practical
sub-128 X tile as slower and measured `y128/w8` as 4.69% faster than `y64/w4`.
The admissible direction is therefore less per-output work or register
pressure while preserving the winning geometry and route.

## Candidate D: Packed Half2 Scale Product

The RDNA4 WMMA Q4_K/Q5_K dot body currently converts both stored half2 scale
pairs to `float2`, then performs two FP32 scale products per accumulator. The
existing CUDA/HIP vecdot path already uses the narrower operation order:

1. multiply the two stored half2 scale pairs with `__hmul2`;
2. convert the packed result once to `float2`;
3. retain FP32 accumulation of the integer dot and min correction.

Candidate D first applied that operation order to the separately compiled
Q4_K/Q5_K RDNA4 MMQ instances. It does not change tile dimensions, LDS layout,
integer WMMA, output accumulation type, tensor placement, KV, or device
transport. The extra FP16 rounding in the scale product makes correctness and
quality gates mandatory. Admission requires a clean backend-op comparison and
either lower registers/instruction cost or a positive resident-proxy A/B
before the expensive production Q4_K_M lane.

The focused Q4_K and Q5_K `MUL_MAT n=64` backend-op cases both passed against
the CPU reference. On the exact `N=1024,x=128,y=128` specialization, Q4_K
registers fell `236 -> 233`, but Q5_K rose `240 -> 256`; LDS and one-block/CU
occupancy were unchanged. The Q5_K variant is therefore removed before speed
testing. The production candidate is compile-gated only for Q4_K, which
accounts for about 88% of the measured MMQ time on the resident proxy.

### Candidate D Initial Result

The first Q4-only production request processed `16384/29696` prompt tokens
before the 45-second hard timeout. That initially looked catastrophic versus
the early 19-20 second control corridor. The post-revert control then measured
only `246.13 prompt tok/s` and `7.78 decode tok/s`, while ROCm1 exposed only
`10428 MiB` total memory instead of its normal approximately `15428 MiB`.
Process inspection identified `Sovereign` running concurrently. The initial
candidate and the post-control are therefore a same-load sequence, not clean
idle evidence; the early idle controls and diagnostic trace cannot be used as
their comparator.

The repeated Q4-only candidate under confirmed Sovereign load completed at
`178.13 prompt tok/s` with only `10279 MiB` visible on ROCm1. Its neighboring
post-revert control reached `246.13 prompt tok/s`, but ROCm1 reported
`10428 MiB`; the game load moved materially between the two requests. Neither
run is a speed comparator. The Q4-only compile gate was retained only for the
idle repeat below.

## Idle Repeat Results

Each candidate was measured in a separate prompt-only
`control -> candidate -> post-control` bracket. Process checks found neither
`llama-server` nor `Sovereign` before each launch. Every server log completed
gracefully and reported the full `15428 MiB` total on both GPUs.

| Candidate | Pre-control | Candidate | Post-control | Control mean | Delta | Control spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: q8 chunk `8192` | `1690.04` | `1666.32` | `1717.89` | `1703.97` | `-2.21%` | `1.63%` |
| C: D256 float cols32 | `1713.33` | `1482.68` | `1716.23` | `1714.78` | `-13.53%` | `0.17%` |
| D: Q4_K half2 scale | `1689.06` | `1545.62` | `1721.75` | `1705.41` | `-9.37%` | `1.92%` |

Prompt throughput is in tokens per second. All three brackets satisfy the
maximum 3% control-spread gate. No candidate reaches the positive 3% admission
threshold, so none advances to `r3` or decode confirmation.

Canonical idle artifacts:

- A: `d090-idle-control-preA-r1-20260720-220351`,
  `d090-idle-chunk8192-A-r1-20260720-220501`,
  `d090-idle-control-postA-r1-20260720-220554`.
- C: `d090-idle-control-preC-r1-20260720-220653`,
  `d090-idle-cols32-C-r1-20260720-220825`,
  `d090-idle-control-postC-r1-20260720-220937`.
- D: `d090-idle-control-preD-r1-20260720-221056`,
  `d090-idle-q4-half2-D-r1-20260720-221156`,
  `d090-idle-control-postD-r1-20260720-221301`.

## Final Decision

Reject A, C and D. Restore the fixed q8 chunk `4096`, the D256 float
16-column WMMA selector, and the FP32 Q4_K scale product. Remove the temporary
runtime override, compile options, candidate branches and VS Code repeat tasks.
The old external-load artifacts remain route/resource evidence only and are
superseded for speed decisions by the idle brackets above.
