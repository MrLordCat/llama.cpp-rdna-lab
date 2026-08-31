# ggml-HIP feature gap and experiment queue

This is the operational answer to “which ROCm 7.2 features are not being used
efficiently?” Priority reflects expected information/performance value for the
locked Qwen3.8 lane, not the size of the API.

## Current coverage

| ROCm/HIP capability | Current fork | Assessment |
| --- | --- | --- |
| streams and async copies | heavily used | covered |
| events and cross-stream waits | used | covered |
| graph capture/update/replay | used | covered; graph replay already stable |
| pinned host memory | used with default flags | covered; flags remain a narrow staging question |
| managed memory | opt-in with fallback | keep off for the production dGPU lane |
| P2P | API exists; capability false on current pair | closed until runtime/driver change |
| FP8 kernels | used in attention/KV routes | covered |
| rocWMMA | enabled for Flash Attention | covered |
| rocBLAS | integrated fallback/dense route | covered; Q6_K solution-index tuning rejected (G07) |
| MMQ quantized prefill | RDNA4 Q4_K/Q5_K selector | covered; G08 lowers Q4_K MMQ max_ne11 to 256 |
| hipBLASLt | standalone exact-shape scout; runtime gate rejected | diagnostic/offline tuning only |
| rocPRIM/hipCUB | isolated references only | conditional operation-specific research |
| async allocation/memory pools | absent; Beta/Linux-only implementation in 7.2 | defer until Windows support, then load/startup research |
| stream priorities | absent | multi-workload latency feature, not single-request throughput fix |
| RGP HIP profiling | supported, not installed/used | highest-value gap |
| ROCProfiler/RCCL/AI libraries | unavailable on native Windows | out of scope |

## P0: establish trustworthy ROCm 7.2 evidence

### G00 — separate HIP 7.2 control build

Create `build-rocm72` with the same options and `AMDGPU_TARGETS=gfx1201` as
`build-rocm`. Preserve the HIP 7.1 build. Run the locked single/dual L0 and
coherence gate before enabling a new feature.

**Why first:** the knowledge base targets 7.2, but the current binary is 7.1.
Without this control every new result confounds compiler/runtime and feature.

### G02 — paired narrow Q8_0 GDN microbenchmark (closed 2026-08-30)

Added `scripts/research/g02_gdn_pair_mmvq_probe.cpp` (built with ROCm 7.2
`hipcc`, gfx1201, `build-rocm72/bin/g02-gdn-pair-probe.exe`). It measures the
exact GDN contract: pair of Q8_0 `5120 -> 48` matvecs, 48 layers, one warp
per row, K=160 Q8_0 blocks (34 B each).

| Metric | Two launches (A) | One fused two-output (B) |
| --- | ---: | ---: |
| GPU ms per layer pair | 0.001791 | 0.001366 |
| CPU enqueue ms per layer pair | 0.005728 | 0.002901 |
| Per-token GPU (48 layers) | 0.0860 ms | 0.0656 ms |
| Per-token CPU enqueue (48 layers) | 0.2749 ms | 0.1393 ms |

- Fused output is **bit-identical** to the two separate launches
  (`max_abs_diff=0`), CPU reference exact; kernel 29 regs, 0 shared.
- GPU-side saving: 0.0204 ms/token. CPU enqueue saving: 0.1357 ms/token.
- Against the L1 decode wall (~38 ms/token at 26 tok/s) the projected
  whole-token upside is **~0.05% (GPU) / ~0.35% (CPU-bound)** — below the 3%
  gate. Even the optimistic 1-2% estimate from the decode README is not
  reproduced by isolated measurement.

Verdict: **deferred with a hard reason.** Fusion does not clear the 3% gate;
the launch-count saving is real but negligible relative to the 38 ms decode
wall. Prefill stays untouched. The probe is kept so the estimate can be
rebuilt if a decode-bound profile later shows launch gaps dominating.

### G01 — RGP single-versus-dual capture (closed 2026-08-30: portable CLI cannot attach HIP)

Install RGP/RDP, capture the same short decode region on single ROCm0 and dual
ROCm1/ROCm0, then build a device-time table for MMVQ, GDN, FA, copies, and idle
queue gaps.

**Decision:** select the next kernel only from a material device-time/resource
signal. This prevents another broad geometry sweep.

Status 2026-08-30 (final) — RDTS 2026-05-28 (portable ZIP, ~394 MB) at
`C:\Users\krist\AppData\Local\Programs\GPUOpen\RadeonDeveloperToolSuite-2026-05-28-1806`;
elevated (UAC-approved) run confirmed:
- `RadeonDeveloperServiceCLI --port 27300` (elevated) + `RadeonDeveloperPanelCLI
  --remote-host 127.0.0.1 --remote-port 27300` → `Capture API initialized
  successfully`, both RX 9070 XT enumerated (Adrenalin 26.7.1, 2×15.9 GB).
- HIP attach FAILS in this package: with `-p llama-server` and with a pure
  HIP probe (`g06-hipblaslt-single-probe.exe`, hundreds of dispatches), no
  `.rgp` is produced and the CLI keeps waiting; `llama-server` is not in the
  default blocklist (verified via `--list-blocklist`).
- Likely cause: the portable ZIP lacks the installer `Modules/` artifacts —
  GUI `RadeonDeveloperPanel.exe` logs `No modules directory set. Skip loading
  dynamic modules` — so the API-capture client modules never initialize.
Verdict: G01 is NOT reachable through the portable RDP CLI on Windows. Device
time stays unavailable; `GGML_TRACE_*` stays the only per-kernel timing.

Next steps (only if a full installer/module set becomes available): re-run the
elevated CLI attach test; or verify manually in the elevated GUI panel
(Applications → add `llama-server`, enable profiling, start server, capture).
Do not re-download the portable ZIP for this gap.

### G02 — paired narrow Q8_0 GDN microbenchmark

The decode route already identifies 96 narrow `5120 -> 48` alpha/beta matvec
nodes per token. Compare the two existing launches with one two-output kernel,
including output equivalence and resource reporting.

**Decision:** integrate only if the isolated saving predicts at least the
lane's 3% whole-token gate. Keep prefill on its current route.

### G00 — HIP 7.2 control build and A/B (closed 2026-08-30)

Separate HIP 7.2 build `build-rocm72` versus HIP 7.1 control `build-rocm`,
same commit `75f7e87dc`, dual L0/L1 r1:

| Level | Metric | HIP 7.1 | HIP 7.2 | Delta |
| --- | --- | ---: | ---: | ---: |
| L0 | Prefill tok/s | 1498.02 | 1503.94 | +0.40% |
| L0 | Decode tok/s | 27.41 | 26.89 | -1.90% |
| L1 | Prefill tok/s | 1774.51 | 1772.67 | -0.10% |
| L1 | Decode tok/s | 26.26 | 26.00 | -0.99% |

Verdict: toolchain version does not influence the results beyond noise.
HIP 7.2 is a valid alternative build; HIP 7.1 remains the decode control.
Full record: [06_HIP72_ROLLOUT_CHECKLIST.md](06_HIP72_ROLLOUT_CHECKLIST.md).

## P1: bounded runtime/library probes

### G03 — HIP 7.2 P2P capability refresh

Re-run the standalone bidirectional capability/correctness probe once on the
new SDK/driver combination. If either direction reports unsupported, stop.
Never override the runtime result.

### G03 — HIP 7.2 P2P capability refresh (closed 2026-08-30)

Re-built `scripts/research/rocm_peer_copy_probe.cpp` against HIP SDK 7.2
(`build-rocm72/bin/rocm-peer-copy-probe.exe`, clang-cl 7.2) and ran
`--capabilities` on idle GPUs:

```text
device=0/1: AMD Radeon RX 9070 XT gfx1201
p2p src=0 dst=1  can_access=0  access_supported=0
p2p src=1 dst=0  can_access=0  access_supported=0
```

Verdict: no change after the driver/runtime update. The E295 copy ladder is
not started; peer copy remains closed, host staging stays the only cross-GPU
route. Reopening condition: a future HIP/driver change with `can_access=1`
in at least one direction.

### G04 — hardware queue count A/B (closed 2026-08-30)

HIP 7.2 build, same model/context/KV contract as G00, L0/L1, one shot:

| `GPU_MAX_HW_QUEUES` | L0 prefill | L0 decode | L1 prefill | L1 decode |
| --- | ---: | ---: | ---: | ---: |
| 4 (default, G00) | 1503.94 | 26.89 | 1772.67 | 26.00 |
| 1 | 1475.30 (-1.90%) | 24.64 (-8.36%) | 1758.70 (-0.79%) | 26.65 (+2.51%) |
| 2 | 1495.00 (-0.59%) | 27.39 (+1.85%) | 1768.37 (-0.24%) | 25.91 (-0.36%) |

Verdict: no coherent pattern; queue=1 clearly hurts L0 decode, queue=2 is
noise. The default queue count (4) stays. No code change is justified.

### G05 — async staging trace, not redesign

Test `GPU_MAX_HW_QUEUES=1,2,4` on the unchanged HIP 7.2 dual control. Record
prompt, decode, correctness, and residency. This is a no-code runtime probe;
the expected upside is low because each token's layer halves are serial.

### G05 — async staging trace, not redesign

Use `GGML_TRACE_ROCM_ASYNC_CROSS_DEVICE_STAGE` with the existing opt-in path to
verify slot reuse, tensors, and fallback frequency. Only if the trace shows a
registration/fallback problem should an env-gated `hipHostMallocPortable`
allocation be tested. The prior async-staging decode result was neutral/negative.

### G05 — async staging trace (closed 2026-08-30)

Ran L0 with `GGML_ROCM_ASYNC_CROSS_DEVICE_STAGE=1` and
`GGML_TRACE_ROCM_ASYNC_CROSS_DEVICE_STAGE=1` on the HIP 7.2 build.

Trace facts (172 staging events, no synchronous fallback):

- Only two inter-device tensors: `l_out-32` and `attn_inp_kq_mask (copy)`.
- All 8 pinned slots are reused evenly (21-22 times each); no fallback to
  the synchronous path (`synchronous fallback` / `staging unavailable` absent).
- Decode-sized `l_out` copy is 20,480 bytes; prefill spikes 11-21 MB
  (one-shot K/V cache), as expected.
- Result L0: 1472.72 ptps / 26.14 dtps vs G00 HIP 7.2 default
  1503.94 / 26.89 (decode -2.8%). Consistent with the previously closed
  event-chained staging route (decode was neutral/negative).

Verdict: the async path works correctly and uses all slots, but does not
improve decode. No `hipHostMallocPortable` experiment is justified;
the route stays closed for decode.

### G06 — exact-shape hipBLASLt prefill probe (closed 2026-08-30)

First prove that the Windows 7.2 package contains supported `gfx1201` kernels.
Then run offline tuning on exact dense prefill shapes and include conversion
and workspace overhead. Do not use this result to reroute N=1 quantized decode.

Status 2026-08-30 (extended to maximum) — `7.2/bin/hipblaslt/library/` contains
`TensileLibrary_..._gfx1201.co/.dat` and `Kernels.so-000-gfx1201.hsaco`, so
supported kernels are present. E249's grouped API still returns
"no algorithms" (unchanged), but the **plain** `hipblasLtMatmul` path
(`Gemm` in `hipblaslt-ext.hpp`) returns 842 heuristic algorithms and finds
supported `ISA1201` kernels.

`scripts/research/g06_hipblaslt_single_probe.cpp` extended (HIP 7.2):
- `--max-algos 0` now scans **all 842 supported algorithms** (phase 1, iters 4),
  then re-benchmarks the top-N winners (phase 2, warmup 4 / iters 32 — the
  offline-tune result a runtime backend would cache).
- `--bias` adds a fused f32 bias epilogue; `--compute-fast16`/`--output-f16`
  check alternate contracts; `--tune-splitk`/`--tune-wgm` force
  `GemmTuning` split-K / workgroup mapping; `--workspace-mb` varies the
  workspace budget.

The first exploratory shape table used matrix dimensions in probe order rather
than the exact `runtime_compute.inc` `cublasGemmEx(m,n,k)` order. It proved that
`ISA1201` solutions exist, but it was not a valid wall-time ceiling. A fresh
route trace corrected the active Q6_K prefill forms:

- attention QKV: `m=10240,n=1024,k=5120`;
- FFN down: `m=5120,n=1024,k=17408`.

Exact full-scan results (GPU0, workspace 64 MB, f16 inputs, f32 output and
compute):

| runtime GEMM (m·n·k) | hipBLASLt | rocBLAS default | best rocBLAS solution | Δ vs default | Δ vs best rocBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5120×1024×17408 | **0.9820 ms** (88251) | 1.1359 ms | 1.0428 ms (88667) | **−13.6%** | **−5.8%** |
| 10240×1024×5120 | **0.5825 ms** (88255) | 0.7667 ms | 0.6066 ms (88138) | **−24.0%** | **−4.0%** |

The temporary default-off runtime gate used `GGML_CUDA_HIPBLASLT=1`, a
`gfx1201` guard, 64 MiB workspace per device, exact-shape solution caching,
and fail-closed fallback to `hipblasGemmEx`. Trace run
`g07-blaslt-compiled-cand-l0-r1` proves 93 dispatches of algo 88251 and 72 of
algo 88255 across the dual-GPU L0 request.

Correctness passed before the speed decision. The long-prompt greedy control
and candidate outputs are byte-identical (`31` bytes), SHA-256
`33859FFC4CD7AF5D4C5869753061A776BCB14CAF0B2919957B5E7D7929D81D2C`.

No-trace neighbor A/B on the rebuilt HIP 7.2 binary:

| Run | Prompt tok/s | Decode tok/s | Aggregate TPS | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| control r1 (`g07-blaslt-compiled-ctl-l0-r1`) | 1439.2807 | 24.9948 | 11.9935 | 2775.692 |
| candidate (`g07-blaslt-compiled-cand-notrace-l0-r1`) | 1420.8122 | 24.7056 | 11.8468 | 2811.772 |
| control r2 (`g07-blaslt-compiled-ctl-l0-r2`) | 1427.8306 | 24.5364 | 11.8380 | 2797.951 |

Against the two-control mean, candidate prompt is `−0.89%`, decode `−0.24%`,
aggregate `−0.58%`, and TTFT `+0.90%`. Even an optimistic serialized sum of
the standalone point deltas is only `27.6 ms`, below `0.99%` of the `2786.82
ms` control TTFT; dual-device overlap lowers the exposed ceiling further.

Verdict: **closed — real point wins, no wall win.** Do not promote or retain a
runtime hipBLASLt dependency for these Q6_K buckets. The proxy/CMake linkage
was removed and the normal `hipblasGemmEx` server rebuilt. Keep only
`scripts/research/g06_hipblaslt_single_probe.cpp` as an offline diagnostic.
Forced Q6_K MMQ also regressed prefill by `13.6%`, so the existing Q6_K
selector remains unchanged.

### G07 — setup cost isolation and rocBLAS solution-index (closed 2026-08-30)

Why the hipBLASLt wall gate failed. The standalone probe was extended with a
runtime-mirroring setup bench (`--setup-bench`):

- extension wrapper per call (`setProblem` + `isAlgoSupported` + `initialize`)
  adds `0.092-0.142 ms` wall over the hot `gemm.run()`;
- raw cached `hipblasLtMatmul` with one-time descriptors adds only
  `0.040-0.055 ms` (device `0.6640` vs hot `0.6239` ms on
  `10240x1024x5120`), but the point win itself is only `0.09-0.10 ms` there,
  so even cached raw hipBLASLt cannot overtake rocBLAS default on the
  N=1024 Q6_K buckets by more than noise after call overhead.

Next, the same nine hot Q6_K shapes were screened with `rocblas_gemm_ex`
`solution_index` (full 4024-solution scan, device 0 and device 1):

| m·n·k | rocBLAS default | best solution | Δ | solution id |
| --- | ---: | ---: | ---: | ---: |
| 5120x549x17408 | 0.8409 | 0.7034 | **−16.3%** | 88114 |
| 5120x919x17408 | 1.1739 | 1.0003 | **−14.8%** | 88027 |
| 5120x1024x17408 | 1.1887 | 1.0461 | **−12.0%** | 88667 |
| 10240x549x5120 | 0.5642 | 0.3904 | **−30.8%** | 88031 |
| 10240x919x5120 | 0.7570 | 0.5953 | **−21.4%** | 88123 |
| 10240x1024x5120 | 0.7598 | 0.6024 | **−20.7%** | 88113 |
| 1024x549x5120 | 0.0634 | 0.0433 | **−31.7%** | 88483 |
| 1024x919x5120 | 0.0843 | 0.0688 | **−18.4%** | 88228 |
| 1024x1024x5120 | 0.1211 | 0.0712 | **−41.2%** | 88185 |

`g06-rocblas-solution-scout` re-checked the two headline N=1024 forms on
device 1: 88113/88667 remain faster than rocBLAS default there too, so the
tuned kernels are not device-specific.

A default-off runtime gate (`GGML_EXPERIMENTAL_ROCBLAS_Q6_SOLUTIONS=1`,
separate rocblas handle per device, fail-closed fallback to hipBLAS) applied
the nine ids. Trace `g07-rocblas-sol-trace-l0-r1` confirms 315 calls, all
`status=0`, exactly matching the scanned shapes. Long-prompt greedy output
is byte-identical to the control (SHA-256
`D1506E2C4B603C1C7DD5CF45F274CF5AF08F313790FB839D1A58358A3AB53075`).

r3 wall ladder on the locked L0 lane (no trace, rebuilt 7.2 binary):

| Variant | Prompt tok/s | Decode tok/s | Aggregate TPS | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| control r1 | 1446.5691 | 25.0042 | 12.0272 | 2761.707 |
| candidate r1 | 1442.1435 | 25.2802 | 12.0713 | 2770.182 |
| control r2 | 1449.8850 | 25.1255 | 12.0695 | 2755.391 |
| candidate r2 | 1438.2284 | 25.1622 | 12.0273 | 2777.723 |
| control r3 | 1438.4065 | 25.1625 | 12.0282 | 2777.379 |
| candidate r3 | 1436.1892 | 25.2228 | 12.0322 | 2781.667 |

Against the three-control mean (1444.95/25.10/12.04, TTFT 2764.83 ms):
candidate prompt `−0.42%`, decode `+0.50%`, aggregate `+0.02%`, TTFT
`+0.42%`. The optimistic serialized ceiling across all nine forms is
`26.1 ms` ≈ `0.95%` of TTFT and is below the r1-r3 noise band; dual-device
overlap lowers the exposed ceiling further. Every candidate run is below
every control run on prompt, so there is no consistent positive transfer.

Verdict: **closed — no runtime path wins on this lane.** Point wins are real
and device-stable, but their combined wall ceiling is under 1% for Q6_K
prefill. The runtime source probe (including rocblas include/handle/destroy
and the dispatch branch) was reverted; the normal `hipblasGemmEx` server was
rebuilt. Keep both standalone probes as offline diagnostics. A positive Q6_K
step must come from a fused/dequant-free compute body, not library solution
selection; forced MMQ was already measured `−13.6%`.

### G08 — Q4_K prefill route: MMQ vs dequant+hipBLAS (closed 2026-08-30, promoted)

Route census on the locked L0 (3995 prompts + 64 decode tokens) from
`g07-route-l0-r1`: `mul_mat_q_direct|q4_K` is 1430 calls and, in per-node
sync measurement, `94%` of all MMQ time; `mul_mat_vec_q_direct|q4_K` decode is
1310 calls and reports `100%` occupancy.

Per-node resource trace (`GGML_TRACE_MMQ_TIMING` + `_RESOURCES`, PRE_SYNC)
shows every Q4_K MMQ prefill form bounded to `1` block/SM: `nbytes_shared`
`55744-57856` of `65536` (`85-88%`), `regs=212-236`, `occupancy_pct=12.50`,
`waves_per_sm=8`. MMVQ decode instead uses `occupancy_pct=100` with `7168`
bytes shared (`regs=61-76`), so latency hiding is not the decode problem.

Standalone hipBLASLt/rocBLAS measurements of the exact MMQ shapes (f16
inputs; GEMM volume `m*n*k`):

| Shape (m·n·k) | MMQ device ms | hipBLASLt best ms | |
| --- | ---: | ---: | --- |
| 17408x1024x5120 | 2.031 | **1.038** | ~2x faster |
| 6144x1024x5120 | 1.659 | **0.354** | ~4.7x faster |
| 10240x1024x5120 | 1.278 | **0.597** | ~2.1x faster |

An env-only A/B (`GGML_MMQ_RDNA4_Q4K_MAX_NE11=256`, no source change) moved
the Q4_K prefill forms with `ne11=549/919/1024` onto the existing
dequant+`hipblasGemmEx` path; route trace confirms `cublas_backend|q4_K`
(1430 calls, all prefill) and unchanged `mul_mat_vec_q_direct|q4_K` decode.
Greedy output is byte-identical (SHA-256
`D1506E2C4B603C1C7DD5CF45F274CF5AF08F313790FB839D1A58358A3AB53075`).

r3 neighbor ladder (Qwen3.8-27B Q4_K_M, dual ROCm1/ROCm0, q8_0 KV):

| Variant | Prompt tok/s | Decode tok/s | Aggregate | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| control r1 | 1443.87 | 24.85 | 11.98 | 2766.88 |
| candidate r1 | 1497.00 | 25.22 | 12.29 | 2668.68 |
| control r2 | 1439.22 | 25.15 | 12.03 | 2775.81 |
| candidate r2 | 1488.20 | 25.13 | 12.24 | 2684.45 |
| control r3 | 1442.58 | 25.19 | 12.05 | 2769.34 |
| candidate r3 | 1482.78 | 25.10 | 12.20 | 2694.26 |

Versus the three-control mean (1441.89/25.06/12.02/2770.68): candidate
prompt `+3.29%`, decode `+0.35%`, aggregate `+1.83%`, TTFT `−3.18%`.
Every candidate run is faster than every control run. Cross-check on
Qwen3.6-27B Q4_K_M: `1460.59 -> 1493.10` prompt tok/s (`+2.23%`). Current
build with the new default (`256`) stays in the same band
(`g08-q4k-promote-l0-r1`: `1476.52 / 24.99`).

Verdict: **promoted.** `ggml_rdna4_q4k_mmq_max_ne11()` default lowered from
`1024` to `256` (mmq.cu, env override preserved), so RDNA4 Q4_K large-batch
prefill follows the dequant+hipBLAS path while MMQ remains for `ne11<=256`
and for the small-batch Q4_K_S behavior E070 documented. Q5_K retains the
`ne11<=1024` gate until it is measured separately.

### G09 — f32 cublas GEMM on RDNA4: f16 inputs + f32 accumulate (closed 2026-08-30, promoted)

Route census of the locked L0 (`g08-f32-cublas-trace-l0-r1`,
`GGML_TRACE_CUBLAS_SPLIT_TIMING`) showed `832` `cublas_backend|f32` calls
totalling about `628 ms` device time, concentrated on:

- `48x5120` GDN projections (ssm_alpha/ssm_beta/...): 288 calls at n=1024
  (`0.59 ms avg`, both devices),
- `64x64` / `256x256` rotation K/V (RoPE): 32+48+24 calls at n=384/98304/24576,
- repeated per-layer for 36 layers.

All of those used the **pure f32 GEMM** path (`path=f32`) because
`use_fp16` did not admit `GGML_TYPE_F32`; RDNA4 has no f32 MFMA, so the
f32 kernels run at roughly `0.4-0.85 TF/s` there. Standalone hipBLASLt on
the f16-input equivalent (`48x1024x5120`) is `0.0265 ms` versus
`0.59 ms` measured for f32 — an order-of-magnitude gap on the input
conversion plus MFMA GEMM.

A default-off prototype routed RDNA4 f32 GEMMs through the existing
dequant-style machinery: `to_fp16` conversion of src0 and src1, then
`cublasGemmEx` f16 inputs with `CUBLAS_COMPUTE_32F` and f32 output
(route trace confirms `path=fp16 compute=32f` for all 832 f32 rows).

Wall A/B (Qwen3.8-27B Q4_K_M, dual ROCm, L0, no tracing):

| Variant | Prompt tok/s | Decode tok/s | Aggregate | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| control r1 | 1487.29 | 25.11 | 12.23 | 2686.09 |
| candidate r1 | 1557.59 | 25.19 | 12.54 | 2564.86 |
| control r2 | 1473.87 | 25.21 | 12.19 | 2710.55 |
| candidate r2 | 1561.77 | 25.21 | 12.56 | 2558.00 |
| control r3 | 1479.58 | 25.25 | 12.23 | 2700.09 |
| candidate r3 | 1558.82 | 25.20 | 12.54 | 2562.83 |

Versus the three-control mean (1480.25/25.19/12.21/2698.91): candidate
prompt `+5.35%`, decode `+0.04%` (noise), aggregate `+2.71%`,
TTFT `−5.08%`; every candidate run is faster than every control.
Cross-check Qwen3.6-27B Q4_K_M `1498.79 -> 1569.98` (`+4.75%`).
Greedy output is byte-identical (SHA-256
`D1506E2C4B603C1C7DD5CF45F274CF5AF08F313790FB839D1A58358A3AB53075`) for
both routes. Env escape works: `GGML_ROCM_F32_GEMM_F16=0` returns
`1482.31 prompt tok/s` in the control band; the promoted default build
(`g08-f32gemm-promoted-check`) gives `1568.72` without any env.

Verdict: **promoted.** `runtime_compute.inc` now routes RDNA4
`src0=GGML_TYPE_F32` cublas GEMMs to f16 inputs with f32 accumulate by
default (`GGML_ROCM_F32_GEMM_F16=0` for rollback). The remaining `f32`
decode route `mul_mat_vec_f_direct|f32` (`5120x48 n=1`) still uses a
specialized f32 MMVF kernel and is not covered by this change; a
conversion + MMVF half-path is the next candidate for decode, not
prefill.

### G10 — fused two-output decode MMVF pair for GDN (closed 2026-08-30, opt-in only)

The remaining decode f32 lane is the paired narrow GDN matvecs
`ssm_alpha` + `ssm_beta` (`5120 -> 48`, one shared `attn_norm` input,
48 GDN layers, n=1). Per-node MMVF timing (`GGML_TRACE_MMVF_TIMING` +
`GGML_HIP_DISABLE_GRAPHS=1`) shows 96 `5120x48` calls per token at
~0.069 ms average (~6.6 ms/token) and 48-96 `256x256 n=4` Kv rotation
calls at ~0.076 ms. A fused two-output launch (new
`ggml_cuda_mul_mat_vec_f_pair()` kernel in `mmvf.cu`, graph-level pair
detection in `runtime_graph.inc` with concurrent-event bookkeeping for
the skipped beta node) replaces each pair with one launch.

Isolated HIP probe (`scripts/research/g10_gdn_pair_probe.cpp`, gfx1201):
- deterministic bit-exactness 48/48 for both alpha and beta with the
  same inner-loop order as two separate `mul_mat_vec_f<float>` calls;
- per-pair GPU time `0.0038 ms` fused vs `0.0042 ms` as two launches
  (only ~9% saved — the geometry is launch/latency-bound, not
  compute-bound).

Server integration pitfalls found while validating:
- `cudaStreamSynchronize` trace fails on graph capture; capture must be
  disabled for MMVF timing;
- skipping the beta node silently broke the concurrent-event
  bookkeeping (fork/join events are emitted for the last skipped node
  by the main loop); the beta consume path must call
  `try_launch_concurrent_event(node)` itself.

Wall A/B (Qwen3.8-27B Q4_K_M, L0, greedy byte-identical
`D1506E2C4B603...`):

| Variant | Prompt tok/s | Decode tok/s | Aggregate |
| --- | ---: | ---: | ---: |
| ON r1 | 1570.42 | 25.096 | 12.5635 |
| ON r2 | 1563.85 | 24.360 | 12.3508 |
| ON r3 | 1566.12 | 25.062 | 12.5377 |
| OFF r2 | 1561.65 | 25.201 | 12.5546 |
| OFF r3 | 1563.94 | 24.673 | 12.4310 |

Means: ON 1566.80/24.84/12.48 vs OFF 1562.79/24.94/12.49 — prompt
`+0.26%` (prefill noise; the fusion only touches decode), decode
`−0.40%`, aggregate `−0.08%`. No measurable gain: with CUDA-graph
capture the per-node launch overhead is already minimal and the kernel
is latency-bound.

Verdict: **closed — no wall win; kept opt-in** (`GGML_ROCM_GDN_PAIR=1`,
default OFF) with the standalone probe for later work. Not promoted;
the launch-bound remainder of the decode MMVF lane is not a
kernel-resource problem.

### G11 — Q5_K/Q6_K single-token MMVQ decode body (closed 2026-08-31)

The locked UD/f8 L0 route (`Qwen3.8-27B-UD-Q4_K_M`, dual
`ROCm1,ROCm0`, 3995 prompt + 64 decode tokens) confirms that these are
real decode costs, but not occupancy-limited kernels. A PRE_SYNC/resource
trace over warmup + L0 decode reports:

- Q5_K: 9520 N=1 calls, robust median-sum `1238.00 ms`; hot shapes are
  fused `17408x5120` (`0.155 ms`), non-fused `6144x5120` (`0.099 ms`),
  and fused/non-fused `5120x17408` (`0.286/0.172 ms`).
- Q6_K: 1884 N=1 calls, robust median-sum `311.38 ms`; the output head
  `5120x248320` dominates at `1.782 ms` per call. Its 994.63 MiB Q6_K
  payload implies `585.3 GB/s`, about `91.4%` of the RX 9070 XT
  640-GB/s physical bandwidth.
- Both paths already report 100% occupancy and no LDS pressure: Q5_K
  uses 29 regs non-fused / 42 fused; Q6_K uses 26 / 36.

Five exact-body candidates were measured and removed:

1. shared-q8 gate/up pair streaming: weighted fused shapes ~`+0.4%`
  slower (individual Q5/Q6 buckets from `-2.8%` to `+3.0%`);
2. Q5_K reuse of `block_q8_1.ds.y`: not exact (it stores the original
  float sum, not the quantized subgroup sum) and decode `27.80` vs
  `28.28 tok/s`;
3. Q6_K regular packed-byte subtract instead of saturating subtract:
  byte-identical but `27.89` vs `28.24 tok/s`;
4. exact Q5_K subgroup-sum sidecar: byte-identical, but added loads and
  registers move weighted Q5 kernel time `1238.00 -> 1290.78 ms`
  (`+4.26%`), including `+7-11%` on the 5120x17408/10240 buckets;
5. compact fused Q5_K pair-dot (shared q8 loads and dot2, sequential
  weights): byte-identical and lowers fused regs `42 -> 39`, yet fused
  weighted time is `546.31 -> 551.46 ms` (`+0.94%`) and the short wall
  smoke is `27.23` vs `28.09 tok/s`.

ROCm 7.2 exposes `sdot8`, but it is an eight-way **int4** dot. Q5_K
weights multiplied by Q8 activations would require splitting Q8 into
two nibble planes, so it does not reduce the dot count and adds packing.

Verdict: **closed — keep the current MMVQ bodies and geometry.** No G11
runtime gate or kernel remains. Q6_K output is already at the bandwidth
ceiling; a material Q6 gain needs fewer weight bytes (a different model
or storage contract). Q5_K needs a load-time/prepacked weight layout
that removes scale/high-bit decode without materially increasing bytes;
nearby launch, occupancy, q8-reuse, sum-cache and packed-subtract edits
are exhausted.

## P2: use only after profiling admits them

| Candidate | Admission signal | Expected scope |
| --- | --- | --- |
| rocPRIM/hipCUB reduction | RGP shows a generic reduction/sort among expensive events | operation-specific |
| `hipMallocAsync` / pools | a later Windows SDK documents support, then allocation/fragmentation is measured | primarily load/startup |
| stream priorities | concurrent workload latency is the stated target | responsiveness, not max TPS |
| RGA offline analysis | RGP identifies a hot kernel and accepts its code object | kernel resource diagnosis |
| host registration of CPU-mapped weights | a real internal caller is wired and copy/access traffic is measured | residency/transfer experiment, high RAM risk |

## Closed or rejected without a material change

- forced P2P when `can_access=0`;
- generic graph rework or graph upload without a measured launch gap;
- managed/mapped host memory as replacement VRAM for hot weights;
- `ROCBLAS_USE_HIPBLASLT=1` as a decode optimization;
- RCCL, ROCProfiler, CK/AITER, MIOpen, or MIGraphX on native Windows;
- serialized/debug runs as performance evidence;
- combined changes that prevent attribution.

## Experiment record template

Each new item should record:

```text
ID / date / branch / commit
HIP SDK / compiler / Adrenalin / GPU order
model / context / batch / ubatch / KV / split
single changed mechanism
adjacent control and candidate prompt/decode
correctness and shutdown result
VRAM/Shared-memory behavior
device/profile evidence (if instrumented)
keep / reject / defer and reopening condition
```

The locked benchmark and promotion gates are defined in
[the decode lane](../decode/README.md).
