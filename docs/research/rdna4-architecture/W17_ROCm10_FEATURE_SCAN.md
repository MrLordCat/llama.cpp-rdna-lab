# W17: ROCm 10.0 feature scan for gfx1201 decode candidates

Date: 2026-09-07

Scope: after "candidates must be ROCm-10-specific" instruction - scan what the
new toolchain/hardware stack actually adds, and whether it is useful for the
current decode hot paths (MMVQ weight stream, FATTN FP8 K/V, GDN state).
Source-only + compile/ISA probes; no GPU benchmark. Local SDKs compared:
`/home/chris/rocm` (10.0.0) and `/home/chris/rocm714` (7.14.1).

## Environment

| item | ROCm 10.0.0 | ROCm 7.14.1 |
| --- | --- | --- |
| LLVM | 23.0.0git `8f497e0992fb7513f7f78a6f6b6f1056c375e961` | 23.0.0git `f30ae3e6b680ad23dcf522e668c6d4c5c90838d0` + `PATCHED:440716f8b87` |
| rocWMMA | 2.2.1 | 2.2.1 |
| hipBLASLt | 1.4 | 1.4 |
| HIP runtime | amdhip64 7.15.26333 | core-7.14 |
| rocprofv3 | present (`/home/chris/rocm/bin/rocprofv3`) | not checked |

Both SDKs use the same LLVM 23 line, so "what the new version gives" is
mostly about expressibility/ISA/runtime, not a new major LLVM.

## Candidate scan

### 1. Vector `th:TH_LOAD_NT` (b64/b128) - EXPRESSIBLE, MEASURED-REJECTED

- `llvm-mc` accepts `global_load_b64 ... th:TH_LOAD_NT` on gfx1201.
- C++ probe (ROCm 10 clang): `__builtin_nontemporal_load` on `uint64_t`
  emits `global_load_b64 v[0:1], v2, s[0:1] th:TH_LOAD_NT`; on native
  `float4`-like `ext_vector_type(4)` it emits `global_load_b128 ... th:TH_LOAD_NT`.
- Same result on 7.14.1 (LLVM 23); on ROCm 7.1 it was silently dropped - so
  it is an LLVM-23 / ROCm-10-line feature, not 10.0-only.
- H80 already measured the same underlying path (KV reads marked NT):
  FP8 prefill 1884.37 -> 1353.35 tok/s (-28.2%), decode 26.27 -> 20.27
  (-22.8%), server run reverted. The prototype note confirms LLVM
  collapsed the manual scalar loads into `global_load_b64 ... th:TH_LOAD_NT`
  in the produced code - i.e. the regression is not a "scalar vs vector"
  artifact; marking KV NT is negative regardless of width.
- VERDICT: availability confirmed, adoption **closed** (measured negative).

### 2. Prefetch instruction ISA - ABSENT

- Probing gfx1201 in llvm-mc: all forms FAIL - no `s_prefetch`,
  `global_prefetch`, `v_prefetch`, `s_prefetch_l1/l2`.
- `__builtin_prefetch` lowers to LLVM `llvm.prefetch.p1` (works), but W16
  measured implementing it = -0.64% decode. Hardware exposes no distinct
  prefetch op on gfx1201; the LLVM intrinsic lowers to ordinary loads.

### 3. `hipEventDisableTiming` / coalesced `hipEventRecord` (release note "Improved HIP performance")

- ggml HIP uses events only in debug instrumentation
  (`runtime_graph.inc` `trace_graph_device_timing`, env-gated); the
  production decode path does not record timing events. No hot-path
  benefit; no adoption.

### 4. `hipMemGetDefaultMemPool` (CUDA parity, release note)

- API only; ggml-hip allocations do not route through the default pool.
  No measured performance path; not a decode candidate (memory management,
  not compute).

### 5. Cluster launch attributes (`hipLaunchAttributeClusterDimension`,
  `hipFuncAttributeCluster*`, `hipClusterSchedulingPolicy`)

- Present in both SDK headers; `clusterLaunch` field exists in device props.
  Not new in 10.0 vs 7.14 (identical headers), runtime value unverified here.
- Applicability to MMVQ: blocks cover disjoint output rows (row0), so
  weight reads are disjoint per block; the shared operand (`y`) is small and
  already cached. No candidate without a geometry change that would need a
  prototype.

### 6. `hipDeviceResource` / SM partitioning APIs (new difference in 10.0 headers)

- New APIs: split SM resources into groups; intended for Instinct GPU
  partitioning / MIG-like use. Single-process dual-GPU decode does not
  partition CUs. Not a candidate.

### 7. Composable Kernel a8w8 GEMM improvements (release note)

- Targeted at AMD Instinct MI355X long-sequence GEMM shapes; not gfx1201,
  and our decode lane is MMVQ (`ncols_dst=1`), not a GEMM. Not a candidate.

### 8. ROCprofiler-SDK: HIP graph per-node attribution (`rocprofv3 --hip-graph-trace`) - INSTALLED, USEFUL (instrument)

- Release note + verified CLI in 10.0: emits one record per `hipGraphLaunch`
  including `graph_exec_id` and `kernel_dispatch_count`; with `--kernel-trace`
  dispatches are attributed to graph nodes. This is the only immediately
  useful ROCm-10-specific addition for our research pipeline: per-node
  GPU time on decode graphs (D100-style) - without it, decode graph node
  attribution is not available through the old profiler.
- Adoption: opt-in tooling integration (not a runtime kernel change); helps
  future MMVQ/weight-stream and GDN device-trace candidates stay inside the
  ROCm 10 validation loop.
- VERDICT: CANDIDATE (diagnostic tooling).

### 9. Python API for rocprof-trace-decoder (ATT/SQTT)

- Available (`rocprof-trace-decoder` cmake package in 10.0). Lower priority
  than 8: useful for SQTT analysis after per-node attribution finds a hot
  kernel. Instrument candidate.

## Instrumentation rollout (adopted, first device profile 2026-09-07)

Adopted item 8. A second run (short diagnostic: ctx 4096, 19 prompt tokens /
11 generated, f8_e4m3 KV, flash on, spec none, ROCm1,ROCm0 -sm layer,
two devices) produced a working device-level kernel profile.

Invocation (validated):

```bash
rocprofv3 -d <outdir> -o <name> --hip-graph-trace --hip-trace --kernel-trace \
    --stats --summary -D -u msec --summary-output-file <outdir>/summary.txt \
    --output-format json -- <llama-server ...>
# stop the server with SIGTERM; only then is the summary flushed.
python3 scripts/research/rocprof_summary_top.py <outdir>/summary.txt --top 20 --graph
```

Caveats learned:
- The result is written on process exit (signal 15); the profiler flushes on
  TERM, and the server needs a second TERM after that in practice.
- `--output-format json` produces a large file (~70 MB for this short run)
  because of `--hip-trace`; for summary-only use `--stats --summary -D`
  without `--output-format json`.
- A single `summary.txt` contains a KERNEL_DISPATCH table **and** an
  all-domain table after it; `rocprof_summary_top.py` stops at the first
  section boundary (without it every kernel row is double-counted).

First results (kernel time only, includes prompt+decode of the short run):

| kernel group | total ms | % of 353.42 | calls |
| --- | --- | --- | --- |
| `mul_mat_vec_q` (MMVQ) q4_K fused ncols=1 | 113.64 | 32.15 | 779 |
| MMVQ q4_K non-fused ncols=1 | 38.20 | 10.81 | 791 |
| MMVQ q6_K non-fused ncols=1 | 28.70 | 8.12 | 233 |
| `mul_mat_q` q4_K ncols=16 (prompt MMQ) | 27.67 | 7.83 | 286 |
| MMVQ q6_K fused ncols=1 | 27.38 | 7.75 | 226 |
| MMVQ q4_K ncols=2 | 26.22 | 7.42 | 286 |
| MMVQ q5_K + q6_K ncols=2 + q5/MMQ residue | ~35.5 | ~10.1 | - |
| `rms_norm_f32` | 7.39 | 2.09 | 1161 |
| `k_get_rows_float` | 5.57 | 1.57 | 882 |
| `gated_delta_net_cuda<128,...>` | 5.35 | **1.51** | 432 |
| `__amd_rocclr_copyBuffer` | 5.03 | 1.42 | 716 |
| `quantize_q8_1` | 4.27 | 1.21 | 2762 |
| `flash_attn_ext_f16<...>` | 2.00 | **0.56** | 144 |

Interpretation:
- MMVQ family (all row/dot variants) = **~72.6% of kernel time**; the
  weight-stream / decode MUL_MAT direction (W13/W16) is the correct primary
  target - not a misread of the trace.
- `gated_delta_net` = 1.51% on-device: the W12 trace 13.5% was
  sync-inflated, W14 modeled ~5-8% worst-case, measured real share is even
  lower. no GDN prototype priority.
- `flash_attn_ext_f16` = 0.56%: the FP8 K/V WMMA path is no longer a
  hot spot under this contract; H80 negative result stands.
- `__amd_rocclr_copyBuffer` (716) + `hipMemcpyAsync` (1150 calls, 51.8% of
  HIP_API time) are the next visible "other" cluster: copy/quantize motion,
  not compute - a candidate only if MMVQ gain is exhausted.

## Summary

- The new version's *performance* surface for gfx1201 decode is small:
  vector TH hints (expressible, but the only measured use, H80, regressed)
  and no prefetch ISA. Everything else in the 10.0 release notes is
  tooling, memory-pool parity, or Instinct-only GEMM.
- Per the ROCm-10-specific requirement, the next adoptable candidate is the
  **profiler per-node graph attribution (item 8)**, then SQTT decoding
  (item 9) - both tooling, both useful to keep later decode candidates
  (weight stream, GDN device trace) validated under ROCm 10.
- No new compute-kernel candidate from ROCm 10 itself; if the user wants
  compute work next, it must remain an existing C1-family follow-up
  (weight-stream BW) tested on ROCm 10 and marked as such.
- TOOLING ADOPTED: `scripts/research/rocprof_summary_top.py` (rank
  kernels / HIP_GRAPH from rocprofv3 summary); first device profile
  recorded above.
