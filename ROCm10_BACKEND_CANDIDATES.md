# ROCm 10 backend candidates (audit 2026-09-07)

Audit of the installed ROCm 10 / TheRock stack (`/home/chris/rocm`,
HIP 7.15.26333, rocBLAS 5.6.x, hipBLASLt 1.4.x, LLVM/AMD clang 23.0.0git,
`gfx1201`) against the current ggml-HIP backend in this worktree. The
goal: list only capabilities that are (a) newly available versus the
Windows HIP 7.1/7.2 baseline and (b) have a plausible >=3% whole-lane
ceiling, with a concrete integration path.

## What changed in ROCm 10 (vs the Windows 7.1/7.2 baseline)

| Item | Windows 7.1/7.2 | ROCm 10 here | Evidence |
| --- | --- | --- | --- |
| Cache-policy (TH) on global loads for gfx1201 | NOT expressible by the compiler (`H80` blocked: `__builtin_nontemporal_load` silently dropped) | **expressible**: `global_load_u8 ..., th:TH_LOAD_NT` | `/tmp/nt_probe2-hip-amdgcn-amd-amdhsa-gfx1201.s` shows `th:TH_LOAD_NT` only for the `__builtin_nontemporal_load` version |
| `hipMallocAsync` / `hipMemPool` (stream-ordered) | Beta / Linux-only; deferred | **available** (`hipMallocAsync`, `hipMemPoolCreate`, `hipDeviceSetMemPool` in `hip_runtime_api.h`) | header audit |
| `hipStreamSetAttribute` access-policy window | n/a | available | header audit |
| hipBLASLt ext APIs on gfx11xx/gfx12xx | added in hipBLASLt 1.2.0 (ROCm 7.2) | available (`hipblaslt-ext.hpp`, `hipblaslt-ext-op.h`) | header audit |
| `hipMemAdvise` | only coarse-grain opt-in via env | available; already used for managed-memory hint | `runtime_device.inc:345` |

## Candidate 1 (highest priority): H80 reopen — streaming KV loads in native FP8 FA

- **Why it matters:** W09/W12: the native FP8 WMMA decode kernel re-reads the
  whole KV every token (3 GiB/GPU at 49K, 6 GiB at 98K), use-once per token;
  L2 (64 MB) is the shared resource with the heavier Q4_K_M weight stream.
  Marking the KV `global_load_u8` stream as non-retained could preserve L2
  residency for weights.
- **History:** H80 (PHASE2_PLAN 2.1, `W09`) was closed as *BLOCKED by
  toolchain* on ROCm 7.1: `__builtin_nontemporal_load` was silently dropped
  and llvm-mc accepted no `th`/`glc`/`slc`/`nt` modifier. The text said
  "Deferred until a toolchain with an expressible TH field".
- **ROCm 10 proof:** compiled a gfx1201 kernel with and without
  `__builtin_nontemporal_load`:
  - base: `global_load_u8 v2, v[2:3], off`
  - nt: `global_load_u8 v2, v[2:3], off th:TH_LOAD_NT`
  - assembled object bytes (`/tmp/th_real.o`, `llvm-mc -filetype=obj`):
    NT `EE04007C 00100002 00000002` vs base `EE04007C 00000002 00000002` —
    the TH field bit is actually encoded (not just printed in the asm).
  (source `/tmp/nt_probe2-hip-amdgcn-amd-amdhsa-gfx1201.s`).
- **Integration path:** the KV loads live behind `wmma::load_matrix_sync`
  in `fattn-wmma-f16.cu` (`fattn_wmma_f16_direct_q8_rdna4`); rocWMMA does not
  expose a cache-policy option, so a prototype must replace the K/V
  `load_matrix_sync` with a manual fragment build using
  `__builtin_nontemporal_load` for the fp8 (u8) path (KQ and V legs), gated
  behind an env var (e.g. `GGML_HIP_F8_KV_STREAMING=1`), default off.
- **Risks:** W09 counter-hypothesis — L2 may already be effectively
  streaming the 48x-oversized working set, so hints may change little. The
  WMMA fragment layout must be preserved exactly (correctness gate first).
- **Next step if accepted:** one bounded prototype, `test-backend-ops`
  FP8 FA correctness, then adjacent 49K A/B / 98K confirmation.

## Candidate 2 (lower priority): stream-ordered `hipMallocAsync` / pools

- Available in ROCm 10 and currently unused (`ggml_cuda_device_malloc` in
  `runtime_device.inc:338` uses plain `cudaMalloc`).
- Prior decision in `docs/research/rocm72/05_GGML_GAP_MAP.md`: deferred
  "until Windows support"; that gate no longer applies on Linux.
- Expected ceiling: load/startup and buffer fragmentation only, not the
  steady-state decode token. Per `01_RUNTIME_WINDOWS.md`: "neither removes
  the serial dependency between two GPUs for one token".
- Recommend only after a trace shows a real allocation in the steady-state
  path; otherwise keep off.

## Candidate 3 (diagnostic only): hipBLASLt exact-shape / ext APIs

- hipBLASLt 1.4.1 has `gfx1201` code objects and the ext API set for
  gfx12xx, but the prior G07/exact-shape runtime integration on ROCm 7.2
  was **rejected**: per-call `setProblem + isAlgoSupported + initialize`
  cost `0.09-0.14 ms` and ate the point win; even raw `hipblasLtMatmul`
  kept `0.04-0.06 ms` overhead. The current 1.4.x `initialize` may be
  cheaper, but the prior conclusion stands: keep as offline diagnostic,
  not a production selector, unless a new throughput measure shows
  overhead below the point gain.

## Candidate 4 (exploratory): access-policy window on KV/weight streams

- `hipStreamSetAttribute` + `hipLaunchAttributeAccessPolicyWindow` exist.
- Would target the same L2-sharing problem as Candidate 1, but the window
  API applies to a stream/launch granularity and is less precise than a
  per-load TH bit. Only after Candidate 1 shows a material effect.

## Candidate 5 (done, no action): P2P

- Linux P2P is already the default path and our L1-L3 A/B (P2P ON vs OFF)
  shows P2P-ON is 30-33% faster decode on ROCm 10. Nothing new to enable;
  keep `GGML_CUDA_NO_PEER_COPY_RUNTIME=1` as the documented rollback.

## What was NOT adopted / rejected already

- FP8 for dense matmul (not FlashAttention): file inventory shows
  `GGML_TYPE_F8_E4M3` wiring only for convert/copy/FA and set-rows, not for
  `gemm`/`mul_mat`; the FFN weight stream is quantized Q4_K_M, and a
  quantized-to-FP8 dense path would need conversion that costs more than
  the GEMM saves (same conclusion as the rocBLAS materialization note in
  `03_KERNELS_LIBRARIES.md`).
- rocWMMA is already used for Flash Attention; the FP8 WMMA body is enabled.
- Q4_K prefill MMQ vs dequant+hipBLAS route is already settled by G08.
- rocBLAS solution-index selection was rejected (G07/D100).

## Suggested next action

Run the Candidate 1 (H80) bounded prototype: manual nontemporal KV loads in
the gfx12 FP8 WMMA decode kernel, env-gated, correctness-tested with
`test-backend-ops` (FP8 FA), then 49K adjacent A/B and 98K confirmation.
This is the only item on the list with a modeled >=3% whole-lane ceiling and
a hard prior "blocked by toolchain" state that ROCm 10 actually changed.
