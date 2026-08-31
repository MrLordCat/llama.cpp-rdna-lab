# Kernels, precision, and ROCm libraries

## Start with the workload shape

ROCm's performance guide separates compute-bound, memory-bound, and
overhead-bound work. Qwen3.8 crosses all three categories:

- large-batch prefill contains GEMM-like, compute-heavy shapes;
- single-token quantized decode is primarily weight-streaming matrix-vector
  work, mixed with recurrent and attention kernels;
- thousands of nodes per token make launch/dependency overhead relevant, but
  HIP graphs already reduce the host-launch portion.

A library that improves GEMM is therefore a prefill candidate first. It is
not evidence of faster N=1 decode.

## RDNA4 execution model

HIP exposes threads, wavefronts, workgroups, registers, LDS, caches, and global
memory. RDNA uses native wave32. More waves are useful only while registers,
LDS, memory bandwidth, or dependencies do not become the actual limit.

The useful kernel questions are concrete:

- Are global loads coalesced and wide enough?
- Is the kernel limited by memory bandwidth or instruction issue?
- Do VGPR or LDS requirements reduce occupancy?
- Are too few workgroups launched to fill the CUs?
- Does a fusion remove meaningful traffic/launches without raising register
  pressure enough to lose occupancy?

The experimental `mwavefrontsize64` compiler option is not supported by the
HIP runtime. Wave size is not a universal runtime switch.

## Precision and matrix instructions

HIP 7.2 exposes low-precision scalar/vector types and intrinsics including
FP8. FP8 availability does not mean every operation is matrix-core
accelerated. FP4 E2M1 has no native acceleration on RDNA4 according to the HIP
low-precision table.

Generic HIP warp-matrix functions are explicitly unsupported. The supported
portable library route is rocWMMA, a header library that exposes matrix-core
fragments and operations on supported hardware.

This fork already uses both important paths:

- native FP8 Flash Attention kernels;
- rocWMMA Flash Attention, controlled by `GGML_HIP_ROCWMMA_FATTN=ON` and
  enabled in the current CMake cache.

A generic “enable WMMA/FP8” task is therefore already closed. New work must
identify a specific non-WMMA kernel/shape and prove that the conversion,
packing, and precision contract make matrix instructions beneficial.

## hipBLASLt

hipBLASLt provides flexible GEMM, heuristics, fused epilogues, grouped GEMM,
offline kernel tuning, and an Origami/Stream-K selection path.

### Offline tuning

`HIPBLASLT_LOG_MASK=32` can emit `hipblaslt-bench` commands for observed GEMM
problems. The tuning utility finds a solution index and stores a tuning file;
`HIPBLASLT_TUNING_OVERRIDE_FILE` loads the chosen solutions. Indices are tied
to a particular library release and device architecture and must not be
copied across SDKs or GPUs.

### Stream-K

`TENSILE_SOLUTION_SELECTION_METHOD=2` selects Origami/Stream-K where the
installed library contains compatible kernels. Stream-K divides aggregate K
iterations across physical resources and is intended to give consistent GEMM
performance across awkward/non-uniform shapes. Its documented data-type and
GPU support varies by architecture.

### Applicability here

The 7.2 Windows package does contain `gfx1201` code objects and the plain
hipBLASLt GEMM API exposes 842 candidate algorithms. Exact Q6_K prefill forms
showed strong point wins (`−13.6%` and `−24.0%` versus rocBLAS default), but a
temporary exact-shape runtime integration did not improve the L0 lane:
prompt `−0.89%`, decode `−0.24%`, aggregate `−0.58%` versus the two-control
mean. The maximum serialized point saving was below 1% of TTFT.

The runtime prototype was removed after byte-identical greedy correctness and
the negative wall gate. Per-call setup (`setProblem` + `isAlgoSupported` +
`initialize`) accounted for `0.09-0.14 ms`; even a one-time-descriptor raw
`hipblasLtMatmul` kept about `0.04-0.06 ms` overhead, which eats most of the
N=1024 point win. Keep hipBLASLt as an offline exact-shape diagnostic, not a
`ggml-hip` dependency or production selector. The old
`ROCBLAS_USE_HIPBLASLT=1` result remains non-evidence because it does not
activate this explicit tuned path.

## rocBLAS

rocBLAS provides BLAS operations and is present on Windows. It remains useful
for dense fallback/GEMM shapes, but llama.cpp's quantized decode kernels are
specialized to fuse dequantization with matrix-vector work. Routing those
weights through rocBLAS normally requires materialization or conversion, which
can cost more bandwidth than the GEMM saves.

A `rocblas_gemm_ex` `solution_index` screen (G07) found offline solutions up
to `−41%` faster on Q6_K prefill shapes (`−12%` to `−31%` on the N=1024
headline buckets). A default-off runtime gate routed all nine hot shapes; a
r3 dual-GPU A/B measured prompt `−0.42%`, TTFT `+0.42%` and aggregate
`+0.02%` versus the control mean. Combined point ceiling is `26.1 ms`
(`0.95%` of TTFT), below noise. The runtime probe was reverted; the existing
default GEMM path is not changed by solution selection.

## rocPRIM and hipCUB

rocPRIM provides warp-, block-, and device-level reductions, scans, radix
operations, and related primitives. `hipCUB` is the CUDA-compatible facade.
The source contains only isolated references/comments, not a broad primitives
integration.

These libraries are candidates only for a measured standalone reduction/sort
bottleneck. Replacing a short fused kernel with a generic device primitive can
add temporary storage and launches. The old source comment that hipCUB would
require moving from C++11 to C++14 is stale: the current project requires
C++17, but performance and workspace gates still apply.

## Cooperative groups and cross-lane operations

HIP supports cooperative groups and warp shuffles/ballots. On AMD, ballot
returns a 64-bit mask. The documentation notes faster cross-lane operations
when masks contain a contiguous active region without holes. The fork already
uses warp-level operations throughout quantized kernels; any change must be
guided by ISA/event evidence rather than translated CUDA folklore.

## Libraries that are not native-Windows options

- RCCL and other communication libraries: unavailable in the Windows HIP SDK.
- MIOpen and MIGraphX: unavailable in the Windows HIP SDK component matrix.
- Composable Kernel/AITER performance claims aimed at Instinct/Linux do not
  establish a supported Windows Radeon route.

## Sources

- [HIP performance optimization](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/understand/performance_optimization.html)
- [HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/performance_guidelines.html)
- [HIP hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/understand/hardware_implementation.html)
- [HIP low-precision types](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/low_fp_types.html)
- [HIP C++ language extensions](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_cpp_language_extensions.html)
- [rocWMMA 2.2 documentation](https://rocm.docs.amd.com/projects/rocWMMA/en/docs-7.2.0/index.html)
- [hipBLASLt 1.2.1 documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.2.0/index.html)
- [hipBLASLt offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.2.0/how-to/how-to-use-hipblaslt-offline-tuning.html)
- [hipBLASLt Stream-K](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.2.0/how-to/how-to-use-streamk.html)
- [rocBLAS 5.2 documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/docs-7.2.0/index.html)
- [rocPRIM 4.2 documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-7.2.0/index.html)
