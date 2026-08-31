# ROCm 7.2 knowledge base for the Windows RDNA4 lane

Date: 2026-08-30

Scope: native Windows 11, two Radeon RX 9070 XT (`gfx1201`), Qwen3.8,
`ggml-hip`, non-speculative decode. This directory turns the official ROCm
7.2 documentation into a map of features that are usable, already used, or
worth testing in this fork.

## Read this first

ROCm documentation describes several different products. They are not
interchangeable:

1. **ROCm 7.2 on Linux** is the complete stack described by the main release
   notes.
2. **HIP SDK for Windows 7.2** is a subset: compiler, closed-source HIP
   runtime, math libraries, primitive libraries, HIPIFY, debugger integration,
   and Radeon GPU Profiler. Communication and AI libraries, `rocminfo`, ROCm
   SMI, and the CMake HIP language are not part of that Windows product.
3. **The current `build-rocm` directory** was generated with HIP SDK 7.1 even
   though 7.2 is installed. A feature documented for 7.2 is not a fact about
   the current binary until it is reproduced with a separate 7.2 build.

The RX 9070 XT is officially supported by HIP SDK 7.2 for runtime and SDK use
on Windows. Generic Linux guidance is recorded here only when its Windows
applicability is known or can be tested safely.

## Contents

- [Runtime and Windows stack](01_RUNTIME_WINDOWS.md): execution model,
  streams, events, graphs, environment variables, and component boundaries.
- [Memory and multi-GPU](02_MEMORY_MULTI_GPU.md): VRAM, pinned and managed
  memory, P2P, and the actual host-staged split path in this fork.
- [Kernels and libraries](03_KERNELS_LIBRARIES.md): RDNA4 execution,
  rocWMMA, FP8, hipBLASLt, rocBLAS, and primitives.
- [Profiling and tooling](04_PROFILING_TOOLING.md): the supported Windows
  replacement for Linux-only ROCProfiler, plus an L0 capture plan.
- [Feature gap and experiment queue](05_GGML_GAP_MAP.md): prioritized answers
  to “what ROCm capability are we not using yet?”
- [HIP 7.2 rollout checklist](06_HIP72_ROLLOUT_CHECKLIST.md): build, smoke,
   benchmark, and promotion gates for the new Windows binary.

Related local evidence remains authoritative for existing benchmark claims:

- [ROCm route map](../ROCM_ROUTE_MAP.md);
- [RDNA4 architecture track](../rdna4-architecture/README.md);
- [decode lane baseline and gates](../decode/README.md).

## Main conclusions

1. **The biggest unused capability is measurement, not a hidden speed flag.**
   Radeon GPU Profiler 2.7 supports HIP compute on RX 9000 under Windows 11.
   It exposes device event timing, wavefront occupancy, instruction timing,
   queue synchronization, and ISA. It is not currently installed on this
   machine.
2. **The runtime basics are already used well.** The fork uses asynchronous
   streams/copies, events, graph capture/update/replay, pinned host memory,
   occupancy queries, FP8 kernels, and rocWMMA Flash Attention.
3. **Dual-GPU decode is dependency-serial.** A token crosses the layer-split
   boundary between two graph halves. P2P is unavailable on the present
   HIP/driver combination, so the valid route is pinned host staging. The fork
   already contains an opt-in event-chained staging implementation; it did not
   improve decode in the existing L0 gate.
4. **hipBLASLt is useful here as an offline diagnostic, not a runtime win.**
   The installed 7.2 package has working `gfx1201` kernels and exact Q6_K
   prefill points beat rocBLAS, but per-call setup overhead (`0.09-0.14 ms`)
   and a sub-1% combined ceiling make the explicit dual-GPU runtime A/B
   neutral/negative (`−0.42%` prompt versus the three-control mean for the
   cached-style G07 probe). Both runtime prototypes were removed; single-token
   decode remains on quantized MMVQ and fused recurrent/attention work.
5. **Q4_K large-batch prefill prefers dequant+hipBLAS over MMQ.** G08 shows
   every large-batch Q4_K MMQ form at 1 block/SM (`LDS 88%`, `regs 236`,
   occupancy `12.50%`) and hipBLASLt 2-4.7x faster on the same GEMM volumes;
   moving `ne11 549/919/1024` onto the existing `cublas_backend` path raised
   r3 prompt throughput `+3.29%` on Qwen3.8-27B Q4_K_M (`+2.23%` on
   Qwen3.6-27B Q4_K_M), byte-identical greedy. The Q4_K MMQ selector
   default is now `ne11<=256`; MMVQ decode is untouched.
6. **RDNA4 f32 GEMMs must use f16 inputs with f32 accumulation.** G09
   (2026-08-30) shows 832 `cublas_backend|f32` calls (~628 ms of GDN
   projections and RoPE rotation K/V) executing on the pure f32 path at
   ~0.4-0.85 TF/s, while the f16-input equivalent is an order of
   magnitude faster. The promoted route gives prompt `+5.35%` on
   Qwen3.8-27B Q4_K_M (`+4.75%` Qwen3.6-27B), byte-identical greedy, with
   an env rollback (`GGML_ROCM_F32_GEMM_F16=0`) that measures back into
   the control band. The decode `mul_mat_vec_f_direct|f32` lane is not
   covered yet and is the next candidate.
7. **Fusing the paired decode GDN MMVFs is bit-exact but does not win
   wall time.** G10 (2026-08-30) merged `ssm_alpha`+`ssm_beta`
   (`5120->48`, n=1) into one two-output `ggml_cuda_mul_mat_vec_f_pair()`
   launch: standalone probe is bit-exact and only ~9% faster per pair
   (0.0038 vs 0.0042 ms) because the geometry is launch/latency-bound, and
   neighbor A/B on the locked lane shows decode ON `24.84` vs OFF `24.94`
   (no gain; graph capture already minimizes per-node launch overhead).
   Kept opt-in (`GGML_ROCM_GDN_PAIR=1`, default OFF); not promoted.
8. **Q5_K/Q6_K N=1 MMVQ is not occupancy-limited.** G11 (2026-08-31)
   measures both at 100% occupancy with low registers. Q6 output head
   already reaches `585.3 GB/s` (~91.4% of physical bandwidth). Five
   exact Q5/Q6 body candidates were neutral/negative; the exact Q5 qsum
   sidecar regressed weighted kernel time `+4.26%`, and compact pair-dot
   regressed `+0.94%` despite lowering regs `42 -> 39`. All G11 runtime
   code was removed. Q6 now requires fewer stored bytes; Q5 requires a
   byte-neutral prepacked scale/high-bit layout, not another geometry or
   q8-reuse tweak.
5. **Memory pools and async allocation are absent and not Windows-ready in
   HIP 7.2.** The reference marks the API Beta, Linux-implemented, and under
   development on Windows. Even after support arrives, its likely benefit is
   allocation/fragmentation during load, not the steady-state token graph.
6. **Linux-only features are not experiments for this lane.** RCCL,
   ROCProfiler/`rocprofv3`, MIOpen/MIGraphX, and generic managed-memory advice
   do not become usable on native Windows just because they appear in the
   ROCm 7.2 manual.

## Evidence policy

Every proposed change must pass the locked L0 contract in the decode lane.
Documentation is used to generate a hypothesis, never a performance claim.
Keep three labels distinct:

- **documented**: AMD describes the API or tool;
- **available**: the Windows 7.2 runtime/tool reports support on this host;
- **beneficial**: an adjacent, correct benchmark clears the lane gate.

## Official entry points

- [ROCm 7.2 release notes](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html)
  (the page explicitly applies to Linux).
- [HIP 7.2 documentation](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/index.html).
- [HIP SDK for Windows 7.2 component support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html).
- [HIP SDK for Windows system requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html).
- [Radeon GPU Profiler manual](https://gpuopen.com/manuals/rgp_manual/rgp_manual-index/).
