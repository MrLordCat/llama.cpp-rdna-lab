# Memory and multi-GPU behavior

## Memory classes

| Memory/API | What it provides | Practical meaning for this lane |
| --- | --- | --- |
| `hipMalloc` VRAM | device-local allocation | preferred steady-state weights, KV, and scratch |
| pageable host RAM | ordinary CPU allocation | copies are staged by the runtime and are slower |
| `hipHostMalloc` pinned RAM | page-locked host allocation | enables efficient asynchronous H2D/D2H copies; consumes a scarce OS resource |
| `hipHostRegister` | pins an existing host range | useful only when the registered range is actually copied/accessed by HIP |
| mapped host memory | device can address host pages | avoids an explicit copy but pays PCIe latency/bandwidth; generally wrong for hot weights |
| `hipMallocManaged` | managed address space/page migration | convenient, but page faults/migration are dangerous for deterministic dGPU inference |
| stream-ordered pools | allocation lifetime follows stream order | Beta/Linux-implemented and under development on Windows in HIP 7.2; not a current lane feature |

AMD documents pinned transfers as potentially much faster than pageable
transfers, but warns that excessive pinned allocation reduces available system
memory. Mapped host memory is not equivalent to VRAM and should not be used as
a generic “spill accelerator.”

The official host-allocation flags include `Portable`, `Mapped`,
`WriteCombined`, `Coherent`, and `NonCoherent`. `NumaUser` policy is Linux
implemented and under development on Windows, so it is excluded here.

## What the fork already does

[runtime_buffers.inc](../../../ggml/src/ggml-cuda/runtime/runtime_buffers.inc)
implements the common host allocator with `cudaMallocHost`; the HIP shim maps
that call to `hipHostMalloc(..., hipHostMallocDefault)`. It falls back to
pageable RAM when pinned allocation fails or `GGML_CUDA_NO_PINNED` is set.

Managed memory is opt-in and has an explicit Windows-compatible fallback to
ordinary device allocation when unsupported. It should remain off for the
performance lane: WDDM residency pressure is already visible and managed
migration would make timing less predictable.

The backend exports a `hipHostRegister`-based registration hook using portable
and read-only flags, but repository-wide search found no internal model-load
caller. Setting `GGML_CUDA_REGISTER_HOST` alone therefore does not prove that
the 682 MiB CPU-mapped model range is registered or accelerated.

## Two-GPU data path

The locked dual-ROCm topology is a layer split:

```text
CPU input -> ROCm1 layers 0..32 -> host-staged boundary -> ROCm0 layers 33..64/output
```

The second graph half cannot start its dependent work before the boundary
tensor from the first half is ready. More streams do not remove that edge.

### P2P

HIP supports peer access in the general API, and real P2P can avoid host
staging. On this exact Windows dual-RX 9070 XT setup, the HIP 7.1 capability
probe reported `can_access=0` and `access_supported=0` in both directions.
Forced peer copy previously corrupted output and destabilized the driver.

The only valid reopening condition is a material HIP/driver change followed by
the standalone capability and correctness probe. Documentation is not evidence
that P2P became available.

### Host-staged path

There are two relevant implementations:

1. [runtime_backend.inc](../../../ggml/src/ggml-cuda/runtime/runtime_backend.inc)
   contains a Windows/HIP opt-in path selected by
   `GGML_ROCM_ASYNC_CROSS_DEVICE_STAGE`. It reuses multiple pinned slots,
   performs async D2H on the source stream, records an event, waits on that
   event in the destination stream, and enqueues async H2D. It falls back
   safely when cross-device event waiting is unavailable.
2. [runtime_buffers.inc](../../../ggml/src/ggml-cuda/runtime/runtime_buffers.inc)
   contains a simpler thread-local pinned staging buffer for a different copy
   interface and synchronizes the destination per-thread stream.

The existing L0 experiment found that event-chained staging improved 30K
prompt throughput by 2.65% but changed decode by -0.45%. It is therefore a
closed decode route, not an unimplemented optimization.

Both allocators currently use `hipHostMallocDefault`. A narrowly scoped
`hipHostMallocPortable` experiment may be useful only to test cross-device
registration/consistency under HIP 7.2. It has no documented reason to remove
the measured device-work gap, so its expected TPS upside is low.

## Safe investigation order

1. Build a separate HIP 7.2 control and reproduce single/dual L0.
2. Re-run the standalone bidirectional P2P capability test. Stop if either
   direction remains unsupported.
3. Use `GGML_TRACE_ROCM_ASYNC_CROSS_DEVICE_STAGE` to count tensors/bytes/slots
   and verify whether the opt-in path stays asynchronous.
4. Profile the normal path with RGP. Separate boundary-copy duration from the
   wait for ROCm1 graph completion.
5. Test host-allocation flags only in the staging allocator, behind an
   environment gate, with a pageable/default-pinned control and RAM accounting.

Do not combine allocation flags, P2P overrides, graph changes, and layer-ratio
changes in one run. A valid result changes one mechanism at a time.

## Sources

- [HIP memory management](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/memory_management.html)
- [HIP unified memory](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/memory_management/unified_memory.html)
- [HIP multi-device programming](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/multi_device.html)
- [HIP stream-ordered allocator API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html)
- [decode lane transfer evidence](../decode/README.md)
