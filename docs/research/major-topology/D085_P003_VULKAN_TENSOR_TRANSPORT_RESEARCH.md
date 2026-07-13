# D085 P003 Vulkan Tensor Transport Research

## Question

Is there an existing Vulkan or Windows transport that can make two separate
Radeon GPUs competitive in `-sm tensor`, without routing every all-reduce
through host memory?

## Short Answer

No production-ready Vulkan collective library or upstream llama.cpp transport
was found for two separate AMD Vulkan devices on native Windows.

The remaining practical software route is low-bit communication compression.
It does not create peer-to-peer access, but it can reduce the bytes crossing
PCIe for every tensor-parallel boundary by roughly 2x to 4x.

## Why The Obvious Vulkan Routes Do Not Work

### Separate-device external memory

`VK_KHR_external_memory_win32` is not a cross-physical-device VRAM transport.
The Vulkan specification requires imported Win32 memory to have been created on
the same underlying physical device as the importing `VkDevice`:

- https://docs.vulkan.org/spec/latest/chapters/memory.html

It can share an allocation across APIs, processes, instances, or logical
devices that refer to the same physical GPU. It cannot connect the local VRAM
of Vulkan0 to Vulkan1 in this machine.

### Vulkan device groups

Vulkan peer memory is available only inside one logical device created from a
multi-physical-device group. Peer read/write support is then queried with
`vkGetDeviceGroupPeerMemoryFeatures`:

- https://docs.vulkan.org/guide/latest/extensions/device_groups.html
- https://docs.vulkan.org/spec/latest/chapters/memory.html#memory-device-groups

The current Windows AMD driver enumerates the two RX 9070 XT cards as two
single-device groups. Application code cannot combine groups that the driver
keeps separate. A full device-group llama.cpp implementation would therefore
have no usable group on this system.

### D3D12 cross-adapter heaps

D3D12 has cross-adapter shared heaps, but Microsoft documents that they reside
only in system memory and that the L0 pool is inefficient for discrete/NUMA
adapters:

- https://learn.microsoft.com/en-us/windows/win32/direct3d12/shared-heaps

This is the same class of route as the tested coherent host-visible Vulkan
buffer. That experiment was slower than device-local compression plus DMA, so
a D3D12 bridge is not a target-closing route.

## Existing Collective Libraries

Upstream llama.cpp uses NCCL for optimized CUDA reductions and can use RCCL for
ROCm when built with `-DGGML_HIP_RCCL=ON`. Its own multi-GPU guide explicitly
guarantees good tensor-mode performance only for NVIDIA/CUDA and warns that
other backends may be suboptimal:

- https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md

RCCL is not available in the native Windows HIP SDK. AMD lists communication
libraries as available on Linux and unavailable on Windows:

- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html

Linux is a valid alternate experiment. RX 9070 XT is in AMD's supported ROCm
7.2.1 Radeon matrix:

- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html

RCCL on consumer multi-GPU topologies still needs a smoke test before treating
it as reliable. A current dual-RX-7900-XTX report shows the first collective
failing on one ROCm/RCCL release and working on an earlier generation:

- https://github.com/ROCm/ROCm/issues/6074

## Upstream llama.cpp State

No Vulkan all-reduce transport or Vulkan collective library was found in
upstream. The upstream guide describes `tensor` as communication-heavy and
experimental outside CUDA/NCCL. A reported dual-W7900 Vulkan slowdown was
closed stale without a transport fix:

- https://github.com/ggml-org/llama.cpp/issues/16767

The Qwen 3.5/3.6 tensor-granularity fix in PR 23843 addresses a heterogeneous
quantization split with three GPUs. It does not reduce the 127 communication
boundaries or help the two-GPU transport here:

- https://github.com/ggml-org/llama.cpp/pull/23843

## A Software Route That Matches This Bottleneck

Several independent inference studies compress the row-parallel partial
activations before the collective:

- Communication Compression for Tensor Parallel LLM Inference reports about
  3.3x compression with small perplexity impact and up to 2x lower TTFT:
  https://arxiv.org/abs/2411.09510
- Towards Low-bit Communication keeps selected outlier channels in BF16 and
  sends the rest in INT4, averaging under 4.2 bits/value while preserving about
  98.0% of Gemma 2 27B and 99.5% of Llama 2 13B task performance:
  https://arxiv.org/abs/2411.07942
- Flash Communication reports more than 3x faster intra-node communication and
  up to 2x lower TTFT with low-bit activation communication:
  https://arxiv.org/abs/2412.04964

These are research implementations rather than a Vulkan library, but the
algorithm fits the existing native Vulkan communicator exactly: compression
and decompression shaders already surround a host-mediated reduction.

## Expected Ceiling On This Machine

The current BF16 path transfers 10 MiB partials and measures approximately:

| Phase per large boundary | Time |
| --- | ---: |
| Device readback | 1.2 ms |
| AVX2 host reduction | 0.9 ms |
| Device upload | 1.3 ms |
| Total | 3.4 ms |

Qwen3.6-27B requires 127 such boundaries per 1024-token ubatch. BF16 therefore
adds about 0.43 seconds per ubatch before other scheduling costs.

An 8-bit route can halve transfer bytes, but quantization and scale handling
remain. A realistic first estimate is roughly 1.9-2.5 ms per large boundary,
which would likely move the 12k prompt result from about 1032 tok/s into the
1250-1450 tok/s range. A calibrated 4-5 bit route has a higher ceiling, but it
must pass quality tests and is still unlikely to reach 2000 tok/s by itself.

True device-local P2P or RCCL remains necessary to make tensor mode clearly
beat the measured Vulkan layer result of 1826 tok/s.

## Recommended Prototype

1. Add an opt-in `GGML_VK_TENSOR_ALLREDUCE_Q8=1` transport.
2. Use symmetric block-wise INT8 with per-block FP16 scales on each partial.
3. Quantize/dequantize on each GPU; perform a two-device AVX2 scaled reduction
   on the pinned host buffer.
4. Upload the reduced result in Q8 plus scales and dequantize on each GPU, so
   both transfer directions benefit.
5. Gate promotion on numerical comparison, coherent long generation, a
   perplexity check, and a measured improvement over BF16.
6. Only after Q8 is stable, evaluate an INT4 plus BF16-outlier-channel route.

This is a real transport optimization, not a scheduler workaround. It attacks
the measured byte volume while keeping the mathematically required tensor-
parallel reductions intact.

## Decision

- Do not spend time on `VK_KHR_external_memory_win32` for cross-GPU VRAM; the
  Vulkan validity rule excludes this use.
- Do not build a D3D12 bridge for performance; its cross-adapter storage is
  system memory.
- Keep Vulkan `layer` as the production path on Windows.
- Use Q8 communication compression as the next experimental Vulkan tensor
  milestone.
- Treat Linux ROCm plus RCCL as the only currently available route to a true
  driver-backed AMD collective on this hardware.
