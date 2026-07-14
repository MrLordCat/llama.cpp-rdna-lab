# Qwen3.6 Vulkan Paths

## Baseline Text Inference

Qwen3.6 text inference exercises a focused subset of the generic Vulkan backend:

- quantized and floating-point `MUL_MAT` routes;
- `FLASH_ATTN_EXT` and KV-cache reads/writes;
- RMS normalization and fused RMS/MUL/ROPE patterns;
- RoPE, copy/view/set-rows, gather, elementwise and activation operations;
- output projection and sampling support operations;
- `GATED_DELTA_NET` recurrent layers.

Prompt evaluation stresses large-N quantized matmul, Flash Attention, GDN, and
buffer residency. Decode stresses small-N matvec/matmul, synchronization, and
launch overhead. A change that helps one phase may regress the other.

## MTP

MTP does not use a separate Vulkan backend. It changes graph shapes and cadence:

- target verification creates multi-column decode matmuls;
- next-token hidden-state transport adds output and copy pressure;
- accepted draft depth changes how often graphs are recorded and submitted;
- small verify batches can select a different MMVQ/MMQ route from baseline.

MTP work therefore belongs mainly in `vk_dispatch.inc`, `vk_graph.inc`, and
`vk_backend_execution.inc`. Any MTP speed claim must be compared with an adjacent
`spec=none` run using the same model, prompt, KV, split, and background load.

## Vision

The Qwen mmproj path additionally needs convolution, im2col, pooling, image
tensor copies, and ordinary matmul/normalization. Do not remove generic vision
operations merely because they are absent from a text-only trace.

Validation for an operation-pruning change must include at least one real image
request through the server, not only successful model loading.

## Optional Operation Families

The backend also contains RWKV, SSM, optimizer/training, diffusion, and other
generic operation handlers. They are compiled in normal builds but are not
executed by the Qwen3.6 text/MTP workload. Removing them should be a separate
profile-driven project based on an observed operation allowlist.

## Vendor Versus Extension Names

Vendor checks and extension names are different concepts:

- `vendorID == VK_VENDOR_ID_NVIDIA` is an NVIDIA device branch.
- `VK_NV_cooperative_matrix2` is an extension API originally defined by NVIDIA.

An AMD driver and the shader toolchain can expose an extension with an `NV`
name. The RX 9070 XT path may benefit from cooperative-matrix2 support, so an
AMD-only cleanup must not delete every symbol beginning with `VK_NV_`.

Likewise, AMD-only pruning must preserve generic Vulkan fallbacks and both AMD
proprietary/open-source driver paths unless the supported platform contract is
explicitly narrowed further.

## Determining Active Code

A single trace cannot prove code is dead. Build an allowlist from all supported
workloads:

1. baseline text prompt and decode;
2. MTP with several draft depths;
3. long-context prompt evaluation;
4. dual-GPU in both device orders;
5. vision with a real image;
6. context save/restore and KV format variants used by the GUI.

Record operation names, pipeline names, fusion names, shader routes, and copy
paths. Only code absent from the full matrix is a candidate for a compile-time
minimal profile, and removal still requires a fallback analysis.
