# Vulkan Runtime Architecture

## Build Pipeline

`ggml/src/ggml-vulkan/CMakeLists.txt` builds two related parts:

1. `vulkan-shaders-gen` compiles the GLSL shader sources.
2. `ggml-vulkan` embeds the generated SPIR-V and builds the runtime backend.

The shader sources are already independent files. The runtime modules are
ordered includes assembled by `ggml-vulkan.cpp`; CMake lists them as
`HEADER_FILE_ONLY` so IDEs expose the full module tree without compiling the
fragments twice.

Vulkan-Hpp setup must remain at the beginning of `vk_common.inc`. In particular,
`VULKAN_HPP_DEFAULT_DISPATCHER` must be defined consistently before including
`vulkan.hpp`. Moving this setup casually into multiple translation units can
create dispatcher and initialization-order bugs.

## Runtime Lifecycle

### 1. Registry and instance

`ggml_backend_vk_reg()` initializes the Vulkan instance and exposes devices to
the generic GGML backend registry. Instance initialization:

- enumerates physical devices;
- removes unsupported devices;
- resolves duplicate representations exposed by different drivers;
- records memory-budget support;
- establishes a stable Vulkan device index.

### 2. Device initialization

`ggml_vk_get_device()` creates the long-lived `vk_device_struct`. It owns:

- Vulkan physical/logical device handles;
- compute and transfer queues;
- feature and extension capability flags;
- device-local shader pipelines;
- allocation and memory logging state;
- reusable command pools and synchronization primitives.

Architecture classification is capability-based. The current enum name
`AMD_RDNA3` also covers the RDNA generation selected by the mixed signed-dot
capability check, including the current RX 9070 XT path. Do not remove or rename
that branch based only on the marketing generation name.

### 3. Shader registry

`ggml_vk_load_shaders()` creates pipeline descriptors for the capabilities of a
device. Pipelines carry `needed` and `compiled` state, allowing the backend to
compile requested variants without treating every available route as active in
every workload.

The registry selects specialization constants for subgroup size, tile shape,
accumulator type, quant format, alignment, and optional cooperative-matrix
paths. These choices are device state and must not be inferred again in random
operation handlers.

### 4. Buffer ownership and transfers

The backend exposes device, host-visible, and pinned/direct-host buffer types.
`vk_buffer_struct` owns the Vulkan allocation and mapping state. Transfer helpers
choose among direct host access, staged copies, transfer-queue submissions, and
compute-queue copies.

For dual GPU, a tensor still belongs to one backend buffer. Cross-device copies
are orchestrated by the generic scheduler and backend copy interfaces. Changes
to staging or synchronization must therefore be tested with both device orders,
not only a single GPU.

### 5. Graph recording

The generic scheduler passes a GGML graph to the backend. The Vulkan path:

1. optionally reorders independent nodes while preserving fusion patterns;
2. determines supported fusions and preallocation requirements;
3. maps every executable node to an operation encoder;
4. records dispatches and barriers into command buffers;
5. submits at synchronization or overlap boundaries;
6. retires temporary semaphores, events, and command buffers.

`vk_graph.inc` owns graph recording and temporary allocation. Fusion recognition,
backend callbacks, and synchronization live in `vk_backend_execution.inc`.

## State Ownership

| State | Lifetime | Owner |
| --- | --- | --- |
| Vulkan instance and device index map | process | `vk_instance_t` |
| Logical device, queues, pipelines | physical device | `vk_device_struct` |
| Graph command pools and preallocations | backend context | `ggml_backend_vk_context` |
| Vulkan allocation | backend buffer | `vk_buffer_struct` |
| Command-buffer recording state | submission/context | `vk_context_struct` |

## Synchronization Invariants

- A command buffer cannot be reset while a recorded event still refers to its
  use counter.
- Transfer-queue work must signal the timeline semaphore before compute consumes
  copied data.
- Host reads must observe completion before copying mapped or staged data.
- Preallocation resize must submit and synchronize outstanding users first.
- Graph optimization must preserve data dependencies and known fusion adjacency.
- Driver workarounds belong near route selection or capability probing and need
  a narrow hardware/driver guard.

## Future Translation-Unit Split

If separate `.cpp` compilation becomes desirable, introduce a private
`vk_internal.hpp` first. Move leaf modules in this order:

1. result checks and test helpers;
2. device capability helpers;
3. buffer backend wrappers;
4. device and transfer implementation;
5. graph/backend implementation;
6. dispatch and operation hot paths last.

Each step must pass symbol visibility, full Vulkan build, model output, and
performance gates before moving the next module.
