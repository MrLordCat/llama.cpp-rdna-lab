# E295: Reliable ROCm peer copy on Windows

## Scope

- Platform: Windows, ROCm/HIP 7.1, two Radeon RX 9070 XT (gfx1201)
- Target: remove the dual-GPU layer-boundary host copy without accepting
  silent tensor corruption, a hard server abort, or an avoidable driver reset
- Current production route: pinned `VRAM -> RAM -> VRAM` staging
- Status: design and standalone probe compile complete; capability gate run,
  direct peer access rejected by HIP in both directions

## Capability result (2026-07-14)

The capability-only probe completed normally and detected:

| Direction | `can_access` | `access_supported` | Result |
|---|---:|---:|---|
| GPU0 (`0000:0e:00.0`) -> GPU1 (`0000:0b:00.0`) | 0 | 0 | rejected |
| GPU1 (`0000:0b:00.0`) -> GPU0 (`0000:0e:00.0`) | 0 | 0 | rejected |

Both devices were reported as RX 9070 XT / gfx1201. Performance-rank and native
atomic P2P attributes returned `hipErrorInvalidResourceHandle`; array access
was also zero. No buffers were allocated, no peer mapping was enabled, and no
peer copy was submitted.

This closes the direct-P2P gate for the installed Windows driver and HIP 7.1
runtime. Calling `hipMemcpyPeerAsync` anyway would not establish a supported
direct route. HIP may stage unsupported multi-device copies through host memory,
and the previous forced route already produced corruption/driver instability.
Do not bypass `can_access=0`.

Re-run the capability gate only after a material platform change such as a HIP
runtime/driver update. Until it reports access in the required direction, keep
the pinned host-staging route and focus optimization work there.

## Why a capability bit is insufficient

HIP exposes the directional APIs required for peer access:

- `hipDeviceCanAccessPeer`
- `hipDeviceGetP2PAttribute`
- `hipDeviceEnablePeerAccess`
- `hipMemcpyPeerAsync`

`hipDeviceCanAccessPeer=1` means that the address-space route is supported. It
does not certify the motherboard, BIOS/IOMMU configuration, PCIe path, Windows
driver, or repeated-copy correctness. Upstream llama.cpp made P2P an explicit
opt-in in April 2026 after systems that passed the capability query produced
crashes or corrupted output.

The Windows risk is relevant to this exact GPU family. AMD's HIP SDK 7.1.1
release notes list intermittent failures in HIP memory API calls on Radeon RX
9070 products. AMD's Radeon mGPU guidance also recommends equal CPU-connected
PCIe lane widths, PCIe 3.0 atomics support, and avoiding chipset-connected
slots; mixed graphics and compute workloads can cause GPU resets.

Therefore, no software-only preflight can turn this route into an unconditional
guarantee. The practical reliability contract is:

1. remain disabled by default;
2. reject unsupported or inconsistent directed pairs before model startup;
3. detect API failures and silent corruption with deterministic canaries;
4. keep host staging as the normal fallback before any peer operation is
   submitted;
5. stop cleanly rather than continue from a potentially poisoned HIP context
   after an in-flight peer failure.

## Current fork audit

The present implementation has five reliability gaps.

1. `GGML_ROCM_ENABLE_PEER_COPY` allows `hipMemcpyPeerAsync`, while the separate
   `GGML_CUDA_P2P` variable enables peer mappings. The copy and setup decisions
   can disagree.
2. `ggml_cuda_peer_copy_enabled()` is one global boolean. HIP capability and
   access are directional and must be tracked for every `src -> dst` pair.
3. Peer API calls use `CUDA_CHECK`. A runtime error aborts the process instead
   of quarantining the pair or returning a controlled backend failure.
4. There is no deterministic copy self-test. Successful initialization and
   model loading do not detect silent corruption.
5. The async scheduler path always uses a source-side event followed by a wait
   on the destination stream. This is valid in the abstract HIP stream model,
   but has not been isolated from the peer-copy operation on this Windows
   driver. Copy correctness and cross-device event correctness must be tested
   separately.

The current stream ownership is otherwise appropriate: the peer copy is issued
on the source device stream. HIP documents that the device associated with the
stream performs the copy and recommends a stream attached to the physical
source device.

## Reliability state machine

Use one state per directed device pair:

| State | Meaning | Runtime route |
|---|---|---|
| `disabled` | no explicit opt-in | host staging |
| `unsupported` | capability or access attribute rejects the pair | host staging |
| `setup_failed` | peer mapping could not be enabled | host staging |
| `sync_validated` | direct copy passed with host-observed completion | direct copy plus source completion wait |
| `event_validated` | copy and cross-device event ordering both passed | fully async peer route |
| `quarantined` | API error, checksum mismatch, timeout, or device loss | no more P2P in the process |

States are directional. `GPU1 -> GPU0` passing does not promote
`GPU0 -> GPU1`. Promotion is process-local and must be repeated after a driver,
BIOS, motherboard, GPU order, or ROCm change.

An error before a peer operation is submitted can safely select host staging.
An error returned after submission or by a completion event may indicate a
sticky or partially executed operation. The request must fail cleanly; it must
not immediately recopy the same tensor through RAM and continue as if the HIP
context were healthy.

## Standalone probe

Added `scripts/research/rocm_peer_copy_probe.cpp`. It has no model dependency
and makes no HIP calls when launched without an explicit mode.

Safety properties:

- capability reporting and data copies are separate modes;
- a data copy also requires `--acknowledge-driver-risk`;
- one direction and one size are tested per process;
- source and destination buffers have 4 KiB guards on both sides;
- every byte and a 64-bit hash are verified after each iteration;
- source patterns and destination sentinels change each iteration;
- Ctrl+C requests a soft stop after the in-flight operation, with no hard-kill
  path;
- no host fallback is attempted after a submitted peer operation reports an
  error.

The two copy modes isolate synchronization risk:

- `--copy-sync`: records and host-polls source completion before reading the
  destination. This is the first production candidate because it removes the
  host payload transfer without relying on a cross-device event wait.
- `--copy-event`: reproduces the llama.cpp source-event/destination-wait route
  and then validates the result through the destination stream.

The probe is host-only and links directly to `amdhip64`; no HIP kernels are
needed. The verified build command is:

```powershell
& 'C:\Program Files\AMD\ROCm\7.1\bin\clang-cl.exe' `
  /nologo /std:c++17 /O2 /EHsc /D__HIP_PLATFORM_AMD__ `
  /I'C:\Program Files\AMD\ROCm\7.1\include' `
  scripts\research\rocm_peer_copy_probe.cpp `
  /link /libpath:'C:\Program Files\AMD\ROCm\7.1\lib' amdhip64.lib `
  /out:build-rocm-full\bin\rocm-peer-copy-probe.exe
```

Do not run the copy modes while a game or another compute workload is active.

## Validation ladder

Run each step only after the previous step passes. A failure ends the session
and leaves production peer copy disabled. The current system stopped at step 1.

1. Capability-only report. Require `can_access=1` and
   `access_supported=1` for both directions.
2. `copy-sync`, one iteration, `GPU1 -> GPU0`, 20,480 bytes.
3. `copy-sync`, one iteration, reverse direction, 20,480 bytes.
4. Repeat both directions at 43,520 bytes, 1 MiB, 20 MiB, and 44 MiB.
5. Repeat the decode-sized cases 100 times, then 1,000 times.
6. Repeat the prompt-sized cases 10 times, then 100 times.
7. Run the same ladder with `copy-event`.
8. Reboot and repeat the short ladder to reject one-process-only success.
9. Add an opt-in runtime `sync_validated` route and compare deterministic model
   output against host staging before measuring speed.
10. Promote `event_validated` only if it is both correct and materially faster
    than the sync route.

Initial commands, to be run later on idle GPUs:

```powershell
build-rocm-full\bin\rocm-peer-copy-probe.exe --capabilities

build-rocm-full\bin\rocm-peer-copy-probe.exe --copy-sync `
  --acknowledge-driver-risk --src 1 --dst 0 --bytes 20480 --iterations 1
```

Run the reverse direction in a separate process. Do not jump directly to a
large buffer or repeated event-mode test.

## Runtime integration design

Replace the two independent environment gates with one ROCm-specific mode,
initially off:

```text
GGML_ROCM_PEER_MODE=off|sync|event
```

The backend should initialize a directed-pair table once and log, for every
used pair:

- device IDs and PCI bus IDs;
- capability and P2P attributes;
- peer-enable result;
- canary mode, size, iteration count, and hashes;
- final selected route and quarantine reason.

`sync` should be implemented first. It queues `hipMemcpyPeerAsync` on the
source stream, obtains source-side completion, and only then allows the
destination graph to proceed. It sacrifices some host overlap but removes both
the D2H and H2D payload copies and avoids cross-device event dependence.

`event` may retain the current source event and destination wait only after the
exact sequence passes the standalone probe. Events must retain the default
system visibility behavior; do not use flags that disable the system fence.

For either mode, `CUDA_CHECK` is unsuitable in optional-route setup. Expected
capability/setup failures should populate the pair state and select host
staging. In-flight errors require a controlled backend/request failure and pair
quarantine, not an abort and not transparent continuation.

## Performance interpretation

Correctness does not prove that the runtime used direct PCIe DMA. HIP documents
that some memory APIs can still stage through host memory when peer access was
not enabled. A successful capability query, successful peer enable, and clean
checksums establish eligibility and correctness; large-buffer timing versus the
known host-staged path is still needed to infer that the direct route is active.

The first performance target is not tensor parallelism. It is the existing
single layer boundary in `-sm layer`: remove the approximately 0.85-1.0 ms
host-staged crossing measured per decode token and recover part of the observed
31.61 single-GPU versus 25.54 dual-GPU ROCm decode gap without reducing prompt
evaluation.

## Sources

- [HIP 7.1 peer-to-peer API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/doxygen/html/group___peer_to_peer.html)
- [HIP 7.1 multi-device management](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/how-to/hip_runtime_api/multi_device.html)
- [AMD HIP SDK for Windows 7.1.1 release notes](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/about/releasenotes.html)
- [AMD Radeon mGPU setup guidance](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1.1/docs/install/installrad/native_linux/mgpu.html)
- [llama.cpp P2P explicit opt-in, PR #21910](https://github.com/ggml-org/llama.cpp/pull/21910)
- [llama.cpp broken ROCm P2P workaround, PR #6208](https://github.com/ggml-org/llama.cpp/pull/6208)
