# Fork Details

Fork-only capabilities, backend fixes beyond upstream, and the recommended
runtime profiles. This file collects the larger README sections so the main
`README.md` stays compact; production defaults and lane contracts are in
[PERFORMANCE.md](PERFORMANCE.md).

## Key Fork Features

- PyQt6 GUI for dependency checks, builds, server launch, monitoring, and logs.
- Vulkan/ROCm-aware benchmark and autotune UI with live prompt progress.
- OpenAI-compatible `llama-server` for local applications and coding agents.
- Dual-GPU layer placement and explicit output-device controls.
- Upstream-style Qwen3.5/3.6/3.8 MTP pipeline with backend-resident NextN handoff.
- ROCm KV-only sparse-history MTP with a bounded long-prompt prefill cost.
- RDNA4 Q3_K prompt and small-N decode kernel specializations.
- Native Prism `PQ2_0` GGUF loading, CPU support, and optimized HIP MMQ/MMVQ
  kernels for Ternary Bonsai.
- Vision support through a compatible `mmproj-*.gguf` projector.
- Prompt checkpoints, cache controls, benchmark history, and diagnostic traces.
- DFlash integration for research; it is not currently the recommended runtime
  profile.

## Fork-Only Backend Fixes

The following production paths are local to this fork. They were checked
against the neighboring stock `ggml-org/llama.cpp` checkout at commit
`f955e394b` (2026-07-15); the named controls and implementations are absent
there. Upstream changes quickly, so this is a snapshot rather than a permanent
claim about future llama.cpp releases.

### Vulkan Fixes

- **AMD large cooperative matmul route.** The proprietary Windows AMD driver
  can use the large cooperative-matrix pipelines and fork-tuned `bn256`
  variant instead of being limited to the conservative small/medium route.
  It is automatic on the tested discrete RDNA device. Use
  `GGML_VK_DISABLE_AMD_LARGE_MATMUL=1` for rollback;
  `GGML_VK_AMD_LARGE_MATMUL_VARIANT` selects a research variant.
- **Explicit output and MTP placement.** `LLAMA_OUTPUT_DEVICE` places the large
  output/vocabulary tensors on the intended card. NextN tensors are placed on
  the first Vulkan device, and the expensive four-copy MTP pipeline scheduler
  is disabled by default. Diagnostic rollbacks are
  `LLAMA_VK_MTP_NEXTN_MAIN_DEVICE=0` and
  `LLAMA_MTP_PIPELINE_PARALLEL=1`. See
  [E274](docs/research/experiments/E274_vulkan_dual_mtp_nextn_placement.md) and
  [E280](docs/research/experiments/E280_vulkan_gpu1_primary_residency.md).
- **Warm MTP verification topology.** Startup prepares verification widths
  `1..n_max+1`, retains the warmed token-generation scheduler across prompt
  processing, and avoids invalidating it with prompt-only output reservation
  changes. Windows/AMD widths 5-8 are split into safe `4 + remainder`
  dispatches instead of using the driver-crashing specialization or the slow
  generic fallback. Set `LLAMA_VK_MTP_VERIFY_WARMUP=0` to disable the path. See
  [D086](docs/research/major-topology/D086_P003_VULKAN_MTP_TG_WARM_CACHE.md).
- **Batched recurrent-checkpoint reads.** Vulkan groups checkpoint tensor reads
  by backend and performs one staged transfer/synchronization per GPU instead
  of synchronizing every tensor. The measured incremental-tail checkpoint time
  fell 17.9%, with prompt TPS up 8.9%. Set
  `LLAMA_CHECKPOINT_BATCH_READ=0` for the sequential path. See
  [E279](docs/research/experiments/E279_vulkan_batched_recurrent_checkpoint.md).

### ROCm/HIP Fixes

- **RDNA4 rocWMMA FlashAttention.** Fresh HIP builds discover the bundled
  rocWMMA 7.1 headers and enable the D=256 WMMA path. The matched 53.5K prompt
  improved from 1091.68 to 1557.94 tok/s. Configure with
  `-DGGML_HIP_ROCWMMA_FATTN=OFF` for the generic-tile rollback. See
  [E293](docs/research/experiments/E293_rocm_rdna4_rocwmma_fattn_restore.md).
- **Q3_K and PQ2_0 kernels.** Packed Q3_K conversion/staging and RDNA4 small-N
  MMQ/MMVQ specializations cover the primary Qwen model. The fork also adds
  the Prism `PQ2_0` GGUF type plus CPU and native HIP kernels for Ternary
  Bonsai. `GGML_CUDA_Q3K_PADDED_DEQUANT_PACKED=0` restores the older Q3_K
  staging path. See
  [E292](docs/research/experiments/E292_rocm_q3k_packed_dequant_probe.md) and
  [E331](docs/research/experiments/E331_bonsai_pq2_ubatch_decode_isolation.md).
- **Bounded quantized-KV FlashAttention memory.** Quantized K/V conversion
  scratch is graph-owned instead of accumulating in the non-VMM HIP pool. For
  long Q8 K/V contexts, a 4096-token chunked WMMA route replaces full-context
  F16 staging and combines chunk softmax results online. Set
  `GGML_ROCM_FATTN_Q8_CHUNKED_WMMA=0` to disable it. See
  [E334](docs/research/experiments/E334_rocm_quantized_kv_scratch_reservation.md)
  and [E337](docs/research/experiments/E337_rocm_q8_chunked_wmma.md).
- **Windows dual-GPU safety and staging.** Direct HIP peer copy is quarantined
  by default on Windows because the tested driver path was not reliable; the
  backend uses explicit host-staged transfers. `GGML_ROCM_ENABLE_PEER_COPY=1`
  is a diagnostic opt-in, not a production recommendation. The independently
  gated `GGML_ROCM_ASYNC_CROSS_DEVICE_STAGE=1` overlaps the safe staged layer
  boundary. See
  [E295](docs/research/experiments/E295_rocm_windows_peer_copy_reliability.md)
  and [E313](docs/research/experiments/E313_rocm_async_cross_device_stage.md).
- **Long-prompt MTP transport.** ROCm keeps NextN hidden states on the backend,
  prefills only KV work, and retains sparse long-range history plus the recent
  tail. Deferred sparse blocks are flushed before staging reuse, preventing a
  duplicate final-window decode. `LLAMA_MTP_DEVICE_HANDOFF=0` restores host
  handoff; `LLAMA_SPEC_PREFILL_SPARSE_CHUNK=0` removes sparse anchors. See
  [E315](docs/research/experiments/E315_rocm_long_context_mtp_sparse_history.md)
  and [E338](docs/research/experiments/E338_rocm_dual_long_context_scheduler_residency.md).
- **Single-request scheduler residency.** ROCm defaults to one split-graph copy
  instead of four for this fork's `-np 1` workload. On Q4 98K this reduced
  prefill Dedicated/Shared from 23.85/5.46 to 22.05/3.20 GiB without reducing
  prompt throughput. `GGML_SCHED_PIPELINE_COPIES=2` or `4` restores extra
  copies for controlled concurrent-request experiments. See
  [E338](docs/research/experiments/E338_rocm_dual_long_context_scheduler_residency.md).

## Recommended Runtime Profiles

### Vulkan Dual GPU

Use GPU1 as the output device. The profile below is the measured long-context
route; the short headline lane instead uses `Vulkan0,Vulkan1`. Keep the chosen
order fixed when comparing configurations:

```powershell
$env:LLAMA_OUTPUT_DEVICE = "Vulkan1"
$env:GGML_VK_FORCE_AMD_LARGE_MATMUL = "1"

build-vulkan\bin\llama-server.exe `
  -m models\Qwen3.6-27B-Q4_K_M.gguf `
  -c 131072 -b 8192 -ub 1024 -ngl 999 `
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on `
  -dev Vulkan1,Vulkan0 -sm layer -ts 1,1 `
  --spec-type none
```

Equal split is the current conservative general default; use autotune for a
specific context and residency target. For MTP, replace the final line with:

```powershell
--spec-type draft-mtp --spec-draft-n-max 3
```

The server's built-in MTP prefill window is 256 tokens. Override it only for a
controlled comparison:

```powershell
$env:LLAMA_SPEC_PREFILL_WINDOW = "512"
```

### ROCm Dual GPU

The reference MTP device order is:

```text
-dev ROCm1,ROCm0 -sm layer -ts 1,1
```

Direct HIP peer copy remains disabled by default on Windows/RDNA4. The safe
host-staged split route is used instead. Do not enable
`GGML_ROCM_ENABLE_PEER_COPY=1` as a production default without a fresh
correctness and driver-stability validation.

For prompt-heavy dual-GPU testing, the event-chained host-staging prototype is
available without enabling peer access:

```powershell
$env:GGML_ROCM_ASYNC_CROSS_DEVICE_STAGE = "1"
```

It improved the matched 30K prompt lane by about 2.7% and left mean decode
within noise. It remains opt-in pending larger-context driver validation. With
the reference ROCm order, leave `LLAMA_OUTPUT_DEVICE` unset: forcing output to
`ROCm1` adds a return transfer after the ROCm0 layers and severely reduces
long-prompt evaluation throughput.

The production long-context MTP profile needs no additional environment
variables:

```powershell
build-rocm-full\bin\llama-server.exe `
  -m models\Qwen3.6-27B-Q4_K_M.gguf `
  -c 49152 -b 8192 -ub 1024 -ngl 999 `
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on `
  -dev ROCm1,ROCm0 -sm layer -ts 1,1 `
  --spec-type draft-mtp --spec-draft-n-max 3
```

ROCm builds default to a 4096-row sparse anchor every 32768 prompt positions,
the latest 256 rows, KV-only draft prefill, staging preallocation, and
event-ordered device hidden-state handoff. For `-np 1`, ROCm also defaults to
one pipeline scheduler graph copy to avoid retaining duplicate long-context
arenas. Multi-request experiments can override this with
`GGML_SCHED_PIPELINE_COPIES=2` or `4`. Set
`LLAMA_SPEC_PREFILL_SPARSE_CHUNK=0` to disable the sparse anchors or
`LLAMA_MTP_DEVICE_HANDOFF=0` to restore the host hidden-state path for a
diagnostic comparison.

### Ternary Bonsai PQ2 Runtime Profile

Bonsai does not use Qwen NextN MTP. Its recommended dual-GPU starting profile
uses the same explicit ROCm order and the large prefill ubatch validated in
E331:

```powershell
build-rocm-full\bin\llama-server.exe `
  -m models\Ternary-Bonsai-27B-PQ2_0.gguf `
  -c 49152 -b 8192 -ub 1024 -ngl 999 `
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on `
  -dev ROCm1,ROCm0 -sm layer -ts 1,1 `
  --spec-type none
```

The model also fits on one 16 GB GPU. Use `-dev ROCm1` for the single-GPU
control. Dual GPU substantially raises prompt throughput, while single GPU can
retain a modest decode advantage because it avoids the layer-boundary transfer.
Vulkan does not yet implement the fork's `PQ2_0` kernels.

### Why GPU Order Matters

`-sm layer` is pipeline/layer placement, not symmetric tensor parallelism. The
first and second entries do not receive identical work: token embeddings,
repeating-layer ranges, recurrent state, output tensors, MTP NextN staging, and
scheduler copy boundaries are placed according to graph ownership and device
order. `LLAMA_OUTPUT_DEVICE` changes output placement but does not make the
rest of that topology symmetric.

Consequently, swapping two identical GPUs can change both transfer direction
and which device owns a synchronization-heavy graph boundary. On this machine,
Vulkan's short lane uses `Vulkan0,Vulkan1`, while the matched long lane uses
`Vulkan1,Vulkan0`; both place output on Vulkan1. ROCm's measured general order
is `ROCm1,ROCm0`. A mature tensor-parallel implementation would reduce this
asymmetry, but the current supported production mode is layer split.
