# llama.cpp-rdna-lab

`llama.cpp-rdna-lab` is a hardware-focused fork of
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) for local AI on
Windows with two AMD RDNA4 GPUs. It combines a maintained subset of the
`llama.cpp` runtime with a PyQt6 desktop application, reproducible
benchmark/autotune tooling, long-context work, and AMD-specific Vulkan and
ROCm/HIP optimizations.

The desktop application is branded **RDNA LLM Studio**. The repository keeps
upstream `llama-*` executable names and source structure so that runtime
compatibility and upstream synchronization remain straightforward.

The primary workload is agentic coding with Qwen3.6-27B: large cold prompts,
single-user requests, long contexts, tool use, vision, and speculative decode.
The main performance priority is prompt evaluation. MTP is kept only when its
decode gain does not impose an unacceptable prefill cost.

> This is a specialized research and production fork, not a drop-in replacement
> for every upstream platform. Results and defaults are tuned for the reference
> dual-RX 9070 XT machine described below.

## At a Glance

| Area | Current focus |
| --- | --- |
| Host platform | Windows 11 on AMD AM4 |
| Accelerators | 2x Radeon RX 9070 XT 16 GB (`gfx1201`) |
| Backends | ROCm/HIP, Vulkan, and CPU |
| Primary model | Qwen3.6-27B Q4_K_M with MTP |
| Experimental model | Ternary Bonsai 27B `PQ2_0` on CPU and ROCm |
| Main objective | Maximum cold prompt evaluation without sacrificing useful decode speed |
| Serving | OpenAI-compatible `llama-server` plus a PyQt6 desktop GUI |

The fork is currently substantially faster than the measured stock upstream
checkout on the same long-prompt contract. See
[Fork vs Stock Upstream](#fork-vs-stock-upstream) for the exact matched runs.

## Contents

- [Project Goals](#project-goals)
- [Supported Backends and Models](#supported-backends-and-models)
- [Reference System](#reference-system)
- [Current Performance](#current-performance)
- [Fork vs Stock Upstream](#fork-vs-stock-upstream)
- [Key Fork Features](#key-fork-features)
- [Quick Start](#quick-start)
- [Build Requirements](#build-requirements)
- [Recommended Runtime Profiles](#recommended-runtime-profiles)
- [MTP Behavior](#mtp-behavior)
- [Vision](#vision)
- [Benchmarking](#benchmarking)
- [Repository Layout](#repository-layout)
- [Development](#development)
- [Upstream and License](#upstream-and-license)

## Project Goals

- Maximize Qwen3.6 prompt-evaluation throughput for agent workloads.
- Use both GPUs without moving the active working set into system RAM.
- Make MTP improve decode while keeping long-prompt prefill close to baseline.
- Provide a practical GUI for building, launching, monitoring, and autotuning.
- Keep performance claims reproducible through cold, lane-locked benchmarks.
- Keep the fork maintainable by carrying only the backends and upstream changes
  that are useful on this machine.

Current non-goals include broad accelerator portability and native support for
NVIDIA CUDA, Metal, SYCL, OpenCL, CANN, or other removed upstream backends.

## Supported Backends and Models

| Backend | Role | Status |
| --- | --- | --- |
| ROCm/HIP | Primary prompt-eval, long-context MTP, and RDNA4 runtime | Supported and preferred for prompt-heavy MTP work |
| Vulkan | General AMD runtime and backend comparison | Supported; competitive for decode-heavy work; q8/MTP path fixed (D094) |
| CPU | Fallback, conversion, sanity checks, and tests | Supported |

ROCm still builds HIP-compatible kernels from `ggml/src/ggml-cuda`. That is an
internal HIP implementation detail and does not mean that this fork supports
NVIDIA hardware. See [Supported Backends](docs/SUPPORTED_BACKENDS.md).

### Model and Format Matrix

| Model / feature | CPU | ROCm/HIP | Vulkan | Notes |
| --- | --- | --- | --- | --- |
| Qwen3.6 GGUF, including Q3_K_S and Q4_K | Yes | Yes | Yes | Primary supported family |
| Qwen3.6 NextN MTP | Yes | Yes | Yes | Requires an MTP-enabled GGUF |
| Ternary Bonsai 27B `PQ2_0` | Yes | Yes | Not yet | Native loader, CPU kernels, and HIP MMQ/MMVQ path |
| Qwen3.6 vision projector | Yes | Yes | Yes | Use a matching `mmproj-*.gguf` |
| DFlash | Research | Research | Research | Not a recommended production profile |

D094 (2026-08-07, tested on `Qwen3.6-27B-Q4_K_M.gguf` and
`Qwen3.6-27B-Q3_K_S_mtp.gguf`, 2x RX 9070 XT): the Vulkan q8_0 vec/mmq
numerical divergence vs ROCm was root-caused and fixed (CUDA-style dp4a
accumulation, round-half-away q8_1 quantize, mmq variant-B math). MTP
acceptance recovered from 0.33 to 0.80+ (52k-token drafts; target 0.53) and
the f16-KV 49K lane now beats ROCm
(prompt 1719.92 vs 1679.20 tok/s, decode 43.10 vs 32.33, aggregate 6.1358 vs
5.7655 TPS). See [BENCHMARKS.md](BENCHMARKS.md) and
[Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md).

Q4_K_M is the primary practical Qwen model on this 2x16 GB machine. The
one-copy ROCm scheduler and bounded Q8 Flash Attention route make its measured
49K and 98K lanes viable. Q3_K_S remains the secondary choice for maximum
context/VRAM headroom, vision, and Q3-specific kernel research. `PQ2_0` is an
experimental Prism format and should not be confused with conventional `Q2_0`
quantization.

## Reference System

- Windows 11
- AMD Ryzen 7 5800X3D, 8 cores / 16 threads
- 64 GB system RAM
- 2x AMD Radeon RX 9070 XT, 16 GB VRAM each, RDNA4 `gfx1201`
- AMD ROCm/HIP SDK 7.1 for Windows
- AMD proprietary Vulkan driver
- Main model: `Qwen3.6-27B-Q4_K_M.gguf`
- Vision projector: `mmproj-F16.gguf`

The two GPUs are normally used with layer split, not tensor split. GPU1 is the
preferred output device because GPU0 also drives the desktop. Device order is
backend- and workload-sensitive even with two identical cards, and the best
Vulkan order is not identical for every lane. Exact routes are recorded with
each benchmark instead of being presented as a universal default. PCIe
topology, driver version, background GPU load, KV type, and model residency can
materially change the numbers below.

## Current Performance

Snapshot date: **2026-08-13**.

The fixed headline tables below retain their original model-scoped contracts.
Q3_K_S rows are historical/secondary evidence; the current primary Q4_K_M
rows are identified explicitly in the near-capacity and matched 49K sections.
All use FlashAttention, one server slot, cold prompt processing, no
prompt-cache reuse, and no prime pass. Compare `none` and `MTP` only inside the
same model, backend, and lane.

### Q4_K_M Vulkan q8 vs native FP8

The current dual-Vulkan refresh uses `Vulkan1,Vulkan0`, layer split `1,1`,
`b8192/ub1024` and identical cold repo-snapshot prompts. The 12K/49K rows and
98K spec-none rows use 128 output tokens. The current 98K MTP rows use the D097
256-token adjacent bracket: q8 is the center of its two controls and FP8 uses
the production last-12-f16 policy. FP8 uses the native P5 FlashAttention route.

| Context | KV | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 12,288 | q8_0 | none | 7,889 / 128 | 1672.17 | **28.47** | 13.80 | - |
| 12,288 | f8_e4m3 P5 | none | 7,889 / 128 | **1791.64** | 28.03 | **14.21** | - |
| 12,288 | q8_0 + last8 f16 | MTP n2 | 7,889 / 128 | 1686.74 | 45.55 | 17.01 | 65.45% |
| 12,288 | f8_e4m3 P5 + last8 f16 | MTP n2 | 7,889 / 128 | **1702.94** | **50.69** | **17.79** | **75.25%** |
| 49,152 | q8_0 | none | 30,723 / 128 | 1519.42 | **27.21** | 5.1146 | - |
| 49,152 | f8_e4m3 P5 | none | 30,723 / 128 | **1677.42** | 25.60 | **5.4627** | - |
| 49,152 | q8_0 + last8 f16 | MTP n2 | 30,723 / 128 | 1656.10 | 43.59 | 5.9288 | 66.06% |
| 49,152 | f8_e4m3 P5 + last8 f16 | MTP n2 | 30,723 / 128 | **1679.76** | **44.51** | **6.0176** | **67.59%** |
| 98,304 | q8_0 | none | 57,530 / 128 | 1314.53 | **25.07** | 2.6090 | - |
| 98,304 | f8_e4m3 P5 | none | 57,530 / 128 | **1480.18** | 21.98 | **2.8522** | - |
| 98,304 | q8_0 + last8 f16 (r2 center) | MTP n2 | 57,530 / 256 | 1422.71 | **41.96** | 5.4736 | 72.60% |
| 98,304 | **f8_e4m3 P5 + last12 f16** | MTP n2 | 57,530 / 256 | **1510.95** | 41.79 | **5.7618** | **73.79%** |

FP8 is the faster prompt-heavy profile: spec-none prompt throughput rises by
7.1%/10.4%/12.6% at 12K/49K/98K and main KV allocation falls by 5.88%.
With MTP it improves aggregate TPS at all three contexts. At 98K the current
last-12 policy reaches `73.79%` acceptance versus the q8 center's `72.60%`,
while improving prompt by `+6.20%` and aggregate TPS by `+5.27%`. The cost is
5376 versus 4704 MiB main KV. The superseded FP8 last8 result
(`1465.98/32.40/2.9494`, 51.61%) remains only in the D097 diagnosis record.
Full methodology, memory rows, corrected historical KV labels, and artifacts are in
[Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md).

### Q4_K_M ROCm q8 vs native FP8

D098 adds byte-compatible HIP E4M3 cache conversion and a guarded native
gfx1201 `fp8 x fp8 -> fp32` FlashAttention body. The selected eight-wave body
uses `154-156 VGPR`, `29568 B` LDS and no spills/scratch. Focused reference,
KQ-only and full-native prefill/decode tests pass `6/6`.

| Context | KV | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 49,152 | q8_0 center | none | 30,187 / 256 | 1686.75 | 21.83 | 8.62 | - |
| 49,152 | **f8_e4m3 native** | none | 30,187 / 256 | **1769.37** | **22.97** | **9.05** | - |
| 49,152 | q8_0 + last8 f16 center | MTP n2 | 30,187 / 128 | 1672.03 | 37.96 | 5.95 | 77.78% |
| 49,152 | **f8_e4m3 native + last8 f16** | MTP n2 | 30,187 / 128 | **1751.58** | **41.98** | **6.29** | **83.16%** |
| 98,304 | q8_0 + last8 f16 center | MTP n2 | 57,893 / 128 | 1443.54 | 33.50 | 2.905 | 79.59% |
| 98,304 | **f8_e4m3 native + last12 f16** | MTP n2 | 57,893 / 128 | **1482.86** | **35.82** | **2.99** | **81.25%** |

The 49K same-binary spec-none bracket gives full FP8 `+4.9%` prompt,
`+5.2%` decode and `+5.0%` aggregate versus q8. MTP also passes quality:
49K gains `+5.38 pp` acceptance and 98K gains `+1.66 pp`, with stable D091
`ROCm1,ROCm0` placement. The guarded RDNA4 D=256 F8/F8 route is enabled by
default; set `GGML_ROCM_FATTN_F8_NATIVE_KQ=0` for complete rollback or
`GGML_ROCM_FATTN_F8_NATIVE_V=0` for KQ-only diagnosis. Details and artifacts:
[D098](docs/research/major-topology/D098_Q4KM_ROCM_FP8_KICKOFF.md).

### Qwen3.6-35B-A3B Q4_K_M Vulkan q8 vs native FP8

The local `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` is 22.66 GB and therefore uses
both 16 GB GPUs plus WDDM-managed residency. This first 35B checkpoint uses
`Vulkan1,Vulkan0`, layer split `1,1`, `ctx=32768`, `b8192/ub8192`, one slot,
FlashAttention, no warmup/reuse/prime, and two cold repo-snapshot tasks with
21,381/21,362 prompt tokens and 128 output tokens each. Results are single-run
diagnostic rows, not an r3 promotion.

| KV | Mode | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance | Main KV MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| q8_0 | none | 1962.97 | 61.45 | 9.73 | - | 340 |
| **f8_e4m3 P5** | none | **2048.05** | **70.10** | **10.36** | - | **320** |
| q8_0 + last8 f16 | MTP n2 | 317.80 | **71.04** | 1.85 | 81.25% | 580 + 64 draft |
| **f8_e4m3 P5 + last8 f16** | MTP n2 | **325.17** | 67.36 | **1.89** | **86.02%** | **576 + 64 draft** |

On the recommended `spec=none` profile, native FP8 improves prompt by 4.33%,
decode by 14.08%, and aggregate TPS by 6.47% while reducing main KV by 5.88%.
MTP n2 is not recommended for prompt-heavy 35B use on this machine: its target
and draft contexts release inactive prompt-processing schedulers, and the
`ub8192` working set triggers visible WDDM dedicated-VRAM eviction/reload on
one GPU. Prompt throughput falls to roughly 320 tok/s even though generation
acceptance remains high. The four observed memory drops across this benchmark
match two scheduler lifecycles for each of the two tasks; they are not model
reloads initiated by the benchmark harness. Artifacts use
`d098-vk35b-32k-{q8,f8}-{none,mtp2}-r1`.

### Benchmark Launch Parameters

| Lane | Context | Actual prompt | Output | Batch / UBatch | KV | Repeats |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Vulkan short | 12,288 | 7,842 | 128 | 8192 / 1024 | q8_0 / q8_0 | 3 |
| ROCm short | 12,288 | 6,393 | 256 | 8192 / 1024 | q8_0 / q8_0 | 3 |
| Matched long, both backends | 49,152 | 29,561 | 128 | 8192 / 1024 | q8_0 / q8_0 | 1 |
| ROCm extended long | 65,536 | 41,058 | 128 | 8192 / 1024 | q8_0 / q8_0 | 1 |
| ROCm near-capacity | 131,072 | 72,295 | 64 | 8192 / 1024 | q8_0 / q8_0 | 1 |

Every row also uses `-np 1 -ngl 999 --flash-attn on --no-warmup -fit off`, seed
42, top-p 0.9, `--cache-ram 0`, `--ctx-checkpoints 0`, and no prompt reuse. The
short lanes use temperature 0.2; the deterministic long lanes
use temperature 0.0. The matched long lane injects 96,000 repository-snapshot
characters and produces 29,561 prompt tokens. The extended ROCm lane requests
147,456 characters; the current tree reaches its 144,287-character safe cap
and produces 41,058 prompt tokens.

Device routes are part of the benchmark contract. The short Vulkan lane uses
`-dev Vulkan0,Vulkan1`; the matched long and stock-comparison lanes use
`-dev Vulkan1,Vulkan0`. Both use `LLAMA_OUTPUT_DEVICE=Vulkan1`, layer split,
equal tensor split, and `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`. ROCm uses
`-dev ROCm1,ROCm0 -sm layer -ts 1,1` with direct peer copy disabled. MTP rows
add `--spec-type draft-mtp`; depth is 3 except for the ROCm short lane, where
the measured best is `--spec-draft-n-max 4`. ROCm MTP uses KV-only sparse
history by default: 4096 rows every 32768 prompt positions plus the latest 256
rows. Vulkan uses the 256-token recent window and host hidden-state handoff.
ROCm uses one pipeline scheduler graph copy for this single-request workload.

### Headline Fork Advantage

This compact view preserves the original matched 29,563-token A/B snapshot. It
is included so the stock comparison remains a coherent historical measurement;
the current ROCm long rebaseline is in the detailed tables below. Full
methodology for the archived comparison is in
[Fork vs Stock Upstream](#fork-vs-stock-upstream).

| Backend | Mode | Stock prompt / decode TPS | Fork prompt / decode TPS | Fork change |
| --- | --- | ---: | ---: | ---: |
| Vulkan | `none` | 930.11 / 21.58 | **1556.89 / 35.45** | **+67.39% / +64.27%** |
| Vulkan | MTP n3 | 861.48 / 17.77 | **1508.01 / 45.20** | **+75.05% / +154.36%** |
| ROCm | `none` | 1285.42 / 22.30 | **1787.94 / 25.21** | **+39.09% / +13.05%** |
| ROCm | MTP n3 | 1102.92 / 41.57 | **1721.97 / 42.02** | **+56.13% / +1.08%** |

### Short Prompt Lanes

| Backend | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | `none`, r3 mean | 7,842 / 128 | **1783.49** | 38.17 | 16.42 | `Vulkan0,Vulkan1`, `ctx=12288`, q8/q8 KV |
| Vulkan | MTP n3, r3 mean | 7,842 / 128 | 1724.73 | **51.82** | **17.99** | 60.05% acceptance; backend-resident NextN |
| ROCm | `none`, r3 mean | 6,393 / 256 | **1850.13** | 27.67 | 20.07 | `ROCm1,ROCm0`, `ctx=12288`, q8/q8 KV |
| ROCm | MTP n4, r3 mean | 6,393 / 256 | 1794.17 | **41.39** | **26.12** | 59.27% acceptance; backend-resident NextN |

In this lane, Vulkan MTP changes prompt/decode/aggregate throughput by
`-3.29% / +35.78% / +9.55%`. ROCm MTP changes them by
`-3.02% / +49.58% / +30.14%`. The refreshed ROCm artifacts start with
`e330-rocm-dual-q3-12k-`.

### Long Prompt Lanes

| Backend | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | `none` | 29,563 / 128 | 1556.89 | 35.45 | 5.65 | `ctx=49152`, `b8192/ub1024`, q8/q8 KV |
| Vulkan | MTP n3 | 29,563 / 128 | 1508.01 | **45.20** | 5.69 | 52.38% acceptance; backend-specific host handoff |
| ROCm | `none` | 29,561 / 128 | **1734.14** | 25.77 | 5.79 | `ctx=49152`, `b8192/ub1024`, q8/q8 KV |
| ROCm | MTP n3 | 29,561 / 128 | 1672.05 | 35.42 | **5.99** | 63.08% acceptance; sparse KV-only history |

On the matched 29.5k lane, Vulkan MTP changes prompt/decode throughput by
`-3.14% / +27.50%`. The current ROCm rebaseline changes
prompt/decode/aggregate throughput by `-3.58% / +37.45% / +3.35%`. ROCm MTP
is 10.9% faster in prompt evaluation and 5.3% faster in aggregate than the
recorded Vulkan MTP row, while Vulkan retains a 27.6% decode advantage.

### Ternary Bonsai PQ2 Performance

The first table preserves the pre-optimization functional baseline for
`Ternary-Bonsai-27B-PQ2_0.gguf`. It uses the same ROCm benchmark contracts as
the Qwen rows: FlashAttention, `b8192/ub1024`, q8/q8 KV, one slot, full GPU
offload, cold prompts, no cache reuse, and `spec=none`. Single GPU means
`ROCm1`; dual GPU means `ROCm1,ROCm0 -sm layer -ts 1,1`.

| Model | GPUs | Lane | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B Q3_K_S | dual | short, r3 mean | 6,393 / 256 | 1850.13 | 27.67 | 20.07 |
| Bonsai-27B PQ2 | single | short, r3 mean | 6,393 / 256 | 1189.20 | **50.30** | 24.39 |
| Bonsai-27B PQ2 | dual | short, r3 mean | 6,393 / 256 | **1858.69** | 45.40 | **28.06** |
| Qwen3.6-27B Q3_K_S | dual | matched long | 29,561 / 128 | **1734.14** | 25.77 | 5.79 |
| Bonsai-27B PQ2 | single | matched long | 29,561 / 128 | 1046.07 | **41.55** | 4.08 |
| Bonsai-27B PQ2 | dual | matched long | 29,561 / 128 | 1779.50 | 37.72 | **6.38** |

The long rows are directly comparable: both inject the same 96,000-character
repository snapshot and differ by only two tokenizer tokens. The refreshed
short Qwen and Bonsai rows are also directly comparable: both consume the same
current 18,851-character snapshot and produce 6,393 prompt tokens.

For Bonsai, dual GPU raises prompt throughput by 56.3% on the short lane and
70.1% on the matched long lane. The layer boundary reduces decode by 9.7% and
9.2%, respectively, but dual remains faster in aggregate. These rows preserve
the initial functional PQ2 HIP port before kernel optimization. Artifact labels
start with `e322-bonsai-pq2-`.

The current native PQ2 path includes a dedicated HIP MMQ/MMVQ implementation.
A later controlled long-prompt run isolated `ubatch` from device placement:

| Devices | Prompt / output | Batch / UBatch | Prompt TPS | Decode TPS | Aggregate TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ROCm1,ROCm0` | 32,085 / 128 | 8192 / 128 | 1067.99 | 36.35 | 3.81 |
| `ROCm1,ROCm0` | 32,085 / 128 | 8192 / 1024 | **1819.10** | **36.59** | **6.04** |

Raising `ubatch` from 128 to 1024 improved prompt throughput by 70.33% without
reducing decode. The server releases the inactive prompt-processing scheduler
after prefill and uses a separate one-token generation graph, so a large
prefill ubatch does not need a decode workaround. The earlier lower GUI decode
result came from the old automatic `ROCm0,ROCm1` order. New GUI configurations
now default to the measured `ROCm1,ROCm0` route while keeping every manual
device order available. See
[E331: Bonsai PQ2 ubatch/decode isolation](docs/research/experiments/E331_bonsai_pq2_ubatch_decode_isolation.md).

### Fork vs Stock Upstream

This section is an archived A/B snapshot. The same earlier 29,563-token lane
was run against stock
[`ggml-org/llama.cpp` commit `f955e394b`](https://github.com/ggml-org/llama.cpp/commit/f955e394b)
from a neighboring clean checkout. The model, generated prompt, output length,
sampling, context, batch/ubatch, KV types, device order, layer split, and server
cache settings matched inside that snapshot. Its fork rows intentionally remain
unchanged; current ROCm rows use the refreshed 29,561-token repository snapshot.

| Implementation | Backend | Mode | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Stock `f955e394b` | Vulkan | `none` | 930.11 | 21.58 | 3.38 | - |
| Stock `f955e394b` | Vulkan | MTP n3 | 861.48 | 17.77 | 3.08 | 71.67% |
| This fork | Vulkan | `none` | **1556.89** | **35.45** | **5.65** | - |
| This fork | Vulkan | MTP n3 | **1508.01** | **45.20** | **5.69** | 52.38% |
| Stock `f955e394b` | ROCm | `none` | 1285.42 | 22.30 | 4.44 | - |
| Stock `f955e394b` | ROCm | MTP n3 | 1102.92 | 41.57 | 4.27 | 78.07% |
| This fork | ROCm | `none` | **1787.94** | **25.21** | **5.91** | - |
| This fork | ROCm | MTP n3 | **1721.97** | **42.02** | **6.31** | 75.86% |

Against stock, the fork improves Vulkan `none` prompt/decode throughput by
`+67.39% / +64.27%` and Vulkan MTP by `+75.05% / +154.36%`. The ROCm gains are
`+39.09% / +13.05%` for `none` and `+56.13% / +1.08%` for MTP. Stock ROCm MTP
already has strong acceptance and decode, but reduces prompt throughput by
14.2%; the fork's sparse KV-only history keeps nearly the same decode rate
while recovering most of that prompt cost and raising aggregate throughput by
47.78% over stock ROCm MTP.

The stock Vulkan build used GCC 13.2 and Vulkan SDK 1.4.350. The stock ROCm
build used HIP SDK 7.1/Clang 21, `gfx1201`, MFMA, no HIP VMM, upstream's default
generic FlashAttention path, and direct peer copy disabled for this Windows
dual-GPU system. A build-only MSVC 14.44 header selection was required because
HIP SDK 7.1 is incompatible with the installed MSVC 14.51 `<cmath>`; no stock
model, graph, kernel, scheduler, or speculative-decoding source was changed.
The stock tree does not implement the fork-specific `LLAMA_OUTPUT_DEVICE` or
`GGML_VK_FORCE_AMD_LARGE_MATMUL` controls.

### Extended ROCm Long Prompt

| Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 41,058 / 128 | **1630.59** | 24.96 | **4.2096** | - |
| MTP n3 | 41,058 / 128 | 1546.88 | **33.92** | 4.2062 | 68.00% |

At 41.1k tokens, sparse-history MTP costs 5.13% prompt throughput and gains
35.90% decode throughput. Aggregate throughput is effectively neutral
(`-0.08%`) for this 128-token answer, so MTP remains most useful when the
generated answer is longer. The current artifacts use the `e335-rocm-q3ks-`
prefix.

### Near-Capacity ROCm Prompt

This lane uses `ctx=131072`, a 278,083-character repository snapshot, 72,295
prompt tokens, 64 output tokens, and the production one-copy ROCm scheduler.

| Mode | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance | Prefill dedicated | Prefill Shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | **1439.89** | 21.90 | **1.20** | - | 19.35 GiB | 3.51 GiB |
| MTP n3 | 1363.95 | **32.53** | 1.16 | 74.14% | 21.71 GiB | 3.58 GiB |

MTP costs 5.27% prompt throughput and gains 48.5% decode throughput. Shared
changes by only about 62 MiB during prefill, so n3 does not create a separate
RAM-residency cliff here. The 64-token request remains prompt-dominated; MTP's
wall-time benefit starts with longer generated answers.

The MTP-enabled Q4_K_M GGUF was also validated at `ctx=98304` with a 59,045
token prompt and 64 output tokens. `none` measured `1493.21/19.15` prompt/decode
tok/s; MTP n3 measured `1435.97/35.44` with 80.00% acceptance. MTP therefore
costs 3.83% prompt throughput and gains 85.1% decode. Prefill Shared changes
only from 3.204 to 3.261 GiB, while the additional 1.91 GiB is Dedicated.

The current Q4_K_M kernel profile was revalidated on the matched 49K lane with
a 29,561-token prompt and 128 output tokens. Baseline measures `1778.59/21.98`
prompt/decode tok/s and `5.6829` aggregate TPS. MTP n3 measures
`1731.71/39.58`, `6.2802` aggregate TPS, and 74.36% acceptance: a 2.64% prompt
cost, 80.11% decode gain, and 10.51% end-to-end gain. E344 widens only Q4_K and
Q5_K prompt MMQ geometry; E345 keeps Q6_K prompt on hipBLAS and improves its
RDNA4 MMVQ decode policy.

After these fixed-lane tables were recorded, E292 promoted a packed HIP Q3_K
staging kernel. Matched A/B runs improved ROCm prompt evaluation by
`+0.72%` to `+1.52%` across 7.8k-30.1k-token prompts. The table values remain
unchanged because the repository snapshot, and therefore exact prompt token
count, had changed by the time E292 was measured. Set
`GGML_CUDA_Q3K_PADDED_DEQUANT_PACKED=0` to restore the previous staging kernel.

E293 then restored the rocWMMA FlashAttention path that was disabled in fresh
ROCm build caches. On the full production profile, a matched 11,561-token r3
lane improved prompt/decode/aggregate throughput from
`1713.61 / 28.02 / 2.1696` to `1930.26 / 30.71 / 2.4403 tok/s`. On a matched
30,075-token lane, prompt evaluation improved `1369.24 -> 1761.34 tok/s`
(`+28.64%`) and server evaluation time fell `22.54 -> 17.65 s`; decode was
neutral within single-run noise. At `ctx=131072`, a matched 53,523-token prompt
improved `1091.68 -> 1557.94 tok/s` (`+42.71%`) and wall time fell
`49.85 -> 35.16 s`. Fresh HIP builds now enable rocWMMA by default and
discover the bundled headers automatically. Configure with
`-DGGML_HIP_ROCWMMA_FATTN=OFF` for the generic-tile rollback.

E337 removes the remaining context-sized F16 staging for the RDNA4 Q8 K/V
rocWMMA path. It converts one bounded 4096-token K/V chunk, reuses the fast
WMMA kernel, and combines chunk-local softmax outputs online. A matched
one-card 49K/29.5K lane recovered 216 MiB (`1282 -> 1066 MiB` unaccounted)
while prompt/decode throughput stayed neutral (`1044.47/31.07 ->
1045.61/31.31 tok/s`). The automatic policy keeps short contexts on the
standard WMMA route.

E338 identifies the larger dual-GPU Shared source as duplicated split-graph
scheduler arenas, not a growing KV cache. ROCm now uses one graph copy by
default for `-np 1`. In the Q4 98K lane this reduced prefill Dedicated/Shared
from `23.85/5.46` to `22.05/3.20 GiB` without reducing prompt throughput. The
environment variable `GGML_SCHED_PIPELINE_COPIES=1..4` remains available for
controlled multi-request experiments.

E315 adds ROCm KV-only sparse MTP history and event-ordered backend handoff.
The long-prompt acceptance improvement is not a ROCm numerical workaround:
matched target-prefix traces showed equal backend acceptance when both paths
received the same history. The new policy retains selected long-range KV blocks
without evaluating the entire draft layer over the prompt. It raises acceptance
to 75.86% at 29.5k and 68.55% at 41.1k on its recorded output. The current
E335 rebaseline measures 63.08% and 68.00%, respectively; acceptance is output
and prompt-content dependent, so the archived and current samples are kept
separate.

Q4_K_M and UD-Q4_K_XL are supported. Windows still reports WDDM Shared for the
27B Q4 long-context working set, but E337/E338 removed the old active-residency
cliff: the Q4_K_M 98K lane improved from 553.50 to 1493.21 prompt tok/s while
Shared fell from 6.25 to 3.20 GiB. Q4_K_M is now the primary production and
performance model. Q3_K_S remains available when vision or maximum context
headroom matters; its 2000 prompt tok/s program remains model-specific history.

Evidence:

- [E331: Bonsai PQ2 ubatch/decode isolation](docs/research/experiments/E331_bonsai_pq2_ubatch_decode_isolation.md)
- [E337: bounded ROCm Q8 FlashAttention WMMA](docs/research/experiments/E337_rocm_q8_chunked_wmma.md)
- [E338: ROCm dual-GPU long-context scheduler residency](docs/research/experiments/E338_rocm_dual_long_context_scheduler_residency.md)
- [E344: ROCm Q4_K/Q5_K MMQ geometry](docs/research/experiments/E344_rocm_q4q5_type_specific_mmq_geometry.md)
- [E345: ROCm Q6_K route and MMVQ policy](docs/research/experiments/E345_rocm_q6_route_and_smallk.md)
- [E291: ROCm long-context Q3_K decode and memory](docs/research/experiments/E291_rocm_long_context_q3k_decode_and_memory.md)
- [E292: ROCm packed padded-Q3_K dequant](docs/research/experiments/E292_rocm_q3k_packed_dequant_probe.md)
- [E293: ROCm RDNA4 rocWMMA FlashAttention restore](docs/research/experiments/E293_rocm_rdna4_rocwmma_fattn_restore.md)
- [E315: ROCm long-context MTP sparse history](docs/research/experiments/E315_rocm_long_context_mtp_sparse_history.md)
- [E289: ROCm Q3_K packed subtract](docs/research/experiments/E289_rocm_q3k_packed_sub4.md)
- [E284: matched 49K-context README lane](docs/research/experiments/E284_matched_49k_context_readme_lane.md)
- [E283: clean README revalidation](docs/research/experiments/E283_clean_readme_revalidation.md)
- [E282: MTP device hidden-state handoff](docs/research/experiments/E282_mtp_device_hidden_handoff.md)
- [D078: ROCm RDNA4 small-N DP4A MTP](docs/research/major-topology/D078_P002_ROCM_MTP_SMALLN_DP4A_MMQ.md)
- [D080: Vulkan layer-stage balance](docs/research/major-topology/D080_P003_VULKAN_LAYER_STAGE_BALANCE.md)
- [Canonical benchmark history](build_logs/agent-workload/BENCH_RUNS.csv)
- [Benchmark notes](BENCHMARKS.md)

## Key Fork Features

- PyQt6 GUI for dependency checks, builds, server launch, monitoring, and logs.
- Vulkan/ROCm-aware benchmark and autotune UI with live prompt progress.
- OpenAI-compatible `llama-server` for local applications and coding agents.
- Dual-GPU layer placement and explicit output-device controls.
- Upstream-style Qwen3.6 MTP pipeline with backend-resident NextN handoff.
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

## Quick Start

Install Python GUI dependencies and launch the application from the repository
root:

```powershell
python -m pip install -r gui/requirements-gui.txt
python run.py
```

`run.bat` and `start-gui.bat` are also available. In the GUI:

1. Open **Build & Setup** and configure Vulkan, ROCm/HIP, or CPU.
2. Build `llama-server` or select an existing compatible build.
3. Open **Launch Server** and select a local GGUF model.
4. Start with `Spec: None` to establish a baseline.
5. For an MTP-enabled GGUF, select MTP and use depth 3 as the current general
   Vulkan/ROCm starting point.
6. For vision, enable the projector and select `models/mmproj-F16.gguf`.
7. In **Benchmark / Autotune**, use the recommended explicit device order for
   reproducible dual-GPU tests. `Auto` remains useful for discovery, but it is
   not a stable benchmark contract.
8. Validate batch, ubatch, KV, split, and spec settings at the intended context
   length. Short-prompt winners do not automatically remain best at 49K.

Model files are not part of the source tree history. Put local GGUF files in
`models/` or select them from another local directory.

## Build Requirements

The reference builds are Windows x64 builds. A clean machine needs:

- Git and 64-bit Python 3.11 or newer with `pip`;
- CMake 3.14 or newer and Ninja (tested with CMake 3.29 and Ninja 1.12);
- Visual Studio Build Tools 2022 with **Desktop development with C++**, the
  MSVC v143 toolset, and a Windows 10 or 11 SDK;
- the current AMD display driver, including the Vulkan runtime;
- full LunarG Vulkan SDK with `glslc`, `spirv-as`, `spirv-dis`, and
  `spirv-val` for Vulkan/FP8 shader builds;
- AMD ROCm/HIP SDK 7.1 for Windows for ROCm builds;
- Strawberry Perl for Windows ROCm configuration and the reference MinGW
  Vulkan toolchain;
- OpenSSL development files. HTTPS is enabled by default; use
  `-DLLAMA_OPENSSL=OFF` only when HTTPS/model downloads are not required.

The tested Vulkan build uses the GCC 13.2 MinGW-w64 toolchain bundled with
Strawberry Perl. A MinGW executable also needs `libgcc_s_seh-1.dll`,
`libstdc++-6.dll`, and `libwinpthread-1.dll` either beside the executable or on
`PATH`. The GUI launch environment handles the configured toolchain; for a
manual launch, put `C:\Strawberry\c\bin` before other MinGW installations on
`PATH` to avoid loading incompatible runtime DLLs.

The tested ROCm build uses `clang.exe` and `clang++.exe` from HIP SDK 7.1, not
MSVC as the compiler, but still links against MSVC v143 and Windows SDK host
libraries. Strawberry Perl is also required. A full HIP compilation is memory
intensive; 64 GB RAM and `-j 4` are recommended for this fork. Allow roughly
30 GB of free disk space for source, two build trees, and one local model.

Install the Python side and verify the native tools before opening the GUI:

```powershell
python -m pip install --upgrade pip
python -m pip install -r gui/requirements-gui.txt
cmake --version
ninja --version
glslc --version
spirv-as --version
spirv-dis --version
spirv-val --version
```

The GUI's **Build & Setup** tab checks the configured dependencies and creates
backend-specific build directories. Manual equivalents are shown below.

### CPU

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4 --target llama-server
```

### Vulkan

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\<version>"
$env:PATH = "$env:VULKAN_SDK\Bin;C:\Strawberry\c\bin;$env:PATH"

cmake -S . -B build-vulkan -G Ninja `
  -DGGML_VULKAN=ON `
  -DCMAKE_C_COMPILER=C:\Strawberry\c\bin\gcc.exe `
  -DCMAKE_CXX_COMPILER=C:\Strawberry\c\bin\g++.exe `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 4 --target llama-server
```

### ROCm/HIP on Windows RDNA4

```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.1"
$env:ROCM_PATH = $env:HIP_PATH
$env:CMAKE_PREFIX_PATH = "$env:HIP_PATH\lib\cmake"
$env:PATH = "$env:HIP_PATH\bin;C:\Strawberry\perl\bin;C:\Strawberry\c\bin;$env:PATH"

cmake -S . -B build-rocm -G Ninja `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS=gfx1201 `
  -DGGML_HIP_MMQ_MFMA=ON `
  -DGGML_HIP_ROCWMMA_FATTN=ON `
  -DGGML_HIP_NO_VMM=ON `
  -DGGML_OPENMP=OFF `
  -DCMAKE_C_COMPILER="$env:HIP_PATH\bin\clang.exe" `
  -DCMAKE_CXX_COMPILER="$env:HIP_PATH\bin\clang++.exe" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm -j 4 --target llama-server
```

ROCm uses clang from the HIP SDK but still needs the Windows SDK and MSVC host
libraries. Missing `kernel32.lib`, `msvcrtd.lib`, or similar files indicates an
incomplete Build Tools environment. See the full [Build Guide](docs/build.md).
The fork includes rocWMMA 7.1 headers under `third_party/rocwmma`; no separate
rocWMMA SDK install is required for the command above.

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

## MTP Behavior

MTP accelerates token generation; it does not make the target model's prompt
prefill free. ROCm uses selected long-range KV blocks plus the recent prompt
tail and keeps NextN hidden states on their backend, avoiding a complete draft
prefill and the previous GPU-to-RAM-to-GPU round trip. Vulkan uses a host
handoff by default because keeping unmasked NextN output resident over the
whole Vulkan prompt was substantially slower.

Practical rules:

- Use at least 128 output tokens when benchmarking MTP. Very short runs are
  dominated by the first target-verification graph.
- Compare MTP and `none` with the same model, prompt, output length, KV type,
  batch/ubatch, device split, and background load.
- Depth 3 is the current robust starting point. Higher depth is not
  automatically faster because acceptance falls and verification batches grow.
- For prompt-dominated requests with short answers, `none` can still win wall
  time even when MTP decode is much faster.
- Non-zero Windows Shared memory is not by itself proof that MTP is reading KV
  from RAM. Check for a throughput cliff and compare process Dedicated/Shared;
  the current 72K lane adds only about 62 MiB Shared during MTP prefill.
- Set `LLAMA_MTP_DEVICE_HANDOFF=0` only as a diagnostic rollback to the old host
  hidden-state path.

## Vision

Qwen3.6 vision requires a projector that matches the text model architecture
and embedding dimension. In the GUI, enable Vision and select
`models/mmproj-F16.gguf`. The equivalent server argument is:

```text
--mmproj models/mmproj-F16.gguf
```

Use `Spec: None` for the first image request so vision-pipeline issues can be
separated from speculative decoding.

## Benchmarking

The canonical runner starts an isolated OpenAI-compatible server, injects a
real repository snapshot, records prompt/decode timings, and updates the live
history files:

```powershell
python scripts/agent_workload_bench.py --help
```

Important history files:

- `build_logs/agent-workload/BENCH_RUNS.csv`
- `build_logs/agent-workload/BENCH_RECENT.md`
- `build_logs/agent-workload/BENCH_LANES.md`
- `docs/research/RESULTS_LOG.md`

Performance work should use neighboring controls. Background GPU applications,
driver power state, warm shader caches, prompt-cache reuse, or a different
output length can otherwise create a false improvement. Record an explicit
`-dev` route for every dual-GPU result; the GUI now defaults new ROCm and Vulkan
benchmark configurations to the measured recommended order instead of `Auto`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `gui/` | PyQt6 desktop application |
| `src/`, `common/`, `include/` | llama runtime and speculative pipeline |
| `ggml/src/ggml-vulkan/` | Vulkan backend and generated shaders |
| `ggml/src/ggml-hip/` | ROCm/HIP build integration |
| `ggml/src/ggml-cuda/` | Shared HIP-compatible kernel implementation |
| `ggml/src/ggml-cpu/` | CPU backend |
| `scripts/agent_workload_bench.py` | Benchmark and autotune runner |
| `docs/research/` | Accepted, rejected, and diagnostic performance work |
| `docs/vulkan/` | Vulkan architecture and validation rules |

## Development

Read [AGENTS.md](AGENTS.md) before changing the fork. Upstream changes are
ported selectively according to [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md); removed
backends are not restored automatically during synchronization.

When reporting performance, include the model, backend, device order, split,
context, actual prompt tokens, output tokens, batch/ubatch, KV types, speculative
mode, cache policy, and background load. A faster isolated number is useful only
when its lane and tradeoffs are visible.

## Upstream and License

The runtime is derived from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
Upstream changes are reviewed and ported selectively so they do not silently
restore removed backends or invalidate AMD-specific behavior. This repository
is distributed under the [MIT License](LICENSE); bundled third-party components
retain their own notices and licenses.
