# Current Performance

Snapshot date: **2026-08-14** (Qwen3.8-27B-Q4_K_M rebaseline).

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
The Qwen3.8 rebaseline includes the W12 f8 decode indexing fix
(`a013f5230`); earlier Qwen3.6 f8 rows predate it and stay in
[Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md).

| Context | KV | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 12,288 | q8_0 | none | 7,958 / 128 | 1637.77 | **28.31** | 13.5852 | - |
| 12,288 | f8_e4m3 P5 | none | 7,958 / 128 | **1666.95** | 28.02 | **13.6357** | - |
| 12,288 | q8_0 + last8 f16 | MTP n2 | 7,958 / 128 | 1654.23 | **53.89** | **17.6927** | **84.2%** |
| 12,288 | f8_e4m3 P5 + last8 f16 | MTP n2 | 7,958 / 128 | 1599.87 | 48.26 | 16.5878 | 80.4% |
| 49,152 | q8_0 | none | 30,764 / 128 | 1532.79 | **26.05** | 5.1016 | - |
| 49,152 | f8_e4m3 P5 | none | 30,764 / 128 | 1524.51 | 24.78 | 5.0278 | - |
| 49,152 | q8_0 + last8 f16 | MTP n2 | 30,764 / 128 | 1637.44 | **48.56** | **5.9465** | **81.8%** |
| 49,152 | f8_e4m3 P5 + last8 f16 | MTP n2 | 30,764 / 128 | **1648.44** | 46.63 | **5.9526** | 74.4% |
| 98,304 | q8_0 | none | 58,186 / 128 | 1355.34 | **24.55** | **2.6480** | - |
| 98,304 | f8_e4m3 P5 | none | 58,186 / 128 | **1366.27** | 22.82 | 2.6460 | - |
| 98,304 | q8_0 + last8 f16 (r1) | MTP n2 | 58,120 / 256 | 1447.99 | **42.87** | 5.5278 | **71.5%** |
| 98,304 | **f8_e4m3 P5 + last12 f16** | MTP n2 | 58,120 / 256 | **1470.42** | 39.77 | **5.5502** | 60.8% |

On Qwen3.8 the FP8 prompt advantage over q8 has shrunk to roughly parity:
spec-none prompt throughput changes by `+1.8%`, `-0.5%` and `+0.8%` at
12K/49K/98K, while spec-none decode is `-1.0%`, `-4.9%` and `-7.1%`. MTP
aggregate changes by `-6.2%`, `+0.1%` and `+0.4%`. FP8 keeps the 5.88% main-KV
saving at every context, and MTP acceptance stays usable at 12K/49K
(`80.4%`/`74.4%` versus q8 `84.2%`/`81.8%`), but at 98K the last-12 f16 policy
lands at `60.8%` versus the q8 center's `71.5%`. The 98K last-12 f16 MTP
profile is therefore context-research material on Qwen3.8, not the default
recommendation. Artifacts use `q38-rb-vk-{12k,49k,98k}-{q8,f8}-{none,mtp2}-r1`
(98K q8 spec-none is r2 after a harness-timeout retry).

### Q4_K_M ROCm q8 vs native FP8

D098 adds byte-compatible HIP E4M3 cache conversion and a guarded native
gfx1201 `fp8 x fp8 -> fp32` FlashAttention body. The selected eight-wave body
uses `154-156 VGPR`, `29568 B` LDS and no spills/scratch. Focused reference,
KQ-only and full-native prefill/decode tests pass `6/6`. Qwen3.8 rows below
use the same contracts as the Qwen3.6 D098 table; the 49K spec-none bracket
keeps `b512/ub512` and 256 output tokens.

| Context | KV | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 49,152 | q8_0 center | none | 30,764 / 256 | 1647.17 | 21.53 | 8.3481 | - |
| 49,152 | **f8_e4m3 native** | none | 30,764 / 256 | **1713.67** | **22.20** | **8.6528** | - |
| 49,152 | q8_0 + last8 f16 center | MTP n2 | 30,764 / 128 | 1625.29 | 35.14 | 5.6373 | 70.7% |
| 49,152 | **f8_e4m3 native + last8 f16** | MTP n2 | 30,764 / 128 | **1716.79** | **39.96** | **6.0332** | **78.2%** |
| 98,304 | q8_0 + last8 f16 center | MTP n2 | 58,186 / 128 | 1431.43 | 29.47 | 2.8343 | **65.8%** |
| 98,304 | **f8_e4m3 native + last12 f16** | MTP n2 | 58,186 / 128 | **1481.94** | **31.60** | **2.9461** | 65.3% |

The 49K same-binary spec-none bracket gives full FP8 `+4.0%` prompt,
`+3.1%` decode and `+3.7%` aggregate versus q8. MTP also passes quality at
49K: `+5.6%` prompt, `+13.7%` decode and `+7.5 pp` acceptance. At 98K FP8
keeps `+3.5%` prompt and `+7.2%` decode, with acceptance parity
(`-0.5 pp`). The guarded RDNA4 D=256 F8/F8 route is enabled by default; set
`GGML_ROCM_FATTN_F8_NATIVE_KQ=0` for complete rollback or
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
| Vulkan short | 12,288 | 7,958 | 128 | 8192 / 1024 | q8_0 / q8_0 | 1 |
| ROCm short | 12,288 | 6,393 | 256 | 8192 / 1024 | q8_0 / q8_0 | 3 |
| Matched long, both backends | 49,152 | 30,764 | 128 | 8192 / 1024 | q8_0 / q8_0 | 1 |
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
