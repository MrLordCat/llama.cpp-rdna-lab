# Qwen3.8-27B Q4_K_M Results

Status (2026-08-14): Qwen3.8-27B-Q4_K_M is now the primary production and
performance model of the fork. The rebaseline below re-runs the full README
Q4_K_M matrix (Vulkan q8/f8 at 12K/49K/98K, ROCm q8/f8 at 49K/98K, spec-none
and MTP n2) on the Qwen3.8 GGUF with the same contracts. It also includes the
W12 f8 decode indexing fix (`a013f5230`): the earlier Qwen3.6 f8 rows predate
that fix and stay in the historical sections. The safe default research lane
remains dual-ROCm `ctx=49152` with q8 KV and an adjacent spec-none control.

ROCm short and 49K rows were refreshed on 2026-07-16 after the E344 Q4_K/Q5_K
prompt geometry and E345 Q6_K decode policy. The 98K production row remains the
E337/E338 bounded-Q8 and one-copy scheduler measurement. No foreground GPU
workload was active during the 2026-08-11 Vulkan refresh.

Updated 2026-08-07 (D094): Vulkan q8/MTP path fixed. The numerical divergence
in the Vulkan q8_0 vec/mmq path (CUDA-style dp4a accumulation order,
round-half-away quantize, mmq variant-B math) was resolved; MTP acceptance
recovered from 0.33 to 0.80+ (target 0.53) and Vulkan now beats ROCm on the
f16-KV 49K autotune lane. Fresh GUI autotune rows (f16 KV, ctx 49152, no game)
are listed below.

## Test system

- 2x AMD Radeon RX 9070 XT 16 GB
- `Qwen3.8-27B-Q4_K_M.gguf` (17.1 GiB)
- Windows 11, ROCm/HIP 7.1 and Vulkan
- dual-GPU layer split, one server slot, full GPU offload
- FlashAttention, `b8192/ub1024` (ROCm 49K spec-none bracket: `b512/ub512`)
- current Vulkan matrix: q8_0/q8_0 and f8_e4m3/f8_e4m3 KV
- MTP n2 refresh rows: last 8 full-attention KV layers are f16; current 98K
	FP8 MTP automatically uses last 12 after D097
- cold prompts, no cache reuse, no warmup, seed 42
- artifacts: `q38-rb-vk-{12k,49k,98k}-{q8,f8}-{none,mtp2}-r1`
	(98K q8 spec-none = r2 after a harness 45 s hard-timeout retry) and
	`q38-rb-rc-{49k,98k}-{q8,f8}-{none,mtp2}-r1`

## Current Vulkan q8 vs FP8 refresh (2026-08-14, Qwen3.8 rebaseline)

All rows use the same binary, `Vulkan1,Vulkan0`, layer split `1,1`, one slot,
`b8192/ub1024`, FlashAttention and cold/no-reuse/no-prime execution. The
12K/49K rows and 98K spec-none rows use 128 output tokens. The current 98K MTP
rows use the D097 256-token adjacent bracket: q8 is the center of its two
controls and FP8 uses the production last-12-f16 policy.
`GGML_VK_FA_F8_P5=1` affects only FP8; q8 ignores it.

| Context | KV | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance | Main KV MiB |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12,288 | q8_0 | none | 7,958 / 128 | 1637.77 | **28.31** | 13.5852 | - | 408 |
| 12,288 | f8_e4m3 P5 | none | 7,958 / 128 | **1666.95** | 28.02 | **13.6357** | - | **384** |
| 12,288 | q8_0 + last8 f16 | MTP n2 | 7,958 / 128 | 1654.23 | **53.89** | **17.6927** | **84.2%** | 588 |
| 12,288 | f8_e4m3 P5 + last8 f16 | MTP n2 | 7,958 / 128 | 1599.87 | 48.26 | 16.5878 | 80.4% | **576** |
| 49,152 | q8_0 | none | 30,764 / 128 | 1532.79 | **26.05** | 5.1016 | - | 1632 |
| 49,152 | f8_e4m3 P5 | none | 30,764 / 128 | 1524.51 | 24.78 | 5.0278 | - | **1536** |
| 49,152 | q8_0 + last8 f16 | MTP n2 | 30,764 / 128 | 1637.44 | **48.56** | **5.9465** | **81.8%** | 2352 |
| 49,152 | f8_e4m3 P5 + last8 f16 | MTP n2 | 30,764 / 128 | **1648.44** | 46.63 | **5.9526** | 74.4% | **2304** |
| 98,304 | q8_0 | none | 58,186 / 128 | 1355.34 | **24.55** | **2.6480** | - | 3264 |
| 98,304 | f8_e4m3 P5 | none | 58,186 / 128 | **1366.27** | 22.82 | 2.6460 | - | **3072** |
| 98,304 | q8_0 + last8 f16 (r1) | MTP n2 | 58,120 / 256 | 1447.99 | **42.87** | 5.5278 | **71.5%** | **4704** |
| 98,304 | **f8_e4m3 P5 + last12 f16** | MTP n2 | 58,120 / 256 | **1470.42** | 39.77 | **5.5502** | 60.8% | 5376 |

On Qwen3.8 the FP8 prompt edge over q8 is roughly parity: spec-none prompt
throughput changes by `+1.8%`, `-0.5%` and `+0.8%` at 12K/49K/98K; spec-none
decode changes by `-1.0%`, `-4.9%` and `-7.1%`; aggregate by `+0.4%`,
`-1.5%` and `-0.1%`. MTP aggregate changes by `-6.2%`, `+0.1%` and `+0.4%`.
Acceptance: FP8 MTP lands at `80.4%`/`74.4%`/`60.8%` versus q8
`84.2%`/`81.8%`/`71.5%` at 12K/49K/98K. At 98K the last-12 f16 policy loses
`10.7 pp` to the q8 center on Qwen3.8, so it stays research material there;
at 12K/49K FP8 remains the memory-saving option (main KV -5.88% spec-none,
-2.04% with the matched last-8 f16 MTP policy) with comparable aggregate TPS.

### ROCm q8 vs native FP8 (2026-08-14, Qwen3.8 rebaseline)

Same D098 guarded native gfx1201 F8/F8 body as the Qwen3.6 table.
`ROCm1,ROCm0 -sm layer -ts 1,1`; the 49K spec-none bracket uses `b512/ub512`
and 256 output tokens.

| Context | KV | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 49,152 | q8_0 center | none | 30,764 / 256 | 1647.17 | 21.53 | 8.3481 | - |
| 49,152 | **f8_e4m3 native** | none | 30,764 / 256 | **1713.67** | **22.20** | **8.6528** | - |
| 49,152 | q8_0 + last8 f16 center | MTP n2 | 30,764 / 128 | 1625.29 | 35.14 | 5.6373 | 70.7% |
| 49,152 | **f8_e4m3 native + last8 f16** | MTP n2 | 30,764 / 128 | **1716.79** | **39.96** | **6.0332** | **78.2%** |
| 98,304 | q8_0 + last8 f16 center | MTP n2 | 58,186 / 128 | 1431.43 | 29.47 | 2.8343 | **65.8%** |
| 98,304 | **f8_e4m3 native + last12 f16** | MTP n2 | 58,186 / 128 | **1481.94** | **31.60** | **2.9461** | 65.3% |

49K spec-none: FP8 `+4.0%` prompt, `+3.1%` decode, `+3.7%` aggregate. 49K MTP:
`+5.6%` prompt, `+13.7%` decode, `+7.5 pp` acceptance. 98K MTP: `+3.5%` prompt,
`+7.2%` decode, acceptance parity `-0.5 pp`.

## Qwen3.6 historical refresh (2026-08-11, superseded by the rebaseline above)

## D097 98K FP8 MTP acceptance recovery (2026-08-11)

The superseded 128-token FP8 last8 refresh diagnosed the problem and remains in
the D095/D097 research artifacts, not in the current headline table above. Raw
E4M3 is not more precise than q8_0 for this KV distribution: it has three
mantissa bits and no block scale, while q8_0 has an int8 payload plus a scale
per 32 values. A real-KV scout measured raw-E4M3 attention-logit MSE about
29.9x higher than q8_0.

The fixed 57,530-token/256-output adjacent bracket is:

| KV policy | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance | Main KV MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| q8 last8, control A | 1430.89 | 42.09 | 5.5039 | 72.60% | 4704 |
| **FP8 P5 last12** | **1510.95** | 41.79 | **5.7618** | **73.79%** | 5376 |
| FP8/q8 bridge M6 + last8 | 1427.45 | **47.05** | 5.5687 | **90.61%** | **4680** |
| q8 last8, control B | 1414.53 | 41.83 | 5.4432 | 72.60% | 4704 |

FP8 last12 beats the q8 control center by `+6.20%` prompt and `+5.27%`
aggregate, matches decode within `-0.41%`, and raises acceptance by `+1.19`
percentage points. It costs `+672 MiB` main KV versus q8. Therefore only
FP8+MTP with `ctx >= 98304` automatically selects last12; q8 and shorter FP8
contexts remain last8. `LLAMA_VK_MTP_KV_LAST_F16` remains the explicit
override/rollback. The M6 bridge is retained as a default-off generation-heavy
research profile, not the general default. See
[D097](docs/research/major-topology/D097_Q4KM_VULKAN_FP8_LONG_ACCEPTANCE.md).

## Historical cross-backend performance

Audit correction (2026-08-11): the 2026-08-07 Vulkan rows below were described
as q8_0, but their canonical CSV and server logs show q4_0/q4_0 KV. They remain
historical reference numbers; the explicitly typed table above supersedes them
for current q8/FP8 decisions. ROCm rows below are q8_0/q8_0.

| Backend | Mode | Context | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ROCm | none, r3 mean | 12,288 | 6,393 / 256 | **1798.03** | 24.00 | 17.9169 | - |
| ROCm | MTP n3, r3 mean | 12,288 | 6,393 / 256 | 1671.69 | **47.17** | **27.4421** | **79.06%** |
| Vulkan | none | 12,288 | 7,889 / 128 | **1647.76** | 28.64 | 13.76 | - |
| Vulkan | MTP n3 | 12,288 | 7,889 / 128 | 1595.95 | **53.61** | **17.33** | 71.1% |
| ROCm | none, r2 mean | 49,152 | 29,561 / 128 | **1778.59** | 21.98 | 5.6829 | - |
| ROCm | MTP n3, r2 mean | 49,152 | 29,561 / 128 | 1731.71 | **39.58** | **6.2802** | **74.36%** |
| Vulkan | none | 49,152 | 30,723 / 128 | **1461.26** | 26.96 | 4.95 | - |
| Vulkan | MTP n3 | 49,152 | 30,723 / 128 | 1452.57 | **48.36** | **5.34** | 70.5% |
| Vulkan | none | 98,304 | 57,530 / 128 | **1211.62** | 25.43 | 2.43 | - |
| Vulkan | MTP n3 | 98,304 | 57,530 / 128 | 1205.65 | **39.95** | **2.51** | 65.6% |
| ROCm | none | 98,304 | 59,045 / 64 | **1493.21** | 19.15 | **1.4890** | - |
| ROCm | MTP n3 | 98,304 | 59,045 / 64 | 1435.97 | **35.44** | 1.4872 | **80.00%** |

At 6.4k prompt tokens, current ROCm MTP loses 7.03% prompt throughput, gains
96.54% decode throughput, and gains 53.15% aggregate throughput for a 256-token
answer. At 29.5k prompt tokens, it loses only 2.64% prompt throughput, gains
80.11% decode throughput, and gains 10.51% aggregate throughput. Historical
Vulkan q4_0 MTP (validated 2026-08-07) loses 3.1% prompt throughput and gains
87.1% decode throughput at 7.9k tokens; at 30.7k it loses 0.6% prompt and gains
79.4% decode. Those Vulkan rows use repo-snapshot prompts
7,889/30,723/57,530 tokens; see `Q4_REFRESH_WIP_2026-08-07.md` in the ignored
benchmark artifact directory. The current q8/FP8 table above replaces them for
KV-format decisions.

At 59k prompt tokens, ROCm MTP loses 3.83% prompt throughput and gains 85.1%
decode throughput. The 64-token request is exactly at the amortization
boundary: wall time changes only from 42.98 to 43.04 seconds. Longer generated
answers favor MTP.

## Long-context residency

| Backend | Context | Actual prompt | Prompt TPS | Decode TPS | Dedicated peak | Shared peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ROCm, current performance / E335 memory | 49,152 | 29,561 | 1778.59 | 21.98 | 21.21 GiB | 2.60 GiB |
| Vulkan, historical q4_0 KV | 49,152 | 30,723 | 1461.26 | 26.96 | 18.09 GiB | 0.29 GiB |
| ROCm, pre-E337 spill | 98,304 | 59,004 | 553.50 | 17.64 | 24.45 GiB | 6.25 GiB |
| ROCm, E338 one copy, none | 98,304 | 59,045 | **1493.21** | 19.15 | 22.05 GiB | 3.20 GiB |
| ROCm, E338 one copy, MTP n3 | 98,304 | 59,045 | 1435.97 | **35.44** | 23.96 GiB | 3.26 GiB |
| Vulkan, historical q4_0 KV | 98,304 | 57,530 | 1211.62 | 25.43 | 20.06 GiB | 0.54 GiB |
| Vulkan, historical q8_0 KV | 131,072 | 75,979 | 1051.67 | **23.02** | 21.38 GiB | 0.70 GiB |

## D094 refresh (2026-08-06/07, f16 KV, GUI autotune, no game)

| Backend | Context | KV | Prompt TPS | Decode TPS | Aggregate TPS | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Vulkan | 49,152 | f16 | **1719.92** | **43.10** | **6.1358** | +7.8% prompt vs 2026-08-05 (1595.06) |
| ROCm | 49,152 | f16 | 1679.20 | 32.33 | 5.7655 | same lane, same day |

Vulkan is now ahead of ROCm on this lane: prompt +2.4%, decode +33%, aggregate
+6.4%. The gain is graph-level (concat fast path is env-gated; FA prealloc and
q8_0-KV pre-dequant benefit q8-KV lanes); accuracy fixes carry no speed cost.

The current 49K performance row is E353; its Dedicated/Shared peaks are the
matched-placement E335 sampler values because E353 did not run the WDDM memory
monitor. The kernel changes do not add persistent buffers.

ROCm remains the faster prompt-evaluation backend at `ctx=49152`. The old
`ctx=98304` run reached a real residency cliff: Shared peaked at 6.25 GiB and
prompt throughput fell to 553.50 tok/s. E337 removed context-sized Q8
FlashAttention staging, and E338 reduced the single-request ROCm scheduler from
four graph copies to one. The matched 64-token 98K control now measures
1493.21 tok/s, 2.70x the old spill result and 27.5% above the recorded Vulkan
prompt row.

Windows still reports 3.20 GiB Shared. This is largely pageable WDDM backing
for HIP graph allocations rather than proof of active RAM reads: dedicated
residency remains high and prompt throughput no longer has the spill shape.
The KV cache itself is allocated at context creation and does not grow as the
prompt is processed. Enabling MTP n3 raises prefill Dedicated by 1.91 GiB but
Shared by only 0.057 GiB, confirming that its extra working set remains local.

The 131K ROCm rows were deliberately not repeated after this result. The older
pre-E334 controls were 447.59 tok/s with equal `1:1` placement and 1379.14
tok/s with the memory-aware `27:37` placement. They remain useful historical
evidence for placement sensitivity, but they are not current post-fix numbers.

E334 identified and fixed one ROCm-only growth source: TILE FlashAttention
converted quantized K/V to context-sized F16 scratch through the non-VMM HIP
pool, and old sizes could accumulate as the graph grew. Reserving the scratch
in the graph bounds that allocator behavior. E337 then replaced the active
context-sized staging with bounded 4096-token scratch for RDNA4 Q8 K/V and
recovered 216 MiB in a one-card Q3 control with neutral throughput. The Q4
98K row above includes both E337 and E338. Q4 still has much less headroom than
Q3, especially with MTP or vision, but it no longer enters the old 553 tok/s
residency cliff on this tested no-MTP lane.

Q4_K_M cannot be fully resident on one 16 GiB card. The GPU model tensors alone
use about 15.25 GiB before KV, recurrent state, compute buffers, driver
reservations, and desktop usage.

The refreshed short/49K artifacts use the `e352-`, `e353-`, and `e357-q4km-`
prefixes. The sampled 49K residency artifacts use `e335-rocm-q4km-`; the 98K
scheduler-residency artifacts use `e338-rocm-dual-q4km-`. The benchmark
registry identifies the configured build directory as `build-rocm-full`. The
complete original methodology and historical per-device measurements are
recorded in
[E332: Qwen3.6-27B Q4_K_M performance and residency](docs/research/experiments/E332_qwen36_q4km_performance_and_residency.md)
and the follow-up
[E333: ROCm Q4_K_M memory-aware split](docs/research/experiments/E333_rocm_q4km_memory_aware_split.md)
and
[E334: ROCm quantized-KV scratch reservation](docs/research/experiments/E334_rocm_quantized_kv_scratch_reservation.md).
The post-fix rebaseline and current per-device peaks are recorded in
[E335: ROCm post-reservation rebaseline](docs/research/experiments/E335_rocm_post_reservation_rebaseline.md).
The bounded active-staging follow-up is recorded in
[E337: bounded ROCm Q8 FlashAttention WMMA](docs/research/experiments/E337_rocm_q8_chunked_wmma.md).
The scheduler-residency follow-up is recorded in
[E338: ROCm dual-GPU long-context scheduler residency](docs/research/experiments/E338_rocm_dual_long_context_scheduler_residency.md).
The current quantized prompt/decode kernel work is recorded in
[E344: Q4_K/Q5_K MMQ geometry](docs/research/experiments/E344_rocm_q4q5_type_specific_mmq_geometry.md)
and
[E345: Q6_K MMVQ policy](docs/research/experiments/E345_rocm_q6_route_and_smallk.md).
