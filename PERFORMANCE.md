# Current Performance

Snapshot: **2026-09-09**.

Reference machine: 2x AMD Radeon RX 9070 XT (16 GB, `gfx1201`), AMD Ryzen 7
5800X3D, 64 GB RAM, dual-GPU layer split.

Platform note:
- **Windows** rows are production measurements on Windows 11 (ROCm/HIP 7.1,
  AMD Vulkan) and are kept from the last verified reference runs; fresh
  Windows re-runs are planned, until then these are the Windows reference.
- **Linux** rows were measured on Linux with ROCm 10 (2026-09-09, W20-W26) and
  are explicitly labeled with their L1/L2 lane; Windows repeats are pending.

Older benchmark tables (fork-vs-stock snapshots, Bonsai, extended/near-capacity,
pre-Qwen3.8 rows) were removed from this file awaiting the Windows re-runs.
Archived history stays in [BENCHMARKS.md](BENCHMARKS.md) and
[Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md).

All rows use FlashAttention, one server slot, cold prompt processing, no
prompt-cache reuse, no prime pass. Compare `none` and `MTP` only inside the
same model, backend, lane, and KV policy. Model quality (PPL) for each format
is listed in [Model Quality](#model-quality) — performance must not be
compared without it, because MXFP4/NVFP4 trade quality for speed.

### Q4_K_M Vulkan q8 vs native FP8 (Windows)

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

### Q4_K_M ROCm q8 vs native FP8 (Windows)

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

### Qwen3.6-35B-A3B Q4_K_M Vulkan q8 vs native FP8 (Windows, diagnostic)

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

## Linux reference (2026-09-09) — L1 / L2, ROCm 10

Measured on Linux with ROCm 10 on the same 2x RX 9070 XT hardware; Windows
re-runs are planned. Contract (bench2 `rdna-lab`): `batch 8192 / ubatch 1024`,
KV `f8_e4m3 / f8_e4m3`, FlashAttention, `ROCm1,ROCm0 -sm layer -ts 1,1`,
`-ngl 999`, one slot, cold prompt, `seed 42`, temp 0.2, top-p 0.9,
`--no-warmup`, repo-snapshot prompts:
- L1: 8,450 prompt / 128 output (ctx 49152)
- L2: 33,865 prompt / 256 output (ctx 49152)

| Model / format | Lane | Spec | Prompt TPS | Decode TPS | Aggregate TPS | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B-Q4_K_M (UD) | L2 | none | 1765.82 | 23.53 | 8.517 | Q4 reference (UD file, 16.46 GiB) |
| Qwen3.8-27B-MXFP4-requant | L1 | none | **2253.06** | 29.38 | **15.787** | W26 MMQ default; W24 nwarps=8 |
| Qwen3.8-27B-MXFP4-requant | L2 | none | **2076.46** | 25.92 | **9.777** | W26 MMQ default; W24 nwarps=8 |
| Qwen3.8-27B-MXFP4-requant | L2 | MTP n3 | 1666.09 | **49.195** | 10.028 | acceptance 169/256 (66.0%) |
| Qwen3.8-27B-MXFP4-hybrid-attnQ6 | L2 | none | 1769.91 | 24.26 | 8.623 | attn/output/tokembd Q6_K + rest MXFP4 |
| Qwen3.8-27B-NVFP4-native | L2 | none | 488.95 | 24.86 | 3.218 | prefill -4x; quality best of FP4 |

Results summary (same binary A-B-A):
- **W24** `calc_nwarps` RDNA4 ncols=1: MXFP4 -> `nwarps=8`
  (decode L1 +5.0%, L2 +3.0%, MTP +1.8%).
- **W26** `ggml_rdna4_mxfp4_mmq_max_ne11` default 4096 + MXFP4 case in the
  RDNA4 `should_use_mmq` switch: prefill L1 **+20.8%**, L2 **+17.4%**, decode
  neutral (`w26mmq-a1/b/a2-l1,l2`).
- **W26b** Q4_K control (UD-Q4_K_M): prefill 1765.82 / decode 23.53; no Q4_K
  regression (routing unchanged).
- **W24/W25** MTP acceptance is a draft-state ceiling: profile
  `#acc rate/pos = (0.800, 0.600, 0.432)`; `n_max=2` 41.57, `n_max=4` 44.16,
  `p_min=0.5` 44.25 — n3 (49.195) stays best. Draft up-quantization (Q6_K) and
  `small_k` for MXFP4 measured negative (docs below).

## Model Quality

Perplexity is the quality gate for this repo. The MXFP4/NVFP4 rows trade
quality for speed; keep this table in view when comparing with the Q4_K_M
baseline. PPL contract: `llama-perplexity -c 512 -b 512 -ub 512 --chunks 512
-ngl 999 -dev ROCm1,ROCm0 -sm layer -ts 1,1 --flash-attn on --no-warmup
--no-mmap` on the repo corpus (~240k tokens).

| Model / format | bpw | Size | PPL (512 chunks) | Δ vs Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B-Q4_K_M | 5.01 | 17.11 GiB | **6.7802 ± 0.05045** | — (baseline) |
| Qwen3.8-27B-MXFP4-requant (Q4_K->MXFP4) | 4.25 | 13.85 GiB | 7.1937 ± 0.05471 | **+6.10%** |
| Qwen3.8-27B-MXFP4-native (BF16->MXFP4) | 4.25 | 13.85 GiB | 7.1391 ± 0.05414 | **+5.29%** |
| Qwen3.8-27B-MXFP4-hybrid-attnQ6 | 4.85 | 15.75 GiB | **6.9925 ± 0.05375** | **+3.13%** |
| Qwen3.8-27B-NVFP4-native | 4.50 | 14.67 GiB | **6.9840 ± 0.05201** | **+3.01%** |

Reading the trade-off:
- **Q4_K_M** stays the production quality baseline (best PPL, 5.01 bpw).
- **MXFP4-native / hybrid** are speed-first: decode +3-5% (W20/W24) and
  prefill +17-21% (W26) at +3.1-5.3% PPL.
- **NVFP4-native** has the best quality of the FP4 family (+3.01% PPL,
  4.50 bpw) but its prefill is ~4x slower — not usable for prompt-heavy lanes.
- imatrix for MXFP4/NVFP4 is not implemented (`quantize_mxfp4` ignores
  `quant_weights` in this fork and upstream), so these are the honest format
  limits.

## Change log (recent, affecting the tables above)

- 2026-09-09 (W20-W26): MXFP4 decode geometry (`nwarps=8`), MXFP4 prefill MMQ
  routing (+17-21%), MTP acceptance profile (negative for p_min/n_max/Q6
  draft), native BF16->MXFP4/NVFP4 quality (PPL table above).
- Older benchmarks were removed 2026-09-09 pending Windows re-runs. History:
  [BENCHMARKS.md](BENCHMARKS.md), [Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md),
  [docs/research/RESULTS_LOG.md](docs/research/RESULTS_LOG.md),
  [docs/research/rdna4-architecture/](docs/research/rdna4-architecture/).

Archived benchmark history (removed from this file on 2026-09-09 pending
Windows re-runs) is preserved in
[BENCHMARKS.md](BENCHMARKS.md), [Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md) and the
canonical [build_logs/agent-workload/BENCH_RUNS.csv](build_logs/agent-workload/BENCH_RUNS.csv).
