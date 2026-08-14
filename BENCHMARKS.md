# Benchmarks

## D098 ROCm Native Full FP8 (2026-08-13)

The guarded gfx1201 D=256 F8/F8 FlashAttention route is now native and enabled
by default on ROCm. Serializing the packed-P conversion and VKQ merge removed
`196` VGPR spills plus `788 B/thread` scratch; the selected eight-wave body
uses `154-156 VGPR`, zero spills/scratch and `29568 B` LDS. Focused reference,
native-KQ and native-KQ+V prefill/decode pass `6/6`.

On the final same-binary 49K spec-none q8/FP8/q8 bracket, q8 centered at
`1686.75/21.83/8.62` prompt/decode/aggregate and FP8 reached
`1769.37/22.97/9.05`: `+4.9%/+5.2%/+5.0%`. MTP n2 also passed acceptance:
49K FP8 reached `1751.58/41.98/6.29`, `83.16%` versus q8-center
`1672.03/37.96/5.95`, `77.78%`; 98K FP8 last12 reached
`1482.86/35.82/2.99`, `81.25%` versus q8-center `1443.54/33.50/2.905`,
`79.59%`. D091 device order and placement stayed stable.

Rollback: `GGML_ROCM_FATTN_F8_NATIVE_KQ=0`; KQ-only bisect:
`GGML_ROCM_FATTN_F8_NATIVE_V=0`. Full evidence is in
[D098](docs/research/major-topology/D098_Q4KM_ROCM_FP8_KICKOFF.md).

## D097 Vulkan FP8 98K Acceptance Recovery (2026-08-11)

The long-run FP8 acceptance loss was a precision-policy problem, not a P5
kernel regression. Raw E4M3 has only three mantissa bits; q8_0's per-32 scale
makes its captured attention-logit MSE about 29.9x lower. With the old matched
last-8-f16 policy, repeated 256-token controls were deterministic: FP8 accepted
`140/230` drafts (60.87%) versus q8 `151/208` (72.60%).

The accepted context-scoped fix uses last-12 f16 only for FP8+MTP at
`ctx >= 98304`. Against the center of adjacent q8 controls it measured
`1510.95/41.79/5.7618` prompt/decode/aggregate TPS and 73.79% acceptance:
`+6.20%` prompt, `-0.41%` decode, `+5.27%` aggregate and `+1.19` acceptance
points. Main KV is 5376 MiB versus q8's 4704 MiB. q8 and shorter FP8 contexts
stay at last8; `LLAMA_VK_MTP_KV_LAST_F16` remains a rollback. Full sweep and
the faster default-off M6 q8-bridge profile are documented in
[D097](docs/research/major-topology/D097_Q4KM_VULKAN_FP8_LONG_ACCEPTANCE.md).

## D095 Vulkan Q4_K_M q8/FP8 Refresh (2026-08-11)

The D095 diagnostic matrix uses `Vulkan1,Vulkan0`, layer split `1,1`,
`b8192/ub1024`, 128 output tokens, cold/no-reuse/no-prime execution and matched
repo-snapshot prompts. Native P5 FP8 raises spec-none prompt throughput over q8
by `+7.1%/+10.4%/+12.6%` at 12K/49K/98K while reducing the main KV allocation
by 5.88%. Prompt-heavy aggregate TPS rises `+3.0%/+6.8%/+9.3%`.

With matched hybrid last-8-f16 MTP n2, FP8 improves aggregate TPS by 4.6% at
12K and 1.5% at 49K. The historical 98K N8 row ties wall TPS (`2.9494` vs
`2.9407`) but drops FP8 acceptance to 51.61% versus q8 68.87%; D097 above
supersedes that policy. The headline table in
[Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md) now substitutes the current 98K q8
control center and FP8 last12 result; the old N8 row remains only in the
diagnosis record. Audit note: the older 2026-08-07
Vulkan rows were actually q4_0 KV despite being described as q8; the current
table is explicitly typed and supersedes them for KV-format decisions.

## D094 Vulkan q8/MTP Fix (2026-08-06/07)

The Vulkan q8_0 mat-vec/mmq numerical divergence vs ROCm was root-caused and
fixed (int dp4a accumulation order, round-half-away-from-zero q8_1 quantize,
mmq variant-B math, SPV regeneration). MTP acceptance recovered from 0.33 to
0.80+ on 52k-token drafts (target 0.53):

| Backend | Prompt (tokens) | Acceptance | tps | Config |
| --- | ---: | ---: | ---: | --- |
| ROCm | 52,427 | **86.0%** (80/93) | 0.29 | dual layer -ts 1,1, mtp n2 |
| Vulkan | 52,427 | **80.4%** (78/97) | 0.60 | dual layer -ts 1,1, mtp n2 |
| ROCm | 13,924 | **69.8%** (74/106) | 0.84 | same, game on background |
| Vulkan | 13,924 | **64.0%** (71/111) | 1.41 | same, game on background |

Vulkan acceptance now tracks ROCm (gap ~5-6 pp) and both are far above the
0.53 target; Vulkan wall completion is 2x faster under dual-GPU layer split.
F16-KV 49K lane (GUI autotune, no game): Vulkan 1719.92/43.10/6.1358
prompt/decode/aggregate TPS vs ROCm 1679.20/32.33/5.7655 (Vulkan +2.4%/+33%/+6.4%).

## Primary Q4_K_M Baseline (2026-07-20)

The project baseline is now `Qwen3.8-27B-Q4_K_M.gguf`. The safe production
research lane is dual ROCm, `ctx=49152,b8192/ub1024,q8_0/q8_0`, one slot,
FlashAttention, cold/no-reuse/no-warmup, and `-dev ROCm1,ROCm0 -sm layer -ts
1,1`. The adjacent spec-none row measures `1778.59/21.98` prompt/decode tok/s;
MTP n3 measures `1731.71/39.58`, `6.2802` aggregate TPS, and 74.36% acceptance.
See [Q4_K_M_RESULTS.md](Q4_K_M_RESULTS.md) and
[D089](docs/research/major-topology/D089_Q4_K_M_PRIMARY_BASELINE_PROMOTION.md).

The Q3 sections below retain their original model-specific contracts. They are
secondary/historical evidence and must not be used as Q4 baseline rows.

## D079-D080 Vulkan Q3_K_S 56k Prompt Baseline (2026-07-12)

The parked secondary Q3 prompt-eval lane uses non-MTP `Qwen3.6-27B-Q3_K_S.gguf`,
56,456 prompt tokens, `ctx=131072`, `b8192/ub1024`, q8/q8 KV, FlashAttention,
`spec=none`, no warmup/reuse/prime, and thinking on. Equal dual layer split
measured `1276.93 prompt tok/s`. Balancing the output-heavy Vulkan1 stage with
`-dev Vulkan1,Vulkan0 -sm layer -ts 5,6` reached `1350.01 tok/s` on cold run 1
(`+5.72%`); r3 mean was `1327.82`, with zero errors. This is the retained P003
baseline if the historical `2000 prompt tok/s` target is explicitly reopened.

D079 perf evidence attributes 46.6% of parsed time to Q3_K matmul and 46.4% to
q8/q8 FlashAttention. Vulkan tensor split is rejected for this program:
`540.18` versus `1809.02 tok/s` on the small F16-KV control, with 127 generic
fallback allreduces per ubatch and no native cross-device allreduce.

## E280 Vulkan GPU1 Residency (2026-07-11)

For the local dual-RX 9070 XT Vulkan server, use explicit
`-dev Vulkan1,Vulkan0 -sm layer -ts 1,1` and GUI output placement on Vulkan1.
`LLAMA_OUTPUT_DEVICE=Vulkan1` moved `1004 MiB` of accounted model memory away
from display-attached GPU0 while retaining `1793.75 tok/s` prompt versus
`1860.21 tok/s` in the r1 control. Literal `LLAMA_KV_DEVICE=Vulkan1` does put
the full `4352 MiB` q8 KV cache on GPU1, but prompt fell to `814.82 tok/s`
because layer split must transfer attention data across devices. All-KV remains
an explicit residency option, not the performance default. A `54757`-token
validation at `b8192/ub1024` completed at `1313.68 tok/s`; accounted memory was
`Vulkan1 9604 MiB / Vulkan0 7972 MiB`.

Главный локальный benchmark для этой ветки:

```powershell
python scripts\agent_workload_bench.py
```

Он запускает короткую симуляцию агентной работы через OpenAI-compatible `llama-server`: triage diff, code review, ROCm log diagnosis и маленький patch simulation. По умолчанию инструмент ищет ROCm server binary в:

```text
build-rocm\bin\llama-server.exe
build-rocm\bin\Release\llama-server.exe
```

и модель в:

```text
models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
models\Qwen3.6-27B-Q3_K_S.gguf
models\Qwen3.5-9B-Q6_K.gguf
```

Результаты пишутся в:

```text
build_logs\agent-workload\<label>.csv
build_logs\agent-workload\<label>.jsonl
build_logs\agent-workload\<label>.server.log
```

По умолчанию runner выбирает свободный порт сам. Для уже запущенного сервера укажи `--no-start --port 8080`.

Для активной 130k performance lane runner по умолчанию держит thinking включённым. Для явного отключения thinking укажи `--disable-thinking`.

## ROCm RDNA4 MTP small-N DP4A (2026-07-11)

Для dual RX 9070 XT добавлен отдельный Q3_K DP4A MMQ-маршрут для малых
verification batch MTP (`N=2..4`). Он включён по умолчанию только на HIP RDNA4,
только для dense Q3_K и не затрагивает prompt/large-N WMMA. Откат:
`GGML_RDNA4_Q3K_SMALLN_DP4A=0`. Ненулевое значение также включает
экспериментальный `N=5`, который пока не рекомендуется.

На cold-first lane `ctx=12288,b=8192,ub=1024,q8/q8,max_tokens=256`, thinking
on, no reuse/no prime, MTP `n_max=3` вырос с `34.9170` до `41.2505` decode
tok/s (`+18.14%`) и достиг `1.65x` относительно spec-none baseline
`25.02 tok/s`. Новый default подтверждён отдельным r1: `41.35 tok/s`.
`n_max=2` дал `39.06`, а `n_max=4` только `35.06`, поэтому текущая рекомендация
для ROCm: `--spec-type draft-mtp --spec-draft-n-max 3`.

На практическом long-prompt lane `ctx=131072`, 56,305 prompt tokens и 128 output
tokens: spec none `1088.67` prompt / `19.02` decode tok/s; MTP n3 `1045.62`
prompt / `26.85` decode tok/s (`1.41x` decode, acceptance `68.55%`). Общий wall
TPS почти одинаков (`2.1859` vs `2.1799`), потому что 52-54 секунд prefill
доминируют над 4.8-6.7 секунд decode, а MTP сам prefill не ускоряет.

## Vulkan MTP long-KV autotune correction (2026-07-11)

Прежний GUI autotune с `max_tokens=16` занижал Vulkan MTP: первый target
verify после длинного prefill занимал около `630 ms`, тогда как следующие
verify-раунды занимали `42-49 ms`. Поэтому результат `18.02 decode tok/s` для
MTP n3 в основном измерял одноразовый переход PP -> TG, а не устойчивый decode.

На одинаковом prompt в 38,757 токенов и `max_tokens=128` spec-none получил
`1449.41` prompt / `29.15` decode tok/s, а MTP n3 — `1401.10` prompt / `38.78`
decode tok/s. Это `1.33x` по decode при `3.3%` prompt tax. GUI autotune теперь
использует 128 output tokens для всех spec-режимов, чтобы отдельные none/MTP
запуски оставались сопоставимыми. Фазовая диагностика доступна только через
`LLAMA_SPEC_SERVER_PHASE_TIMING=1` и по умолчанию ничего не логирует.
Финальная пересборка без warmup-прототипа подтвердила MTP результат:
`1399.85` prompt / `38.61` decode tok/s.

## Vulkan recurrent checkpoint batching (2026-07-11)

E279 добавил batch tensor-read callback для Vulkan state serialization. На
повторно используемом 7k prefix и 15 новых prompt-токенах последовательный
checkpoint занимал `33.362 ms`, а весь prompt `180.553 ms` (`83.08 tok/s`).
Batch-маршрут сгруппировал 96 R/S reads как 50/46 по двум GPU: checkpoint
`27.376 ms`, prompt `165.787 ms` (`90.48 tok/s`, `+8.9%`). Deterministic branch
rollback дал идентичный текст. Откат: `LLAMA_CHECKPOINT_BATCH_READ=0`.

Первоначальные `757-790 ms` на checkpoint внутри длинного первого prefill были
не чистым transfer: первый read синхронизировал ещё не завершившийся
1024-token prompt graph. Поэтому основной провал 39-40 token tail около 98k KV
остаётся small-N long-KV FlashAttention задачей H63, а не checkpoint задачей.

## MTP single-request sanity (2026-07-07)

После ROCm dual-GPU peer-copy fix и MTP pending-row/argmax правок проверен именно
single-request режим, не параллельные слоты: `Qwen3.6-27B-Q3_K_S_mtp.gguf`,
ROCm, `-np 1`, `ctx=8192`, `b512/ub128`, `q4_0/q4_0`, FlashAttention on,
thinking on, no reuse, quick tasks, `max_tokens=64`.

| Label | Spec | Wall TPS | Notes |
| --- | --- | ---: | --- |
| `mtp-single-none-pending-v2` | `none` | `19.15` | baseline on the same MTP GGUF |
| `mtp-single-mtp-n8-argmax-v3` | `mtp --spec-draft-n-max 8` | `23.12` | `+20.7%`; acceptance `51/89` then cumulative `98/202`; direct argmax replaced the MTP-only greedy sampler chain |

`--spec-draft-n-max 4` was slower (`15.41 TPS`), and `12` was below `8`
(`21.16 TPS`) on the same short lane, so `8` is the current practical MTP sanity
knob for this model/shape. This is not a 130k headline claim.

## Retention boundary

Detailed benchmark history before `2026-07-01` was removed from the active
fork. Current raw history lives in `build_logs/agent-workload/BENCH_RUNS.csv`;
accepted and rejected implementation findings remain in `docs/research/`.

## ROCm Qwen3.6 MTP Depth-8 Decode Profile (E266, 2026-07-07)

After checking the Unsloth Qwen3.6 MTP guide, the local ROCm issue was
reframed: acceptance was healthy, but the earlier `n_max=1..2` settings did not
give enough target-verify batching to outrun ROCm overhead. With the MTP
hook-prefill no-logits path, bulk hidden copy, and GPU argmax path in place,
`--spec-draft-n-max 8` is the first confirmed positive two-GPU ROCm profile.

Lane: `Qwen3.6-27B-Q3_K_S_mtp.gguf`, ROCm default two-GPU layer split,
`ctx=4096`, `b512/ub128`, `q4_0/q4_0`, FlashAttention on,
`quick:triage_diff`, `max_tokens=256`, `temperature=0.0`, no reuse/no prime,
thinking enabled.

| Mode | Label | Aggregate TPS | Decode tok/s | Prompt tok/s | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | `mtp-temp0-postbuild-none-confirm3` | `25.6412` | `26.29` | `744.72` | - |
| MTP `n_max=8` | `mtp-temp0-postbuild-n8-confirm3` | `42.1258` | `44.68` | `515.81` | `54.33%` |

The confirmed r3 delta is `1.64x` aggregate completion TPS and `1.70x` decode
tok/s, which matches the lower part of the Unsloth `1.4-2.2x` expectation for a
generation-heavy run. A small repo-context probe, safe-capped at `1577` prompt
tokens under `ctx=4096`, also stayed positive: baseline `22.3927` TPS /
`26.36` decode tok/s versus MTP `n8` `29.3322` TPS / `41.54` decode tok/s.

Depth scan around the winner:

| MTP depth | Aggregate TPS | Decode tok/s | Acceptance | Decision |
| ---: | ---: | ---: | ---: | --- |
| `4` | `24.9690` | `26.00` | `76.19%` | below baseline |
| `6` | `21.5639` | `22.34` | `62.42%` | reject |
| `8` | `41.5827` | `44.55` | `54.33%` | best local point |
| `10` | `37.8592` | `40.41` | `44.61%` | positive but slower than n8 |
| `12` | `37.2331` | `39.73` | `38.59%` | positive but slower than n8 |

Decision: keep the code fixes and use `--spec-type mtp --spec-draft-n-max 8`
as the current measured ROCm two-GPU generation-heavy profile for this local
MTP GGUF. Do not promote it as a 130k/60k-prompt headline yet: MTP still slows
prompt eval because hook-prefill advances the MTP context, so long-prompt
claims need a separate same-lane A/B.

## ROCm Qwen3.6 MTP Big-Prompt Gate (E267, 2026-07-08)

Follow-up to E266 on the practical large-prompt lane:
`Qwen3.6-27B-Q3_K_S_mtp.gguf`, ROCm default two-GPU layer split,
`ctx=131072`, `b512/ub128`, `q4_0/q4_0`, FlashAttention on,
`quick:triage_diff`, `real-context-chars=152000`, `56371` prompt tokens,
`max_tokens=256`, `temperature=0.0`, no reuse/no prime, thinking enabled.

| Mode | Label | Aggregate TPS | Wall s | Prompt tok/s | Prompt ms | Decode tok/s | Decode ms | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `mtp-bigprompt-none-r1` | `2.1774` | `117.5701` | `537.93` | `104793.21` | `20.19` | `12682.62` | - |
| MTP `n_max=8` | `mtp-bigprompt-n8-r1` | `1.5997` | `160.0303` | `386.54` | `145835.45` | `18.15` | `14102.51` | `54.62%` |

Delta: aggregate completion TPS `-26.5%`, prompt eval throughput `-28.1%`,
decode throughput `-10.1%`. The MTP run adds `41.04 s` of prompt eval time on
the same `56371`-token prompt. Acceptance is healthy (`207/379`, `54.62%`), so
the gate fails on runtime overhead, not draft quality.

Decision: reject MTP `n_max=8` as a cold large-prompt default for the current
code. Keep E266 as the generation-heavy profile, but the 130k/60k-prompt route
needs an MTP prefill/hook optimization or a proven safe lazy MTP initialization
before any speedup claim.

## ROCm Qwen3.6 MTP Windowed NextN + Dual Layer Profile (E268, 2026-07-09)

After the upstream-style NextN extraction port, MTP no longer needs the old
per-verify target hook over the whole prompt. The practical launch profile is:
`Qwen3.6-27B-Q3_K_S_mtp.gguf`, ROCm, `ctx=8192`, `b512/ub128`, q4 KV,
FlashAttention on, `max_tokens=256`, `temperature=0.0`, no reuse/no prime,
thinking enabled, `--spec-type draft-mtp --spec-draft-n-max 8`.

Important device note: `-sm none` is single-GPU mode. For the real two-card
launch use `-dev ROCm1,ROCm0 -sm layer -ts 1,1`; ROCm1-only is just a clean
diagnostic lane when GPU0 is busy.

| Mode | Device/split | Label | Aggregate TPS | Decode tok/s | Prompt tok/s | Acceptance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline | `ROCm1 -sm none` | `rocm1-mtp-polish-mt256-none-r1` | `28.9357` | `29.93` | `598.78` | - |
| MTP n8 | `ROCm1 -sm none` | `rocm1-mtp-polish-mt256-n8-r1` | `42.6461` | `45.31` | `503.76` | `57.89%` |
| baseline | `ROCm1,ROCm0 -sm layer -ts 1,1` | `rocm-dual-layer-mtp-polish-mt256-none-r1` | `24.3710` | `25.06` | `618.77` | - |
| MTP n8 | `ROCm1,ROCm0 -sm layer -ts 1,1` | `rocm-dual-layer-mtp-polish-mt256-n8-r1` | `39.5312` | `41.71` | `516.74` | `57.89%` |

Dual-layer delta: `+62.2%` aggregate completion TPS and `+66.4%` decode tok/s.
Absolute dual baseline is lower than ROCm1-only because Windows ROCm peer copies
are disabled and cross-device transfers are host-staged, but the dual profile
keeps the model/KV resident across both cards and avoids the one-card VRAM/RAM
spill path.

## ROCm Dual Split Baseline Host-Stage Copy (E269, 2026-07-09)

E269 isolates the `--spec-type none` slowdown when the same MTP-capable Qwen3.6
model is split across both GPUs. Lane: `Qwen3.6-27B-Q3_K_S_mtp.gguf`, ROCm,
`ctx=8192`, `b512/ub128`, q4 KV, FlashAttention on, `max_tokens=256`,
`temperature=0.0`, no reuse/no prime, thinking enabled.

The kept code change is a safe ROCm/Windows fallback in the CUDA/HIP buffer copy
path: when direct peer copy is disabled, cross-device CUDA-buffer copies are
handled through a thread-local host staging buffer instead of falling through to
the slower generic tensor get/set path. Direct `GGML_ROCM_ENABLE_PEER_COPY=1`
remains rejected: the diagnostic run stopped after one empty token.

| Variant | Device/split | Label | Runs | Aggregate TPS | Decode tok/s | Prompt tok/s | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| single diagnostic | `ROCm1 -sm none` | `rocm1-mtp-polish-mt256-none-r1` | 1 | `28.9357` | `29.93` | `598.78` | diagnostic only |
| old dual baseline | `ROCm1,ROCm0 -sm layer -ts 1,1` | `rocm-dual-layer-mtp-polish-mt256-none-r1` | 1 | `24.3710` | `25.06` | `618.77` | baseline |
| host-stage buffer copy | `ROCm1,ROCm0 -sm layer -ts 1,1` | `rocm-dual-split-bufferhostcopy-mt256-none-r3` | 3 | `25.6137` | `26.26` | `749.89` | keep |
| placement scout | `ROCm0,ROCm1 -sm layer -ts 1,3 -mg 1` | `rocm-dual-split-bufferhostcopy-dev01-mg1-ts1_3-mt256-none-r1` | 1 | `25.4359` | `26.17` | `606.56` | optional profile |
| direct peer-copy opt-in | `ROCm0,ROCm1 -sm layer -ts 1,3 -mg 1` | `rocm-dual-split-peercopy-dev01-mg1-mt64-none-r1` | 1 | invalid | invalid | prompt ok | reject |

Decision: keep the safe buffer-copy fallback. It recovers about `+5.10%` wall
TPS and `+4.80%` decode tok/s versus the old dual baseline, but it does not yet
restore the single-GPU diagnostic speed. The remaining gap is still the ROCm
Windows host-staged split/sync cost, so further work should target safer pinned
host staging or a correctness fix for HIP peer copies before any default peer
copy change.

## MTP Exact Prefill Tail + ROCm Verify Route (E275, 2026-07-11)

The server now splits the final prompt batch exactly at the 512-token MTP tail.
On the 49k lane this changes the MTP-enabled region from the full 6206-row final
batch to exactly 512 rows. Fresh cold-first results with 38888 prompt tokens:

| Backend | Mode | Prompt tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| ROCm | none | `1275.42` | `20.88` |
| ROCm | MTP n4 | `1233.79` | `26.75` |
| Vulkan | none | `1448.29` | `28.80` |
| Vulkan | MTP n2 | `1407.52` | `34.06` |

The prompt penalty is now about `3%` instead of the earlier double-digit loss.
ROCm tracing also found that target verify `N=3` spent `94.44 ms` in the legacy
multi-column Q3_K MMVQ route. Keeping the tuned MMVQ path for `N=1` and routing
RDNA4 Q3_K `N>=2` to MMQ reduced it to `65.12 ms`. On the clean 12k/256-token
lane, ROCm MTP n4 reached `35.58` decode tok/s versus `25.23` baseline
(`+41.0%`) and `20.57` aggregate TPS versus `17.16` (`+19.9%`).

Use MTP n4 for ROCm on this model and n2 for Vulkan. The remaining route toward
`1.6x` ROCm decode is a fused multi-column Q3_K verify body; deeper drafts and
pipeline-parallel toggles were measured and do not close the gap.
