# Stormrage Benchmark Shape vs Local Build (updated 2026-05-13)

Сводка по воспроизведению benchmark shape из `Stormrage34/llama.cpp-turboquant-hip` на локальном `build-rocm-vec`.

Файл изначально был создан 2026-05-12, но обновлён после локального внедрения реальных `TKV2/3/4`, direct TKV FlashAttention, mixed K/V route и специализированного `TKV4 set_rows`.

## Что перенесли и что это дало

Из Stormrage/storage-направления в локальный форк оказались реально полезны:

1. Реальные KV-типы `turbo2/turbo3/turbo4` как локальные `GGML_TYPE_TKV2_0`, `GGML_TYPE_TKV3_0`, `GGML_TYPE_TKV4_0`, а не alias на старые типы.
2. `GGML_OP_TURBO_WHT`: WHT-преобразование Q перед direct FATTN и inverse-WHT на выходе, когда V хранится в TKV.
3. Direct compressed-KV FlashAttention для TKV decode.
4. Hybrid default для Turbo4: direct decode + F16/WMMA prefill. Full-direct prefill на большом `ubatch` оказался хуже.
5. Специализированный `TKV4 set_rows` kernel для записи KV cache.
6. Mixed TKV/Q8 route (`turbo4/q8_0`, `q8_0/turbo4`) как opt-in режим.

Измеримый эффект на активной real-workload lane (`v2-review`, `ctx=12288`, `b=6144`, `ub=1024`, no-reuse, thinking on, `spec=none`):

| KV cache | Mode | Runs | Aggregate TPS | Delta vs q4 |
| --- | --- | ---: | ---: | ---: |
| q4_0/q4_0 | baseline | 3 | 11.17 | baseline |
| turbo4_0/turbo4_0 | hybrid default + specialized TKV4 set_rows | 3 | 10.38 | -7.1% |
| turbo4_0/q8_0 | mixed direct decode, F16 prefill | 3 | 10.60 | -5.1% |

Итог: Turbo4 пока не обогнал `q4_0` по wall TPS, но разрыв сокращён с ранних `~26%` на underfilled `ub=192` до `~7%` для memory-saving `turbo4/turbo4` и до `~5%` для opt-in `turbo4/q8_0`. Direct path также сделал TKV практически применимым: fallback `GGML_TKV_DIRECT_FATTN=0` для `turbo4/turbo4` на диагностической lane был `3.10 TPS`, direct route был `6.68 TPS`.

Что не оставили: full-direct prefill (`7.70 TPS`, хуже hybrid), warp-level pack/reduction и sign-LUT micro-ideas для set_rows/WHT не дали воспроизводимого выигрыша и были откатаны.

## Внешний профиль Stormrage

Профиль из `scripts/run_rdna2_bench.sh` внешнего репо:

- `p=512,2048,4096`
- `n=128`
- `b=256`
- `ub=128`
- `ctk=turbo4`
- `ctv=turbo2`
- `fa=1`
- `mmp=0`
- `t=8`
- `ngl=99`
- `fit-target=2048`, `fitc=4096`
- `r=3`

Локальный бинари: `build-rocm-vec/bin/llama-bench.exe`.

Локальные модели:

- `models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (MoE)
- `models/Qwen3.6-27B-Q3_K_S.gguf` (dense)

## Важное ограничение сопоставимости

Это operational-сравнение одного benchmark shape, не строгое apples-to-apples A/B.

Внешние числа Stormrage:

- GPU: RX 6800 XT, RDNA2, `gfx1030`.
- Основной claim: Qwen3.6-35B-MoE-IQ4_XS.
- Дополнительный путь: RDNA2-specific MoE accelerator (`RDNA2_MATMUL_OPT_V1`).
- KV: Stormrage `turbo4/turbo2` с их внутренней семантикой.

Локальные числа:

- GPU: RX 9070 XT, RDNA4, `gfx1201`.
- ROCm/HIP SDK 7.1.
- Модели: локальные `Qwen3.6-35B-A3B-UD-IQ3_XXS` и `Qwen3.6-27B-Q3_K_S`.
- KV: локальные реальные `TKV4/TKV2` после порта.

## Reference claims из Stormrage README

- RX 6800 XT + Qwen3.6-35B-MoE-IQ4_XS:
  - baseline: `~480 pp`, `~57 tg`
  - stable RDNA2: `~540 pp`, `~55 tg`
  - +MoE accelerator: `~1772 +/- 6 pp`, `~52 +/- 7 tg`
- Dense 27B summary: `~480 pp`, `~27 tg`

## Локальный recheck: original Stormrage shape (`b=256`, `ub=128`)

| Source / GPU | Model / KV | pp512 | pp2048 | pp4096 | tg128 |
| --- | --- | ---: | ---: | ---: | ---: |
| Stormrage README, RX 6800 XT | MoE baseline | ~480 | n/a | n/a | ~57 |
| Stormrage README, RX 6800 XT | MoE stable RDNA2 | ~540 | n/a | n/a | ~55 |
| Stormrage README, RX 6800 XT | MoE + RDNA2_MATMUL_OPT_V1 | ~1772 +/- 6 | n/a | n/a | ~52 +/- 7 |
| Stormrage README, RX 6800 XT | Dense 27B summary | ~480 | n/a | n/a | ~27 |
| Local RX 9070 XT | MoE35B q4_0/q4_0 | 1318.83 | 1275.92 | 1239.98 | 102.76 |
| Local RX 9070 XT | MoE35B turbo4_0/turbo2_0 | 1143.86 | 1064.55 | 992.07 | 56.71 |
| Local RX 9070 XT | Dense27B q4_0/q4_0 | 795.66 | 787.07 | 776.22 | 28.59 |
| Local RX 9070 XT | Dense27B turbo4_0/turbo2_0 | 636.45 | 608.08 | 554.85 | 20.49 |

Вывод для original shape: локальный `turbo4/turbo2` теперь воспроизводится как реальные `TKV4/TKV2`, но в `b=256`, `ub=128` он медленнее локального `q4_0/q4_0`. Внешний большой MoE prefill gain Stormrage не является общим TurboKV gain; он завязан на RDNA2 MoE accelerator path.

## Extra bench: Stormrage shape с `b=1024`, `ub=1024`

Поскольку `ub=1024` при `b=256` не раскрывается, extra bench снят как `b=1024`, `ub=1024`, остальные параметры оставлены как в Stormrage shape: `p=512,2048,4096`, `n=128`, `fa=1`, `mmp=0`, `t=8`, `ngl=99`, `fit-target=2048`, `fitc=4096`, `r=3`.

| Local RX 9070 XT | KV | pp512 | pp2048 | pp4096 | tg128 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense27B Q3_K_S | q4_0/q4_0 | 1079.38 | 1244.60 | 1225.79 | 28.85 |
| Dense27B Q3_K_S | turbo4_0/turbo4_0 | 1006.08 | 1172.52 | 1135.15 | 20.95 |
| Dense27B Q3_K_S | turbo4_0/turbo2_0 | 997.35 | 1168.99 | 1133.96 | 20.78 |
| MoE35B IQ3_XXS | q4_0/q4_0 | 2807.61 | 3549.80 | 3500.76 | 102.50 |
| MoE35B IQ3_XXS | turbo4_0/turbo2_0 | 2590.18 | 3290.59 | 3182.46 | 56.28 |

Вывод для extra `ub=1024`: на RX 9070 XT большой `ubatch` резко поднимает MoE prefill и для `q4_0`, и для `TKV4/TKV2`. `turbo4/turbo2` на MoE35B достигает `3290.59 pp2048` и `3182.46 pp4096`, то есть как storage/fit режим он выглядит полезным для чужих сравнений. Но локальный `q4_0/q4_0` всё ещё быстрее в том же shape, особенно на decode (`102.50 tg128` против `56.28`).

Для dense27B extra `ub=1024` тоже улучшает prefill относительно original `ub=128`, но `turbo4` остаётся decode-bound около `21 tg128`, тогда как q4 держит около `28.85 tg128`.

## Почему MoE accelerator не переносится одной кнопкой

Технически перенести можно, но это не безопасный простой порт и не доказанный RDNA4 win.

Что есть в Stormrage:

- Compile-time gate: `-DRDNA2_MATMUL_OPT_V1=1`.
- Runtime gate: env `RDNA2_MATMUL_OPT_V1=1`.
- Hardware gate: `GGML_CUDA_CC_IS_RDNA2(cc)`, то есть только `gfx1030`/RDNA2.
- Кодовая зона: `ggml/src/ggml-cuda/mmq.cuh`.
- Механизм: LDS double-buffer для `tile_x`, `tile_x_next`, +1 LDS bank pad, плюс `amdgpu_waves_per_eu(4, 8)` для стабилизации occupancy.

Почему это не прямой перенос на RX 9070 XT:

1. Stormrage сам жёстко ограничивает accelerator RDNA2 (`gfx1030`), потому что фикс нацелен на RDNA2 LDS bank conflicts и register-spill variance.
2. RDNA4 (`gfx1201`) имеет другие kernel routes и occupancy-поведение; слепое расширение gate с RDNA2 на RDNA4 может ухудшить MMQ/WMMA/MFMA path.
3. Наши extra `ub=1024` MoE замеры уже дают очень высокий prefill без этого accelerator: `q4_0/q4_0` до `3500.76 pp4096`, `turbo4/turbo2` до `3182.46 pp4096`.
4. Узкое место Turbo4 на наших practical lanes сейчас больше похоже на decode/direct-vec и prefill dequant/WHT overhead, а не на тот же RDNA2 MoE MMQ bottleneck.

Правильный путь, если всё же пробовать: отдельный guarded эксперимент `RDNA4_MOE_MMQ_EXPERIMENT=1`, без включения по умолчанию, с A/B только на MoE lane (`q4_0/q4_0` и `turbo4/turbo2`, `b/ub=1024`) и negative control на dense27B. То есть это не “невозможно”, а “требует отдельной RDNA4 MoE validation, иначе легко получить регресс”.

## Быстрые выводы

- Stormrage TurboKV идеи для storage/direct-FATTN мы практически исчерпали: реальные `TKV2/3/4`, WHT graph op, direct FATTN, mixed route и `TKV4 set_rows` уже внедрены или проверены.
- В practical prompt-heavy lane лучший memory-saving Turbo4 всё ещё медленнее q4 примерно на `7%`; mixed `turbo4/q8_0` сужает gap до `~5%`, но требует больше KV памяти.
- В external Stormrage shape с `ub=128` локальный `TKV4/TKV2` не быстрее q4.
- В extra shape с `ub=1024` MoE prefill очень сильный и может быть полезен для внешних сравнений, но q4 baseline всё равно быстрее на той же машине.
- Оставшаяся Stormrage-идея с реальным потенциалом - RDNA2 MoE LDS double-buffer accelerator, но для нас это отдельный RDNA4 MoE experiment, не продолжение Turbo4 storage-порта.

## Артефакты

Original Stormrage shape (`b=256`, `ub=128`):

- `build_logs/agent-workload/stormrage-shape-current-moe35b-q4-q4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-shape-current-moe35b-turbo4-turbo2-20260513.jsonl`
- `build_logs/agent-workload/stormrage-shape-current-dense27b-q4-q4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-shape-current-dense27b-turbo4-turbo2-20260513.jsonl`

Extra `b=1024`, `ub=1024`:

- `build_logs/agent-workload/stormrage-extra-ub1024-dense27b-q4-q4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-extra-ub1024-dense27b-turbo4-turbo4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-extra-ub1024-dense27b-turbo4-turbo2-20260513.jsonl`
- `build_logs/agent-workload/stormrage-extra-ub1024-moe35b-q4-q4-20260513.jsonl`
- `build_logs/agent-workload/stormrage-extra-ub1024-moe35b-turbo4-turbo2-20260513.jsonl`

Historical 2026-05-12 artifacts are still useful only as pre-port/alias-era references:

- `build_logs/agent-workload/stormrage-shape-local-moe35b-q4kv.jsonl`
- `build_logs/agent-workload/stormrage-shape-local-moe35b-f16kv.jsonl`
- `build_logs/agent-workload/stormrage-shape-local-dense27b-q4kv.jsonl`
- `build_logs/agent-workload/stormrage-shape-local-dense27b-f16kv.jsonl`
- `build_logs/agent-workload/turbo3-dense27b-shape-rerun-clean.jsonl`
- `build_logs/agent-workload/turbo4-dense27b-shape-rerun-clean.jsonl`

## Команды

Original Stormrage shape:

```bash
build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-27B-Q3_K_S.gguf \
  -p 512,2048,4096 -n 128 -b 256 -ub 128 -ctk turbo4 -ctv turbo2 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl

build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  -p 512,2048,4096 -n 128 -b 256 -ub 128 -ctk turbo4 -ctv turbo2 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl
```

Extra `b=1024`, `ub=1024`:

```bash
build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-27B-Q3_K_S.gguf \
  -p 512,2048,4096 -n 128 -b 1024 -ub 1024 -ctk turbo4 -ctv turbo4 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl

build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-27B-Q3_K_S.gguf \
  -p 512,2048,4096 -n 128 -b 1024 -ub 1024 -ctk turbo4 -ctv turbo2 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl

build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  -p 512,2048,4096 -n 128 -b 1024 -ub 1024 -ctk turbo4 -ctv turbo2 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl
```# Stormrage Benchmark Shape vs Local Build (2026-05-12)

Сводка по воспроизведению benchmark shape из `Stormrage34/llama.cpp-turboquant-hip` на локальном `build-rocm-vec`.

## Что сравнивали

- Внешний профиль (из `scripts/run_rdna2_bench.sh`):
  - `p=512,2048,4096`
  - `n=128`
  - `b=256`
  - `ub=128`
  - `fa=1`
  - `fit-target=2048`, `fitc=4096`
  - `r=3`
- Локальный бинари: `build-rocm-vec/bin/llama-bench.exe`
- Локальные модели:
  - `models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (MoE)
  - `models/Qwen3.6-27B-Q3_K_S.gguf` (dense)

## Важное ограничение сопоставимости

У внешнего форка в бенче используется `-ctk turbo4 -ctv turbo2`, но там это отдельные TurboKV реализации с другой внутренней семантикой и типами.

В нашем текущем build для сравнения использованы:

- `q4_0/q4_0`
- `f16/f16`
- `tbq3/tbq4` в отдельном сравнительном столбце (через текущие локальные alias-пути).

## Reference claims из внешнего README

- RX 6800 XT + Qwen3.6-35B-MoE-IQ4_XS:
  - baseline: `~480 pp`, `~57 tg`
  - stable RDNA2: `~540 pp`, `~55 tg`
  - +MoE accelerator: `~1772 ± 6 pp`, `~52 ± 7 tg`
- Dense 27B (их summary): `~480 pp`, `~27 tg`

## Локальные результаты (наш build-rocm-vec)

| Run | pp512 | pp2048 | pp4096 | tg128 | KV |
| --- | ---: | ---: | ---: | ---: | --- |
| MoE35B IQ3_XXS | 1336.74 | 1262.82 | 1246.78 | 99.68 | q4_0/q4_0 |
| MoE35B IQ3_XXS | 1307.25 | 1267.26 | 1253.30 | 102.69 | f16/f16 |
| Dense27B Q3_K_S | 794.90 | 785.89 | 775.53 | 28.59 | q4_0/q4_0 |
| Dense27B Q3_K_S | 810.54 | 804.04 | 757.18 | 23.66 | f16/f16 |

## Прямое сравнение: внешний репо vs наш build

Ниже сопоставление в одном месте для быстрого чтения. Внешние числа взяты из README форка Stormrage (RX 6800 XT, MoE IQ4_XS, TurboKV `turbo4/turbo2`).

| Metric | Stormrage34 (README) | My repo (q4_0/q4_0) | My repo (f16/f16) | My repo (tbq3/tbq4) |
| --- | ---: | ---: | ---: | ---: |
| MoE35B prefill pp512 (baseline) | ~480 | 1336.74 | 1307.25 |  1127.93 / 1310.89 |
| MoE35B prefill pp512 (stable RDNA2) | ~540 | 1336.74 | 1307.25 |  1127.93 /  1310.89 |
| MoE35B prefill pp512 (+MoE accelerator) | ~1772 +/- 6 | 1336.74 | 1307.25 |  1127.93 /  1310.89 |
| MoE35B decode tg128 (baseline) | ~57 | 99.68 | 102.69 |  52.18 /  99.20 |
| MoE35B decode tg128 (stable RDNA2) | ~55 | 99.68 | 102.69 |  52.18 /  99.20 |
| MoE35B decode tg128 (+MoE accelerator) | ~52 +/- 7 | 99.68 | 102.69 |  52.18 / 99.20 |
| Dense27B prefill pp512 | ~480 | 794.90 | 810.54 | 666.04 /  790.12 |
| Dense27B decode tg128 | ~27 | 28.59 | 23.66 |  20.03 /  28.40 |

Примечание: это удобное operational-сравнение, а не строго apples-to-apples A/B. Отличаются GPU (RDNA2 vs RDNA4), модель/квант, а также KV path (TurboKV против стандартных KV типов).

## TurboQuant benchmark (наш build, dense27B)

Сняты повторные clean-прогоны (после проверки, что нет фоновых `llama-*` процессов) в том же benchmark shape (`p=512,2048,4096`, `n=128`, `b=256`, `ub=128`, `fa=1`, `r=3`) на `models/Qwen3.6-27B-Q3_K_S.gguf`:

- `turbo3/turbo3` (в текущем коде резолвится в `tq3_0/tq3_0`)
- `turbo4/turbo4` (в текущем коде резолвится в `q4_0/q4_0`)

| Run | pp512 | pp2048 | pp4096 | tg128 |
| --- | ---: | ---: | ---: | ---: |
| turbo3/turbo3 | 666.04 | 612.69 | 552.06 | 20.03 |
| turbo4/turbo4 | 790.12 | 782.80 | 773.08 | 28.40 |

| Метрика | turbo3 vs turbo4 |
| --- | ---: |
| pp512 | -15.70% |
| pp2048 | -21.73% |
| pp4096 | -28.59% |
| tg128 | -29.47% |

Вывод: в текущей реализации на нашем ROCm build преимуществ у `turbo3` нет; `turbo4` устойчиво быстрее на всех точках.

## GUI-style one-run autotune (best preset base)

Запущен requested one-run autotune «как в GUI» на базе применяемого пресета для `Qwen3.6-27B-Q3_K_S.gguf` (берётся первым совпадением из `gui/model_presets.json`):

- `ctx=32768`, `batch=5120`, `ubatch=1024`, `spec=ngram-mod`, `runs=1`.

Результаты:

| Run | aggregate_tps | mean_task_tps | Notes |
| --- | ---: | ---: | --- |
| gui-autotune-bestpreset-turbo3-r1 | 0.0115 | 0.0115 | completed, no errors |
| gui-autotune-bestpreset-turbo4-r1 | 0.0191 | 0.0191 | completed, no errors |

В этой конфигурации `turbo4` быстрее `turbo3` примерно на `+66%` по aggregate TPS (`0.0191 / 0.0115`).

## Быстрые выводы

- Dense decode на `q4_0/q4_0` (`28.59 tg`) близок к их заявленному `~27 tg` corridor.
- Dense prefill у нас заметно выше их summary `~480 pp`.
- MoE decode у нас (`~100 tg`) существенно выше их README цифр (`~52-57 tg`), но это не apples-to-apples:
  - другая GPU (RDNA4 vs RDNA2),
  - другой quant/model build,
  - другой KV path (без `turbo4/turbo2`).

## Артефакты

- `build_logs/agent-workload/stormrage-shape-local-moe35b-q4kv.jsonl`
- `build_logs/agent-workload/stormrage-shape-local-moe35b-f16kv.jsonl`
- `build_logs/agent-workload/stormrage-shape-local-dense27b-q4kv.jsonl`
- `build_logs/agent-workload/stormrage-shape-local-dense27b-f16kv.jsonl`
- `build_logs/agent-workload/turbo3-dense27b-shape-rerun-clean.jsonl`
- `build_logs/agent-workload/turbo4-dense27b-shape-rerun-clean.jsonl`
- `build_logs/agent-workload/turbo3-moe35b-shape-rerun-clean.jsonl`
- `build_logs/agent-workload/turbo4-moe35b-shape-rerun-clean.jsonl`
- `build_logs/agent-workload/gui-autotune-bestpreset-turbo3-r1-autotune-summary.csv`
- `build_logs/agent-workload/gui-autotune-bestpreset-turbo4-r1-autotune-summary.csv`

## Команды (локально выполненные)

```bash
build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  -p 512,2048,4096 -n 128 -b 256 -ub 128 -ctk q4_0 -ctv q4_0 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl

build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  -p 512,2048,4096 -n 128 -b 256 -ub 128 -ctk f16 -ctv f16 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl

build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-27B-Q3_K_S.gguf \
  -p 512,2048,4096 -n 128 -b 256 -ub 128 -ctk q4_0 -ctv q4_0 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl

build-rocm-vec/bin/llama-bench.exe -m models/Qwen3.6-27B-Q3_K_S.gguf \
  -p 512,2048,4096 -n 128 -b 256 -ub 128 -ctk f16 -ctv f16 \
  -fa 1 -mmp 0 -t 8 -ngl 99 -fitt 2048 -fitc 4096 -r 3 -o jsonl
```
