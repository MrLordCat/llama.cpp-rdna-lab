# ROCm Acceleration Plan

Дата плана: 2026-05-07.

Фокус: Windows 11 + AMD Radeon RX 9070 XT (`gfx1201`) + HIP SDK 7.1 + локальный fork `llama.cpp-with-GUI`.

## Current State

Уже подтверждено на текущем `master`:

- ROCm GUI configure снова проходит на текущем дереве.
- Short agent-workload benchmark добавлен и уже даёт repeatable метрики.
- `ngram-mod` поддерживается текущим `llama-server` и уже даёт прирост на Qwen3.6.
- MTP core/runtime уже присутствует в дереве, но локальный MTP benchmark ещё не завершён из-за отсутствия проверенного MTP-enabled GGUF.

## Measured Baseline On Current Build

Текущий build commit: `5facfaea9`.

| Model | Baseline agg TPS | ngram-mod agg TPS | Delta |
| --- | ---: | ---: | ---: |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | `37.454` | `41.007` | `+9.5%` |
| `Qwen3.6-27B-Q3_K_S.gguf` | `12.055` | `13.547` | `+12.4%` |

Практический вывод: для coding-agent сценариев `ngram-mod` уже стоит считать рабочим ускорением, а не только экспериментом.

## Priority 1: Finish MTP Validation

Цель: понять, даёт ли `--spec-type mtp --spec-draft-n-max 3` выигрыш именно на этой машине для text-only Qwen3.6 workloads.

Шаги:

1. Найти локально проверяемый MTP-enabled GGUF для `Qwen3.6-35B-A3B` или `Qwen3.6-27B`.
2. Запустить text-only smoke benchmark без `--mmproj`.
3. Сравнить не только TG, но и весь wall-time agent workload.
4. Если prefill regression слишком велика, не поднимать MTP в GUI как основной режим.

Критерий успеха:

- no crash;
- no OOM;
- measurable gain по aggregate completion TPS или total wall time;
- приемлемый PP regression.

## Priority 2: Promote ngram-mod To First-Class GUI Feature

Это уже не исследование, а продуктовая интеграция.

Что стоит сделать:

1. Добавить GUI preset `Qwen3.6 coding ngram-mod`.
2. Добавить в GUI отдельный блок `Speculative Decoding` с `none / ngram-mod / mtp`.
3. Для `ngram-mod` дать quick preset-кнопку, а не заставлять пользователя вручную писать long args.
4. Хранить параметры `n-match`, `n-min`, `n-max` в preset metadata.

Почему это важно:

- выигрыш уже подтверждён benchmark-ом;
- это снижает порог использования ускорения;
- это не требует MTP GGUF и не зависит от мультимодальности.

## Priority 3: Add GUI Benchmark Mode

Основная вкладка запуска уже перегружена, поэтому benchmark mode лучше не впихивать в тот же form-block.

Практичный вариант:

1. Отдельная вкладка `Bench` или отдельный dialog из Launch Server.
2. Два режима:
   - `Quick benchmark`: 4 short agent prompts из `scripts/agent_workload_bench.py`.
   - `Full benchmark`: расширенный prompt set или повторные runs.
3. Возможность выбрать:
   - build directory / binary;
   - model;
   - backend profile;
   - speculative mode;
   - KV cache type;
   - ctx / batch / ubatch.
4. Автоматическая запись CSV/JSONL/server.log и md-summary по build id.

Минимальный UX outcome:

- пользователь из GUI запускает бенч без ручного терминала;
- можно сравнить `baseline vs ngram-mod vs mtp` в одном месте.

## Priority 4: KV Cache Strategy Matrix

На 16 GB VRAM стоит оформить не один default, а несколько режимов.

Рекомендуемая матрица:

1. `Short coding / max quality`:
   - `--cache-type-k q8_0`
   - `--cache-type-v q8_0`
   - `-c 32768`
2. `Long context / safer VRAM`:
   - `--cache-type-k q4_0`
   - `--cache-type-v q4_0`
   - `-c 32768` или выше
3. `Experimental speed path`:
   - протестировать `tq3_0` или другие локальные TurboQuant-связки там, где это реально поддержано и не ломает качество.

Что нужно сделать:

- оформить benchmark matrix `q8_0 vs q4_0` на тех же prompt sets;
- не полагаться на субъективную «скорость на глаз»;
- отдельно сравнивать dense и MoE Qwen.

## Priority 5: Batch And UBatch Sweep

Сейчас используется хороший, но не доказанный optimum: `-b 2048 -ub 2048`.

Что стоит протестировать:

1. `2048 / 2048`
2. `4096 / 2048`
3. `4096 / 4096`
4. при необходимости `1024 / 1024` как low-VRAM fallback

Цель:

- найти sweet spot для PP/TG на `gfx1201`;
- измерить отдельно baseline и `ngram-mod`;
- записать результат в presets и docs.

## Priority 6: ROCm Build Variants Worth Testing

Под эту машину есть смысл проверить не только model/runtime args, но и build-level варианты.

Кандидаты:

1. current ROCm build with HIP 7.1
2. build with and without ccache side effects on iterative work
3. build with alternate batch defaults in GUI presets
4. build with selective compile-time backend trimming if это сокращает binary complexity и startup overhead

Важно: не расползаться в десятки билдов. Каждый новый build должен получать свой id и md-summary.

## Priority 7: Draft-Model Speculative Path For Non-MTP Models

Если MTP GGUF не найдётся быстро или даст слабый wall-time результат, следующая practical ветка ускорения:

1. подобрать маленькую совместимую draft-model для Qwen text workloads;
2. benchmark через `--model-draft` path;
3. сравнить с `ngram-mod` и baseline.

Почему это полезно:

- работает даже без MTP-enabled GGUF;
- может быть лучшим вариантом для non-MTP моделей;
- даёт ещё один GUI profile для advanced users.

## Priority 8: Prompt-Processing Aware Evaluation

Для agent workloads нельзя смотреть только на generation speed.

Нужно сохранять и сравнивать:

1. aggregate completion TPS;
2. wall time всего benchmark;
3. prompt processing tok/s из server log;
4. generation tok/s из server log;
5. first token latency;
6. VRAM behavior;
7. crash / warning patterns.

Если ускорение улучшает TG, но режет PP, для coding-agent use case оно может быть плохим tradeoff.

## Recommended Next Implementation Order

1. Найти MTP-enabled Qwen GGUF и завершить первый text-only MTP benchmark.
2. Параллельно сделать GUI preset и UI controls для `ngram-mod`.
3. Добавить в GUI benchmark mode отдельно от Launch Server main form.
4. Прогнать `q8_0 vs q4_0` и `b/ub` sweep на текущем build.
5. После этого обновить `gui/model_presets.json` уже benchmark-обоснованными defaults.

## Do Not Over-Invest Yet

Пока рано:

- делать MTP default mode;
- смешивать MTP и VLM/mmproj;
- расползаться в CUDA/Metal optimization before ROCm path is benchmark-stable;
- массово тюнить low-level HIP kernels без доказанного bottleneck на benchmark logs.

## New Findings (2026-05-07)

### Clean environment is mandatory

Фоновые `llama-server` процессы искажают результаты сильнее, чем большинство runtime-тюнингов.

Практика для всех новых замеров:

1. Убивать фоновые server процессы перед стартом.
2. Использовать `--background-server-policy fail`.
3. Считать baseline недействительным, если preflight не прошёл.

### 32K vs 64K on Qwen3.5-9B-Q6_K

Контрольный A/B прогон на одинаковых параметрах (`q8_0`, `b=2048`, `ub=1024`, `np=1`, `flash-attn=on`):

- `research-ctx32`: aggregate `62.84 TPS`
- `research-ctx64`: aggregate `62.70 TPS`

Разница около `-0.22%`, что в пределах шумового коридора.

Из server logs:

- PP: ~`1289` vs ~`1315` tok/s
- TG: ~`67.84` vs ~`67.97` tok/s

Вывод: для этой модели и этого профиля длинный контекст до 64K не является главным bottleneck.

### ngram-mod acceptance is workload-dependent

Есть сценарии, где `ngram-mod` инициализируется, но не принимает draft tokens (`#acc tokens = 0`).
Это объясняет случаи, когда режим почти не ускоряет или не ускоряет вовсе.

Следствие: speculative режимы нужно оценивать не только по aggregate TPS, но и по acceptance rate.

### Priority Model: Qwen3.6-27B (Phase-1, 32K+)

Прогнан Phase-1 autotune (`ctx=32K/49K/65K/131K`, `b=1024`, `ub=1024`, `kv=q8_0/q4_0`, `none/ngram-mod`).

Лучший конфиг:

- `ctx=65536`
- `batch=1024`
- `ubatch=1024`
- `kv=q4_0`
- `spec=ngram-mod`
- aggregate `~19.70 TPS`

Близкий fallback без speculative:

- `ctx=65536`, `kv=q4_0`, `spec=none`
- aggregate `~19.69 TPS`

Практический вывод:

1. Для 27B наиболее сильный сдвиг пришёл от перехода к `q4_0` и `ctx=65536`.
2. `ngram-mod` может давать дополнительный разгон, но его эффект зависит от history/acceptance и может быть нестабильным между runs.
3. Production default для 27B стоит держать как `q4_0 + 65536 + np1`, а `ngram-mod` включать как speed mode.

### Priority Model: Qwen3.6-27B (Phase-2, batch/ubatch sweep)

Дозамерен batch/ubatch sweep в зоне `ctx=65536`, `kv=q4_0`:

- `b=4096`, `ub=2048`, `spec=none` -> aggregate `~19.76 TPS` (best)
- `b=4096`, `ub=2048`, `spec=ngram-mod` -> aggregate `~19.74 TPS` (почти равно)

Наблюдения:

1. `b=4096` с `ub=2048` стабильно обходит `b=1024`/`b=2048` для этого workload.
2. В этом режиме `ngram-mod` не даёт устойчивого преимущества и может быть нейтральным.
3. Для production default 27B в GUI можно фиксировать `none` как базовый speculative mode, а `ngram-mod` оставлять как optional toggle.

Итоговый рекомендуемый production профиль для 27B:

- `ctx=65536`
- `batch=4096`
- `ubatch=2048`
- `kv=q4_0`
- `parallel=1`
- `flash-attn=on`

## Comprehensive Research Program

Ниже — рабочая программа с приоритетами и критериями решения.

### Phase A: Reproducible measurement pipeline

Цель: исключить шум и сравнивать изменения честно.

1. Для каждого сценария запускать минимум `runs=3`.
2. Сохранять: aggregate TPS, mean/median TPS, stdev, PP/TG, acceptance rate.
3. Сценарий признаётся улучшением только если прирост > `3%` при сопоставимой стабильности.

### Phase B: Runtime matrix for long context (32K+)

Цель: найти реальные runtime-optimum, а не локальные случайности.

Матрица:

1. Context: `32768`, `49152`, `65536`, `131072`.
2. KV cache: `q8_0` vs `q4_0`.
3. Spec mode: `none` vs `ngram-mod`.
4. Batch/UBatch: `1024/1024`, `2048/1024`, `2048/2048`, `4096/2048`.

Критерий:

- выбрать top-3 конфигурации по aggregate TPS и перепроверить их на `runs=5`.

### Phase C: Bottleneck attribution

Цель: понять, где теряется время — PP, TG, spec acceptance или memory pressure.

1. Парсить server logs автоматически: PP tok/s, TG tok/s, acceptance, OOM/warnings.
2. Отдельно считать корреляцию между acceptance и конечным TPS.
3. Классифицировать bottleneck тип:
   - PP-bound
   - TG-bound
   - acceptance-bound
   - memory-fit-bound

### Phase D: Build-level experiments (ROCm)

Цель: отделить runtime-тюнинг от compile-time эффектов.

1. Контрольный build и variant build сравнивать на идентичном runtime профиле.
2. Варианты:
   - different HIP compile flags already available in GUI;
   - alternative job parallelism at build time only for compile speed (не смешивать с inference выводами);
   - отдельный build-id для каждого набора флагов.

### Phase E: MTP track (experimental)

Цель: проверить фундаментальный потенциал ускорения generation-heavy workloads.

1. Только text-only, без mmproj.
2. Только `np=1`.
3. Старт с `--spec-draft-n-max 3`.
4. Сравнение против лучшего non-MTP long-context профиля из Phase B.

Критерий перехода в GUI-default:

- стабильный выигрыш по wall-time и aggregate TPS без серьёзного PP regression.

## Execution Order For Next Sessions

1. Завершить full autotune matrix на Qwen3.5-9B (32K-131K).
2. Повторить для Qwen3.6-27B и Qwen3.6-35B-A3B.
3. Построить bottleneck summary table (PP/TG/acceptance).
4. Выбрать 1-2 production presets на модель.
5. Только потом переходить к MTP ветке как к более рискованной оптимизации.
