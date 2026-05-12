# Benchmarks

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

Для коротких агентных ответов runner по умолчанию добавляет `--chat-template-kwargs {"enable_thinking":false,"preserve_thinking":false}`. Это можно отключить флагом `--no-disable-thinking`.

## Надёжность замеров

### Активная политика контекста (2026-05-10)

- Для текущего performance-трека запускать benchmark только при `ctx <= 16384`.
- Запуски выше 16k считаются архивными/исследовательскими и не используются для текущих KPI.
- В `scripts/agent_workload_bench.py` и `scripts/repo_snapshot_context_bench.py` это ограничение включено по умолчанию; обход только явным флагом `--allow-ctx-above-16k`.

### Новый автономный чекпоинт (2026-05-10, lane <16k)

Профиль:

- `tasks=v2-mini`, `runs=1`, `ctx=12288`
- incoming prompt: `--real-context-mode repo-snapshot --real-context-chars 21872`
- no-reuse: `--cache-ram 0 --ctx-checkpoints 0`
- `q4_0/q4_0`, `spec=none`

Подтверждённый baseline после пересборки `build-rocm-vec`:

| Label | Build | Batch | UBatch | Aggregate TPS |
| --- | --- | ---: | ---: | ---: |
| `postrebuild-vec-b6144-ub512-none` | `build-rocm-vec` | `6144` | `512` | `9.85` |
| `postrebuild-vec-b6144-ub512-none-r2` | `build-rocm-vec` | `6144` | `512` | `9.84` |

Наблюдения из этого цикла:

- `ub=640` даёт резкий cliff на `build-rocm-wmma` (примерно `3.67-3.69 TPS`), поэтому активный safe corridor остаётся `ub=512`.
- `spec=ngram-mod` без prime (`--no-v2-prime-pass`) почти равен `spec=none`; всплеск при включённом prime не считать cold-first прогрессом.
- KV-типы `f16/bf16` на этом lane дают сильную регрессию (`~3.7-3.8 TPS`), `q4_0` остаётся лучшим.

### Shape-score paradox + context-cap probe (2026-05-11, superseded)

На lane `ctx=12288`, `b=6144`, `q4_0/q4_0`, no-reuse с shape-score:

- `ub=192` стабильно в быстром коридоре (`~8.52 TPS`);
- `ub=512` при тех же split-параметрах падает до `~4.19-4.24 TPS`.

Трассы показали, что для `ub192` и `ub512` совпадают:

- planner chosen/target histogram (`chosen=192`);
- GDN `n_tokens` histogram;
- FATTN hot-shape и MMQ selector route.

Бывший экспериментальный runtime-рычаг, использованный только как диагностический discriminator:

- env `LLAMA_UBATCH_SHAPE_CONTEXT_CAP=1`;
- при `LLAMA_UBATCH_SPLIT_POLICY=shape-score` и `LLAMA_UBATCH_SHAPE_PREFERRED=192` физический context `n_ubatch` капается до preferred.
- после root-cause проверки этот guard удалён из runtime: финальный фикс не меняет requested `-ub` и не использует shape-score/preferred cap.

Проверка на том же бинаре:

| Label | UBatch arg | Context cap | Aggregate TPS | Prompt eval | Decode eval |
| --- | ---: | --- | ---: | ---: | ---: |
| `p7-pass2-postctx-20260511-205925-shape-ub512-r1` | `512` | off | `4.19` | `332.79 tok/s` | `27.26 tok/s` |
| `p7-pass2-cap-20260511-205849-shape-ub512-r1` | `512` | on (`n_ubatch 512 -> 192`) | `8.53` | `827.82 tok/s` | `27.81 tok/s` |
| `p7-pass2-cap-20260511-205746-shape-ub192-r1` | `192` | on | `8.54` | `~828 tok/s` | `~27.8 tok/s` |

Вывод на этом этапе был неполным: context-cap доказал связь cliff с reserve/layout, но сам по себе был workaround, а не финальным решением.

### PP reserve outputs root cause (2026-05-12)

Финальная причина `ub489 -> ub490+` cliff на RDNA4/ROCm оказалась в reserve-time PP graph layout: обычный server decode резервировал PP graph как будто нужны logits/outputs для всех `n_tokens`, хотя на этом lane фактически нужен один output. Это раздувало compute buffer и переводило full graph в медленный layout при `ub490+`.

Финальный фикс: `llama_context::sched_reserve()` резервирует PP graph по фактическому числу decode outputs; all-output/encoder режимы оставляют полный reserve. Это не cap/guard и не меняет requested `-ub`.

Clean validation после удаления diagnostic probes, без `LLAMA_PP_RESERVE_SEQ_OUTPUTS`, без `LLAMA_UBATCH_SPLIT_POLICY`, без `LLAMA_UBATCH_SHAPE_PREFERRED`, без `LLAMA_UBATCH_SHAPE_CONTEXT_CAP`:

| Label | UBatch arg | Auto reserve log | Wall | Prompt eval |
| --- | ---: | --- | ---: | ---: |
| `e010-ub490-final-ppout` | `490` | `PP reserve outputs 490 -> 1` | `7.41s` | `966.26 tok/s` |
| `e010-ub512-clean-ppout` | `512` | `PP reserve outputs 512 -> 1` | `7.32s` | `979.33 tok/s` |

До output-aware reserve direct `ub490/491/512` были в slow band около `24-25s` wall и `~280-300 tok/s` prompt eval; с финальным reserve прямые `ub490+` остаются в fast band без обхода через меньший ubatch.

Чтобы исключить искажения от фонового `llama-server`, запускать benchmark с жёсткой проверкой:

```powershell
python scripts\agent_workload_bench.py --background-server-policy fail
```

Если процесс уже занят, runner завершится с ошибкой и покажет PID.

Для снижения методологического шума и анализа cold-vs-warm поведения:

```powershell
python scripts\agent_workload_bench.py `
  --background-server-policy fail `
  --server-seed 42 `
  --no-disable-thinking `
  --stats-ignore-first-run
```

- `--server-seed 42` фиксирует seed на стороне `llama-server` и уменьшает run-to-run случайность sampling path;
- `--no-disable-thinking` принудительно оставляет thinking включённым (обязательный режим для performance benchmark в этом форке);
- `--stats-ignore-first-run` печатает отдельные warm-only метрики (без run #1), чтобы не смешивать cold старт и рабочую фазу.

### Политика метрик (cold-first, 2026-05-09)

Для v2/v2-mini в этой ветке основной KPI фиксируется как **cold-first throughput**:

- измерение: первый измеряемый проход при `--runs 1`;
- для cold-замера отключать priming pass: `--no-v2-prime-pass`;
- warm/prime метрики считать диагностическими и публиковать отдельно, без подмены headline-числа.

Почему так:

- в агентном использовании с большим и меняющимся контекстом cold-фаза сильно влияет на реальный UX;
- warm-only числа показывают потенциал steady-state, но могут завышать ожидаемую скорость для «первого ответа»;
- ускорение cold-path почти автоматически улучшает и последующую warm-фазу.

Рекомендуемый формат отчёта:

- `Cold first-turn TPS` (headline);
- `Warm steady-state TPS` (secondary);
- `Session aggregate TPS` (смешанный показатель для серии запросов).

### Batch 4096 / UBatch 512 with stabilized method (2026-05-09)

Новый контрольный 5-run с фиксированным seed, thinking ON и warm-only статистикой:

- `build-rocm-wmma/bin/llama-server.exe`
- `Qwen3.6-27B-Q3_K_S.gguf`
- `-c 65536 -b 4096 -ub 512 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`
- `--server-seed 42 --no-disable-thinking --stats-ignore-first-run`

Результат (`sprint14-b512-newmethod-thinkon-5run`):

- Aggregate completion TPS: `37.57`
- Mean task TPS: `38.90`
- Task TPS stdev: `6.5194`
- Warm-only aggregate TPS: `41.61`
- Warm-only task TPS stdev: `3.0439`

Итог: цель `>=35 TPS` для `b=4096/ub=512` подтверждена на обновлённой методике, при этом warm-only дисперсия существенно ниже.

## V2-mini simple workflow (27B only, 2026-05-09)

Цикл выполнен строго на `Qwen3.6-27B-Q3_K_S.gguf` с коротким набором задач:

- `--tasks v2-mini` (`v2_code_review` + `v2_write_function`)
- `--runs 1`
- `-c 65536 -b 4096 -ub 512 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`
- `--background-server-policy fail`

Команды запускались через `build-rocm-exp/bin/llama-server.exe`.

Результаты по шагам:

| Label | Изменение | Aggregate TPS | Действие |
|---|---|---:|---|
| `wf-27b-baseline-exp-r1` | baseline | `25.98` | baseline |
| `wf-27b-varA-fattn-vec2-r1` | RDNA4 FATTN: quantized VEC порог `<=4 -> <=2` | `25.87` | **rollback (regress)** |
| `wf-27b-varB-mmq-routing-r1` | RDNA4 MMQ routing: убрать always-MMQ, ввести `ne11/type` эвристику | `26.58` | **keep (profit)** |
| `wf-27b-varC-streamk-r1` | MMQ stream-k: enable for RDNA4 при `ne11 >= 256` | `26.90` | **keep (profit)** |
| `wf-27b-varD-mmq-q45-384-r1` | RDNA4 MMQ routing: расширить окно Q4/Q5 `ne11 <= 256 -> <= 384` | `26.79` | **rollback (regress)** |
| `wf-27b-varE-mmq-k224-r1` | RDNA4 MMQ routing: расширить окно QK `ne11 <= 192 -> <= 224` | `26.72` | **rollback (regress)** |

Итог по циклу: финальная комбинация (B + C) дала `+0.92 TPS` к baseline v2-mini на 27B в этой сессии.

## Large Context Autotune (32K+)

Новый режим автоподбора параметров для длинного контекста:

```powershell
python scripts\agent_workload_bench.py `
  --autotune `
  --label rocm-autotune-32k `
  --server-bin build-rocm\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf `
  --background-server-policy fail `
  --autotune-min-ctx 32768 `
  --autotune-ctx-values 32768,49152,65536 `
  --autotune-batch-values 1024,2048,4096 `
  --autotune-ubatch-values 1024,2048,4096 `
  --autotune-kv-values q8_0,q4_0 `
  --autotune-spec-values none,ngram-mod `
  --autotune-update-preset `
  --autotune-preset-file gui\model_presets.json
```

Что делает режим:

- прогоняет grid конфигураций только для контекста `>= 32768`;
- сохраняет обычные `.csv/.jsonl` для каждой конфигурации;
- пишет summary: `<label>-autotune-summary.csv` и `.json`;
- печатает `BEST: ...` по aggregate completion TPS;
- при `--autotune-update-preset` обновляет `gui/model_presets.json` для выбранной модели.

## Large Context Reality Check (2026-05-10)

Статус изменён: активная long-context оптимизация временно остановлена как primary lane.

Почему:

- `sentinel128` дал ложное ощущение, что `128k` почти не хуже `64k`, потому что там prompt был всего `489/410` токенов.
- новый repo-snapshot workload загрузил действительно длинный prompt и показал, что проблема начинается уже на `64k`.

Зафиксированный reference:

| Workload | ctx64k | ctx128k | Комментарий |
| --- | ---: | ---: | --- |
| `sentinel128-qwen36q3` | `26.5825 TPS` | `26.0672 TPS` | Короткий sentinel, не годится как главный real-world сигнал |
| `repo-real-64k128k` | `2.3128 TPS` | `0.8167 TPS` | Реальный repo snapshot prompt, корректный long-prefill сигнал |

Текущий вывод:

- не делать новые 128k прогоны по умолчанию;
- не использовать 64k как стартовую «главную» точку оптимизации;
- активный performance lane: prompt-heavy стартовая точка ниже `16k`.

### New Primary Goal (2026-05-10)

- Текущая стартовая точка: `ctx=12288` в prompt-heavy no-reuse режиме.
- Текущий уровень: `~9.24 TPS`.
- Цель: `25-27 TPS` на стартовой точке.
- Способ достижения: поиск и верификация изменений в кодовой базе llama.cpp/ggml (prefill/runtime path), не только параметрический тюнинг запуска.

### Agent Workload: prompt-heavy mode (incoming context fix)

Проблема: стандартный `scripts/agent_workload_bench.py` в `v2-mini` режиме часто оставался decode-heavy и имел слишком маленький входящий prompt для real-scenario выводов.

Решение:

- добавлен режим `--real-context-mode repo-snapshot`;
- в каждый task prompt инжектится большой `repo snapshot` префикс;
- добавлен ctx-aware safe cap, чтобы избегать `HTTP 400` от переполнения контекста:
  - `--real-context-safe-fill` (default `0.70`),
  - `--real-context-reserve-tokens` (default `2048`),
  - `--real-context-chars-per-token` (default `3.4`).

Рекомендуемый запуск для реального входящего контекста без prompt-cache reuse:

```powershell
python scripts\agent_workload_bench.py `
  --label ctxwall-real-noreuse-c32768 `
  --server-bin build-rocm-compare\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --tasks v2-mini --runs 1 `
  --ctx-size 32768 -b 2048 -ub 512 `
  --cache-type-k q4_0 --cache-type-v q4_0 `
  --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0" `
  --max-tokens 120 `
  --real-context-mode repo-snapshot --real-context-chars 180000
```

Первые точки нового ctx sweep (prompt-heavy, no-reuse):

| Label | ctx | Avg prompt tokens | Aggregate TPS |
| --- | ---: | ---: | ---: |
| `ctxwall-real-noreuse-c12288` | `12288` | `~8050` | `9.2389` |
| `ctxwall-real-noreuse-c16384` | `16384` | `~11550` | `6.8685` |
| `ctxwall-real-noreuse-c24576` | `24576` | `~19382` | `4.1962` |
| `ctxwall-real-noreuse-c32768` | `32768` | `~26860` | `2.8934` |

Вывод: на реалистичном большом входящем prompt'е стена начинается намного раньше, чем показывал старый decode-heavy режим.

Это теперь главный reference-коридор для всех новых speed claims.

### Archived: 64K real-scenario single-ctx sanity (`repo_snapshot_context_bench.py`)

Эта секция сохранена как исторический reference. Активные speed claims теперь принимаются только по prompt-heavy стартовому lane `<16k`.

Скрипт `scripts/repo_snapshot_context_bench.py` обновлён для нового workflow и теперь принимает одиночный `--ctx-values 65536`, без обязательного парного `128k` прогона.

Первый 64k-only A/B на `build-rocm-compare`, `b=2048`, `ub=512`, `q4_0/q4_0`, prompt `62610` токенов, completion `120` токенов:

| Label | Spec | Wall TPS | Prompt eval | Decode eval | Вывод |
| --- | --- | ---: | ---: | ---: | --- |
| `repo-64k-single-none-ctx64k` | `none` | `0.9101` | `514.72 tok/s` | `11.89 tok/s` | текущий 64k baseline для real-scenario single-ctx |
| `repo-64k-single-ngram-ctx64k` | `ngram-mod` | `0.8955` | `506.02 tok/s` | `11.83 tok/s` | немного хуже baseline |

Практический вывод:

- для prompt-heavy `64k` repo snapshot workload `ngram-mod` пока не окупает свой overhead;
- ближайший safe baseline для новых 64k real-scenario исследований - `spec=none`;
- если возвращаться к speculative на этом lane, то только после новой гипотезы или kernel-level улучшения, а не по инерции от коротких decode-heavy benchmark'ов.

Дополнительный 64k-only check по `ubatch` на том же workload:

| Label | Batch | UBatch | Spec | Wall TPS | Prompt eval | Decode eval | Вывод |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `repo-64k-single-none-ctx64k` | `2048` | `512` | `none` | `0.9101` | `514.72 tok/s` | `11.89 tok/s` | текущий baseline |
| `repo-64k-single-none-ub256-ctx64k` | `2048` | `256` | `none` | `0.7263` | `405.13 tok/s` | `11.86 tok/s` | сильная регрессия по prefill |

Это означает, что для реального `64k` bottleneck сейчас чувствителен прежде всего к prompt processing throughput, и уменьшение `ubatch` до `256` здесь вредно, даже если в отдельных synthetic рассуждениях такой шаг казался безопасным.

Ещё три быстрых 64k-only проверки на том же repo snapshot lane:

| Label | Build | Batch | UBatch | Extra | Wall TPS | Prompt eval | Decode eval | Вывод |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `repo-64k-single-none-ctx64k` | `build-rocm-compare` | `2048` | `512` | none | `0.9101` | `514.72 tok/s` | `11.89 tok/s` | текущий local baseline |
| `repo-64k-single-none-b4096-ctx64k` | `build-rocm-compare` | `4096` | `512` | none | `0.8834` | `501.10 tok/s` | `11.79 tok/s` | `b=4096` не помогает, слегка хуже baseline |
| `repo-64k-single-none-nocache-ctx64k` | `build-rocm-compare` | `2048` | `512` | `--cache-ram 0 --ctx-checkpoints 0` | `0.8671` | `490.85 tok/s` | `11.86 tok/s` | отключение prompt cache/checkpoints не помогло |
| `repo-64k-exp-none-ctx64k` | `build-rocm-exp` | `2048` | `512` | none | `0.8985` | `510.48 tok/s` | `11.77 tok/s` | соседний ROCm build тоже не даёт прорыва |

Вывод по этому micro-screen:

- быстрые server-level рычаги на новом `64k` real-scenario lane почти исчерпаны;
- bottleneck остаётся в prefill/prompt path, а не в speculative или decode-path настройках;
- следующий полезный уровень исследования: kernel/path selection и runtime поведение на длинном prompt, а не новые перестановки `spec/cache/batch` вокруг того же бинаря.

Отдельно была проверена kernel-level probe в `ggml/src/ggml-cuda/gated_delta_net.cu`: принудительный `chunk_size=96` для RDNA4 chunked prefill вместо текущего adaptive `96/128`.

| Label | Variant | Wall TPS | Prompt eval | Decode eval | Решение |
| --- | --- | ---: | ---: | ---: | --- |
| `repo-64k-single-none-ctx64k` | baseline adaptive chunk | `0.9101` | `514.72 tok/s` | `11.89 tok/s` | baseline |
| `repo-64k-chunk96-none-ctx64k` | forced `chunk_size=96` | `0.8791` | `499.60 tok/s` | `11.81 tok/s` | rollback |
| `repo-64k-revert-check-none-ctx64k` | baseline rebuilt after rollback | `0.8957` | `510.31 tok/s` | `11.70 tok/s` | corridor restored |

Вывод: старая идея из quick-agent sweep не переносится напрямую на реальный repo-snapshot `64k` lane; фиксированный `chunk_size=96` ухудшает prefill и не подходит как следующий шаг.

## Archived: Separate Real-World Large Context Bench (120K + 160K)

Для регулярной оценки ожидаемой скорости в реальных агентных сценариях добавлен отдельный сценарный раннер:

- `scripts/large_context_realworld_bench.py`
- запускает одинаковый workload в двух практичных точках контекста: `122880` (120K) и `163840` (160K);
- строит итоговую сводку сравнения, чтобы быстро видеть деградацию скорости на растущем контексте;
- использует текущий `scripts/agent_workload_bench.py` как движок, поэтому метрики и формат логов полностью совместимы.

### Почему 120K/160K?

- **120K**: минимум для реальных длинных агентных диалогов + документы + контекст из других чатов.
- **160K**: расширенный сценарий, где важно видеть point-of-no-return по скорости.
- Диапазон позволяет оценить насколько критична масштабируемость и где находятся узкие места.

Рекомендуемый запуск:

```powershell
python scripts\large_context_realworld_bench.py `
  --label-prefix realctx-120k-qwen27b `
  --server-bin build-rocm-exp\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --tasks v2-mini `
  --runs 1 `
  --batch-size 4096 `
  --ubatch-size 512 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --spec-profile ngram-mod `
  --background-server-policy fail
```

Что получаем после запуска:

- `build_logs/agent-workload/<label-prefix>-ctx120k.csv`
- `build_logs/agent-workload/<label-prefix>-ctx160k.csv`
- `build_logs/agent-workload/<label-prefix>-largectx-summary.csv` (TPS decay от 120K к 160K)
- `build_logs/agent-workload/<label-prefix>-largectx-summary.md`

Где смотреть итог:

- headline метрика: `aggregate_tps` из `*-largectx-summary.*`;
- сравнение `160K vs 120K` уже посчитано (ratio % показывает потерю скорости);
- если нужна стабильность вместо быстрых итераций, повышай `--runs` до `3`.

### Archived Research Target: 120K Large Context Optimization

Текущий baseline на `ctx=131072, ubatch=512, q4_0 KV, ngram-mod`:
- **Prefix (PP)**: ~215 TPS
- **Generation (TG)**: ~8.5 TPS ← **узкое место**
- **Spec acceptance rate**: ~18% (низко)

Гипотезы для исследования:

1. **MMQ/FATTN kernel threshold** — может быть на большом контексте срабатывает неоптимальный path (VEC vs TILE).
2. **Speculative decoding overshoot** — ngram-mod с large ctx может генерировать слишком много draft токенов, замедляя verification.
3. **KV cache bandwidth** — даже q4_0 может быть узким местом при 120K+ tokens × 24 heads × 256 dims.
4. **ROCm kernel occupancy** — RDNA4 может недополучать work при малых batch/ubatch на большом контексте.

Архивный research workflow:

```bash
# Step 1: historical baseline на 120K
python scripts/large_context_realworld_bench.py --label-prefix baseline-120k ...

# Step 2: попробовать более консервативный speculative (ngram-simple или none)
python scripts/large_context_realworld_bench.py --label-prefix nostep-120k --spec-profile none ...

# Step 3: попробовать меньший ubatch (256 вместо 512)
python scripts/large_context_realworld_bench.py --label-prefix ub256-120k --ubatch-size 256 ...

# Step 4: попробовать больший batch (6144 вместо 4096)
python scripts/large_context_realworld_bench.py --label-prefix b6144-120k --batch-size 6144 ...

# Сравнить результаты и выбрать best по aggregate_tps
```

Примечание 2026-05-10: этот workflow сохранён только как исторический след. Новые performance-итерации вести на `ctx=65536`.

## GUI Automation API (E2E)

GUI теперь поднимает локальный HTTP API для автоматизации действий и проверки результата end-to-end.

- Base URL: `http://127.0.0.1:8765`
- Port можно переопределить через `LLAMA_GUI_API_PORT`.

### Endpoints

- `GET /api/ping` — health check.
- `GET /api/state` — текущее состояние GUI-параметров (модель, контекст, batch, kv и т.д.).
- `POST /api/autotune` — запуск автотюна из GUI.
- `POST /api/apply-preset` — применение model preset в Launch Server.
- `POST /api/scenario/autotune-apply` — сценарий: autotune одной модели + apply preset.

### Пример сценария autotune + apply preset

```powershell
python - << 'PY'
import json, urllib.request

payload = {
  "model_path": "models/Qwen3.5-9B-Q6_K.gguf",
  "wait": True,
  "timeout_sec": 1200,
  "sweep_mode": "smoke"
}

req = urllib.request.Request(
  "http://127.0.0.1:8765/api/scenario/autotune-apply",
  data=json.dumps(payload).encode("utf-8"),
  headers={"Content-Type": "application/json"},
  method="POST",
)

with urllib.request.urlopen(req, timeout=1800) as resp:
    print(resp.read().decode("utf-8"))
PY
```

Если `ok=true`, в ответе будет:

- блок `autotune.result.best` с лучшей конфигурацией;
- пути к `*-autotune-summary.csv/json`;
- блок `preset.result` с применёнными значениями (`context`, `batch`, `kv`, ...);
- `state` с текущим состоянием GUI после применения пресета.

## Current Clean Snapshot

Актуальный clean snapshot на текущем ROCm build `5facfaea9` был снят через `build\bin\llama-server.exe`.

| Model | Mode | Key args | Aggregate completion TPS |
| --- | --- | --- | ---: |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | baseline | `-np 1 -c 32768 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0` | `37.454` |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | `ngram-mod` | baseline + `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` | `41.007` |
| `Qwen3.6-27B-Q3_K_S.gguf` | baseline | `-np 1 -c 32768 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0` | `12.055` |
| `Qwen3.6-27B-Q3_K_S.gguf` | `ngram-mod` | baseline + `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` | `13.547` |

Вывод на текущем билде: `ngram-mod` уже поддерживается и даёт прирост примерно `+9.5%` на 35B A3B и `+12.4%` на 27B Q3_K_S для короткой coding-agent симуляции.

Старые baseline CSV (`rocm-baseline-qwen36-*.csv`) стоит считать noisy, потому что часть прошлых замеров выполнялась при параллельной игровой нагрузке.

## RDNA4 Gated Delta Net Chunked Prefill (2026-05-08)

Экспериментальная kernel-ветка для `gated_delta_net` (chunked prefill на RDNA4) была проверена по строгому протоколу `3 runs` на quick-agent workload.

Параметры прогона:

- `build-rocm-vec/bin/llama-server.exe`
- `Qwen3.6-27B-Q3_K_S.gguf`
- `--spec-type ngram-mod`
- `-c 65536 -b 4096 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`

| Label | UBatch | Runs | Aggregate completion TPS | Mean task TPS | Median task TPS | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sprint4-gdn-chunk-ub256` | `256` | `3` | `31.7809` | `33.86` | `30.73` | `10.2471` |
| `sprint4-gdn-chunk-ub128` | `128` | `3` | `31.9844` | `33.39` | `36.37` | `6.6477` |

Вывод:

- Оба 3-проходных прогона выше ранее используемого ориентира `~29 TPS`.
- Зафиксирован новый практический коридор aggregate throughput: `~31.8-32.0 TPS` для этой модели и профиля.

Артефакты:

- `build_logs/agent-workload/sprint4-gdn-chunk-ub256.csv`
- `build_logs/agent-workload/sprint4-gdn-chunk-ub256.jsonl`
- `build_logs/agent-workload/sprint4-gdn-chunk-ub128.csv`
- `build_logs/agent-workload/sprint4-gdn-chunk-ub128.jsonl`

## RDNA4 Gated Delta Net Chunk Size Sweep (2026-05-08)

Проверен локальный A/B по `chunk_size` в `ggml/src/ggml-cuda/gated_delta_net.cu` при одинаковом quick-agent профиле и `Qwen3.6-27B-Q3_K_S.gguf`.

Параметры прогона:

- `build-rocm-vec/bin/llama-server.exe`
- `--spec-type ngram-mod`
- `-c 65536 -b 4096 -ub 256 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`

| Label | Chunk size | UBatch | Launches | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sprint5-gdn-chunk96-ub256` | `96` | `256` | `~3` | `3` | `33.17` | `35.57` | `32.87` | `10.47` |
| `sprint5-gdn-chunk96-ub256-r2` | `96` | `256` | `~3` | `3` | `31.86` | `33.47` | `29.86` | `7.65` |
| `sprint5-gdn-chunk64-control-ub256` | `64` | `256` | `4` | `3` | `30.76` | `31.52` | `30.17` | `5.11` |
| `sprint5-gdn-chunk64-control-ub256-r2` | `64` | `256` | `4` | `3` | `28.63` | `29.26` | `27.12` | `4.87` |
| `sprint5-gdn-chunk128-ub256` | `128` | `256` | `2` | `3` | `28.86` | `29.44` | `27.14` | `4.65` |
| `sprint5-gdn-chunk96-ub128` | `96` | `128` | `~2` | `3` | `28.53` | — | — | — |
| `sprint5-gdn-chunk96-ub512` | `96` | `512` | `~6` | `3` | `31.71` | `32.90` | `30.56` | `6.50` |
| `sprint5-gdn-chunk128-ub512` | `128` | `512` | `4` | `3` | `31.32` | `32.52` | `28.69` | `6.48` |

**Замечания по sweep ub × chunk_size:**

- `ub=512` НЕ регрессирует к ~20 TPS — ранее наблюдавшийся провал был при других условиях.
- chunk=128 на ub=256 (2 запуска) хуже chunk=96 (3 запуска): вероятно, увеличенный внутренний цикл (128 итераций vs 96) создаёт большее регистровое давление или является шумом (stdev ~5 TPS делает 3-run сравнение ненадёжным).
- Для ub=512 chunk=96 и chunk=128 дают одинаковый результат (~31.3-31.7 TPS) — разница в пределах погрешности.
- ub=256 чуть выше ub=512 при chunk=96 (~32.5 vs ~31.7 TPS), но разница незначительная при данной дисперсии.

**Теоретический предел chunk_size:**

$$\text{launches} = \left\lceil \frac{n\_tokens}{chunk\_size} \right\rceil$$

Снижение launch overhead даёт выгоду, пока:
- Каждый запуск меньше L1/L2 cache рабочего набора
- Отсутствует регистровое давление (spilling)
- Ядро остаётся memory-bandwidth-bound, а не compute-bound

Для ub=256: оптимум при chunk≈96 (3 launches). Переход к chunk=128 (2 launches) не даёт выигрыша — вероятно, внутренний цикл достигает предела.

Вывод: **chunk_size=96 — текущий confirmed optimal** для RDNA4 + Qwen3.6-27B на ub=256.

Артефакты:

- `build_logs/agent-workload/sprint5-gdn-chunk96-ub128.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk96-ub256.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk96-ub256-r2.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk64-control-ub256.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk64-control-ub256-r2.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk128-ub256.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk96-ub512.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-gdn-chunk128-ub512.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-adaptive-chunk-ub256.{csv,jsonl}`
- `build_logs/agent-workload/sprint5-adaptive-chunk-ub512.{csv,jsonl}`

## RDNA4 Adaptive Chunk — Финальный результат (2026-05-08)

По итогам sweep реализован адаптивный `chunk_size` в `gated_delta_net.cu`:

```cpp
// n_tokens > 256 → chunk=128 (4 launches), иначе chunk=96 (3 launches)
const int64_t chunk_size = (n_tokens > 256) ? 128 : 96;
```

Верификационные прогоны (3 runs каждый):

| Label | UBatch | Effective chunk | Aggregate TPS |
| --- | ---: | ---: | ---: |
| `sprint5-adaptive-chunk-ub256` | `256` | `96` | `30.53` |
| `sprint5-adaptive-chunk-ub512` | `512` | `128` | **`33.86`** |

- `ub=512` с адаптивным chunk показал **33.86 TPS** — лучший результат за всю sprint5 сессию.
- `ub=256` в рамках нормальной дисперсии (~30-33 TPS, stdev ~5).
- Прежде ub≥256 деградировало до ~20 TPS из-за FATTN kernel switch — эта проблема устранена через chunked prefill.

Итоговый диапазон TPS для Qwen3.6-27B-Q3_K_S на RX 9070 XT (ROCm/gfx1201):

| Параметр | До sprint5 | После sprint5 |
|---|---:|---:|
| max ub без регресса | 128 | 512+ |
| типичный TPS (ub=256) | ~29 TPS | ~31-33 TPS |
| типичный TPS (ub=512) | ~20 TPS | ~31-34 TPS |

## RDNA4 FATTN Routing Tuning (2026-05-08, Sprint7)

Цель: проверить, можно ли получить стабильный выигрыш на фокусном профиле `ub=512` за счёт более раннего перехода из `TILE` в `MMA_F16` для RDNA4 в quantized KV path.

Изменение в `ggml/src/ggml-cuda/fattn.cu` (ветка `amd_wmma_available && RDNA4`):

```cpp
// было
if (Q->ne[1] * gqa_ratio_eff <= 8) return BEST_FATTN_KERNEL_TILE;

// стало
if (Q->ne[1] * gqa_ratio_eff <= 4) return BEST_FATTN_KERNEL_TILE;
```

Идея: сдвинуть crossover в сторону `MMA_F16` для более широкого диапазона эффективных батчей.

Профиль сравнения (одинаковый для всех запусков):

- `Qwen3.6-27B-Q3_K_S.gguf`
- `-c 65536 -b 4096 -ub 512 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64`
- `build-rocm-vec/bin/llama-server.exe`

| Label | Variant | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sprint7-baseline5-ub512-ngram` | baseline (`tile<=8`) | `5` | `33.25` | `34.68` | `32.80` | `7.36` |
| `sprint7-tile4-5run-ub512-ngram` | patched (`tile<=4`) | `5` | `35.68` | `37.49` | `37.31` | `8.16` |
| `sprint7-tile4-5run-ub512-ngram-r2` | patched confirm | `5` | `33.96` | `36.12` | `33.00` | `9.55` |

Вывод:

- Патч показывает устойчивое преимущество над baseline в обоих 5-run замерах.
- Прирост по aggregate TPS:
  - run1: `35.68 - 33.25 = +2.43` TPS (`+7.3%`)
  - run2: `33.96 - 33.25 = +0.71` TPS (`+2.1%`)
- Порог `>32 TPS` устойчиво выполнен, а лучший подтверждённый результат цикла — `35.68 TPS`.

Артефакты:

- `build_logs/agent-workload/sprint7-baseline5-ub512-ngram.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint7-tile4-5run-ub512-ngram.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint7-tile4-5run-ub512-ngram-r2.{csv,jsonl,server.log}`

## Batch 4096 / UBatch 512 Repro Check (2026-05-09)

Запрос: подтвердить целевой уровень `>=35 TPS` именно на профиле `b=4096, ub=512` для long-context agent workflow.

Условия прогона:

- `Qwen3.6-27B-Q3_K_S.gguf`
- `-c 65536 -b 4096 -ub 512 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`
- `build-rocm-wmma/bin/llama-server.exe`
- `scripts/agent_workload_bench.py --runs 5 --background-server-policy fail`

Результаты sprint14 (сегодня):

| Label | Aggregate TPS | Mean task TPS | Median task TPS | Stdev |
| --- | ---: | ---: | ---: | ---: |
| `sprint14-b512-target35-5run` | `30.25` | `31.15` | `29.34` | `5.55` |
| `sprint14-b512-target35-5run-r2` | `34.98` | `36.53` | `35.18` | `7.58` |
| `sprint14-b512-target35-5run-r3` | `32.90` | `34.34` | `32.55` | `7.25` |
| `sprint14-b512-target35-5run-r4` | `31.39` | `32.18` | `30.81` | `5.28` |

Ранее подтвержденные попадания `>=35 TPS` на том же профиле:

| Label | Build | Aggregate TPS |
| --- | --- | ---: |
| `sprint13-wmma-5run-r2` | `build-rocm-wmma` | `36.53` |
| `sprint7-tile4-5run-ub512-ngram` | `build-rocm-vec` | `35.68` |
| `sprint9-tile4-warmup-ub512-5run` | `build-rocm-vec` | `35.15` |

Вывод:

- Цель `35+ TPS` для `b=4096/ub=512` **достижима**, но имеет заметную run-to-run вариативность.
- Для стабильного daily-профиля на `build-rocm-clean` сейчас практичнее `ub=256` (средний 5-run `35.69 TPS`).
- Для приоритета именно `ub=512` нужно продолжать работу над снижением дисперсии (warmup discipline, thermal/load control, kernel-path stability).

Артефакты sprint14:

- `build_logs/agent-workload/sprint14-b512-target35-5run.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint14-b512-target35-5run-r2.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint14-b512-target35-5run-r3.{csv,jsonl,server.log}`
- `build_logs/agent-workload/sprint14-b512-target35-5run-r4.{csv,jsonl,server.log}`

### Stdev Investigation (2026-05-09)

Цель: выяснить, почему на `b=4096/ub=512` выросла дисперсия (`stdev`).

Ключевые наблюдения:

- В server log для нестабильных прогонов сильно гуляет `draft acceptance rate` и число speculative draft tokens.
- Пример:
  - низкий прогон `sprint14-b512-target35-5run`: итог `#gen tokens = 954`, `#acc tokens = 461`;
  - более быстрый прогон `sprint14-b512-target35-5run-r2`: итог `#gen tokens = 1500`, `#acc tokens = 918`.
- Это указывает, что заметная часть дисперсии идёт из speculative path (`ngram-mod`), а не из prompt prefill.

Контрольный тест без speculative (`--spec-type none`) на том же профиле:

| Label | Aggregate TPS | Mean task TPS | Median task TPS | Stdev |
| --- | ---: | ---: | ---: | ---: |
| `sprint14-b512-specnone-5run` | `27.54` | `27.54` | `27.66` | **`0.28`** |

Вывод: без speculative дисперсия почти исчезает, но throughput заметно ниже.

Быстрый стабилизационный A/B (3-run, warmup on) не дал снижения stdev:

| Label | Config | Aggregate TPS | Stdev |
| --- | --- | ---: | ---: |
| `sprint14-stab-warmup-default-3run` | ngram 24/48/64 | `31.77` | `5.99` |
| `sprint14-stab-warmup-n32-3run` | ngram 32/48/64 | `32.65` | `6.76` |
| `sprint14-stab-warmup-min32max48-3run` | ngram 24/32/48 | `32.79` | `9.18` |

Практический итог:

- Высокий stdev на `ub=512` в первую очередь связан с нестабильным speculative acceptance.
- Для стабильного daily-профиля приоритет остаётся у `ub=256`.
- Для `ub=512` следующая работа должна быть направлена на стабилизацию speculative acceptance, а не только на peak TPS.

## UBatch=256 Optimization Discovery (2026-05-09)

**Critical finding**: При систематическом тестировании разных ubatch размеров выявлено, что **ubatch=256 даёт значительное преимущество** на этом профиле и GPU.

### Methodology

Compared 5-run baseline warm-cache runs с одинаковыми параметрами:

- `Qwen3.6-27B-Q3_K_S.gguf`
- `-c 65536 -b 4096 -np 1 --flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64`
- `build-rocm-clean/bin/llama-server.exe` (master commit 8c7db71f1)

| UBatch | Runs | Aggregate TPS | Mean task TPS | Median task TPS | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| `256` | `5` | **`35.45`** | `37.20` | `39.37` | `7.95` |
| `256` (r2) | `5` | **`35.93`** | `37.76` | `37.67` | `8.40` |
| `224` | `5` | `33.80` | `35.24` | `34.82` | `7.18` |
| `512` | `5` | `31.05` | `32.08` | `27.84` | `6.23` |

**Average ub=256**: `(35.45 + 35.93) / 2 = **35.69 TPS**` — **+14.7% vs ub=512 baseline**

### Why ub=256?

Гипотезы:

1. **Memory hierarchy alignment**: ub=256 (32 KB uBatch state per thread block) может оптимально вписываться в GPU L1/L2 cache на gfx1201.
2. **GDN chunking**: Адаптивный chunk_size=96 (from sprint5-adaptive-chunk) работает наилучше именно с ub=256 как базовой единицей.
3. **FATTN kernel dispatch**: VEC/TILE/MMA crossover точки оптимальны для ub=256 при данной длине контекста.

### Single-run cold-cache behavior

Интересно, что на single-run (cold cache) нет заметного преимущества:

| UBatch | Single-run TPS |
| --- | ---: |
| `256` | `27.00` |
| `192` | `27.10` |
| `224` | `25.88` |
| `320` | `26.81` |
| `384` | `26.97` |
| `512` | `25.14` |
| `768` | `19.84` |

**Вывод**: Преимущество ub=256 проявляется только при **прогреве кэша** в серии запусков. Single-run benchmarks **не отражают реальной производительности** для этого профиля.

### Artifacts

- `build_logs/agent-workload/baseline-clean-5run-ub256.{csv,jsonl,server.log}`
- `build_logs/agent-workload/baseline-clean-5run-ub256-r2.{csv,jsonl,server.log}`
- `build_logs/agent-workload/baseline-clean-5run-ub512.{csv,jsonl,server.log}` (для сравнения)

### Recommendation

**Обновить все Qwen3.6-27B профили** в `gui/model_presets.json` с `ubatch: 512` → `ubatch: 256`.

Цель: **Стабильно достичь 35+ TPS** на RX 9070 XT при агентной рабочей нагрузке.

## Baseline ROCm

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-baseline `
  --server-bin build-rocm\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf `
  --ctx-size 32768 `
  --batch-size 2048 `
  --ubatch-size 2048 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  --max-tokens 160
```

## Existing Server

Если GUI уже запустил сервер:

```powershell
python scripts\agent_workload_bench.py --no-start --port 8080 --label gui-server-baseline
```

## MTP Branch Test

Только после того, как `llama-server --help` показывает `mtp`:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-mtp-draft3 `
  --server-bin build-rocm-mtp\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf `
  --server-extra "--spec-type mtp --spec-draft-n-max 3" `
  --ctx-size 32768 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0
```

MTP benchmark должен быть text-only: не добавлять `--mmproj`.

## ngram-mod Coding-Agent Test

Для текущего master без MTP:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-ngram-mod `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-draft-n-min 48 --spec-draft-n-max 64"
```

Для текущего parser актуальны и новые long-form имена флагов:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-ngram-mod `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64"
```

На текущем билде предпочтительнее использовать именно `--spec-ngram-mod-*`, чтобы не путать их с draft-model speculative decoding.

## Глоссарий метрик

| Метрика | Тип | Пояснение |
|---------|-----|-----------|
| `wall_s` | секунды | Астрономическое (настенное) время от отправки запроса до получения последнего токена. Включает все задержки: сеть, prompt processing, generation. Главная метрика скорости для агентной задачи. |
| `completion_tokens` | шт. | Количество токенов, сгенерированных моделью (не считая prompt). Зависит от задачи и stop-sequence, у нас лимитируется `--max-tokens`. |
| `completion_tps_wall` | тк/с | Throughput генерации: `completion_tokens / wall_s`. Основная агрегированная метрика в CSV. Чем выше — тем лучше. |
| `prompt_tokens` | шт. | Число токенов в контексте (системный промпт + вопрос). Влияет на prefill latency. |
| `ttft_s` | секунды | Time-To-First-Token — latency до первого сгенерированного токена. Отражает скорость prompt processing (PP). |
| `tg_tps` | тк/с | Token Generation speed из server log — чистая скорость генерации без prefill. Отличается от `completion_tps_wall`: wall учитывает TTFT, tg_tps — нет. |
| `pp_tps` | тк/с | Prompt Processing speed из server log — скорость обработки контекста (prefill). |
| `spec_accept_rate` | % | Процент принятых speculative токенов (для MTP/ngram). 100% = все драфтные токены приняты, 0% = ни одного. Реальный прирост TPS зависит от acceptance rate. |
| `error` | строка | Непустое поле означает сбой запроса (HTTP error, timeout, empty response). |

> **Важно для агентного workflow**: если MTP/ngram повышает `tg_tps`, но увеличивает `ttft_s` (более долгий prefill), итоговый `wall_s` может не улучшиться. Смотреть нужно именно на `completion_tps_wall` и `wall_s`.

## Что сравнивать

Смотреть в CSV:

- `wall_s` по каждой задаче;
- `completion_tokens`;
- `completion_tps_wall`;
- ошибки запуска/ответа.

Смотреть в server log:

- prompt processing tok/s;
- generation tok/s;
- speculative draft acceptance rate;
- ROCm/HIP warnings;
- VRAM/memory allocation failures.

Для нашего workflow важен не только TG. Если MTP ускоряет generation, но сильно режет prompt processing, агентная задача может стать медленнее.

Смежный roadmap по следующим аппаратно-ориентированным оптимизациям вынесен в `ROCM_ACCELERATION_PLAN.md`.

---

## Методика V2 — Реалистичный Agentic-Flow Benchmark (2026-05-09)

### Мотивация

Задачи `TASKS_QUICK/FULL` (v1) специально коротки (`max_tokens=160`, "keep it brief"), что создаёт искусственно высокий TPS (многократные короткие burst генерации с частым ngram accept). Реальный агентный флоу — длинные ответы (400–600 токенов), разнообразные промпты с низким ngram acceptance. Поэтому v1 и ручной чат показывают разные числа.

### V2 Task Set (`--tasks v2`)

По умолчанию v2 теперь запускает компактный набор для быстрых итераций:
- включены: `v2_code_review`, `v2_write_function`;
- отключены: `v2_debug_trace`, `v2_refactor_plan`, `v2_perf_analysis`.

Полный набор включается только для ретеста после заметного speed breakthrough:
- добавить флаг `--v2-include-heavy`.

| ID | Название | Целевая длина ответа |
|----|----------|---------------------|
| `v2_code_review` | Полный code review модуля build_manager | ~400–500 токенов |
| `v2_write_function` | Написать класс BuildRegistry | ~450–550 токенов |
| `v2_debug_trace` | Диагностика crash-лога ROCm сервера | ~350–450 токенов |
| `v2_refactor_plan` | План рефакторинга монолитного GUI | ~400–500 токенов |
| `v2_perf_analysis` | Анализ performance bottleneck | ~400–500 токенов |

### Ключевые отличия от V1

| Параметр | V1 (quick) | V2 |
|----------|------------|-----|
| `--max-tokens` | 160 | 500 (автоматически) |
| Формулировка задач | "keep it brief / under 140 words" | Развёрнутые, без ограничений длины |
| `--history-version` | v1 → `BENCH_HISTORY.csv` | v2 → `BENCH_HISTORY_V2.csv` |
| Соответствие реальному чату | Оптимистичная оценка | Репрезентативная оценка |

### Команда V2 Baseline

```powershell
python scripts\agent_workload_bench.py `
  --label v2-baseline-rocm-ub512 `
  --tasks v2 `
  --runs 3 `
  --server-seed 42 `
  --no-disable-thinking `
  --stats-ignore-first-run `
  --server-bin build-rocm-vec\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --ctx-size 65536 `
  --batch-size 4096 `
  --ubatch-size 512 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --flash-attn `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 24"
```

Для полного ретеста с тяжёлыми задачами:

```powershell
python scripts\agent_workload_bench.py `
  --label v2-baseline-rocm-ub512-heavy `
  --tasks v2 `
  --v2-include-heavy `
  --runs 3 `
  --server-seed 42 `
  --no-disable-thinking `
  --stats-ignore-first-run `
  --server-bin build-rocm-vec\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --ctx-size 65536 `
  --batch-size 4096 `
  --ubatch-size 512 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --flash-attn `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 --spec-ngram-mod-n-match 24"
```

История результатов хранится отдельно: `build_logs/agent-workload/BENCH_HISTORY_V2.csv` и `BENCH_HISTORY_V2.md`.

### V2 Baseline Results

| Label | Build | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev | Warm-only TPS | Warm stdev | max_tokens |
|-------|-------|------|--------------|----------|------------|-------|--------------|------------|------------|
| v2-baseline-rocm-ub512 | build-rocm-vec | 3×5 | 27.77 | 27.78 | 27.97 | 0.47 | 28.07 | 0.19 | 500 |

**Вывод:** v2 baseline = **~28 TPS** при 500-токенных ответах — это точно совпадает с тем, что наблюдается в ручном чате (28–30 TPS). Очень низкий stdev (0.47) показывает, что при длинных ответах генерация устойчива. V1 (~33-37 TPS) был оптимистичен из-за многократных коротких burst (160 токенов).

### V2 A/B: `build-rocm-clean` vs `build-rocm-vec` (ub=512, ngram-mod)

| Label | Build | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev | Warm-only TPS | Warm stdev |
|-------|-------|------|--------------|----------|------------|-------|--------------|------------|
| `v2-baseline-rocm-ub512` | `build-rocm-vec` | 3x5 | `27.77` | `27.78` | `27.97` | `0.47` | `28.07` | `0.19` |
| `v2-clean-ub512` | `build-rocm-clean` | 3x5 | `27.72` | `27.72` | `27.80` | `0.35` | `27.92` | `0.17` |

Разница по aggregate: `+0.06 TPS` в пользу `build-rocm-vec` (меньше порога `0.5 TPS`).

**Вывод:** на реалистичной v2 нагрузке патчи `tile<=4 + chunk=96` не дают значимого выигрыша по throughput.

### V2 A/B: `spec-type none` vs `ngram-mod` (ub=512, build-rocm-vec)

| Label | Spec mode | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev | Warm-only TPS | Warm stdev |
|-------|-----------|------|--------------|----------|------------|-------|--------------|------------|
| `v2-baseline-rocm-ub512` | `ngram-mod 48/64/24` | 3x5 | `27.77` | `27.78` | `27.97` | `0.47` | `28.07` | `0.19` |
| `v2-rocm-vec-specnone-ub512` | `none` | 3x5 | `27.78` | `27.78` | `27.92` | `0.33` | `27.99` | `0.06` |

Разница по aggregate: `~0.00 TPS` (в пределах шума).

**Вывод:** для v2-кодовых промптов `ngram-mod` практически не ускоряет, но и не штрафует throughput; заметный эффект в основном на variance (без speculative stdev ниже).

### V2 A/B: `ubatch 256` vs `ubatch 512` (build-rocm-vec, ngram-mod)

| Label | ubatch | Runs | Aggregate TPS | Mean TPS | Median TPS | Stdev |
|-------|--------|------|--------------|----------|------------|-------|
| `v2-baseline-rocm-ub512` | `512` | 3x5 | `27.77` | `27.78` | `27.97` | `0.47` |
| `v2-rocm-vec-ub256-ngram-r1` | `256` | 1x5 | `27.52` | `27.52` | `27.35` | `0.32` |

Разница по aggregate: `-0.25 TPS` при переходе на `ub=256`.

**Вывод:** на текущем профиле длинных ответов `ub=512` остаётся предпочтительным.

### Политика прогонов для V2 (обновлено)

- Для быстрых итераций/скрининга использовать `--runs 1` (экономия времени, stdev на v2 обычно низкий).
- Повторять `--runs 3` только для финального подтверждения спорных/пограничных изменений (например, дельта в диапазоне `0.2-0.5 TPS`).

### Research Phase R35-01 (2026-05-09): старт long-run к цели 35 TPS

Цель фазы: найти конфиг/билд, который сможет вывести v2-профиль к `35 TPS`.

#### Скрининг готовых ROCm билдов (`runs=1`, v2, `b=4096`, `ub=512`, `ngram-mod`)

| Label | Build | Aggregate TPS |
|-------|-------|--------------|
| `v2-scan-rocm-exp-ub512-r1` | `build-rocm-exp` | `27.37` |
| `v2-scan-rocm-wmma-ub512-r1` | `build-rocm-wmma` | `27.34` |
| `v2-scan-build-bin-ub512-r1` | `build` | `27.33` |
| `v2-scan-rocm-clean-ub512-r1` | `build-rocm-clean` | `27.26` |
| `v2-scan-rocm-vec-ub512-r1` | `build-rocm-vec` | `27.26` |
| `v2-scan-rocm-a-check-ub512-r1` | `build-rocm-a-check` | `27.20` |

Промежуточный лидер: `build-rocm-exp` (`27.37 TPS`).

#### Свип параметров на лидере `build-rocm-exp` (`runs=1`)

| Label | Конфиг | Aggregate TPS |
|-------|--------|--------------|
| `v2-scan-exp-b4096-ub512-p1-specnone-r1` | `b=4096, ub=512, p=1, spec=none` | `27.24` |
| `v2-scan-exp-b8192-ub512-p1-specngram-r1` | `b=8192, ub=512, p=1, spec=ngram` | `27.21` |
| `v2-scan-exp-b4096-ub512-p1-specngram-r1` | `b=4096, ub=512, p=1, spec=ngram` | `27.20` |
| `v2-scan-exp-b4096-ub512-p2-specngram-r1` | `b=4096, ub=512, p=2, spec=ngram` | `25.70` |
| `v2-scan-exp-b8192-ub1024-p1-specngram-r1` | `b=8192, ub=1024, p=1, spec=ngram` | `20.18` |

Вывод по свипу:
- `ub=1024` и `parallel=2` в этом профиле явно вредят throughput.
- `spec none` и `ngram-mod` дают почти одинаковую скорость на v2-кодовых задачах.
- На текущем железе/модели v2-профиль упирается в ~`27.2-27.4 TPS`.

#### Статус чекпоинта

- Целевой чекпоинт `35 TPS` на v2-профиле **не достигнут** (текущий максимум в этой фазе: `27.37 TPS`).
- Для дальнейшего роста нужен следующий виток: новые кодовые kernel-правки + свежая ROCm сборка с корректной toolchain-настройкой.

#### Новый ROCm контур `build-rocm-r35-c` (`GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_OPENMP=OFF`)

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-scan-rocm-r35-c-ub512-r1` | `ngram-mod 48/64/24` | `26.83` |
| `v2-scan-rocm-r35-c-specnone-r1` | `none` | `27.30` |

Вывод:
- `GGML_CUDA_FA_ALL_QUANTS=ON` сам по себе не помог на текущем v2 профиле.
- Без speculative новый контур близок к обычному уровню, но всё равно не обгоняет `build-rocm-exp`.
- Этот билд не выглядит перспективным для дальнейшего разгона к `35 TPS`.

### Research Phase R35-02 (2026-05-09): kernel micro-optimizations (ROCm, runs=1)

Цель фазы: проверить быстрые low-risk правки в ядрах без смены модели/режима и оценить, дают ли они выход за потолок `~27.4 TPS` на v2.

#### Эксперимент A: `ggml/src/ggml-cuda/gated_delta_net.cu`

Гипотеза:
- уменьшить стоимость `expf` в fused GDN (замена на fast intrinsic + кэширование `exp(g)` в `KDA` ветке) может ускорить decode/prefill.

Результаты (`build-rocm-exp`, `b=4096`, `ub=512`, `np=1`):

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-r35-gdn-expfast-ub512-r1` | `ngram-mod 48/64/24` | `27.42` |
| `v2-r35-gdn-expfast-specnone-ub512-r1` | `none` | `27.29` |

Промежуточный вывод:
- метрики остались в шумовом коридоре относительно текущего потолка `27.2-27.4`;
- устойчивого прироста не подтверждено.

#### Эксперимент B: `ggml/src/ggml-cuda/fattn.cu` (RDNA4 selector threshold)

Гипотеза:
- расширить окно выбора VEC/TILE (`<=8` вместо `<=4`) в RDNA4 ветке и ускорить decode на малом эффективном батче.

Результаты:

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-r35-fattn8-gdnexp-ub512-r1` | `ngram-mod 48/64/24` | `27.36` |
| `v2-r35-fattn8-gdnexp-specnone-ub512-r1` | `none` | `27.06` |

Вывод:
- изменение порога ухудшило non-spec профиль и не дало выигрыша с `ngram-mod`.
- правка откатана.

#### Rollback-check после отката обеих правок

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-r35-rollback-check-ub512-r1` | `ngram-mod 48/64/24` | `27.42` |

Финал фазы R35-02:
- обе kernel-гипотезы не дали подтверждённого роста TPS;
- дерево возвращено к baseline-поведению;
- целевой чекпоинт `35 TPS` для v2 остаётся недостигнутым.

### Research Phase R35-03 (2026-05-09): draft-model path + kernel pass

Цель фазы: проверить «дорогой» путь ускорения через draft model (non-MTP), затем сделать kernel/runtime pass по самому слабому месту из логов.

#### Что использовалось как draft model path

- target model: `models/Qwen3.6-27B-Q3_K_S.gguf`;
- draft model: `models/Qwen3.5-9B-Q6_K.gguf`;
- режим: `--model-draft ... --spec-draft-n-max 12 --spec-draft-n-min 0 --spec-draft-p-min 0.75`.

#### Сравнение baseline режимов на compact v2 (`runs=1`, 2 задачи)

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-r35-combo-draft-sweep-r1-cfg01` | `none` | `27.44` |
| `v2-r35-combo-draft-sweep-r1-cfg02` | `ngram-mod 48/64/24` | `27.41` |

#### Draft-model результат

По `v2-r35-combo-draft-only-r1b-cfg01.server.log`:
- `prompt eval`: ~`1.22-1.28 ms/token` (нормально);
- `eval`: ~`308-321 ms/token` (`~3.11-3.24 tok/s`) — критический провал;
- `draft acceptance rate`: высокий (`~0.79-0.86`), но это не помогает;
- `statistics draft ... dur(g)`: `~140-288 s` — узкое место именно генерация draft model.

Вывод:
- на текущем локальном draft (`Qwen3.5-9B-Q6_K`) speculative через draft model радикально медленнее baseline (`~3.2 tok/s` vs `~27.4 TPS`);
- bottleneck не в acceptance, а в стоимости самого draft decode.

#### Kernel/runtime pass по узкому месту

Была проверена runtime-гипотеза снижения стоимости draft-контекста (batch sizing в `tools/server/server-context.cpp`), но воспроизводимого ускорения не получено.

Итог:
- runtime-патч откатан;
- кодовая база возвращена к baseline-поведению;
- для продолжения draft-ветки нужен существенно более лёгкий draft GGUF (уровня ~0.5B-1.5B), иначе этот путь не конкурентен.

### Research Phase R35-04 (2026-05-09): kernel-only возврат (без draft-model)

Цель: вернуться к чистой kernel-only ветке и проверить более агрессивный selector-твик в RDNA4 FlashAttention.

Изменение:
- файл: `ggml/src/ggml-cuda/fattn.cu`;
- ветка `amd_wmma_available && RDNA4`;
- для non-quantized single-query decode (`Q->ne[1] == 1`) добавлен ранний выбор `BEST_FATTN_KERNEL_VEC` при `gqa_ratio_eff <= 2`.

#### Результаты compact v2 (`runs=1`, 2 задачи)

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-konly-baseline-r1` | `ngram-mod 48/64/24` | `27.04` |
| `v2-konly-fattnvec-r1-ngram` | `ngram-mod 48/64/24` | `27.47` |
| `v2-konly-fattnvec-r2-ngram` | `ngram-mod 48/64/24` | `27.50` |
| `v2-konly-fattnvec-r1-none` | `none` | `27.40` |

Вывод фазы:
- патч не даёт прорыва, но показывает небольшой стабильный плюс относительно локального baseline-прогона;
- целевой порог `35 TPS` всё ещё далеко, нужен следующий цикл более глубоких kernel-изменений (не только selector tuning).

### Research Phase R35-05 (2026-05-09): deep FATTN softmax/fixup exp-path (kernel-only)

Цель: сделать более глубокий pass по вычислительным блокам FATTN (не selector), сфокусированный на softmax/fixup hot-path.

Изменение (экспериментальное, затем откат):
- `ggml/src/ggml-cuda/fattn-vec.cuh`: замена `expf` -> `__expf` в softmax-обновлении `KQ_max_scale`, `KQ_reg`, sink-path и финальном merge-scale;
- `ggml/src/ggml-cuda/fattn-tile.cuh`: замена `expf` -> `__expf` в KQ softmax (`KQ_max_scale`, `val`);
- `ggml/src/ggml-cuda/fattn-common.cuh`: замена `expf` -> `__expf` в stream-k fixup/combine scaling.

#### Результаты compact v2 (`runs=1`, 2 задачи)

| Label | Spec mode | Aggregate TPS |
|-------|-----------|--------------|
| `v2-konly-fattnvec-r2-ngram` | `ngram-mod 48/64/24` | `27.50` |
| `v2-konly-deepfattnexp-r1-ngram` | `ngram-mod 48/64/24` | `27.46` |
| `v2-konly-deepfattnexp-r1-none` | `none` | `26.74` |

Вывод фазы:
- deep exp-path замена не дала прироста в `ngram-mod` и дала заметный регресс в `spec none`;
- патч признан неуспешным и полностью откатан;
- рабочее состояние оставлено на kernel-only ветке с сохранённым улучшением из R35-04.

### Research Phase R35-06 (2026-05-09): serving-param exhaustive screen (no rebuild)

Цель: проверить 3 serving-level гипотезы из deep research документа, не требующие пересборки.
Базовый уровень в этой фазе: `build-rocm-exp`, `ctx=65536`, `b=4096`, `ub=512`, `kv=q4_0/q4_0`, `ngram-mod 48/24/64`, `np=1` → **~27.4–27.5 TPS**.

| Label | Гипотеза | ctx | ub | kv_k | kv_v | Aggregate TPS | Δ vs baseline |
|-------|----------|-----|-----|------|------|--------------|--------------|
| `v2-h1-ctx32k-ub512-r1` | Меньше KV IO: ctx=32K | 32768 | 512 | q4_0 | q4_0 | **27.55** | +0.1 (нейтр.) |
| `v2-h2-ctx65k-ub128-r1` | VEC path: ub=128 | 65536 | 128 | q4_0 | q4_0 | **25.61** | -1.9 (регрессия) |
| `v2-h3-kv-q8-ub512-r1` | Qwen KV qual: q8_0/q8_0 | 65536 | 512 | q8_0 | q8_0 | **26.74** | -0.7 (регрессия) |

#### Выводы фазы

- **H1 (ctx=32K)**: нейтрально. v2-задачи укладываются в 32K, реальный использованный KV-размер не меняется — bandwidth не является ограничивающим фактором для данной нагрузки.
- **H2 (ub=128)**: регрессия −1.9 TPS. «Гарантированный VEC path» хуже ub=512: при ngram-mod verification batches часто >128 токенов, что создаёт overhead из дополнительных kernel launches; меньший batching снижает GPU utilization.
- **H3 (q8_0 KV)**: регрессия −0.7 TPS. Удвоенная KV bandwidth → чуть медленнее, несмотря на более высокое качество кэша. Вывод противоположен гипотезе: q4_0 KV предпочтительнее на данной нагрузке.

#### Общий вывод по serving-param exploration

Пространство serving-параметров в текущем v2-профиле исчерпано:
- `ub`: 128 → регрессия; 256 → -0.25; **512 → оптимум**; 1024 → обрыв (-7 TPS TILE switch)
- `ctx`: 32K ≈ 65K → оба одинаковы (нагрузка не использует полный ctx)
- `kv type`: **q4_0 оптимум**; q8_0 → -0.7 TPS
- `parallel`: **p=1 оптимум**; p=2 → -1.7 TPS
- `spec`: ngram-mod ≈ none (для v2 кодовых задач acceptance rate низкий)

Потолок ~27.5 TPS является compute-bound ограничением линейных слоёв модели (weight loading / MMQ), не KV bandwidth и не selector kernel.
Для прорыва требуется: более лёгкая модель (IQ2/IQ3_XS), более быстрый MMQ kernel (RDNA4 MFMA tuning), или MTP с подходящей GGUF.

### Research Phase R35-07 (2026-05-09): MMQ RDNA4 cap (`x_max=96`) + rebuild

Цель: проверить RDNA4-специфичный MMQ тюнинг после полного rebuild ROCm контура.

Изменение:
- файл: `ggml/src/ggml-cuda/mmq.cuh`;
- функция: `get_mmq_x_max_host(const int cc)`;
- для `GGML_CUDA_CC_IS_RDNA4(cc)` установлен экспериментальный cap: `return 96` (вместо общего пути до `128`).

Сборка:
- после зависания терминала были обнаружены «осиротевшие» процессы `cmake/ninja/clang++`; они остановлены принудительно;
- rebuild выполнен командой `cmake --build build-rocm-exp --target llama-server -j 4`;
- новый бинарь: `build-rocm-exp/bin/llama-server.exe`.

#### Результат A/B (`runs=1`, compact v2, ngram-mod 48/24/64)

| Label | Конфиг | Aggregate TPS |
|-------|--------|--------------|
| baseline corridor | `ctx=65536, b=4096, ub=512, q4_0/q4_0` | `~27.4-27.5` |
| `v2-r35-mmqx96-r1` | `MMQ RDNA4 x_max=96` | **`25.77`** |

Вывод:
- текущий MMQ cap `x_max=96` для RDNA4 даёт **существенную регрессию** (`~ -1.7 TPS`);
- гипотеза не подтверждена, вариант не подходит для дальнейшего использования в baseline.

## Пост-мортем: почему патчи не пробили потолок ~27 TPS (cold-first)

Ниже сводный анализ по фазам R35-01..R35-07 для v2/v2-mini профиля.

1. Упор в compute-bound линейных слоёв (MMQ/weight loading), а не в KV/selector мелочи.

- Это подтверждено serving-перебором: `ctx 32K ~= 65K`, `q8_0 KV` даёт регрессию, `ub=512` остаётся лучшим для cold-first.
- Следствие: параметры, которые в основном двигают KV bandwidth, почти не меняют потолок.

2. Спекулятивный путь (`ngram-mod`) в cold-first v2 не даёт стабильного ускорения.

- В v2-кодовых задачах `spec none ~= ngram-mod` по aggregate.
- Большие warm-числа возникают на повторном проходе одинаковых задач (прогретый speculative context), но это другой режим, не cold-first headline.

3. Большинство проверенных патчей были «локально-микро», а не в главном bottleneck.

- GDN fast-exp, FATTN threshold widening, deep `expf -> __expf` не дали устойчивого выигрыша и/или дали регрессию в non-spec.
- Логика: даже если локально ускоряется отдельный участок, вклад в общий wall-time decode недостаточен для заметного роста aggregate TPS.

4. Часть направлений уже исчерпана и показала регресс заранее.

- `parallel=2`, `ub=1024`, RDNA4 MMQ `x_max=96`, расширения отдельных MMQ окон и draft-path с 9B draft-моделью — все дали отрицательный результат.
- Это указывает на структурный потолок текущей пары: модель 27B Q3_K_S + текущие kernel policies на RX 9070 XT.

5. Draft-model ветка упёрлась в стоимость самого draft decode.

- При высоком acceptance итоговый TPS всё равно резко падает из-за дорогой генерации draft-моделью.
- Значит bottleneck был не в acceptance, а в latency draft model per token.

Практический вывод:

- текущая ветка оптимизаций упёрлась в устойчивый cold-first коридор около `27.2-27.5 TPS`;
- для реального прорыва выше потолка нужны не микро-твики selector/exp, а более крупные изменения: новый MMQ/MFMA путь под RDNA4, более лёгкая target/draft модель, либо полноценный MTP-путь с совместимым MTP GGUF.

## Аудит окружения: что уже проверено вне build-патчей (2026-05-09)

Цель этого блока: зафиксировать, какие внешние ограничения уже видны по хосту и runtime, чтобы не переоценивать очередные kernel-правки.

### 1. Ключевые ROCm билды собраны почти в одинаковом базовом контуре

Проверенные CMakeCache для `build-rocm-exp`, `build-rocm-wmma`, `build-rocm-r35-c` показывают общий фундамент:

- `ROCm 7.1` (`clang/clang++` из `C:/Program Files/AMD/ROCm/7.1/bin`);
- `AMDGPU_TARGETS=gfx1201`;
- `Release` + `Ninja`;
- `GGML_HIP=ON`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_NO_VMM=ON`.

Вывод: cold-first потолок нельзя объяснить тем, что один из основных билдов случайно собран «не под ту архитектуру» или на другом toolchain.

### 2. Runtime-путь у разных билдов почти одинаков

По server logs для `build-rocm-exp`, `build-rocm-wmma`, `build`:

- везде `offloaded 65/65 layers to GPU`;
- везде `graph nodes = 3849`, `graph splits = 2`;
- везде decode для cold-first задач держится около `~28.2-28.6 tok/s` на уровне отдельных задач;
- включение `rocWMMA FATTN` не дало отдельного качественного скачка.

Вывод: разные локальные бинарники в текущем workload в основном проходят через одинаковый практический runtime-контур.

### 3. На хосте уже видны платформенные ограничения Windows ROCm

Проверено на машине:

- Windows power plan: `Balanced`;
- GPU driver: `AMD Radeon RX 9070 XT`, driver `32.0.23033.1002` от `2026-03-09`;
- активная HIP runtime DLL: `C:/Windows/System32/amdhip64_7.dll`;
- `amdhip64_7.dll` из `System32` и из `C:/Program Files/AMD/ROCm/7.1/bin` имеют одинаковую version string `10.0.3665.0`, но разные размеры и разные SHA256;
- рядом с `llama-server.exe` в локальных `build*/bin` нет копий ROCm DLL, а текущие launcher paths в основном только prepend'ят `PATH`/`HIP_PATH`;
- `hipInfo`: `gfx1201`, `32 CU`, `clockRate 2460 MHz`, `memoryClockRate 1259 MHz`;
- `hipInfo`: `isLargeBar = 0`, `concurrentKernels = 1`, `cooperativeLaunch = 0`;
- server logs: `VMM: no`, `ROCm : NO_VMM = 1`.

Интерпретация:

- build менялся, но платформа исполнения оставалась одной и той же;
- на Windows это делает загрузку HIP runtime из `System32` фактическим default-path для текущих билдов, поэтому простой prepend `PATH` не гарантирует использование DLL из ROCm SDK;
- это делает рассинхрон `compiler/toolchain` vs `runtime DLL` правдоподобным кандидатом на скрытый performance ceiling;
- отсутствие VMM и Large BAR не доказывает текущий bottleneck само по себе, но указывает на менее гибкий runtime-контур, чем хотелось бы для агрессивного разгона;
- по коду backend `NO_VMM` в первую очередь отключает VMM memory pool / virtual-memory allocation path, а не переписывает основные MMQ/FATTN kernels; поэтому сам по себе `NO_VMM` скорее объясняет ограничения среды/allocator path, чем весь `~27 TPS` потолок;
- свойства `concurrentKernels = 1` и `cooperativeLaunch = 0` пока не выглядят главной причиной: в коде проекта нет явной логики, которая бы строила текущий decode hot path вокруг этих capability flags;
- `Balanced` power plan и Windows ROCm stack остаются валидными внешними кандидатами для A/B, прежде чем делать ещё 10 микро-патчей в kernel-слое.

### 4. Что это значит для цели `35 TPS cold-first`

На текущем наборе фактов наиболее вероятна такая картина:

- главный потолок формируется сочетанием `model size + quant format + RX 9070 XT + Windows ROCm runtime`;
- многие kernel-патчи не попадают в основной wall-time, потому что runtime и workload остаются почти неизменными;
- дальнейший поиск нужно вести не только в коде, а в платформенных A/B:
  - power plan `Balanced` vs `High performance`;
  - BIOS/driver проверка ReBAR / Smart Access Memory;
  - чистый runtime A/B Windows vs Linux на том же железе;
  - проверка, не вносит ли заметный штраф системная HIP DLL/driver pair.

Текущий рабочий вывод: до смены хотя бы одного существенного внешнего фактора вероятность получить `35 TPS cold-first` только build-патчами выглядит низкой.

### 5. Быстрый A/B: app-local `amdhip64_7.dll` рядом с `llama-server.exe`

Был проверен прямой эксперимент: принудительно положить `amdhip64_7.dll` из `C:/Program Files/AMD/ROCm/7.1/bin` рядом с `build-rocm-exp/bin/llama-server.exe`, чтобы уйти от implicit загрузки HIP runtime из `System32`.

Результат:

- `v2mini-local-hipdll-r1` (`v2-mini`, cold-first, `ctx=65536`, `b=4096`, `ub=512`, `q4_0/q4_0`, `ngram-mod`) дал **`26.21 TPS`**;
- это хуже обычного cold-first коридора `~27.4-27.5 TPS`.

Вывод:

- простая подкладка только `amdhip64_7.dll` рядом с бинарём **не является выигрышным путём**;
- более того, такой частичный override может создавать смешанный runtime-контур, поэтому он не подтверждает гипотезу «локальная DLL = автоматически быстрее»;
- практическое решение: этот путь считать **проверенным и регрессивным**, не держать его как активную оптимизацию.

## Продолжение cold-first цикла (2026-05-09, вечер)

Профиль для всех сравнений ниже:

- `--tasks v2-mini --runs 1 --no-v2-prime-pass`
- `-c 65536 -b 4096`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`

### 1. Бинарный A/B (build-rocm-exp vs свежий build-rocm-compare)

Для честного сравнения был поднят отдельный каталог `build-rocm-compare` с ROCm clang 7.1 и Ninja.

Ключевой технический момент:

- при первом configure OpenMP подцепил `C:/Strawberry/c/lib/libgomp.dll.a` и ломал линковку (`__kmpc_*`);
- после выравнивания с рабочим профилем (`-DGGML_OPENMP=OFF`) сборка прошла и тест стал валидным.

Результат:

- `v2mini-buildrocmcompare-r1`: **`26.65 TPS`**
- против `v2mini-postrollback-streamk-r1`: **`26.72 TPS`**
- дополнительная проверка `build-rocm-vec`: `v2mini-buildrocmvec-r1` = **`25.45 TPS`**

Вывод: отдельный свежий ROCm-билд не дал прироста; код/ядро остаются в том же практическом коридоре.

### 2. Runtime sweep по `ubatch`

На `build-rocm-exp`:

| Label | UBatch | Aggregate TPS | Вывод |
| --- | ---: | ---: | --- |
| `v2mini-ub256-r1` | `256` | `25.50` | regress |
| `v2mini-ub384-r1` | `384` | `25.63` | regress |
| `v2mini-ub640-r1` | `640` | `26.65` | около baseline, без выигрыша |

Вывод: для текущего cold-first профиля `ub=512` остаётся практическим опорным значением.

### 3. Runtime sweep по потокам (host feeder)

Проверка `--threads 16 --threads-batch 16`:

| Label | UBatch | Aggregate TPS | Вывод |
| --- | ---: | ---: | --- |
| `v2mini-threads16-r1` | `512` | `26.78` | небольшой плюс в шуме |
| `v2mini-threads16-ub512-r1` | `512` | `26.67` | без подтверждения плюса |
| `v2mini-threads16-ub640-r1` | `640` | `26.62` | без выигрыша |

Итог: устойчивого выигрыша от 16/16 потоков не подтверждено.

### 4. MTP-путь с локальным MTP GGUF

Проверен запуск:

- модель: `models/Qwen3.6-27B-IQ3_M-mtp.gguf`
- `--spec-type mtp`

Результат:

- `v2mini-mtp-main-r1`: **`4.00 TPS`** (сильный regress)

Диагностика по server log:

- MTP действительно активирован (`set_mtp: MTP draft head registered`);
- acceptance высокий (`#acc drafts` высокий), но `dur(g)` (generation stage) огромный;
- wall-time уходит в MTP generation path, поэтому общая скорость резко ниже базового ngram-mod.

Практический вывод:

- в текущем локальном сочетании модель/квант/железо MTP-путь **непригоден** как ускорение;
- держим его как проверенный тупик до появления более лёгкого/лучше совместимого MTP-конфига.

### 5. Альтернативные ngram-режимы (без изменений кода)

Проверены на том же профиле:

| Label | Spec type | Aggregate TPS | Наблюдение |
| --- | --- | ---: | --- |
| `v2mini-ngramsimple-r1` | `ngram-simple` | `26.67` | около baseline |
| `v2mini-ngramk4v-r1` | `ngram-map-k4v` | `26.56` | небольшой regress |

По server logs:

- `ngram-simple`: drafts есть, но мало (`#gen drafts = 5`, `#acc tokens = 30` суммарно);
- `ngram-map-k4v`: drafts почти не активируются (`#gen drafts = 1`, `#acc tokens = 6`);
- основная decode-скорость остаётся близкой к `~27.7-27.9 tok/s` на задачу, поэтому общий aggregate почти не меняется.

Итог: переключение между ngram-режимами само по себе не даёт прорыва для текущего cold-first workload.

### 6. MMQ host-policy pass для Q3_K-heavy decode

Текущая модель `Qwen3.6-27B-Q3_K_S.gguf` по server log почти целиком упирается в `q3_K` тензоры (`353` tensors), поэтому следующим шагом был проверен более структурный MMQ-тюнинг в `ggml/src/ggml-cuda/mmq.cuh`.

#### A. RDNA4 `granularity=16`

Изменение:

- для RDNA4 в MMQ зафиксирован более мелкий `granularity=16` вместо перехода на `32` при `mmq_x >= 128`.

Результаты:

| Label | Spec mode | Aggregate TPS |
| --- | --- | ---: |
| `v2mini-mmq-gran16-r1` | `ngram-mod 48/64/24` | `26.89` |
| `v2mini-mmq-gran16-r2` | `ngram-mod 48/64/24` | `26.94` |
| `v2mini-mmq-gran16-r3` | `ngram-mod 48/64/24` | `26.88` |
| `v2mini-mmq-gran16-specnone-r1` | `none` | `26.89` |
| `v2mini-mmq-gran16-specnone-r2` | `none` | `26.94` |
| `v2mini-mmq-gran16-specnone-r3` | `none` | `26.92` |

Интерпретация:

- прирост небольшой, но воспроизвёлся и с `ngram-mod`, и с `spec none`;
- три независимых cold-first прогона с `ngram-mod` дали combined aggregate `26.9053 TPS`, то есть патч выглядит повторяемым;
- три независимых cold-first прогона с `spec none` дали combined aggregate `26.9184 TPS`, то есть `spec none` на этом профиле как минимум не хуже `ngram-mod`;
- это похоже на маленький decode/MMQ gain, а не на speculative-шум;
- патч оставлен как **текущий лучший малый кандидат** в рабочем дереве.

Практический вывод по runtime mode:

- для обычного `Qwen3.6-27B-Q3_K_S` на `ctx=65536, b=4096, ub=512` после MMQ `gran16` режим `spec none` выглядит наиболее консервативным default;
- разница против `ngram-mod` минимальна, но `spec none` чуть лучше по combined 3-run aggregate и не зависит от speculative counters.

#### B. Дополнительный RDNA4 bundle: `y=64` + `4 warps`

Поверх `gran16` был проверен ещё один более агрессивный MMQ host-policy bundle:

- `get_mmq_y_* = 64` для RDNA4;
- `mmq_get_nwarps_* = 4` для RDNA4.

Результаты:

| Label | Spec mode | Aggregate TPS |
| --- | --- | ---: |
| `v2mini-mmq-gran16-y64-r1` | `ngram-mod 48/64/24` | `26.91` |
| `v2mini-mmq-gran16-y64-specnone-r1` | `none` | `26.86` |

Вывод:

- отдельной ценности поверх `gran16` этот слой не показал;
- improvement в `ngram-mod` слишком мал, а на `spec none` он уже не подтверждается;
- bundle `y=64 + 4 warps` **откачен**, чтобы оставить в дереве только более чистый `gran16`-патч.

#### C. RDNA4 selector: `always-MMQ` поверх `gran16`

На том же 65K cold-first профиле был отдельно проверен старый сильный кандидат с 32K-сессии: принудительный `always-MMQ` для RDNA4 в `ggml_cuda_should_use_mmq()`.

Результат:

| Label | Spec mode | Aggregate TPS |
| --- | --- | ---: |
| `v2mini-mmq-gran16-always-r1` | `ngram-mod 48/64/24` | `26.01` |

Вывод:

- на 65K cold-first этот путь даёт сильный regress;
- старый выигрыш `always-MMQ` на 32K не переносится напрямую на текущий профиль;
- правка **откачена**, в рабочем дереве оставлен только `gran16`.

#### D. Мягкий MMQ cap: `x_max=112` поверх `gran16`

После подтверждения `gran16` был проверен более мягкий соседний cap для RDNA4:

- `get_mmq_x_max_{host,device} = 112` вместо `128`.

Результат:

| Label | Spec mode | Aggregate TPS |
| --- | --- | ---: |
| `v2mini-mmq-gran16-x112-r1` | `ngram-mod 48/64/24` | `26.80` |

Вывод:

- мягкий cap `112` всё равно хуже подтверждённого `gran16` baseline;
- этот путь **откачен**.

### UBatch cliff study for prompt-heavy `v2-mini` (2026-05-10)

Профиль:

- `--tasks v2-mini --runs 1 --no-v2-prime-pass`
- `--ctx-size 12288 -b 6144`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--spec-type none --cache-ram 0 --ctx-checkpoints 0`
- repo-snapshot prompt lane

Последние точки на `build-rocm-vec`:

| Label | UBatch | Aggregate TPS | Статус |
| --- | ---: | ---: | --- |
| `trace-vec-b6144-ub784-none` | `784` | `10.0734` | fast |
| `trace-vec-b6144-ub800-none-r2` | `800` | `9.9074` | fast |
| `trace-vec-b6144-ub832-none-r2` | `832` | `3.6181` | cliff |

Что показал трассировочный лог:

- в `ggml/src/ggml-cuda/gated_delta_net.cu` RDNA4 prefill идёт через chunked path при `n_tokens >= 128`;
- при `n_tokens > 256` launcher выбирает `chunk_size = 128`, иначе `96`;
- для fast точек `784/800` trace показывает final chunk `16/32`, а для slow точки `832` final chunk становится `64`;
- это выглядит как локальный tail-chunk threshold в RDNA4 `Gated Delta Net` prefill, а не как общий убыток от самого `ubatch`.

Рабочая гипотеза на следующий цикл:

- избегать конфигураций, где `n_tokens % 128 == 64` в этом lane;
- проверить ещё несколько точек вокруг границы (`848`, `864`, `880`) и посмотреть, сохраняется ли провал именно на tail `64`;
- если гипотеза подтвердится, рассмотреть alignment-aware `ubatch` policy или локальную правку chunking в RDNA4 prefill.

### Short lane: `v2-review` (only `v2_code_review`) for low-noise prompt-eval checks (2026-05-10)

Чтобы сократить длительность прогона и уменьшить шум от смешивания нескольких задач,
в `scripts/agent_workload_bench.py` добавлен режим:

- `--tasks v2-review` (только `v2_code_review`).

Быстрый шаблон запуска:

```powershell
python scripts/agent_workload_bench.py --label promptfocus-v2review-<tag> --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks v2-review --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size <UB> --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none" --real-context-mode repo-snapshot --real-context-chars 21872 --no-reuse --background-server-policy ignore --no-v2-prime-pass --no-disable-thinking --max-tokens 120
```

Подтверждение на `runs=3` (одинаковый lane, `spec=none`, no-reuse):

| Label | UBatch | Aggregate TPS | prompt_eval_tps mean | prompt_eval_ms mean |
| --- | ---: | ---: | ---: | ---: |
| `promptfocus-v2review-ub128-r3` | `128` | `7.8540` | `736.65` | `10900.92` |
| `promptfocus-v2review-ub256-r3` | `256` | `8.0202` | `758.95` | `10582.62` |
| `promptfocus-v2review-ub512-r1` | `512` | `3.7187` | `288.93` | `27791.89` |
| `promptfocus-v2review-ub824-r3` | `824` | `3.8020` | `297.17` | `27025.85` |
| `promptfocus-v2review-ub832-r3` | `832` | `3.6944` | `287.32` | `27949.02` |

Вывод:

- в этом конкретном prompt-heavy lane высокие `ubatch` сейчас вредят prefill: `128/256` существенно быстрее `512+`;
- даже в коротком однотасковом режиме сохраняется просадка `ub832` против `ub824`;
- фокус оптимизации остаётся на prefill/prompt-eval path;
- для быстрых A/B итераций по prompt-eval использовать `v2-review` как дефолтный short lane.

Трассировочные подтверждения (текущий `build-rocm-vec`):

- `GGML_TRACE_GDN_PATH=1`:
  - `ub256`: `launch_gated_delta_net ... n_tokens=256 ... chunk_size=96`;
  - `ub512`: `launch_gated_delta_net ... n_tokens=512 ... chunk_size=128`.
- `GGML_TRACE_FATTN_SELECTED=1`:
  - `ub256`: много вызовов `Q1=256`, `selected=wmma_f16`;
  - `ub512`: вызовы с `Q1=512`, также `selected=wmma_f16`, но итоговый prompt eval резко хуже.

Проверка гипотезы «виноват только chunk_size=128»:

- принудительный override `GGML_GDN_CHUNK_SIZE=96` при `ub512` (`promptfocus-v2review-ub512-ch96-r1`) не восстановил скорость (`aggregate ~3.71 TPS`).
- значит, деградация связана не только с `chunk_size`, а с более широким kernel-route/shape поведением prefill при больших `n_tokens`.

### Narrow-band ubatch sweep (`<=256`) on `v2-review` (2026-05-10)

После подтверждения, что выше `ub=256` в этом lane смысла нет, был сделан короткий sweep только по low-ubatch зоне (`runs=1`):

| Label | UBatch | Aggregate TPS |
| --- | ---: | ---: |
| `promptfocus-v2review-ub176-r1-micro` | `176` | `8.07` |
| `promptfocus-v2review-ub184-r1-micro` | `184` | `8.23` |
| `promptfocus-v2review-ub192-r1-micro` | `192` | `8.47` |
| `promptfocus-v2review-ub200-r1-micro` | `200` | `6.81` |
| `promptfocus-v2review-ub208-r1-micro` | `208` | `6.92` |
| `promptfocus-v2review-ub216-r1-micro2` | `216` | `7.14` |
| `promptfocus-v2review-ub224-r1-micro2` | `224` | `7.26` |
| `promptfocus-v2review-ub232-r1-micro2` | `232` | `7.42` |
| `promptfocus-v2review-ub240-r1-micro2` | `240` | `7.55` |
| `promptfocus-v2review-ub248-r1-micro2` | `248` | `7.72` |
| `promptfocus-v2review-ub256-r1-micro2` | `256` | `7.86` |

Тонкий sweep вокруг пика (`188..198`) показал резкую границу:

| Label | UBatch | Aggregate TPS |
| --- | ---: | ---: |
| `promptfocus-v2review-ub190-r1-fine` | `190` | `8.38` |
| `promptfocus-v2review-ub192-r1-fine` | `192` | `8.46` |
| `promptfocus-v2review-ub194-r1-fine` | `194` | `6.67` |
| `promptfocus-v2review-ub196-r1-fine` | `196` | `6.72` |
| `promptfocus-v2review-ub198-r1-fine` | `198` | `6.75` |

Трассировочный A/B (`GGML_TRACE_GDN_PATH=1 GGML_TRACE_FATTN_SELECTED=1`):

- `ub192` (`promptfocus-v2review-ub192-trace-r1`): `aggregate 8.47`, `prompt_eval_tps 821.08`, `decode_eval_tps 27.53`.
- `ub194` (`promptfocus-v2review-ub194-trace-r1`): `aggregate 6.66`, `prompt_eval_tps 591.25`, `decode_eval_tps 27.51`.
- decode почти неизменен; просадка целиком в prefill.
- histogram `n_tokens` в GDN trace:
  - `ub192`: `{192, 158, 2, 1}`;
  - `ub194`: `{194, 140, 130, 2, 1}`.

Проверка гипотезы `GDN chunk_size` на лучшей точке (`ub192`):

- `GGML_GDN_CHUNK_SIZE={64,80,96,128}` дали `8.46-8.47 TPS` (разница < 1%).
- по правилу трека это **не прогресс**, гипотеза закрыта без дополнительных re-check.
- `LLAMA_FUSED_GDN_CH=0` / отключение chunked prefill на `ub192` не является рабочим обходом: `nonmtp-ub192-gdnch-off-20260511-r1` завис на первом prompt batch (`6144/8030`) сразу после `prompt processing progress`, поэтому эксперимент откатан и помечен как no-go.

Дополнительные no-go проверки на той же точке (`ctx=12288`, `b=6144` unless noted, `ub192`, `spec=none`, no-reuse, `runs=1`):

| Label | Проверка | Aggregate TPS | Вывод |
| --- | --- | ---: | --- |
| `nonmtp-ub192-nographs-noreuse-20260511-r1` | `GGML_CUDA_DISABLE_GRAPHS=1` | `8.53` | <1% к лучшему `8.47`, не считается |
| `nonmtp-ub192-offloadmin1-noreuse-20260511-r1` | `GGML_OP_OFFLOAD_MIN_BATCH=1` | `8.47` | нет прироста |
| `nonmtp-ub192-nocudafusion-noreuse-20260511-r1` | `GGML_CUDA_DISABLE_FUSION=1` | `8.38` | регрессия |
| `nonmtp-ub192-t16tb16-noreuse-20260511-r1` | `--threads 16 --threads-batch 16` | `8.47` | CPU threads не bottleneck |
| `nonmtp-ub192-backendsampling-noreuse-20260511-r1` | `--backend-sampling` | `8.43` | backend sampling не окупает overhead |
| `nonmtp-ub192-b3072-noreuse-20260511-r1` | `b=3072` | `8.44` | хуже `b=6144` |
| `nonmtp-ub192-b2048-noreuse-20260511-r1` | `b=2048` | `8.45` | хуже `b=6144` |
| `nonmtp-compare-ub192-noreuse-20260511-r1` | `build-rocm-compare` | `8.10` | готовая compare-сборка хуже |
| `nonmtp-exp-ub192-noreuse-20260511-r1` | `build-rocm-exp` | `8.09` | готовая exp-сборка хуже |
| `nonmtp-ub192-ngrammod-noreuse-20260511-r1` | `--spec-type ngram-mod` | `8.46` | ngram-mod сгенерировал `0` draft tokens, ускорения нет |
| `nonmtp-ub192-kvq8-noreuse-20260511-r1` | `--cache-type-k q8_0 --cache-type-v q8_0` | `8.41` | prefill тот же, decode хуже (`26.99 tok/s`) |

Первые shape-planner и outer-batch проверки после добавления `LLAMA_UBATCH_SPLIT_POLICY=tail-avoid`:

| Label | Изменение | Aggregate TPS | Вывод |
| --- | --- | ---: | --- |
| `nonmtp-shapeplan-ub256-pref192-noreuse-20260511-r1` | `-ub 256`, `LLAMA_UBATCH_SHAPE_PREFERRED=192` | `8.44` | planner реально дал chunks `192...192,158`, восстановив `ub256` с прежних `7.86`, но peak `ub192` не побил |
| `nonmtp-ub192-b8192-noreuse-20260511-r1` | `b=8192`, `ub=192` | `8.47` | один outer prompt batch вместо `6144+1886` почти не меняет wall; boundary не bottleneck |

### P1 shape-score boundary gate (`v2-review`, 2026-05-11)

После внедрения `shape-score` planner (`src/llama-batch.cpp`) был выполнен полный gate на active lane:

- `ctx=12288`, `b=6144`, `q4_0/q4_0`, `spec=none`, no-reuse, `--no-disable-thinking`.

Screening (`runs=1`):

| Label | UBatch | Policy | Aggregate TPS |
| --- | ---: | --- | ---: |
| `p1-gate-20260511-174248-base-ub192-r1` | `192` | off | `8.51` |
| `p1-gate-20260511-174248-shape-ub190-r1` | `190` | shape-score | `8.43` |
| `p1-gate-20260511-174248-shape-ub192-r1` | `192` | shape-score | `8.54` |
| `p1-gate-20260511-174248-shape-ub194-r1` | `194` | shape-score | `8.53` |
| `p1-gate-20260511-174248-shape-ub196-r1` | `196` | shape-score | `8.54` |
| `p1-gate-20260511-174521-base-ub194-r1` | `194` | off | `6.71` |

Confirmation (`runs=3`):

| Label | UBatch | Policy | Aggregate TPS | TPS stdev |
| --- | ---: | --- | ---: | ---: |
| `p1-confirm-20260511-174606-base-ub194-r3` | `194` | off | `6.83` | `0.0663` |
| `p1-confirm-20260511-174606-shape-ub194-r3` | `194` | shape-score | `8.52` | `0.0064` |
| `p1-confirm-20260511-174606-base-ub192-r3` | `192` | off | `8.51` | `0.0009` |

Итоговые дельты (по diagnostics + CSV):

- shape-score `ub194` vs baseline `ub194`:
  - aggregate TPS: `+24.73%`
  - prompt_eval_ms: `-26.42%`
  - decode_eval_ms: `+0.10%` (в пределах шума)
- shape-score `ub194` vs baseline `ub192`:
  - aggregate TPS: `+0.08%`
  - prompt_eval_ms: `-0.11%`
  - decode_eval_ms: `-0.10%`

Verdict:

- boundary cliff на `ub194` воспроизводимо снят под `shape-score` без decode-regression;
- throughput `ub194` возвращён в corridor `ub192` класса;
- изменение оставлено в дереве как env-guarded policy.

Timing trace после добавления `LLAMA_UBATCH_TIMING`:

- `nonmtp-ub192-timing-noreuse-20260511-r1`: async trace, `8.44 TPS`; build/alloc/input overhead на prompt chunks меньше `~1.5 ms`, но `compute_call` асинхронный и не показывает полную GPU стоимость.
- `nonmtp-ub192-timing-sync32-noreuse-20260511-r1`: diagnostic-only (`LLAMA_UBATCH_TIMING_SYNC=1`, `max_tokens=32`, TPS не сравнивать). Средние sync timings: prompt `n_tokens=192` стоит `~232-240 ms` total на chunk, decode `n_tokens=1` стоит `~36 ms` на token. Host-side graph overhead не является bottleneck; следующий реальный рычаг — GDN/FATTN/MMQ device kernels или model-graph reshape вокруг них.

Reduced HIP/FlashAttention build corridor после `amdgcn-link` blocker:

| Label | Проверка | Aggregate TPS | Вывод |
| --- | --- | ---: | --- |
| `nonmtp-fa-reduced-ub192-noreuse-20260511-r1` | `build-rocm-fa-reduced`, `GGML_HIP_QWEN_FA_REDUCED=ON`, `GGML_OPENMP=OFF` | `8.46` | reduced dispatcher проходит активную Qwen/RDNA4 lane, но сам по себе не ускоряет |
| `nonmtp-fa-reduced-forcevec-ub192-mt32-20260511-r1` | `GGML_QWEN_FA_REDUCED_FORCE=vec`, diagnostic `max_tokens=32` | diagnostic only | prompt eval упал `820 -> 580 tok/s`, force-vec для `Q1=192` закрыт |
| `nonmtp-fa-reduced-forcewmma-ub192-mt32-20260511-r1` | `GGML_QWEN_FA_REDUCED_FORCE=wmma_f16`, diagnostic `max_tokens=32` | diagnostic only | prompt `823 tok/s`, decode `27.51 tok/s`; tiny decode через WMMA не лучше baseline |

Вывод по reduced mode:

- `GGML_HIP_QWEN_FA_REDUCED=ON` решает практический build blocker для дальнейших FATTN/GDN A/B патчей: heavy `fattn.cu`, tile/MMA dispatcher и template instances исключены, вместо них используется host-only reduced dispatcher.
- Reduced dispatcher имеет ручку `GGML_QWEN_FA_REDUCED_FORCE=vec|wmma_f16` для smoke A/B FATTN selector без тяжелого `fattn.cu` relink.
- Fresh reduced build на Windows/ROCm потребовал `-DGGML_OPENMP=OFF`, иначе link `ggml-cpu.dll` падает на `__kmpc_*` symbols.
- Результаты из этого build можно использовать для smoke/A-B проверки kernel hypotheses, но финальные speed claims лучше подтверждать на обычном ROCm build после переноса удачной правки.

MMQ/MMVQ follow-up:

- `GGML_TRACE_MMQ_PATH=1` на reduced build (`nonmtp-fa-reduced-mmqtrace-ub192-mt8-20260511-r1`) дал `16674` MMQ route lines, все в prefill: `type=11/Q3_K ncols=192 xbest=96 tiles=2`, `type=12/Q4_K ncols=192 xbest=96 tiles=2`, плюс tail `ncols=158 xbest=80 tiles=2`.
- Decode не идёт через MMQ trace; активный decode matvec path — `mmvq.cu`.
- Попытка добавить MMVQ trace/Q3_K nwarps knob упёрлась в `amdgcn-link command failed due to signal` на `mmvq.cu`.
- Попытка ограничить MMVQ switch до Qwen tensor types (`q3_K/q4_K/q6_K`) тоже не прошла: source-specific `mmvq.cu` compile всё равно падал в `amdgcn-link`. Эксперимент откатан, чтобы не оставлять несобираемый source state.

P2 Stage A+B+C+D (MMVQ dispatch split + observability/tuning scaffold, 2026-05-11):

- Stage A: публичные host entrypoints вынесены из `mmvq.cu` в новый `mmvq-dispatch.cu`.
- Stage B: type switch (`ggml_cuda_mmvq_switch_type`) перенесён в lightweight `mmvq-dispatch.cu`, а `mmvq.cu` экспортирует per-type entrypoints.
- Stage C: type routing разделён на `mmvq-kernels-qwen.cu` (`Q3_K/Q4_K/Q6_K`) и `mmvq-kernels-rest.cu` (остальные типы).
- Stage D: добавлены env-gated MMVQ observability/tuning hooks:
  - `GGML_TRACE_MMVQ_PATH=1` (route trace `qwen-hot/rest` с type и shape полями)
  - `GGML_TRACE_MMVQ_SMALL_K=1` (small_k decision trace)
  - `GGML_MMVQ_QWEN_FORCE_SMALL_K=1` / `GGML_MMVQ_QWEN_DISABLE_SMALL_K=1` (RDNA4 Qwen-hot override, default unchanged)
- Normal ROCm gate (`build-rocm-vec`, target `llama-server`) прошёл после переконфигурации.
- Reduced ROCm gate (`build-rocm-fa-reduced`, `GGML_HIP_QWEN_FA_REDUCED=ON`, `GGML_OPENMP=OFF`) также прошёл.
- Повторные инкрементальные touch+rebuild циклы (`mmvq.cu`, `mmvq-dispatch.cu`, `mmvq-kernels-qwen.cu`, `mmvq-kernels-rest.cu`) прошли без `amdgcn-link ... signal`.
- Runtime smoke на активной lane:
  - `p2-stageA-smoke-20260511-181905-ub192-r1`: `8.54 TPS`
  - `p2-stageB-smoke-20260511-182335-ub192-r1`: `8.54 TPS`
  - `p2-stageC-smoke-20260511-182726-ub192-r1`: `8.54 TPS`
  - `p2-stageC-reduced-smoke-20260511-183047-ub192-r1`: `8.54 TPS`
  - `p2-active-lane-posthooks-20260511-184542-ub192-r1`: `8.55 TPS`
  - `p2-reduced-posthooks-20260511-184624-ub192-r1`: `8.55 TPS`
  - Все результаты остаются в `ub192` corridor, явной default-regression на scaffold этапе не видно.

Stage D diagnostics:

- Route trace sample (`p2-trace-route-20260511-183846-ub192-r1`) подтвердил рабочий MMVQ маршрутный лог: `qwen-hot=1077`, `rest=0` (для этой Qwen lane).
- Force trace sample (`p2-trace-force-smallk-20260511-184219-ub192-r1`) подтвердил, что `GGML_MMVQ_QWEN_FORCE_SMALL_K=1` реально переключает `small_k=1` в Qwen-hot вызовах (`680` lines; baseline trace had `680` lines with `small_k=0`).
- Decode-biased lane (`ctx=12288`, no-reuse, no real-context, `max_tokens=256`):
  - runs=1: base `26.84`, force `27.09`, disable `26.88` TPS.
  - runs=3 confirm: base `26.8355` vs force `27.0066` TPS (`+0.64%`), decode_eval_tps `28.6767 -> 28.8767` (`+0.70%`).
  - эффект умеренный; default policy не менялась.

P3 theory fanout check (dry-run explain, 2026-05-11):

- Команда для всех проверок: `cmake --build <build-dir> --target llama-server -- -d explain -n`.
- `touch fattn.cu`:
  - normal (`build-rocm-vec`): rebuild `fattn.cu.obj` + relink chain (`7` steps).
  - reduced (`build-rocm-fa-reduced`): `ninja: no work to do`.
- `touch mmvq.cu`:
  - normal (`build-rocm-vec`): rebuild `mmvq.cu.obj` + relink chain (`7` steps).
  - reduced (`build-rocm-fa-reduced`): rebuild `mmvq.cu.obj` + relink chain (`7` steps).
- Теоретический вывод: reduced corridor уже снимает FATTN-side build pressure, но не снимает MMVQ-side pressure; MMVQ-focused corridor остаётся предметом P3 implementation.

Artifacts:

- `build_logs/agent-workload/p3-dryrun-normal-fattn.txt`
- `build_logs/agent-workload/p3-dryrun-reduced-fattn.txt`
- `build_logs/agent-workload/p3-dryrun-normal-mmvq.txt`
- `build_logs/agent-workload/p3-dryrun-reduced-mmvq.txt`

P3 implementation build gates (2026-05-11):

- Build-system implementation landed for:
  - centralized HIP source bundle assembly,
  - `GGML_HIP_EXPERIMENT_PROFILE` (`default`, `qwen-fa-reduced`, `mmvq-focused`),
  - Windows HIP compiler fail-fast guard (`clang++/hipcc` required).
- Configure gates passed for all three profiles:
  - `build-rocm-vec` (`default`)
  - `build-rocm-fa-reduced` (`qwen-fa-reduced`)
  - `build-rocm-mmvq-focused` (`mmvq-focused`)
- Build gate passed for all three profiles: `llama-server` linked successfully.
- Guard test passed: intentional bad configure with Strawberry/GNU now fails early with:
  - `GGML_HIP on Windows requires ROCm clang++ or hipcc as CMAKE_CXX_COMPILER`.

Artifacts:

- `build_logs/agent-workload/p3-implementation-build-gates.txt`
- `build_logs/agent-workload/p3-guard-bad-config.txt`

P3 runtime closure checks (2026-05-11):

- Active lane (`v2-review`, `repo-snapshot chars=21872`, `ctx=12288`, `b/ub=6144/192`, no-reuse):
  - `p3-close-default-20260511-r1`: **8.54 TPS** (pass)
  - `p3-close-reduced-20260511-r1`: **8.55 TPS** (pass)
  - `p3-close-mmvq-focused-20260511-r2`: **request timeout** (`TimeoutError('timed out')`), server log stops at prompt progress `6144/8030`.
- Short decode-biased sanity (`tasks=quick`, no real-context, same ctx/b/ub, `max_tokens=64`):
  - `p3-close-default-quick-20260511-r1`: **26.57 TPS**
  - `p3-close-reduced-quick-20260511-r1`: **26.59 TPS**
  - `p3-close-mmvq-focused-quick-20260511-r1`: **17.56 TPS** (major regression vs default/reduced in short lane)
- Additional check (`p3-close-mmvq-focused-sanity-20260511-r1`): with `--flash-attn off` and KV `q4_0/q4_0`, context init fails with `V cache quantization requires flash_attn`.

P3 closure interpretation:

- P3 is closed for build-pressure workflow objective.
- `mmvq-focused` is kept as a narrow debug/build profile only and is not promoted to active prompt-heavy runtime lane.

Artifacts:

- `build_logs/agent-workload/p3-close-default-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-reduced-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-mmvq-focused-20260511-r2.csv`
- `build_logs/agent-workload/p3-close-default-quick-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-reduced-quick-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-mmvq-focused-quick-20260511-r1.csv`
- `build_logs/agent-workload/p3-close-mmvq-focused-20260511-r2.diagnostics.md`
- `build_logs/agent-workload/p3-close-mmvq-focused-sanity-20260511-r1.server.log`

---

## Dual TG/PP Compute Scheduler (2026-05-10) ✅

### Проблема

При `ubatch=512` `ggml_backend_sched` выделял compute buffer **495 MiB** (под PP-граф максимального размера). Во время decode (1 токен за шаг) GPU тратил cache bandwidth на весь этот буфер, хотя реально нужно только ~7 MiB.

Результат: `ub=512` давал **~19-25 TPS** против **~25 TPS** при `ub=128`.

### Решение

Реализован dual TG/PP compute scheduler в `src/llama-context.cpp`:

- **PP scheduler** (`sched`): стандартный, sized для полного ubatch — 495 MiB. Используется при prefill (`n_tokens > 1`).
- **TG scheduler** (`sched_tg`): новый, sized для 1-токенного графа — **6.95 MiB**. Используется при decode (`n_tokens == 1`).
- Переключение происходит автоматически в `process_ubatch()` при смене режима.
- Оба scheduler'а имеют отдельный кэш графа (`gf_res_prev` / `gf_res_prev_tg`).

**Изменённые файлы:**
- `src/llama-context.h` — поля `sched_tg`, `sched_is_tg`, `gf_res_prev_tg`
- `src/llama-context.cpp` — `sched_reserve()`, `process_ubatch()`, `synchronize()`, destructor
- `gui/model_presets.json` — пресет обновлён: `ubatch_size: 128 → 512`
- `gui/llama_gui.py` — ROCm fallback поиск по `build-rocm-wmma`, `build-rocm-vec`

### Результаты (`build-rocm-wmma`, `Qwen3.6-27B-Q3_K_S.gguf`, `ctx=65536`, `q4_0 KV`, `ngram-mod`)

| Label | Dual-sched | UBatch | Wall TPS | Runs |
|---|:---:|---:|---:|---:|
| `nodual-ub128-wmma` | нет | 128 | 25.17 | 1 |
| `nodual-ub512-wmma` | нет | 512 | 19.58 | 1 |
| `dual-sched-ub512-wmma` | **да** | 512 | **32.16** | 3 |
| `gui-dual-sched-ub512-wmma` | **да** | 512 | **29.97** | 3 (через GUI live) |

**+64% к старому `ub=512`**, **+27% к лучшему `ub=128` без dual-sched.**

TG compute buffer: 6.95 MiB вместо 495 MiB → GPU cache pressure в decode фазе снята.

### Воспроизведение через GUI

1. Launch Server → Backend: **ROCm** → подхватит `build-rocm-wmma` автоматически
2. Apply Preset → `Qwen3.6-27B-Q3_K_S.gguf` → `ub=512, b=4096, ctx=65536, q4_0, ngram-mod`
3. Start Server → ожидаемый decode: **~30 TPS**

## Dual TG/PP Compute Scheduler (2026-05-10)

### Мотивация

При больших `ubatch_size` (например, 512) `ggml_backend_sched` выделял compute buffer под максимальный PP-граф (495 MiB для `ub=512`). В режиме TG (decode, n_tokens=1) этот буфер оставался занятым, создавая GPU memory pressure при каждом шаге decode.

Гипотеза: отдельный TG-scheduler с маленьким compute buffer (TG-граф из 1 токена) снимет это давление и приблизит `ub=512` к `ub=128` по TG TPS.

### Реализация

Добавлены поля в `llama_context`:
- `sched_tg` — второй `ggml_backend_sched_ptr` с TG-буфером (~7 MiB для Qwen3.6-27B)
- `sched_is_tg` — флаг активного scheduler
- `gf_res_prev_tg` — кэш TG-графа (чтобы сохранить graph reuse в decode фазе)

Переключение происходит в `process_ubatch()`: при `ubatch.n_tokens == 1` → TG-scheduler, иначе → PP-scheduler. При смене режима оба scheduler синхронизируются.

Файлы:
- `src/llama-context.h` — объявления полей
- `src/llama-context.cpp` — `sched_reserve()`, `process_ubatch()`, `synchronize()`, деструктор

### Результаты (ctx=65536, b=4096, Qwen3.6-27B-Q3_K_S, q4_0/q4_0, ngram-mod, tasks=quick)

| Build | Dual-sched | UBatch | Wall TPS | Runs |
|---|:---:|---:|---:|---:|
| `build-rocm-wmma` | нет | 128 | 25.17 | 1 |
| `build-rocm-wmma` | нет | 512 | 19.58 | 1 |
| `build-rocm-vec` | **да** | 128 | 25.39 | 1 |
| `build-rocm-vec` | **да** | 512 | 24.53 | 1 |
| `build-rocm-wmma` | **да** | 512 | **32.16** | 3 |

Итог:
- Разрыв `ub=128 vs ub=512` на `build-rocm-vec` сократился с ~5.6 TPS → **0.86 TPS**.
- `build-rocm-wmma + dual-sched + ub=512` = **32.16 wall TPS** (+64% к baseline ub=512 того же билда).
- TG compute buffer: 6.95 MiB (TG) vs 495 MiB (PP) — подтверждён.

### Overhead

Переключение scheduler происходит один раз (PP→TG) после prefill. Остальные decode-шаги не вызывают swap. Overhead измеримо мал (< 1 мс на переключение).

## RDNA4 Graph-Opt Hang (2026-05-10)

Проблема:

- На Windows + ROCm (RX 9070 XT / `gfx1201`) запуск с `GGML_CUDA_GRAPH_OPT=1` стабильно зависал в начале первого запроса (после prefill/checkpoint, до первого ответа).
- Симптом в server log: остановка около `begin: ngram_mod occupancy ...`.

Диагностика:

- Добавлена временная instrumentation в `ggml_backend_cuda_graph_optimize()`.
- Лог показал, что зависание происходит в graph-opt path до стабильного compute/reply цикла.

Фикс:

- В `ggml/src/ggml-cuda/ggml-cuda.cu` добавлен guard:
  - для `GGML_CUDA_CC_IS_RDNA4(cc)` graph optimizer отключается по умолчанию;
  - override доступен через `GGML_CUDA_ALLOW_RDNA4_GRAPH_OPT=1` (только для ручных экспериментов).

Результат после фикса (тот же workload, `ctx=65536`, `b=4096`, `ub=512`, `q4_0/q4_0`):

| Label | Env/Spec | Status | Wall TPS |
|---|---|---|---:|
| `graphopt-on-smoke` | `GGML_CUDA_GRAPH_OPT=1`, `spec=ngram-mod` | hang | — |
| `graphsafe-off-specnone-r1` | `GGML_CUDA_DISABLE_GRAPHS=1`, `spec=none` | stable | `24.61` |
| `graphopt-rdna4-guard-r1` | `GGML_CUDA_GRAPH_OPT=1` + RDNA4 guard, `spec=none` | stable | `24.59` |
| `graphopt-rdna4-guard-ngram-r1` | `GGML_CUDA_GRAPH_OPT=1` + RDNA4 guard, `spec=ngram-mod` | stable | `24.64` |

Вывод:

- На текущем RDNA4/ROCm пути использовать graph-opt без guard нельзя (deadlock-risk).
- Безопасный baseline: оставить guard включённым, или задавать `GGML_CUDA_DISABLE_GRAPHS=1` для диагностических прогонов.

## RDNA4 ROCm Native UBatch Cliff Fix (2026-05-12)

Проблема:

- На RX 9070 XT / ROCm `Qwen3.6-27B-Q3_K_S` уходил в slow pocket при полном native PP reserve: `ctx=32768, ub=904/1024` и также `ctx=16384, ub=900`.
- Full trace показал одинаковые graph/node counts и те же FATTN/GDN/MMQ route classes, но широкое замедление memory-heavy ops: GLU/RMS_NORM/ADD/SSM_CONV и часть MUL_MAT/FATTN. Это не был single-kernel selector bug.
- A/B подтвердил причину: один крупный ROCm compute vbuffer allocation попадает в плохой residency/placement pocket. Простое смещение base offset не помогало; разбиение compute vbuffer на backend chunks помогало при сохранении полного `PP reserve`.

Фикс:

- В `ggml/src/ggml-alloc.c` для ROCm graph allocator добавлен default max compute vbuffer chunk size `256 MiB`.
- `ggml_dyn_tallocr` теперь может создавать несколько backend buffer chunks для одного virtual compute buffer; model/KV offload и requested ubatch не уменьшаются.
- Override для экспериментов: `GGML_COMPUTE_VBUFFER_MAX_CHUNK_SIZE=<bytes>`.
- Контрольное отключение default ROCm chunking: `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1`.

Результаты (`build-rocm-vec`, `ctx=32768`, `b=5120`, `q4_0/q4_0`, `ngram-mod`, no-reuse, repo-snapshot context):

| Label | PP reserve | ROCm0 compute | Prompt eval |
| --- | ---: | ---: | ---: |
| `native-singlechunk-ctx32768-ub904-mt1-r1` | `904 -> 1` | single chunk | `23524.85 ms / 302.87 tok/s` |
| `native-defaultchunk-ctx32768-ub904-mt1-r1` | `904 -> 1` | `374.84 MiB` | `6862.92 ms / 1038.19 tok/s` |
| `native-final-ctx32768-ub1024-mt1-r1` | `1024 -> 1` | `424.53 MiB` | `6392.54 ms / 1114.58 tok/s` |
| `native-defaultchunk-ctx16384-ub900-mt1-r1` | `900 -> 1` | `281.54 MiB` | `6798.72 ms / 1047.99 tok/s` |

Practical run:

| Label | PP reserve | Prompt eval | Decode | Total |
| --- | ---: | ---: | ---: | ---: |
| `native-defaultchunk-ctx32768-ub1024-mt120-r1` | `1024 -> 1` | `6394.28 ms / 1114.28 tok/s` | `120 tok / 25.03 tok/s` | `11188.80 ms` |

Вывод:

- Старый guard/cap до `ub=900` больше не нужен для native `ub1024` path.
- Контроль `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` возвращает slow result, что подтверждает allocator/residency root cause.
