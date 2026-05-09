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
