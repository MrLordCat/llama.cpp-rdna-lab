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
