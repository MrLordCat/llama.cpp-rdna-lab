# bench2 — универсальный бенч-инструмент (v2, 2026-08-28)

Один CLI-инструмент для любых бэкендов (ROCm / Vulkan / CPU / RPC), любых
окружений и целей. Заменяет **v1** (`scripts/agent_workload_bench.py`, архивирован
в `build_logs/archive/agent-workload-legacy-2026-08/`).

## Быстрый старт

```bash
# Три команды покрывают 95% случаев:
python scripts/bench2.py run --level 1              # L1 (16K), автобинарник/модель/имя, live-лог
python scripts/bench2.py run --level 2 --runs 3     # L2 (49K) три раза, автопоиск по имени
python scripts/bench2.py run --session-level 2      # агентная сессия SL2 (98K, 10 ходов)

# Поиск/список:
python scripts/bench2.py find --name recheck --type single
python scripts/bench2.py list --recent 10
```

Уровни можно задавать списком и диапазоном: `--level 0,2`, `--level 1-3`.
Имя прогона генерируется автоматически, если не указано `--run-name`,
например `rocm-l2-20260828-0934`.

---

## 1. Сценарии

### Одиночные уровни (`--level N`): один большой промпт + декод

| Уровень | ctx | Промпт (токенов) | Декод | Назначение |
|---|---|---|---|---|
| L0 smoke | 8K | ~4K | 64 | мгновенная проверка запуска |
| L1 test | 16K | ~8K | 128 | короткие тестовые прогоны |
| L2 | 49K | ~31.7K | 256 | стандартный рабочий |
| L3 | 98K | ~66K | 256 | большой |
| L4 | 131K | ~97K | 256 | максимум для 2×16GB |
| L5 | 200K | ~194K | 256 | редкий, может не влезть — предварительная проверка | 

### Агентные сессии (`--session-level N`): 10 ходов, KV растёт

| Сессия | ctx | Вход/ход | Декод/ход | Назначение |
|---|---|---|---|---|
| SL1 light | 32K | ~1K | 128 | деанон/тест |
| SL2 medium | 98K | ~2K | 256 | стандартная |
| SL3 heavy | 131K | ~4K | 512 | тяжёлая + MTP |

Каждый ход — новый запрос агента; контекст сохраняется (`cache_prompt`), KV
переиспользуется; метрики по-ходовые + наклон деградации decode (tок/s на ход).

## 2. Что автоматически

- Определение сервера: `--server-bin` или **авто-поиск** по бэкенду
  (`build-rocm|build-vulkan|build-cpu/bin/llama-server.exe`);
  runtime-PATH дополняется автоматически (ROCm 7.1 bin / Strawberry MinGW).
- Определение модели: `--model` или **авто** (падает на `models/Qwen3.8-27B-Q4_K_M.gguf`,
  иначе первый `models/*.gguf`).
- Имя прогона (если не задано), свободный порт, старт/graceful-останов сервера,
  preflight: запрет параллельного запуска при живом `llama-server.exe`.
- **Live-лог по умолчанию**: прогресс prefill (из `server.log`) и результаты
  каждого уровня/хода печатаются в консоль.

## 3. Данные

Каталог прогона `build_logs/bench/<RUN_NAME>/`:

| Файл | Содержимое |
|---|---|
| `run.json` | эффективный конфиг (backend, commit, серверные флаги, seed, env-редact) |
| `<RUN_NAME>.jsonl` | построчные события (server, level/turn start/done, summary) |
| `metrics.csv` | одна строка на измерение (single level или сессия) |
| `summary.md` | человекочитаемая таблица |
| `session_turns.csv` | только сессии: строка на ход (turn, ctx, prompt, cache_n, decode, tps, wall) |
| `server.log` | сырой серверный лог |

Глобальные индексы: `build_logs/bench/index.csv` (одна строка на прогон) и
`index.md`.

### Ключевые метрики

- `prefill_tps` — **новых** токенов prefill в сек (для сессий с KV-reuse
  вычитается `cache_n`);
- `decode_tps`, `ttft_ms` (обе префилльные задержки), `total_ms`, `aggregate_tps`;
- сессии: `decode_slope` (деградация decode по 10 ходам), `session_turns`.

## 4. Опции CLI (run)

```text
--run-name NAME        имя (авто если пусто)
--level 0|1|2|3|4|5    одиночные уровни (список/диапазон), default 1
--session-level 1|2|3  агентные сессии
--runs N               повторов каждого сценария
--server-bin PATH  |  --attach http://host:port
--backend auto|rocm|vk|cpu   (auto по имени бинарника)
--model PATH, --profile NAME
--context-source synthetic|repo-snapshot|file  --context-file PATH
--batch-size 8192 --ubatch-size 1024   (дефолт профиля rdna-lab)
--kv-k q8_0 --kv-v q8_0 --spec none|mtp --spec-n 2
--flash-attn/--no-flash-attn, --gpu-layers, --parallel
--dev, --sm, --ts, --fit, --seed, --temperature, --top-p
--server-extra "raw args", --results-dir, --health-timeout, --fail-fast
```

## 5. Конфиги (`configs/bench/`)

| Файл | Содержимое |
|---|---|
| `hardware.profiles.json` | профили железа: `default_batch 8192`, `default_ubatch 1024`, `dev/sm/ts` для `rocm/vk/cpu` |
| `levels.json` | таблица уровней + источники контекста |
| `sessions.json` | пресеты агентных сессий |
| `server.defaults.json` | дефолты llama-server (KV q8_0, FA on, `fit off`, `cache_ram 0`, `ctx_checkpoints 0`, seed 42) |

CLI-флаги всегда важнее конфигов; `--profile generic` отключает железо-специфику.

## 6. Важные правила/ограничения

- Используется **`/v1/chat/completions`**, а не raw `/completion`: Qwen-модели
  (thinking) без chat-тегов генерируют `|im_end|>` первым токеном и сервер
  останавливается (диагностировано 2026-08-28).
- `--cache-ram 0 --ctx-checkpoints 0` включены по умолчанию для холодных замеров.
- L2/L3+ используют синтетический контекст; `repo-snapshot` ограничен ~52K
  токенов и пригоден только до L2.
- Перед запуском — запрет активного `llama-server.exe` (preflight).
- Драйверные правила (AGENTS.md): `-fit off`, без `hipMemGetInfo`, graceful stop.

## 7. Статус (2026-08-28)

- [x] каркас + конфиги + CLI + live-лог;
- [x] single L0/L1 протестированы (ROCm): L0 3995 tok → 64 decode (23.9 tps, prefill 1387), L1 7901 → 128 (23.9 tps, prefill 1418);
- [x] session SL1 (10 ходов, KV-reuse): decode 24.1 tps средний, slope −0.097;
- [x] архив v1 (14 ГБ) в `build_logs/archive/agent-workload-legacy-2026-08/`;
- [x] v1-обёртка `scripts/agent_workload_bench.py` → legacy, тесты 6/6;
- [ ] Vulkan-смоук L0 (не выполнен в первом ревью-проходе);
- [ ] GUI-переключение на bench2 (per user: «позже»);
- [ ] полный прогон L2/L3+ и сравнение с каноном после согласования.
