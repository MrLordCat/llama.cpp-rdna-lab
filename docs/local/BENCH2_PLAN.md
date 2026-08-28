# BENCH 2.0 — План (согласован 2026-08-28)

## 1. Цель

Универсальный бенч-инструмент, независимый от бэкенда (ROCm / Vulkan / CPU / RPC),
окружения и цели. Два типа сценариев:
1. **Одиночный уровень** — один большой промпт + декод (tokens).
2. **Агентная сессия** — 10 ходов в одном растущем контексте (KV накапливается),
   видно деградацию decode с ростом контекста.

Уровень прогона выбирается как параметр (`--level 0..5` / `--session-level 1..3`),
без отдельных `suite/single` режимов: можно 1 уровень, список уровней или диапазон.

## 2. Одиночные уровни (single)

| Уровень | Контекст | Промпт (токенов) | Декод (токенов) | Оценка времени |
|---|---|---|---|---|
| L0 smoke | 8K    | ~4K  | 64  | ~15 с  |
| L1 test  | 16K   | ~8K  | 128 | ~40 с  |
| L2       | 49K   | ~31K | 256 | ~3 мин |
| L3       | 98K   | ~65K | 256 | ~7 мин |
| L4       | 131K  | ~95K | 256 | ~10–15 мин |
| L5       | 200K  | ~190K| 256 | 20+ мин (редкий, может не влезть в 2×16GB — предпроверка) |

Настройки по умолчанию (профиль железа, переопределяемые):
- batch **8192** / ubatch **1024** (лучшая конфигурация на этом железе);
- `-dev ROCm1,ROCm0 -sm layer -ts 1,1` — ROCm; `-dev Vulkan1,Vulkan0 -sm layer -ts 1,1` — Vulkan;
- KV q8_0/q8_0, flash-attn, `-fit off`, no-warmup, `-c 0`, seed фиксированный.

## 3. Агентные сессии (10 ходов)

| Сессия | Контекст | Вход на ход | Декод на ход | Нагрузка |
|---|---|---|---|---|
| SL1 лёгкая   | 32K  | ~1K ток | 128 | деанон/тест |
| SL2 средняя  | 98K  | ~2K ток | 256 | стандартная |
| SL3 тяжёлая  | 131K (опц. 200K) | ~4K ток | 512 | максимум + MTP |

Каждый ход: короткий "агентный" запрос, KV растёт; контекст между ходами
сохраняется (одна сессия = ответы подряд на одном контексте).

## 4. Метрики

Одиночный уровень:
- prefill TPS, decode TPS, TTFT (ms), prefill_ms, total_ms, aggregate TPS;
- для MTP — acceptance_ratio, draft_tps, effective_decode_tps.

Сессия:
- per-turn: TTFT, decode TPS, длина контекста, input tokens, decode tokens, wall ms;
- средние/min/max по ходам, наклон деградации decode TPS (turn 1 → 10),
  session aggregate TPS.

## 5. Форматы данных (ОТВЕТ НА ВОПРОС)

**Каталог прогона** — `build_logs/bench/<RUN_NAME>/`:

| Файл | Формат | Что содержит |
|---|---|---|
| `run.json` | JSON | полный эффективный конфиг: run_name, тип, уровень, timestamp, бэкенд, бинарь/commit, модель, ctx/batch/ubatch, KV, spec, env (redacted), флаги сервера, seed |
| `<RUN_NAME>.jsonl` | JSONL | построчные события: server start/ready, prefill (start/end, токены, tps), decode (tps, время), итоговая summary-запись; для сессии — по одному событию на ход |
| `metrics.csv` | CSV | одна строка = один сценарий (level или session): run_name, type, level, backend, model, ctx, prompt_tokens, decoded_tokens, prefill_tps, decode_tps, ttft_ms, total_ms, aggregate_tps, mtp_acc, eff_decode_tps, session_turns, path |
| `summary.md` | Markdown | человекочитаемый отчёт: таблица метрик + конфиг + выводы |
| `server.log` | text | лог сервера (сырой, для диагностики) |
| `artifacts/` | опц. | `responses.jsonl`, `timing*.jsonl` |

**Сессия** — тот же каталог, плюс:
- `session_turns.csv` — строка на ход: turn, ctx_len, input_tokens, decode_tokens, ttft_ms, decode_tps, wall_ms;
- в `metrics.csv` — агрегаты сессии (aggregate, средний decode tps, наклон),
  а в `summary.md` — таблица ходов.

**Глобальные индексы** (для поиска по имени прогона):
- `build_logs/bench/index.csv` — одна строка на прогон: run_name, type, level, timestamp,
  backend, model, ключевые метрики, путь;
- `build_logs/bench/index.md` — та же сводка для чтения.

Поиск: `bench2 find --name <паттерн> --type session --level 3 --backend rocm`.

Архив старых бенчей сохраняется В ИСХОДНОМ ВИДЕ (BENCH_HISTORY.*, BENCH_RUNS.*,
BENCH_*.md, per-run jsonl/csv) — конвертации нет, чтобы ничего не потерять.

## 6. CLI (эскиз)

```text
python scripts/bench2.py run --run-name q38-rocm-l2-a --level 2
python scripts/bench2.py run --run-name d094-vk-session-sl2 --session-level 2
python scripts/bench2.py run --run-name recheck-r3 --level 0,2 --runs 3
python scripts/bench2.py run --run-name smoke --level 0-1 --backend vk
python scripts/bench2.py find --name "l2" --type single --backend rocm
python scripts/bench2.py list --recent 20
```

Общие опции:
- `--server-bin <path>` (старт + graceful teardown) или `--attach <url>`;
- `--model <path>`, `--backend auto|rocm|vk|cpu`, `--profile <имя-железа>`;
- `--level 0..5` / `--session-level 1..3`, `--runs N`;
- `--context-source synthetic|repo-snapshot|file:<path>`;
- `--batch-size/--ubatch-size/--kv/--spec/--flash-attn/--gpu-layers/--parallel/--dev ...` — override;
- `--results-dir` (default `build_logs/bench`), `--run-name` (обязателен для run);
- GPU-free precheck, TERM clean, контроль драйверных правил (`-fit off`, no `hipMemGetInfo`).

## 7. Конфиги

JSON в `configs/bench/`:
- `hardware.profiles.json` — пресеты железа (дефолты batch/ubatch/dev/sm/ts/ctx-лимиты);
- `levels.json` — таблица уровней (ctx, prompt_tokens, decode_tokens);
- `sessions.json` — пресеты сессий (turns, input/decode, ctx);
- `server.defaults.json` — общие defaults (KV, flash-attn, fit off, seed, no-warmup).

CLI-переопределения приоритетнее конфигов.

## 8. Архив старых бенчей

- Переместить `build_logs/agent-workload/*` → `build_logs/archive/agent-workload-legacy-2026-08/`
  (+ README архива с датой и назначением).
- Старый `scripts/agent_workload_bench.py` → `scripts/legacy/`, на старом месте — тонкая
  обёртка-совместимость (GUI/`tool_call_workload_bench.py`/`large_context_*` не ломаем
  в этом шаге; переключение GUI — позже, отдельным решением).
- Зависимые скрипты: поправить импорты; формат старых JSONL/CSV не менять.

## 9. Этапы реализации

1. [x] Конфиги `configs/bench/*.json` + каркас `scripts/bench2.py` (CLI, run-name, indexes).
2. [x] Синтетический контекст-генератор (детерминированный seed) для L0–L5 + `repo-snapshot`/`file`.
3. [x] Single-сценарий: старт/attach сервера, HTTPS/live-лог, метрики, запись run.json/jsonl/csv/md.
4. [x] Session-сценарий: N ходов, растущий KV, per-turn метрики, наклон деградации.
5. [x] `find/list` по index.csv + человекочитаемые summary.
6. [x] Архив старых данных + legacy-обёртка + фикс импортов.
7. [~] Смоук-валидация: L0 (ROCm) ✅, L0 (Vulkan) — осталось, SL1 ✅; согласованные полные прогоны — позже.
8. [x] Документация: `docs/local/BENCH2.md` (+ обновить память).

## 10. Открытые вопросы (решить до/во время реализации)

- [ ] L5 (200K ctx + ~190K токенов промпта) влезает ли на 2×16GB: предварительная
      проверка VRAM/загрузки, иначе L5 = "экспериментальный, с предупреждением".
- [ ] MTP: метрики acceptance считать из `draft_n/accepted` — единый формат для ROCm/VK.
- [x] Синтетический промпт: точный состав (иначе L2/L3 не сопоставимы со старыми
      результатами repo-snapshot) — решение: L2 по умолчанию `repo-snapshot`, L3-L5 `synthetic`
      (обновлено: по умолчанию synthetic для всех; L2 вариант repo-snapshot через `--context-source`).
- [ ] Нужен ли "холодный/соседний" контроль: A/B-бенчи всегда перемежать (правило
      теплового дрейфа), фиксировать в summary "после N прогонов сессии".
- [x] GPU-free precheck (tasklist + Get-Counter) и отказ при активном сервере — реализовано (tasklist + preflight).
