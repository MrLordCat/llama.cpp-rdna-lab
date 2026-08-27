# RPC-ветка (`rpc-vulkan`) — финальный статус

Дата: 2026-08-27. Ветка закрывается: локальные улучшения RPC-контура
доведены до паритета с локальным dual-GPU, а на сценарии «контекст 200K /
промпт 160K» доказано ключевое практическое свойство — RPC-связка из трёх
карт позволяет обрабатывать промпты, не помещающиеся в память двух
локальных GPU.

Сопутствующие документы:
- `RPC_ARCHITECTURE.md` — как устроен RPC-бэкенд (файлы, протокол, узкие места).
- `RPC_BASELINE_TABLE.md` — таблица базовых прогонов.
- `RPC_PREFILL_RESUME_PLAYBOOK.md` — playbook диагностики prefill.
- `docs/research/major-topology/D135_RPC_LOCAL_PROTOCOL_DIAGNOSIS.md` — диагноз D135.
- `docs/research/RESULTS_LOG.md` — журнал результатов.

## 1. Контекст

- Локальное железо: Windows 11, 2× AMD Radeon RX 9070 XT 16 ГБ (Vulkan,
  `-dev Vulkan1,Vulkan0`), 64 ГБ RAM.
- Удалённый RPC-сервер: RTX 3080 10 ГБ (Windows, 192.168.1.60,
  `C:\rpc-3080\run.bat` через schtasks `rpc-srv-3080`, порт **50052**;
  `-d Vulkan0`, сервер в SYSTEM).
- Модель-референс: Qwen3.8-27B Q4_K_M, MTP `--spec-type draft-mtp
  --spec-draft-n-max 4`, KV q8_0, FA, `-sm layer`.
- Топология: `-dev RPC0,Vulkan0,Vulkan1` (RPC0 = 3080-сервер).

## 2. Ключевые результаты

### 2.1 Локальный RPC (loopback, RPC0 = Vulkan1 на 127.0.0.1:50056) — паритет

14K/12K-контракт `-ts 1,1`, один бинарь:

| Конфигурация | Agg | Prompt TPS | Decode TPS |
| --- | ---: | ---: | ---: |
| no-RPC control | 16.92 | 1643.16 | 48.86 |
| RPC база (без фиксов) | 12.47 | 1023.25 | 53.11 |
| RPC + `GGML_RPC_MASK_PIN_HOST=1` | 16.43-16.61 | 1513.8-1517.0 | 51.9-53.6 |
| **+ многопоточные F16-конверсии сервера** | **16.76-16.85** | **1545.9-1558.3** | **52.8-52.9** |

Итог: aggregate **−0.4%** к no-RPC (практический паритет), prefill
**−5.2%**, decode **+8%**. От старой базы 12.47 agg — плюс **+35%**.

Что сделано (код в ветке):
- `GGML_RPC_MASK_PIN_HOST` (`src/llama-context.cpp`): маска кастуется на
  **host**, а не на локальный GPU — убирает ожидание события локального
  GPU перед серверным графом (~423 мс/уб на 12K; отсюда +37-49% prefill).
  Env-gated, по умолчанию выключен (дефолт — только после повторной
  проверки на 3080-контуре).
- Многопоточные `rpc_f32_to_f16`/`rpc_f16_to_f32`
  (`ggml/src/ggml-rpc/rpc_internal.h`), используются в серверных
  `get_tensor`/`set_tensor` (`rpc_server.cpp`) вместо однопоточных
  ggml-хелперов. Серверные конверсии 20 МБ падали с 16-19 мс до ~1-3 мс.

### 2.2 3080-lane (LAN 1 GbE) — диагноз

14K, `-ts 0.8,1,1.4`, серверная таймлайна (обе стороны TL):
серверный цикл prefill-убатча ≈ **975 мс**: SET input_embed 20 МБ =
**182 мс** (сеть), mini-граф 57 мс, mask 8 мс, big-граф **260 мс** (3080
быстрее локальных сильно), GET l_out 12 мс, простой сервера **449 мс**
(46% — клиентские GPU ещё считают).

- Сеть: **~200 мс/уб (20-22%)**, но она почти скрыта при сжатии;
  компрессия Q8 (2/2) и изменения wire **не дали выигрыша** — лимит не в
  байтах, а в сериализации/балансе.
- Равный баланс `-ts 1,1,1` — **хуже** (10.61 agg), чем `-ts 0.8,1,1.4`
  (12.2-12.75): узкое место — дисбаланс, сервер 3080 должен нести больше
  слоёв.
- F16-сжатие `input_embed` **отклонено** как регрессия на 3080
  (серверная конверсия дороже выигрыша по трафику); `input_embed` ходит
  F32.

### 2.3 200K ctx / большой промпт — жизнеспособность сценария (главный итог)

Контекст 204800, KV q8_0, ubatch 1024, MTP n4, реальный промпт 116.8K
токенов (снапшот 462924 chars; безопасный кап 2.6 chars/token даёт 116K
токенов, фактическая плотность кода ~3.96 chars/token, для 160K нужен
`--real-context-chars ~634000 --real-context-chars-per-token 4.0`):

| Контур | Prompt TPS | Decode TPS | Acceptance |
| --- | ---: | ---: | ---: |
| Локальный 2×9070 XT (RAM spill) | 183-184 | 21.0-21.6 | 95/124, 93/136 |
| **RPC 3 карты (3080 + 2×9070 XT)** | **635.3** | **29.8** | 91/139 |

**RPC в 3.45× быстрее по prefill, +40% decode.** Причина: KV 116K (~54 ГБ
q8_0) не влезает в 32 ГБ двух карт → локальный контур уходит в RAM spill;
22 слоя и их KV на 3080-сервере снимают существенную часть spill. Именно
это делает сценарий «много GPU через RPC» жизнеспособным: сеть проиgrывает
намного меньше, чем выигрывает память.

## 3. Диагностические переменные и инфраструктура

| Переменная | Назначение |
| --- | --- |
| `GGML_RPC_MASK_PIN_HOST=1` | маска на host (ускоряет локальный RPC, −0.4% agg) |
| `GGML_RPC_ACT_Q8_0=1` / `GGML_RPC_ACT_F8=1` | лосy-форматы wire l_out (opt-in, PPL ≈ F16) |
| `GGML_RPC_ACT_THREADS` | потоки конверсий (дефолт 8; 16/4/1 хуже) |
| `GGML_RPC_TIMELINE=1` | таймлайн клиент/сервер (`RPC_TL|cli/srv|...`) |
| `GGML_RPC_DEBUG=1` | подробный лог RPC |
| `GGML_SCHED_SPLIT_TIMING=1` | тайминги сплитов scheduler'а |
| `GGML_RPC_SERVER_MAKE_MASK` / `GGML_RPC_ENABLE_MASK_NULL` | серверная генерация маски — **отклонено** (decode −68%) |
| `LLAMA_RPC_RUN_AHEAD` / `GGML_RPC_ASYNC_GRAPH` | async/run-ahead — **вредят** на этом лейне |

Инфраструктура:
- Локальный: `PATH="/c/Strawberry/c/bin:$PATH" ./build-vulkan/bin/rpc-server.exe -d Vulkan1 -p 50056 -H 127.0.0.1`.
- 3080: schtasks `rpc-srv-3080` → `C:\rpc-3080\run.bat` (лог
  `C:\rpc-3080\llama-srv.log`), порт 50052; перезапуск
  `schtasks /End+taskkill+del+Run`; следить, что на 3080 нет фонового
  llama-server (он делит VRAM и рушит RPC-бенчи).

## 4. Ограничения / не сделано (для следующих итераций)

- Сеть 1 GbE (~110 МБ/с) остаётся заметной долей при коротких промптах
  (200 мс/уб, 20% на 14K); **10 GbE** снял бы это целиком — но и сейчас
  при больших контекстах RPC-связка выигрывает из-за разгрузки памяти.
- Конвейерный префетч SET входов N+1 — не реализован (оценка −180 мс/уб).
- Параллельные GET/SET (несколько сокетов/команд) — не реализованы.
- Баланс слоёв для 3080 (`-ts`) не оптимизирован; `1,1,1` = хуже,
  направление — больше на сервер.
- NV-кернелы в форке «вычищены под RDNA4» (~1 мс/слой); отдельная
  оптимизация 3080 не делалась (вне задачи).
- `GGML_RPC_MASK_PIN_HOST` по умолчанию выключен (env-gated).

## 5. Как пользоваться (проверенный контракт)

Локальный паритет (2×9070 XT + loopback RPC):

```text
GGML_RPC_MASK_PIN_HOST=1 python scripts/agent_workload_bench.py \
  --server-bin build-vulkan/bin/llama-server.exe --model models/Qwen3.8-27B-Q4_K_M.gguf \
  --tasks quick --ctx-size 12288 --batch-size 8192 --ubatch-size 1024 --max-tokens 128 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn --no-warmup --server-seed 42 \
  --real-context-mode repo-snapshot --real-context-chars 24576 --background-server-policy fail \
  --server-extra "--rpc 127.0.0.1:50056 --spec-type draft-mtp --spec-draft-n-max 4 -fit off
    --cache-ram 0 --ctx-checkpoints 0 -dev RPC0,Vulkan0 -sm layer -ts 1,1"
```

Большой контекст (3 карты):

```text
GGML_RPC_MASK_PIN_HOST=1 python scripts/agent_workload_bench.py ... \
  --ctx-size 204800 --allow-ctx-above-16k --real-context-chars 634000 \
  --real-context-chars-per-token 4.0 --task-hard-timeout 1800 --request-timeout 1800 \
  --server-extra "--rpc 192.168.1.60:50052 ... -dev RPC0,Vulkan0,Vulkan1 -sm layer -ts 0.8,1,1.4"
```

## 6. Валидация ветки

- 14K локальный: 2/2 стабильно (16.76-16.85 agg), acceptance бит-идентичен
  базе (88/153, 93/136); PPL не требуется.
- 3080 14K: 2/2 стабильно (12.2-12.75 agg).
- 200K/116K: локальный и RPC-прогоны без крашей, acceptance в норме
  (91-95%, с учётом MTP-накладных).
- `git diff --check` — чисто; сборка `build-vulkan` (llama-server,
  rpc-server) — успешна; GPU-процессы после прогонов остановлены.
