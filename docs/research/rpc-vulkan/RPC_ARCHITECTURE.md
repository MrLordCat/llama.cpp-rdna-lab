# RPC (ggml-rpc) — как это работает

Дата: 2026-08-26. Ветка: `rpc-vulkan`. Модель-референс: Qwen3.8-27B Q4_K_M,
dual-GPU Vulkan (VK1+VK0) + RTX 3080 через `--rpc 192.168.1.60:50052`.

Цель документа — дать читаемое описание архитектуры RPC-бэкенда и явно
перечислить, чем RPC-путь отличается от локальной работы и где именно
находятся измеренные узкие места. Для «как читать код»: карта файлов,
слои, таблицы диагностических переменных и доказанные цифры.

## 1. Файловая структура (после рефакторинга 2026-08-26)

```
include/ggml-rpc.h            — публичный API (C): init, is_rpc, buffer_type,
                                get_device_memory, start_server, reg, add_server.
ggml/src/ggml-rpc/
  rpc_types.h                 — протокол: rpc_tensor, rpc_msg_*, enum rpc_cmd,
                                флаги TENSOR_FLAG_*, HASH_THRESHOLD, rpc_cmd_name.
  rpc_internal.h              — общие включает, env-гейты (RPC_DEBUG/RPC_TIMELINE),
                                LOG_DBG/RPC_STATUS_ASSERT, декларации common-слоя,
                                события/буферные контексты, буфер-тип контекст.
  rpc_common.cpp              — НЕ-static общие функции: rpc_wall_ms, fnv_hash,
                                graph_structure_hash, send/recv_msg,
                                parse_endpoint, get_socket, negotiate_hello,
                                упорядоченная очередь отправки (rpc_send_submit),
                                send_rpc_cmd* (direct/async), is_causal_mask_name,
                                buffer_type_name, wait_endpoint.
  rpc_client.cpp              — клиентский ggml_backend: буферы, set/get/cpy,
                                get_alloc_size, сериализация (serialize_tensor,
                                serialize_graph), graph_compute + async,
                                get_async, событийные API, rpc_init/is_rpc.
  rpc_server.cpp              — сервер: class rpc_server (+worker_loop,
                                graph_compute_async/wait), rpc_serve_client,
                                ggml_backend_rpc_start_server, device registry,
                                ggml_backend_rpc_reg/add_server.
  transport.cpp / transport.h — сокетный слой (TCP/RDMA-абстракция, socket_t,
                                caps RPC_CONN_CAPS_SIZE).
  CMakeLists.txt              — единая библиотека ggml-rpc из 4 TU.
scripts/research/rpc_split.py, rpc_split_post.py — генераторы разбивки.
```

Старый монолит `ggml-rpc.cpp` (~3,3 тыс. строк) удалён; вся семантика и
комментарии перенесены без изменений (split только по границам функций,
сборка + два бенч-контроля подтвердили эквивалентность).

## 2. Слои (снизу вверх)

1. **Транспорт** (`transport.cpp`): TCP-сокет большими блоками, сетевые
   фреймы, `RPC_CONN_CAPS_SIZE` (ёмкости стороны: caps/протокол-upgrade).
   На Linux есть опциональный RDMA (GGML_RPC_RDMA); на Windows — ws2_32.
2. **Протокол** (`rpc_types.h`): все сообщения — упакованные структуры
   (packed, передаются как есть). `rpc_tensor` — dоменная копия ggml_tensor
   без указателей (id-переотображение). Команды: HELLO, DEVICE_COUNT,
   ALLOC_BUFFER, GET_ALLOC_SIZE, GET_ALIGNMENT/MAX_SIZE, BUFFER_GET_BASE,
   FREE_BUFFER, BUFFER_CLEAR, SET_TENSOR, SET_TENSOR_HASH, SET_TENSOR_MASK*,
   GET_TENSOR, COPY_TENSOR, INIT_TENSOR, GRAPH_COMPUTE, GRAPH_COMPUTE_ASYNC,
   GRAPH_WAIT, GRAPH_RECOMPUTE, GET_DEVICE_MEMORY.
3. **Cmd-слой** (`rpc_common.cpp`): упорядоченная per-socket очередь
   `rpc_send_submit` + `send_rpc_cmd*`. Приоритет: порядок команд
   сохраняется (важно: данные, присланные позже, не могут «обогнать» граф).
   Async-вариант (`GGML_RPC_ASYNC_GRAPH=1`, `GRAPH_COMPUTE_ASYNC`) уходит в
   worker; `*_ASYNC` и snapshot-копии не блокируют планировщик.
4. **Клиент** (`rpc_client.cpp`): реализует `ggml_backend`/`ggml_backend_buffer`
   интерфейсы. Буфер = `remote_ptr` у сервера; пользовательские тензоры
   «живут» только на сервере, клиентский `data` заведомо не указывает на
   реальные байты (все чтения — через GET_TENSOR).
5. **Сервер** (`rpc_server.cpp`): принимает команды одного сокета
   последовательно в `rpc_serve_client`; графы реконструирует через
   `deserialize_tensor`/`create_node` в собственном ggml-контексте и считает
   на локальных GPU (`devices[]`, переданных в start_server).

## 3. Жизненный цикл одного запроса (prefill)

1. `HELLO` + caps-обмен (один раз на сокет), `DEVICE_COUNT`.
2. Инициализация модели: для каждого веса `ALLOC_BUFFER` → `remote_ptr`;
   для quantized с ne[0]%512!=0 — `INIT_TENSOR`; сами веса — `SET_TENSOR_HASH`
   (хэш + кэш на диске сервера; если файл совпадает — передача скипается).
3. Убатч prefill (каждый = `ubatch 1024`):
   - планировщик ggml-backend-sched строит splits по `-sm layer`;
   - **локальные** splits (VK1, VK0) считаются как обычно на месте;
   - **RPC-граница**: выход предыдущего split (`l_out-<il>`, 10–20 МБ) —
     `SET_TENSOR` (или snapshot-копия через per-socket worker при
     `GGML_RPC_ASYNC_GRAPH=1`), каузальная маска — `SET_TENSOR_MASK_NPAST`
     (только метаданные; сервер сам её строит), KV — обычный `SET_TENSOR`;
   - `GRAPH_COMPUTE` (или `GRAPH_COMPUTE_ASYNC`) — граф-сериализация
     (имена/формы/op) → сервер восстанавливает граф и считает;
   - `GET_TENSOR` для `result_output` (логиты, мало) или `l_out` (через
     `ggml_backend_rpc_get_async` — воркер копирует байты в host-буфер
     без блокировки вызывающего до `ggml_backend_rpc_synchronize`).
4. После prefill декод: те же команды на 1 строке (плюс МТП-цикл → verify-/
   draft-графы, `GRAPH_RECOMPUTE` с структурным хэшем — кэш на сервере,
   пересылка графа повторно не нужна).

## 4. Чем RPC-путь отличается от локального (и что это стоит)

| Аспект | Локальный (VK1+VK0) | RPC-бэкенд (3080) |
|---|---|---|
| Где данные | в VRAM клиента | в VRAM сервера; клиент видит `remote_ptr` |
| Запуск графа | submit GPU-очереди | сериализация графа + сеть + серверный submit |
| Порядок/синхр. | GPU events/fences | логический порядок команд на сокете (всегда сериально per-socket) |
| Доступ к KV | прямая запись/чтение | KV на сервере → каждый убатч: payload через сеть |
| Стоимость «накладных» | ~0 | сетевые копии (SET+GET), сериализация ~0.2–1% |
| Диагностика | VK_PERF-флеймы | GGML_RPC_TIMELINE / RPC_DEBUG / GGML_SCHED_SPLIT_TIMING |

Измеренные накладные (94K, убатч 1024, ts 1,1,1.1):
- копия 20 МБ между VK1→VK0 (локально): 3.6 мс — **не** проблема;
- SET 10–20 МБ на RPC-сервер: ~160 мс при 10МБ/с->100-150 МБ/с (1GbE);
- серверный GRAPH_COMPUTE на 94K: 576 мс/уб (на 33K: MM ~382 мс конст);
- клиентская блокировка на logits: не лимитер (async-get = паритет 94K).

Принципиальное отличие: **локальные бэкенды параллелят сами себя**, RPC —
один поток сервера и один поток сокета, поэтому wall = сумма локальных
графов + серверная очередь, без перекрытия между ubatch (см. §6).

## 5. Ключевые механизмы, уже в коде

- `GGML_RPC_TENSOR_FLAG_ACT_F16` — F32 активации через RPC идут как F16
  (hash-кэш активаций скипается). Опционально `GGML_RPC_ACT_F8=1` для
  промежуточных `l_out-*` (измерено хуже: −2%/−16% decode).
- `GGML_RPC_TENSOR_FLAG_CAUSAL_MASK` + `SET_TENSOR_MASK_NPAST` — маска
  генерируется на сервере (экономия n_kv×n_tokens×2 байт на убатч).
- `SET_TENSOR_HASH` — веса передаются один раз, дальше серверный disk-кэш.
- `GRAPH_RECOMPUTE` + `graph_structure_hash` (структурный FNV) — повторная
  пересылка верifi-графов не нужна (экономия на МТП).
- `GRAPH_COMPUTE_ASYNC` + worker-поток сервера + `GRAPH_WAIT`.
- `rpc_async_copy_submit` — snapshot данных в потоке планировщика, worker
  шлёт payload (фикс дедлока rf5: `ggml_vk_synchronize` не потокобезопасна).
- `rpc_wait_pending_copies` — барьер перед пересозданием буферов
  (GGML_ASSERT «tensor buffer not set»).
- `ggml_backend_rpc_get_async` — упорядоченный воркер для `GET_TENSOR`
  (логиты); клиент не блокируется до `llama_synchronize`.

## 6. Измеренные узкие места (по убыванию вклада)

1. **Серверная очередь (главный лимитер поздних убатчей 94K)**.
   `rpc_serve_client` обрабатывает команды одного сокета последовательно;
   `GET_TENSOR`/`SET_TENSOR`/`GRAPH_COMPUTE` вызывают
   `server.graph_compute_wait()` — т.е. наклад на 576–900 мс/уб при том,
   что сам серверный граф — 576 мс/уб (на 33K+). Рычаг: раздельный поток
   вычислений/очередей или двухсокетный конвейер.
2. **Последовательные GPU-графы клиента**: VK1 (256 мс) → VK0 (276 мс) на
   94K. Cross-device копия `cpy_tensor_async` откатывается на sync.
   Рычаг: run-ahead VK1(N+1)∥VK0(N), требует cross-device event-конвейер.
3. **Сеть 1GbE**: 20 МБ l_out ≈ 160–175 мс на убатч; пиковый wire ~403 МБ
   на 94K (14K-лейн ~86 МБ) — с учётом async-копий частично скрыт.
4. **Граф-сериализация** (клиент `serialize_graph`, сервер `create_node`):
   ~1–3 мс/убатч, незначительно, но в 94K в сумме 100–200 мс.

## 7. Диагностика (и что важно НЕ включать)

| Переменная | Что даёт | Меры предосторожности |
|---|---|---|
| `GGML_RPC_TIMELINE=1` | построчная таймлайна клиент/сервер; средство | печатает каждый убатч — только на короткие прогоны |
| `GGML_RPC_DEBUG=1` | подробные сообщения копий/cовпадений хэшей | сильно шумный |
| `GGML_SCHED_SPLIT_TIMING=1` | split-копии и сигнатуры (когда/что) | без VK_PERF |
| `LLAMA_UBATCH_TIMING=1` | build/alloc/inputs/compute_call по убатчу | — |
| `GGML_RPC_ASYNC_GRAPH=1` | async путь на сервере | работает с snapshot-копиями; без него — последовательный |
| `GGML_RPC_ACT_F8=1` | F8 активации | **не** использовать качество-критично (измерено хуже) |
| `GGML_RPC_BARRIER_DISABLE=1` | снять SYNС-барьер | **опасно**: падение «tensor buffer not set» |

Анализатор таймлайны: `scripts/research/rpc_timeline_analyze.py`.

## 8. Как читать/изменять код

- Протокол меняется только вместе с клиентом и сервером (одинаковая сборка),
  поле `RPC_PROTO_*_VERSION` в `include/ggml-rpc.h` для проверки при HELLO.
- Общие функции — `rpc_common.cpp` + `rpc_internal.h`; сервер — `rpc_server.cpp`;
  клиент — `rpc_client.cpp`; wire-форматы — `rpc_types.h`.
- Общий планировщик `ggml-backend.cpp:1705-1745` — интерфейсный код
  (async-копии cpy_tensor_async), не RPC-специфичный: ВНУТРЕННИЕ проверки
  делаются по признаку «есть ли у бэкенда cpy_tensor_async», а не по типу
  бэкенда. Любые изменения там влияют на локальный Vulkan-путь -> при
  правках обязательно прогонять локальный 94K контроль (см. rf22/rf25).

## 9. Актуальные цифры (2026-08-26, канон)

| Лейн | Конфиг | prompt t/s | decode t/s | Примечание |
|---|---|---|---|---|
| 94K local | spec=none | 1355.34, r25 1351.18 | 24.55/25.05 | канон README.md:162; rf25 — после рефакторинга |
| 94K RPC | MTP n4, ts 1,1,1.1 | 1104–1117 | 19.5 | rf9/rf16; стена: серверная очередь |
| 49K RPC | MTP n4 | 1201–1265 | 23–25 | rf12/rf19 |
| 14K RPC | MTP n4 | 1387.77 (рекорд) | 28.03 | rf15 (async-get) |
