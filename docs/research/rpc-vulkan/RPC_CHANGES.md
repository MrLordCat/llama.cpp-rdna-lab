# RPC Changes Log (rpc-vulkan)

Хронология изменений RPC-контура ветки `rpc-vulkan` (клиент AMD RX 9070 XT ×2
+ сервер RTX 3080 по LAN, 1 GbE). Детальные протоколы/эксперименты — в
`RPC_PREFILL_RESUME_PLAYBOOK.md`, итоговые числа — в
`build_logs/agent-workload/BENCH_RECENT.md`.

Формат: дата — что изменилось — результат (на 2026-08-26).

## 2026-08-22 — восстановление RPC backend для Vulkan offload

- Ветка `rpc-vulkan` создана от `master` (в master RPC удалён).
- Фикс краша сервера `SIGSEGV 139` при загрузке модели: буферную эвристику
  `ggml_backend_rpc_buffer_type_get_alloc_size` заменено на запрос к серверу
  для квантованных тензоров (`ggml-rpc.cpp` ~стр. 610): Vulkan-сервер пишет
  q3_K/q6_K в device-раскладке (+2 байта/блок), host-размер выходил за
  пределы буфера на ~9.4 МБ. Коммит `eb0d3c5ec`.
- Третий рычаг: сдвиг слоёв с дисплейной VK0 на сервер + `SO_SNDBUF 16MB`
  на клиентском сокете (снял блок 88 мс/убатч на дренаже l_out-43).

## 2026-08-23 — 12K цель достигнута

- `alloc_size cache` (fnv по endpoint/type/ne/nb/op/params): alloc 537 → 1.3 мс/убатч
  (srcs в ключ нельзя — у FA-нод kv-вьюхи растут с n_past).
- F16-стык активаций (`GGML_RPC_TENSOR_FLAG_ACT_F16`, `l_out-*`/`result_output`):
  +20% prefill (1969 → 2362 ptps на Qwen3.5-9B 16K), decode +8.5%.
- Маска-оптимизация отключена по умолчанию (`GGML_RPC_ENABLE_MASK_NULL=1`
  включает): NULL-подстановка давала ppl 2.09 — модель «видит будущее»
  (Vulkan-FA с MASK_ENABLE=false = полное внимание). Цена: +27% времени
  PPL-прохода.
- MTP-RPC починен: `RPC_CMD_GRAPH_RECOMPUTE` + структурный hash графа,
  сервер хранит `stored_graphs` по hash (лимит 16); acceptance 1.46% → 59.7%.
- Результат 27B 12K (коммит `62da5f1ab`): 839 → **1314 ptps / 23.6 t/s**
  (цель ≥1277 = 80% соло; PPL 4.0148 = эталон). Конфиг: `-ts 0.9,0.6,1.5`.

## 2026-08-25 — клиент — критический путь; лучи диагноза

- `GGML_RPC_ASYNC_GRAPH=1`: серверный worker + async `GRAPH_COMPUTE_ASYNC`
  (клиент fire-and-forget); +8.3% на 14K при отказе от блокирующего
  RECOMPUTE (1384.98 vs 1278.55 ctl, тот же `-ts 1,0.8,1.2`).
- Без RECOMPUTE (полный async) — A/B: 0.2% (шум), т.к. клиент — критический
  путь (735 мс/уб; сервер 576 мс/уб уже перекрыт).
- Split-timing: узкое место prefill — копии `l_out` (83% стены:
  VK1→VK0 7600 мс + VK0→RPC0 6924 мс из 17.5 с @14K/21 убатч).
- 94K `-ts 1,1,1.1`: 1066.81 t/s (плато, -ts 1.2 = alloc fail на 3080),
  local 98K = 1355.34 → gap −21%.

## 2026-08-26 — рефактор: async-копии; фикс дедлока; 94K +4.3%

- Рефактор-проба «полного async» (снятие входных барьеров + worker-копии
  GPU→RPC в per-socket очереди) → **детерминированный дедлок** на 3-4-м
  декодном убатче. Причина: `ggml_vk_synchronize`
  (`ggml-vulkan/runtime/vk_backend_execution.inc:2`) не потокобезопасна —
  общий `compute_ctx`/`fence`/`submit_pending`; RPC-worker вызывал её
  параллельно с `graph_compute` того же VK-бэкенда.
- ФИКС (код в коммите): **снэпшот в главном потоке планировщика**
  (`ggml_backend_synchronize` + `ggml_backend_tensor_get` в main), per-socket
  worker только шлёт захваченный payload; в работу добавлен `rpc_send_tensor_data`
  (сериализованные metadata захватываются на submit, worker не трогает живые
  ggml-объекты).
- Сняты входные барьеры RPC-сплита: `split4 copy 288 → 0.16 мс`;
  INPUT-копии (leaf_56 и др.) идут async с host-снэпшотом.
- Результаты: 14K 1363.10/27.68 (паритет, шум ±4%), 94K **1116.78**/19.52
  = +4.3% vs 1070.72 (промпт идентичен 58 185 ток).
- Диагностика rf6: `l_out-39` 20 МБ = sync VK0 241-266 мс + get 4.9-6.7 мс —
  остаток = ожидание самого VK0-графа; RPC-путь больше не лимит на 14K.

## Следующая стена (работа в процессе)

- Локальная копия VK1→VK0 (`l_out-21` 20 МБ, 268 мс): путь через host
  staging (`ggml_vk_buffer_copy` MULTI_DEVICE, `vk_transfer.inc:1033`) —
  GPU→host `waitForFences` + host→GPU ≈ 74 МБ/с вместо ожидаемых сотен МБ/с.
- Кандидаты: асинхронный transfer с double-buffer, P2P (external memory,
  крупно), либо перепланировка границы слоёв, чтобы не переносить 20 МБ.
