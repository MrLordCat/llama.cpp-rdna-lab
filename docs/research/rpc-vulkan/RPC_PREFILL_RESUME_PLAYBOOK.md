# RPC / Nvidia-prefill: RESUME PLAYBOOK (handoff)

Дата: 2026-08-23. Ветка: `rpc-vulkan`. Задача: ускорить prompt eval в RPC-схеме
(Qwen3.8-27B Q4_K_M, 12k, клиент = 2×RX 9070 XT Vulkan, сервер = RTX 3080 по LAN).

## Статус одним абзацем (2026-08-23, ЦЕЛЬ ДОСТИГНУТА)

RPC 3-GPU 12K (Vulkan1,Vulkan0,RPC0→3080): **1314 ptps / 23.6 t/s** (r14-final,
чистый прогон) при цели ≥1277 ptps / ≥21.7 t/s (80% соло 1596.8/27.1). PPL
4.0148 = эталон (4.017-4.021). Три рычага дали 839 → 1314 ptps (+57%):

1. **Кэш alloc_size на RPC-клиенте** (ggml-rpc.cpp `g_rpc_alloc_size_cache`,
   ключ = endpoint + type/ne/nb/op/op_params/view_src, srcs НЕ включать — иначе
   FA kv-view растёт с n_past и кэш не попадает): alloc 537→1.3 мс/убатч
   (было: ~1200 round-trip GET_ALLOC_SIZE на сервер при каждом reserve —
   маска растёт каждый убатч → пересборка графа → переreserve).
2. **SO_SNDBUF 16MB на клиентском сокете** (transport.cpp `set_large_send_buf`):
   убрал блокировку 88 мс/убатч (rs_s_copy set ждал дренажа 10.5 МБ l_out-43
   в 1 GbE-буфере); дренаж теперь перекрывается локальным compute следующего
   убатча.
3. **-ts 0.9,0.6,1.5** (слои: VK1 0-19, VK0 20-32, RPC0 33-64=32 слоя на 3080,
   8.9/10 ГБ VRAM): дисплейная VK0 на 40% медленнее VK1 (15.9 vs 11.4 мс/слой
   prefill) — убрали с критического пути, сервер вне пути (его compute
   перекрывается клиентом).

Диагностика: `LLAMA_UBATCH_TIMING=1` (process_ubatch: build/alloc/inputs/
compute_call) + `GGML_SCHED_SPLIT_TIMING=1` (split-копии; копии l_out делают
sync src-бэкенда = ожидание GPU предыдущего сплита — это и есть "скрытое"
GPU-время: compute=submit, compute_sync=реальное GPU-время).
`GGML_RPC_DEBUG=1` slow-copy get/set: копия VK0→RPC0 = 4 мс get + 13 мс set —
копии НЕ горлышко; горлышко = vkQueueWaitIdle в copy-фазе следующего сплита.

## Карта тёплого убатча (r12, ts 0.9,0.7,1.4, 1024 ток, ~700 мс/убатч)

VK1 GPU ~240 мс (20 слоёв, асинхронный submit 56 мс, ожидание в copy-фазе
split3 l_out-19) + staged-копия l_out-19 20 МБ ~20 мс + VK0 GPU ~240 мс
(16 слоёв, ожидание в split4 copy l_out-35) + readback 4 мс + F16 13 мс
(send в буфер, не блокирует) + финальный get result_output 350-440 мс
(первый decode убатч) + хвостовой убатч (другая форма → разовый alloc ~517 мс).

## Что НЕ трогать / уроки

- `GGML_RPC_ENABLE_MASK_NULL` СЛОМАН (ppl-регрессия): NULL-маска = full
  attention в локальном FA. Не включать.
- Правки gui/*.py и docs/research/dflash/*.md — чужие, не трогать.
- rpc-server на 3080 = сборка битпак-маски (клиентские изменения libllama
  серверу не нужны; transport/alloc-cache — клиентские). Деплой на 3080 нужен
  только при изменении ggml-rpc.cpp/server: stop task WinRM, scp rpc-server.exe
  Chriswork@192.168.1.60:C:/rpc-3080/, start, Test-NetConnection :50052.
- Ветка rpc-vulkan: 3080 = Vulkan0 10 ГБ (32 слоя = ~8.9 ГБ — потолок;
  для 160k KV серверные слои уменьшать).

## Открытые идеи (не тестировались)

- Skip output-слоя сервера на промежуточных убатчах (флаг в GRAPH_COMPUTE;
  серверное время и так вне критического пути — выгода малая).
- Ping-pong VK1(N+1)||VK0(N): текущий sched сериализует GPU внутри убатча
  (copy-фаза sync'ит src), но VK1(N+1) уже перекрывается с VK0(N) через
  асинхронный submit; дальше — sub-ubatch 512 (VK1(h2)||VK0(h1)).
- F8 l_out (10.5→5.2 МБ): −10-45 мс/убатч, нужна ppl-валидация.

---

# MTP-DECODE НА RPC: ДИАГНОСТИКА (r15-r36, ЗАКРЫТО 2026-08-23)

Дата: 2026-08-23. Вопрос пользователя: можно ли получить 40 t/s decode с MTP
на RPC-лейне (сейчас spec=none 23.6 t/s).

## РЕЗУЛЬТАТ (r36, loopback, фикс задеплоен)

**MTP через RPC ПОЧИНЕН**: acceptance **0.4% → 59.7%** (89/149, r36), decode
**7.9 → 37.0 t/s**, текст осмысленный (r25: "user user user..." ×120). Корень —
не NVIDIA, не KV, не handoff, а **баг graph-recompute в RPC-протоколе**.

## КОРЕНЬ БАГА (r34): RPC_CMD_GRAPH_RECOMPUTE без идентификации графа

Сервер хранил ОДИН stored_graph на устройство. Клиентский graph_cache (по
memcmp полной структуры ggml_tensor) — ПЕР-КОНТЕКСТ (ggml_backend_rpc_init
создаёт новый backend с новым gc на каждый llama_context). В MTP на одном
RPC0 живут ДВА контекста (target+дraft): клиентский кэш target-контекста
попадает для verify-графа (nodes=2413) → шлёт RECOMPUTE → **сервер
пересчитывает СВОЙ последний полный граф — а это драфт-граф (nodes=1/24/16/3),
присланный между verify-вызовами** → verify-логиты = мусор ("user"), acceptance
1.4%. spec=none не ломался (один контекст → рекомпуты подряд корректны).

Доказательства:
- r34-трейс: `graph_recompute nodes=2413` ×237 (клиент верит в кэш) при том,
  что между verify-вызовами сервер получал полные драфт-графы (nodes=1 ×813,
  24 ×267, 16 ×264 — они НЕ кэшируются: их входы leaf_* меняют remote_ptr).
- r33: `GGML_RPC_NO_GRAPH_CACHE=1` → acc 0.597 / 36.9 t/s — чистый контроль.

## ФИКС (ggml-rpc.cpp, собран и задеплоен в build-vulkan)

1. **Клиент**: `graph_cache` хранит структурный hash графа
   (`graph_structure_hash`: type/op/op_params/ne/nb/name нод + имена src; БЕЗ
   указателей и data — иначе хеши клиента и сервера не совпадут).
2. **Протокол**: `rpc_msg_graph_recompute_req` + `uint64_t hash`, ответ
   `rpc_msg_graph_recompute_rsp { uint8_t result }` (сервер раньше не отвечал).
3. **Сервер**: `stored_graphs[device]` → `unordered_map<hash, stored_graph>`
   (лимит 16, при переполнении clear → клиент получает MISS и шлёт полный
   граф — fallback работает, r35: 1 MISS за прогон). `graph_recompute` ищет
   по hash; не нашёл → result=0 (клиент пересылает полный).
4. Версия протокола НЕ поднята (форк собирает клиент+сервер вместе) — при
   смешивании старого сервера с новым клиентом соединение сломается (старый
   сервер не шлёт rsp).

Проверено: r35 (debug): acc 0.597, 66 recompute hit + 1 MISS; r36 (чистый):
acc 0.597, decode 37.0 t/s. Кэш verify-графа работает (экономит сериализацию
~750 КБ/verify) — производительность не хуже полного отключения кэша.

## ПРОВЕРЕННЫЕ ФАКТЫ (бенчи r15-r36, артефакты в build_logs/agent-workload/)

| Факт | Значение |
|---|---|
| Модель имеет MTP | `nextn_predict_layers=1` в GGUF + head-only nextn (eh_proj/enorm/hnorm/shared_head_norm; блочных nextn_0.* НЕТ — это норма) |
| Локальный MTP (без RPC, r16/r31) | **decode 34.9-47.6 t/s, acceptance 0.44 / 0.69** — работает |
| RPC MTP (3080 r15-r23 / loopback r25-r30) | acceptance 0.4-3% ПРИ ЛЮБЫХ конфигах — сломан (протокол, не NVIDIA: r25 loopback AMD-сервер = 1.46%) |
| spec=none через RPC (r32 loopback) | текст ПРАВИЛЬНЫЙ — target-граф через RPC корректен |
| STORE/SETINPUT CHECK (r21/r23) | maxdiff=0 — h-копии (staging→h) корректны |
| `-devd Vulkan1` (r28) | ИГНОРИРУЕТСЯ для MTP: контекст создаётся с main-девайсами (common/speculative.cpp:2722) |
| `LLAMA_MTP_DEVICE_HANDOFF=0` (r29) | acc тот же 1.46% — handoff не причина |
| `LLAMA_VK_MTP_NEXTN_MAIN_DEVICE=0` (r30) | acc тот же 1.46% — сплит графа не причина |
| Драфт-граф через RPC (r27) | split на 5 кусков: norm-64×2 (сервер), eh_proj (клиент), attn/ffn (сервер, nodes=24/16), head (сервер, nodes=3) — ВСЁ численно корректно по составу; ломался только verify-рекомпут |
| Веса nextn | enorm/hnorm/eh_proj на клиенте (Vulkan0), shared_head_norm/head на сервере (RPC0) — head = output.weight (shared_head_head.weight в GGUF НЕТ) |
| Вход драфт-графа `mtp_h_input` | через RPC_CMD_COPY_TENSOR (серверный copy_tensor, не get/set) — корректно |

## ПРИМЕЧАНИЕ (старая гипотеза)

Ранняя гипотеза "1-нодный драфт-граф (nodes=1) считает неверно на NVIDIA"
ОТВЕРГНУТА: nodes=1 — это только первый сплит (RMS_NORM mtp_h_input); весь
драфт-граф = 1+1+24+16+3 нод. Пин nextn-весов на Vulkan1 (llama-model.cpp
create_tensor) не срабатывает для head-only nextn (tn.bid == -1?) — но это
НЕ причина бага (r30 подтвердил).

## ЧИСТЫЕ ЧИСЛА (loopback, Vulkan1-сервер + Vulkan0,RPC0-клиент, 12k, n=4)

| Прогон | acceptance | decode t/s | примечание |
|---|---|---|---|
| r25-r30 (кэш-баг) | 1.46% (7/479) | ~8 | мусорный текст |
| r31 локальный без RPC | 0.44/0.69 | 40.15 | контроль |
| r33 без graph cache | 89/149 (0.597) | 36.9 | контроль |
| r35+r36 ФИКС (кэш работает) | 89/149 (0.597) | 37.0 | **итог** |

## 160K A/B: local dual против 3-GPU RPC (2026-08-25, r37-r44)

Одинаковый длинный лейн: Qwen3.8-27B Q4_K_M, `-c 163840`, prompt 94651
токен, b8192/ub1024, KV f8 с последними 12 attention-слоями f16, MTP n=2.
Local dual использовал `-dev Vulkan1,Vulkan0 -ts 1,1`; 3-GPU —
`-dev Vulkan1,Vulkan0,RPC0 -ts 1,0.8,0.625` через 1 GbE.

**Baseline-контракт с 2026-08-25:** r38 — canonical local-dual long-prompt
baseline; r39 — canonical 3-GPU RPC long-prompt baseline. Сравнивать дальнейшие
decode/network эксперименты только при том же prompt, KV, MTP depth, split и
отсутствии фоновой GPU-нагрузки либо с отдельным явно записанным A/B.

| Прогон | Prompt t/s | Decode t/s | Acceptance | Wall |
|---|---:|---:|---:|---:|
| r38 local dual, 94651 tok | **1213.88** | **35.50** | 74/104 (0.7115) | 81.83 s |
| r39 + RTX 3080 RPC, 94651 tok | 816.94 | 21.00 | 74/104 (0.7115) | 122.17 s |
| r40 + RTX 3080 RPC, spec=none | 809.32 | 16.91 | n/a | 124.76 s |
| r41 local dual, spec=none | 187.26 | 22.11 | n/a | 511.46 s |

3-GPU освободил VRAM: веса+KV без compute-буферов составили примерно
9551/7804/7908 MiB на Vulkan1/Vulkan0/RPC0 против 11295/13968 MiB на двух
локальных GPU. OOM и прежнего connection reset нет; hash-фикс корректен,
acceptance и текст совпадают с local dual. В этом измерении 3-GPU медленнее на
32.7% по prompt и на 40.8% по decode; вероятный фактор — цена RPC/1 GbE.
Решение принять или отклонить 3-GPU подход не принято: оно остаётся за
пользователем, включая выбор следующих split/network A/B. r37 (тот же
`-c 163840`, короткий prompt 8809) дал 1540.75/48.59 t/s и показывает
отдельную цену длинного заполненного KV.

Разделяющий r40 `spec=none` дал 16.91 t/s: MTP n=2 в r39 ускоряет RPC decode
на 24.2%, а основное узкое место находится в базовом target RPC path. r41-r44
ниже уточняют влияние VRAM residency, сетевых копий и server graph submit.

r41 подтвердил локальный VRAM/residency cliff именно для `spec=none`: третий GPU
в r40 поднял prompt 187.26 -> 809.32 t/s (4.32x), но снизил decode 22.11 ->
16.91 t/s. Это не заменяет canonical local baseline r38: MTP windowed prefill в
r38 не попал в тот же медленный режим и сохранил 1213.88 prompt t/s.

### Decode RPC trace и output-head A/B (r42-r44)

r42/r43 — диагностические прогоны `spec=none`, max 32, с `GGML_RPC_DEBUG=1`;
trace почти не изменил r40 lane. В steady decode (31 RPC-вызов) получено:

| Прогон | Prompt t/s | Decode t/s | Remote graph submit | Output readback |
|---|---:|---:|---:|---:|
| r42 default output на RPC0 | 804.16 | 16.78 | 11.11 ms avg | `result_output`, 0.497 MiB, 6.10 ms avg |
| r43 `LLAMA_OUTPUT_DEVICE=Vulkan1` | 787.41 | 18.84 | 9.21 ms avg | `norm`, 20 KiB, 0.94 ms avg |

Мелкие hidden/mask set-копии в r42 занимали только 0.06/0.11 ms. Основные
явные RPC-центры decode — round-trip remote graph submit и синхронизирующий
readback logits. Перенос output-head на Vulkan1 переместил примерно 1.0 GiB
весов с RPC0 на Vulkan1, заменил 0.497 MiB/token logits на 20 KiB/token hidden
state и дал +12.25% decode при -2.08% prompt в `spec=none`.

r44 — чистый max-128 MTP n=2 A/B с тем же output-head placement: **512.06
prompt / 25.19 decode t/s**, acceptance 74/104 (0.7115), wall 190.16 s. Относительно
r39 это +19.9% decode, но -37.3% prompt. Следовательно, глобальный перенос
output-head не включается по умолчанию: он подтверждает сетевую цену logits,
однако конфликтует с MTP windowed prefill/target-output residency. Решение о
принятии варианта остаётся за пользователем.

**Статус исследования: OPEN.** Decode-регрессия 35.50 -> 21.00 t/s теперь
локализована до target RPC path: r42 показывает около 11.1 ms remote graph
submit и 6.1 ms logits readback на токен. Следующие кандидаты для измерения —
phase-specific output placement (remote в PP, local в TG) без r44
prompt-регрессии и снижение Vulkan server-submit overhead для 852-854-node
decode graph. Любое решение о принятии или отказе от RPC и этих вариантов
делает пользователь.

## КОМАНДЫ (ветка rpc-vulkan, лайв-лог обязателен — tee)

Бенч: `python scripts/agent_workload_bench.py --server-bin build-vulkan/bin/llama-server.exe
--model models/Qwen3.8-27B-Q4_K_M.gguf --label <name> --tasks quick --ctx-size 12288
--batch-size 8192 --ubatch-size 1024 --max-tokens 128 --cache-type-k q8_0 --cache-type-v q8_0
--flash-attn --no-warmup --server-seed 42 --real-context-mode repo-snapshot --real-context-chars 24576
--server-extra "--rpc 192.168.1.60:50052 -fit off --cache-ram 0 --ctx-checkpoints 0
-dev Vulkan1,Vulkan0,RPC0 -sm layer -ts 1,0.8,1.2 --spec-type draft-mtp --spec-draft-n-max 4" | tee log`

Диагностика: `LLAMA_MTP_DEVICE_HANDOFF_TRACE=1` (staging alloc/set_input rows),
`LLAMA_MTP_STORE_CHECK=1` (data compare), `GGML_RPC_DEBUG=1` (все RPC-операции,
nodes=1 ×N = драфт-граф), `GGML_SCHED_SPLIT_TIMING=1`. Acceptance:
`grep "draft acceptance" <label>.server.log` или timings.draft_n_accepted/draft_n
в jsonl. Сборка: `export PATH="/c/Strawberry/c/bin:$PATH" && ninja -C build-vulkan llama-server`.

## Разложение prefill-батча 1.9 с (2026-08-22, 16k, ubatch 1024 — УСТАРЕЛО)

| Фаза | мс | Комментарий |
|---|---:|---|
| get_tensor `l_out-18` 20МБ F32 | 480-490 | ожидание конца compute 3080 + 160 мс сеть |
| get_tensor `attn_inp_kq_mask (copy)` ×2 | 2×(19→167) | растёт с KV — убирается фиксом C2 |
| set_tensor input_embed 20МБ | 25-29 | ~800 МБ/с (не сеть, хэш/кэш) |
| set_tensor `(view)` 4Б | ~177 | блокировка TCP-буфера, стабильно |
| graph_compute ser=430КБ | 36→106 | растёт по вызовам |
| локальные 2 GPU (48 слоёв) | ~650 | не RPC-зависимо |
| серверный compute | 300→510 | растёт с KV |

## Серверный PERF-лог 3080 (GGML_VK_PERF_LOGGER=1, 16k)

- q4_K FFN m=17408 n=1024 k=5120: 76 TFLOPS (2398 мкс × 38) — НОРМА;
  НО деградирует батч-к-батчу: 2398→4074 мкс (+70%) при ТОЙ ЖЕ геометрии;
  q6_K (m=5120 k=17408): 58 TFLOPS (34.3 мс); FLASH_ATTN: 62 TFLOPS,
  13.1→28.4 мс с ростом KV; GDN: 23.1→38.4 мс; SILU/RMS тоже +30%.
- Все ноды деградируют одинаково → подозрение на троттлинг частоты 3080
  (карта в чужом корпусе, TGP 320W, idle 38°C/210МГц — проверено nvidia-smi).
  НЕ ПРОВЕРЕНО под нагрузкой: нужен smi.csv во время живого прогона
  (частота/температура по батчам). Если троттлинг подтвердится — это
  железо-ограничение, сообщить пользователю.

## Тайловый свип NV-coopmat2 (уже сделан, рычаг НЕ закоммичен)

`ggml/src/ggml-vulkan/runtime/vk_shaders.inc`: env `GGML_VK_NV_MMQ_TILE=BLOCK,BM,BN,BK,enable_smaller`
(парсится в coopmat2-ветке; базовые: 256,128,256,64,1). 16k, -ts 28,58,14:
- T1 `256,64,128,32,0` = **571.45 ptps (+6% к базе 540)** — лучший
- T2 `128,64,128,32,0` = 551.89
- T3 `256,128,128,32,0` = 534.53
- T4 `256,64,256,32,0` = 553.07
Потолок этого рычага ~+6%; продолжить можно только если сеть/маска починены
и нужен последний процент.

## Файлы и изменения в рабочей копии (НЕ закоммичены)

- `src/llama-graph.cpp` — B3 (СЛОМАНА, см. выше)
- `ggml/src/ggml-rpc/ggml-rpc.cpp` — клиентский трейс под RPC_DEBUG:
  set_tensor (время), get_tensor (время), graph_compute (nodes/ser/time)
- `ggml/src/ggml-vulkan/runtime/vk_shaders.inc` — GGML_VK_NV_MMQ_TILE
- `build_logs/agent-workload/BENCH_RECENT.md`, `BENCH_RUNS.csv` — обновлены бенчем
- `git diff --check` после правок обязателен; НЕ коммитить без команды юзера.

## Команды

Сборка (MinGW DLL нужны в PATH):
```powershell
PATH="/c/Strawberry/c/bin:$PATH" ninja -C build-vulkan llama-server rpc-server
```

Локальный rpc-server (async-терминал, лог с tee — лайв-лог обязателен на этой ветке):
```powershell
./build-vulkan/bin/rpc-server.exe -d Vulkan1 -p 50052 2>&1 | tee /tmp/rpc-srv.log
```

Воспроизведение ассерта локально (loopback):
```powershell
unset GGML_RPC_DEBUG; export GGML_SCHED_SPLIT_TIMING=1
./build-vulkan/bin/llama-server.exe -m models/Qwen3.8-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 52999 -c 16384 -b 8192 -ub 1024 \
  --cache-type-k q8_0 --cache-type-v q8_0 -ngl 999 --no-warmup \
  --rpc 127.0.0.1:50052 -dev Vulkan0,RPC0 -sm layer -ts 33,32 -fit off
```

Сервер 3080 (192.168.1.60, WinRM Chriswork/4622, ТОЛЬКО прямой запуск в висящей
Invoke-Command-сессии async-терминала — Start-Process убивается после сессии):
```powershell
powershell -NoProfile -Command "Invoke-Command -ComputerName 192.168.1.60 -Credential (New-Object PSCredential('Chriswork',(ConvertTo-SecureString '4622' -AsPlainText -Force))) -ScriptBlock { & 'C:\rpc-3080\rpc-server.exe' -d Vulkan0 -p 50052 -H 0.0.0.0 }"
```
Пакет на 3080: `\\Chris\c\rpc-3080\` (exe + DLL libgcc_s_seh/libgomp/libstdc++/
libwinpthread/libdl; при обновлении exe — убить процесс на 3080, скопировать, запустить).
PERF_LOGGER на сервере: `GGML_VK_PERF_LOGGER=1` в env той же сессии.
Тайлы: `GGML_VK_NV_MMQ_TILE=256,64,128,32,0`.

Бенч (клиентский трейс — через env GGML_RPC_DEBUG=1, лог в build_logs/agent-workload/
<label>.server.log; unset после):
```powershell
python scripts/agent_workload_bench.py --server-bin build-vulkan/bin/llama-server.exe \
  --model models/Qwen3.8-27B-Q4_K_M.gguf --label rpc3080-16k-... \
  --real-context-mode repo-snapshot --real-context-chars 32768 --ctx-size 16384 \
  --batch-size 8192 --ubatch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn --no-warmup --task-hard-timeout 300 \
  --server-extra "--rpc 192.168.1.60:50052 -dev RPC0,Vulkan1,Vulkan0 -sm layer -ts 28,58,14 -fit off --cache-ram 0 --ctx-checkpoints 0 --seed 42 -ngl 99"
```

## Порядок работы следующему агенту

1. Локализовать NULL-буфер (fprintf в ggml-backend.cpp:1583) → подтвердить маску.
2. Откатить B3, применить C2 (маска на локальный бэкенд через
   sched_set_tensor_backend по имени) → сборка → loopback → 16k бенч.
   Целевой эффект: маски нет ни в set, ни в get RPC-трафике.
3. Термал-проверка 3080: nvidia-smi-цикл (2 с) в файл во время живого прогона,
   сравнить частоту с деградацией FFN в PERF-логе. Если троттлинг — стоп и
   к пользователю (железо корпуса).
4. Дальнейшие рычаги (по убыванию): F32→F16 активации l_out/input_embed
   (20МБ→10МБ ×2 стороны ≈ −160 мс/батч, правка в llama-split/RPC-типах);
   2.5G-сеть; тайловый свип только после всего.
5. Каждый шаг — `git diff --check`; итоги в BENCH_RECENT.md/BENCH_RUNS.csv;
   обновить этот playbook перед паузой.

## ОБНОВЛЕНИЕ 2026-08-22 ~20:50 — C2 реализован, но даёт РЕГРЕССИЮ (пауза по просьбе пользователя)

Что сделано и проверено:
- B3 ПОЛНОСТЬЮ откачен (grep `kq_mask = nullptr` в llama-graph.cpp пуст).
- C2 реализован: пин cast-маски `attn_inp_kq_mask (copy)` на первый не-RPC бэкенд
  (`src/llama-context.cpp`, `pin_causal_mask_to_local_backend`, вызовы в graph_reserve
  ~:1960 и process_ubatch ~:3130) + name-matcher `is_causal_mask_name`
  (`ggml/src/ggml-rpc/ggml-rpc.cpp:435`, strstr "#attn_inp_kq_mask" + проверка
  разделителя) в скипе set (клиент) и NULL-подстановке FA src[3] (сервер,
  включая декорированные sched-имена `RPC0[host]#attn_inp_kq_mask (copy)#0`).
- Loopback (новый клиент+сервер, AMD): 0 строк маски в RPC-трафике, decode 27.6 tok/s,
  вывод корректный — механика скипа/подстановки работает.
- Развёрнуто на 3080: rpc-server.exe 66536460 байт (20:24) запущен, порт 50052 ОК.

РЕЗУЛЬТАТ на 3080 (бенчи `rpc3080-16k-c2-r1`/`-r2`, 16k, Q4_K_M, -ts 28,58,14):
- prompt 400.2/402.5/403.1 ptps (бейзлайн 534.8-549.6) — РЕГРЕССИЯ −26%;
- decode 3.49/3.53 tok/s (бейзлайн 22.9-23.2) — РЕГРЕССИЯ −85%.

ПРИЧИНА (RPC_DEBUG-трасса на реальной 3080, /tmp/rpc3080-dbg.log):
- Пин маски на Vulkan1 через sched-эвристику «назначить узлу бэкенд его входов»
  (ggml-backend.cpp pass 3-4) притянул FA полного attention-слоя l3 (слой 3 ∈
  RPC-диапазон [0,28)) НА КЛИЕНТ. Клиент стал тянуть:
  - decode, каждый токен: `cache_k_l3 (view) (permuted)` + `cache_v_l3 ...`
    3.6МБ×2 (~32мс ×2) + `Qcur_full-3 (view)` 48КБ;
  - prefill, каждый убатч: `Qcur_full-3 (view)` 50.3МБ ~440мс ×3 +
    `(reshaped) (view) (permuted)` 25МБ ~273-443мс ×3 + `l_out-18` 20МБ ~544-675мс;
  - серверный `graph_compute nodes=918 ser=407844` 208-211мс.
- В бейзлайне FA l3 считался на сервере: никаких cache_k/v/Qcur get не было,
  decode-трафик = l_out-18 20КБ ~11мс + маска 23.5КБ ×2 ~0.6мс.
- Т.е. C2 убрал маску из сети ценой переноса вычислений FA RPC-слоёв на клиент —
  сделка резко убыточная.

Термал 3080 (smi.csv/smi-bench-c2.csv во время живого прогона): троттлинга НЕТ —
пик 1935МГц (база 1440), макс 49°C, util до 95%, power ≤130W. Подозрение handoff
«FFN +70% из-за перегрева» СНЯТО (деградация FFN батч-к-батчу — не температура).
Предупреждение: параллельный nvidia-smi опрос каждые 2.5с с power.draw глушит GPU
(в r1); для термал-мониторинга — только без power.draw или реже 5с, и НЕ параллельно
с «чистым» замером (r2 без монитора дал те же числа → монитор в r1 не причина).

Кандидаты фикса для следующей сессии (НЕ проверены; выбрать с пользователем):
(a) откатить пин — вернёт бейзлайн 540/23, но вернётся маска-трафик 2×(19→167мс)/батч
    и предпосылка C2 (локальная маска) отпадает;
(b) пинить также FA-узлы по имени `attn_FA-<il>` на бэкенд их слоя (по k/v view_src),
    оставив cast маски локальным — маска на сервер не идёт (скип+NULL), FA остаётся
    на месте, локальные FA получают маску из локального каста;
(c) сервер сам генерирует маску для декорированного имени (fill_mask на сервере),
    пин клиента не трогать — но проблему «FA уехал на клиент» это не решает.

Состояние паузы: локальные процессы остановлены; сервер 3080 работает (оставить).
Несохранённые правки: ggml-rpc.cpp (матчер+трейс), llama-context.cpp (пин),
llama-graph.cpp (нейминг маски `attn_inp_kq_mask`/`_ms`), vk_shaders.inc (tile env —
чужое). Бенч-артефакты: rpc3080-16k-c2-r1/r2 (+ diagnostics/csv/server.log),
BENCH_HISTORY/RECENT/LANES/RUNS обновлены.

## ОБНОВЛЕНИЕ 2026-08-22 ~21:55 — чистое сравнение карт (Qwen3.5-9B Q5_K_M, 16k)

Пользователь сменил тактику: сравнить карты на маленькой модели в одинаковых
контурах. Итог (задача 2 = тёплая; задача 1 на 3080 = холодная, компиляция
NV-шейдеров: 305.9 ptps / 8.68 t/s!):

| Контур | Prefill | Decode |
|---|---:|---:|
| RX 9070 XT, локально (`-dev Vulkan1`) | 3226.8 ptps | 71.3 t/s |
| RTX 3080, локально (llama-server прямо на 3080) | **3267.7 ptps** | **88.7 t/s** |
| 9070 XT через RPC loopback (`-dev RPC0`) | 2895.1 ptps (−10%) | 62.6 t/s (−12%) |
| 3080 через RPC LAN (`-dev RPC0`) | 2037.0 ptps (−38%) | 34.2 t/s (−61%) |

ВЫВОДЫ: (1) карты ПАРИТЕТНЫ, 3080 даже чуть быстрее (decode +24%) — прежняя
оценка «9070 XT в 1.4-1.8× быстрее» была сетевым артефактом; (2) бутылочное
горлышко = RPC-протокол + 1GbE: на LAN decode 3080 падает в 2.6×; для 27B
compute-потенциал 3080 ≈ 3268/2.9 ≈ 1100 ptps против 540 baseline через RPC —
т.е. оверхед ~2×, и главный резерв — RPC-контур (F32-активации 20МБ×2/батч,
маска-трафик, протокол), а НЕ кернелы 3080; (3) NV-шейдерная компиляция
первого прогона = ×10 — в бенчах 3080 брать только тёплые задачи (2+) или
делать прогрев.

Инфраструктура (НОВАЯ, перейти с WinRM):
- SSH на 3080 беспарольный: `ssh Chriswork@192.168.1.60 "..."` (ключ через
  C:\ProgramData\ssh\administrators_authorized_keys — Chriswork админ);
  scp вместо SMB-шары (\\Chris\c ненадёжна для больших файлов).
- llama-server на 3080: schtasks `llama-srv-3080` (SYSTEM, переживает сессии),
  лог C:\rpc-3080\llama-srv.log, порт 53333 + firewall-правило llama-srv-53333,
  батники C:\rpc-3080\start/stop-llama-srv.bat; модель Qwen3.5-9B-Q5_K_M.gguf
  в C:\rpc-3080\models (6.58ГБ).
- Бенч против 3080: agent_workload_bench.py --no-start --host 192.168.1.60 --port 53333.

## Не повторять ошибок

- `GGML_RPC_DEBUG=1` в persistent-терминале не гаснет само — unset перед чистыми
  прогонами (печать на каждый тензор рушит замер).
- Ветка rpc-vulkan: лайв-лог через tee, без `> file 2>&1` для долгих команд.
- Локальный rpc-server держит экзешник залоченным — taskkill //F //IM rpc-server.exe
  перед линковкой.
- `GGML_SCHED_SPLIT_TIMING=1` недостаточно для локализации ассерта (см. выше).

## Состояние 2026-08-22 (конец сессии: FA-пин + F16-стык)

СДЕЛАНО И ПРОВЕРЕНО (50/50 Vulkan1+RPC0, Qwen3.5-9B Q5_K_M, 16k, ubatch 1024,
KV q8_0, spec=none; тёплые цифры = задача 2):
- B0 бейзлайн: 1969.1 ptps / 36.9 t/s (артефакт q9-16k-split5050-vk1-rpc3080-r1).
- B1 (пин FA-узлов "__fattn__-<il>" на бэкенд весов слоя, llama-context.cpp
  pin_causal_mask_to_local_backend + модель; трасса подтвердила: FA остаётся
  на сервере, cache_k/v_l19 get = 0): 1948.9 ptps / 35.6 t/s — НЕЙТРАЛЬНО
  (артефакт q9-16k-split5050-b1-fapin-r1).
- B2 (F16-активации на RPC-стыке: GGML_RPC_TENSOR_FLAG_ACT_F16 в ggml-rpc.cpp,
  имена "#l_out-"/"result_output"; клиент set F32→F16, сервер set F16→F32,
  сервер get F32→F16, клиент get F16→F32; hash-кэш скипается для активаций):
  2361.7 ptps / 40.0 t/s — ПРЕФИЛЛ +19.9%, ДЕКОД +8.5%
  (артефакт q9-16k-split5050-b2-f16act-r1). НЕ ЗАКОММИЧЕНО (сборка готова,
  rpc-server.exe с фиксом задеплоен на 3080).
- Трассировка prefill-убатча (GGML_SCHED_SPLIT_TIMING=1 + RPC_DEBUG, 6901 ток):
  всё строго последовательно: VK1 ~125мс (submit 35мс асинхронно, реальное
  время ждётся при чтении l_out-16 в copy-фазе) + сеть l_out F16 81мс (9.7 send
  + 71 блокировка на серверном recv) + серверный compute 110-190мс (внутри
  get result_output 124-210мс) + меж-убатчевый оверхед ~60мс ≈ 457мс/убатч.
  Маска больше НЕ ходит по сети (set скипнут), но F32-маска всё ещё растёт
  до 25МБ/убатч и копируется CPU→VK1 каждый убатч (мелочь, ~3мс DMA).

СЛЕДУЮЩИЕ ВАРИАНТЫ (не проверены):
(a) пропустить output-слой (2 TFLOP, ~70мс/убатч) на промежуточных убатчах:
    логиты нужны только на последнем; сервер считает result_output каждый
    убатч. Реализация: флаг skip_output в RPC_CMD_GRAPH_COMPUTE + клиент
    знает последний убатч; синхронизация промежуточных — через sync-gc.
(b) конвейер VK1(N+1) || сервер(N) — потенциал −150мс/убатч, нужен ping-pong
    l_out-16 и реструктуризация process_ubatch (большая работа).
(c) decode: бенч 25мс/токен = RPC ~13мс + CPU-сэмплинг харнесса (penalties+dry
    на 248320 vocab) ~12мс; greedy-тест даёт 13мс/токен (76 t/s) — RPC-часть
    декода уже НЕ бутылочное горлышко; 64 t/s недостижимо без правки сэмплинга
    или переноса output на клиента (тогда логиты локальные, но prefill дороже
    на ~73мс/убатч — конфликт).
(d) F8 для l_out-16 (ещё −40мс/убатч) — только с валидацией качества.
Цели: prefill ≥2597.8 ptps (сейчас 2361.7, 90.9% цели), decode ≥64 t/s
(сейчас 40.0, 62.5% цели).
