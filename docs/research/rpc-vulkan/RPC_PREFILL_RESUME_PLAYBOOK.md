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
- F8 l_out (F32 20→F16 10→F8 5 МБ при ubatch 1024) протестирован:
  PPL идентичен, но 14K TPS не вырос; режим оставлен только env-gated.

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

## Деградация RPC с ростом длины промпта (2026-08-25, r47-r53)

Модель Qwen3.8-27B Q4_K_M, MTP n=2, `-c 163840`, b8192/ub1024, KV f8 + 12
слоёв f16, cold/no-warmup. Чистые точки (фон на 3080 выключен):

| Прогон | Путь | Промпт, ток | Prompt t/s | Prompt ms | ms/ub |
|---|---:|---|---:|---:|---:|
| r49 | 3-GPU RPC | 36911 | 1033.2 | 35724 | 991 |
| r39 | 3-GPU RPC | 94651 | 816.9 | 115861 | 1253 |
| r38 | local dual | 94651 | 1213.9 | 77974 | 844 |

Разложение prefill-убатча по логу r42 (`GGML_RPC_DEBUG=1`, spec=none): per-ub
≈ **717 ms + 11.5 мс на 1000 токенов KV**, т.е. от 775 ms при K≈5K до
1716 ms при K≈87K. Сумма по 92 убатчам даёт O(K²/ub) время prefill, поэтому
prompt t/s падает как `1/(a + b*avgK)`. Проверенная эмпирическая модель:

```text
ms на токен = 0.700 + 0.0112 × avgK/1024
r39 94K: прогноз 818 t/s, факт 817 t/s
r49 33K: прогноз 1102 t/s, факт 1033 t/s (холодный старт ~7%)
```

Компоненты линейного члена (по данным r42): передача маски 2048×K байт/убатч
(566 МБ за весь 94K prefill, 2.8 с — малая доля), сетевой перенос активаций
20 МБ/убатч (константа, не наклон), и основной вклад — **серверный
последовательный проход 16 слоёв на RPC0 с FATTN O(K×n_tokens)**.
Фоновые 9B-сервисы на RTX 3080 (llama-srv-3080) умножают время prefill RPC в
~3.4x (r48: 302 t/s на 33K при фоне против 1033 чистым) — перед любым RPC
бенчем этот сервис надо останавливать.

Важное ограничение сравнения: локальный 2-GPU путь сейчас нестабилен —
после серии прогонов он стабильно уходит в «медленный карман» (~165-171 t/s
на 8K и 33K, как r50-r53; breakdown при этом идентичен нормальному r38).
`LLAMA_TG_DROP_PP_SCHED=0` карман не снимает. r38 (94K, 1214 t/s) — пока
единственный надёжный не-карманный локальный контроль. Причина кармана
требует отдельной диагностики и не связана с RPC-наклоном.

### Сеть как ограничитель (2026-08-25)

Канал 1 GbE фактически НЕ ограничитель: по r42 за весь 94K prefill wire-трафик
≈ 2.64 ГБ за 118 с = 22.4 МБ/с (~19% от 1 GbE). Распределение: маска 566 МБ
(packed, raw 9.06 ГБ), активации l_out 2.0 ГБ (raw, из них wire передаётся
как F16 1.0 ГБ + смонтировано в SO_SNDBUF асинхронно), result_output 16 МБ.
Поэтому «увеличить ubatch» НЕ исправляет главный сегмент: серверный проход
O(n_tokens×K) не меняется по суммарной работе, а выигрыш только в
фиксированных накладных (меньше передач l_out, меньше сериализаций графа).
Маска-трафик при этом даже слегка растёт (ΣKV_i = N(N+ub)/2).

Рост объёмов с длиной (замечен пользователем) — это маска: raw 2048×K
байт/убатч (10.5 МБ при K=5K → 178 МБ при K=87K), битовая упаковка /16.
Времена с K — не от сети, а от серверного/клиентского проходов.

### КЛИЕНТ — критический путь (пер-graph op-trace, 2026-08-25)

> Историческая инструментированная трасса. Её абсолютные времена и ранний
> вывод про отсутствие эффекта async уточнены в «Итоговой коррекции» ниже.

Серверный op-trace дал 21.9 с GPU за 36.8 с prefill (576 мс/уб, MM 382 мс
константа, FATTN avg 83 мс, aux 74 мс). Но клиентский per-graph trace
(GGML_VK_PERF_LOGGER, 94 графа на 33K) показал: **клиентские GPU VK1+VK0
= ~735 мс/уб** (MMQ ~580 мс константа; FATTN 6→50 + 2→24 мс по двум картам
= наклон ~3.5 мс/К). Wall/уб ≈ 968 мс. Серверный 576 мс/уб перекрыт с
клиентским лишь частично; клиент — критический путь. Протокол RPC
graph_compute и так fire-and-forget (send без ответа; `RPC_CMD_GRAPH_COMPUTE_ASYNC`
не дал выигрыша: A/B 1073.7 vs 1071.7 t/s = 0.2% шум). Реализованный
async-конвейер (серверный worker-поток + auto-flush перед любой другой
командой, env `GGML_RPC_ASYNC_GRAPH=1`) — CLIENT: не включать по умолчанию:
сервер уже скрыт; остаётся как эксперимент.

A/B KV-тип (гипотеза «f8 плохо идёт на 3080» — ОПРОВЕРГНУТА): q8_0 KV
1058.7 t/s vs f8_e4m3 1031.6–1073.7 t/s (шум ±2%); decode 14.07 vs
12.35–14.39. Mixed не фолбится (NV_coopmat2 есть). Следующие кандидаты:
клиентский MMQ (580 мс/уб константа), ~233 мс/уб накладных (копии
VK1↔VK0 + сокет), клиентский FATTN-наклон.

### Механизм №3: серверная генерация causal-маски (2026-08-25, реализован)

Маска в форке: `ggml_new_tensor_4d(F32, n_kv, n_tokens, 1, n_stream)`,
значения M[i][j] = 0.0 когда `j <= n_past + i` (position клетки == j,
непрерывный 1-seq causal кэш), иначе −inf; «(copy)» = F16-каст.
`llama-kv-cache.cpp:set_input_kq_mask` (rows: data[n_kv*i + j], p0 <= p1).

Реализация (ggml-rpc.cpp, env `GGML_RPC_SERVER_MAKE_MASK=1`): клиент вместо
битпака шлёт `RPC_CMD_SET_TENSOR_MASK_NPAST` | rpc_tensor | offset | n_past,
n_past = n_kv − n_tokens (только для 1-stream causal, иначе старый путь);
сервер генерирует маску в тензор. Проверки: 47 вызовов mask_npast
(256/2/254 → 2048/1024/1024), prompt 1072.3 t/s на 33K (база 1073.7, шум);
**PPL 7.0608 ± 0.23810 = в точности baseline (битпак) 7.0608 ± 0.23810** —
безопасно в отличие от NULL-подстановки (см. GGML_RPC_ENABLE_MASK_NULL).
Выигрыш скромный (маска на 94K ≈ 2.4% prefill-времени), но также убирает
трафик в decode. По умолчанию НЕ включён — решение за пользователем.

### Диагностика 2026-08-25 (чистая A/B: local 2×9070XT vs RPC+3080)

> Исторический промежуточный диагноз. Утверждения про «медленный карман» и
> незакрытый серверный tail superseded итоговой коррекцией ниже.

Все прогнозы ниже сняты **чистым** прогоном (14K/19.7K ток, spec=none,
KV f8, ubatch 1024, real-context 60K):
- local `-dev Vulkan1,Vulkan0 -ts 1,0.8`: **1546.8 / 1546.9 t/s** (два прогона,
  разброс <0.1%) — «медленный карман 165-171» НЕ воспроизводится; карман
  был артефактом инструментации/состояния, не кода.
- rpc `-dev Vulkan1,Vulkan0,RPC0 -ts 1,0.8,0.625`: **1149.5 / 1176.7 t/s**;
  RPC МЕДЛЕННЕЕ local на ~25% (wall 817 vs 607 мс/уб).
- **Инструментация — главный враг**: `GGML_VK_PERF_LOGGER=1` +
  `LLAMA_UBATCH_TIMING=1` + `GGML_SCHED_SPLIT_TIMING=1` роняют local до
  963 t/s (−38%!) и RPC до 1130 (−1.7%). Все прежние «клиент = критический
  путь (735 мс/уб)» сняты под инструментацией и искажены (с перфа: local
  922 мс/уб = RPC 937 мс/уб).
- **Op-карта (чистая логика, VK_PERF на 14K)**: MMQ-блок q4_K/q6_K = 65-73
  TFLOPS (m=5120..17408, n=1024) — потолок де-кванта 9070XT (D076/D077/D094
  «near peak / at peak»), not разгоняемый. FATTN 35 TFLOPS (16-35, D094
  закрыт), CONCAT 24-48 мс, GDN 20-39 мс — вторичны.
- **Узкое место RPC-схемы**: wall = local-client (607 мс/уб, в точности =
  local) + **~210 мс/уб серверного времени, НЕ перекрытого** (ожидание
  серверного graph). Гипотеза: наш auto-flush (async-граф + drain перед
  любой не-async командой) сериализует prefill: каждый следующий убатч
  шлёт KV-set → flush ждёт завершения предыдущего серверного графа.
- Опровергнутые кандидаты: (a) async-граф чистый: 1172.7 (шуm, +0-2%);
  (b) mid-order `-dev Vulkan1,RPC0,Vulkan0 -ts 1.3,0.95,0.5` (RPC в середине):
  885 t/s — хуже (данные последовательны, перекрытия нет; 1-я попытка
  `-ts 0.78,1.25,0.5` убила alloc на 3080: 7.03 ГБ весов + KV 0.96 ГБ);
  (c) ubatch 2048: 1015 t/s — хуже (маска/сервер ×2 перевешивают −50% барьеров).
- Следующий шаг (код): протокольный «no-flush KV set» — KV-записи ubatch
  N+1 пишутся в диапазон, который серверный graph N не читает (prefill KV
  последователен): allow server to start ubatch N+1 graph без ожидания N;
  оценка: +20-25% (RPC → ~1400-1500, паритет с local) — точнее после реализации.

### Инструмент: RPC timeline + split-тайминг (2026-08-25)

Новый диагностический набор (рабочий !):
- `GGML_RPC_TIMELINE=1` — машинный трейс на клиенте и сервере:
  `RPC_TL|cli|<cmd>|<bytes>|<send_ms>|<rsp_ms>|<gap_ms>|t=<wall_ms>` и `RPC_TL|srv|...|<idle_ms>|<proc_ms>|<flush_ms>`; имена тензоров:
  `RPC_TL|name|SET_TENSOR_HASH|<name>|<bytes>|t=...` и `RPC_TL|name|SET_TENSOR|...`.
- Анализатор: `scripts/research/rpc_timeline_analyze.py --client ... --server ...`.
- Правильная комбинация (почти без накладных — различие tps <1%):
  `LLAMA_UBATCH_TIMING=1 GGML_SCHED_SPLIT_TIMING=1 GGML_RPC_TIMELINE=1`
  (НЕ `GGML_VK_PERF_LOGGER` — он искажает замеры, −38% на local!).

### Исторический вывод: split-copy поля включали GPU sync (superseded)

Split-тайминг на 14K (28 splits, 21 ub, wall 17.5 с):
- RPC-контур: **Vulkan0 copy=7600.9 мс** (271 мс/сплит) + **RPC0 copy=6923.6 мс**
  (247 мс/сплит) = 14.5 с = 83% стены; GPU-compute всего 1.57 с (9%!)
  (VK1 1377 мс, VK0 126 мс, RPC0 68 мс, CPU 35 мс).
- Локальный контур (те же 28 сплитов): копии 174.5+187.4 = 0.36 с (2%),
  compute 19.9 с (98%); копия VK1→VK0 6.7 мс (14.4 МБ ~2150 МБ/с).
- Копии: `first=RMS_NORM:norm-28` (VK1→VK0, ~433 мс) и `norm-49` (VK0→RPC0,
  ~391 мс): l_out 14.4 МБ при ~33-36 МБ/с (64× медленнее локальной копии).
- Ранние выводы пересмотрены: 92×`SET_TENSOR_HASH` rsp 4449 мс — это ВЕСА
  модели (blk.49-64: output.weight 1043 МБ + ffn_* 50-73 МБ) при
  инициализации, НЕ префилл; GRAPH_COMPUTE сервер 208 мс/уб полностью
  скрыт (клиент ждёт 0); сеть wire ~403 МБ l_out-48 + 238 малых KV = 23 МБ/с.
- «Префилл параллелится хорошо» подтверждено: суммарное GPU-время в RPC
  ничтожно; всё время съедают последовательные копии/ожидания л_out.
- Следующий шаг: уменьшить l_out-копии: (a) F8 l_out (14.4→7.2 МБ, −50%
  копий); (b) разобраться, почему copy VK1→VK0 через host идёт 33 МБ/с
  вместо 2150 (не является ли ожиданием сетевого хвоста); (c) не
  передавать l_out на RPC0 в промежуточных убатчах, если выход не нужен
  следующему серверному проходу.

### Механизм №1: async-конвейер до selective-drain fix (исторический A/B)

`RPC_CMD_GRAPH_COMPUTE_ASYNC` + серверный worker-поток + auto-flush перед
любой не-async командой; клиент env `GGML_RPC_ASYNC_GRAPH=1`: A/B 33K:
1073.7 vs 1071.7 t/s (0.2% — шум). Причина: клиент (735 мс/уб GPU) —
критический путь, серверный проход (576 мс/уб) уже в основном перекрыт;
graph_compute клиента и так fire-and-forget (send без ответа). По умолчанию
выключен; полезен только если сервер снова станет критическим (больше
слоёв на RPC / двойной буфер l_out).

### Итоговая коррекция 2026-08-25 (актуальный source of truth)

Контракт: Qwen3.8-27B Q4_K_M, prompt 19 707 токенов (real-context 60K),
ctx 163840, ubatch 1024, KV f8_e4m3 с последними 12 слоями f16,
`spec=none`, Vulkan1 сначала, чистые соседние прогоны без тяжёлого op-trace.

1. **«Медленный local-карман» = WDDM VRAM paging, не RPC и не регрессия
  Vulkan-кода.** На дисплейной Vulkan0 внешние процессы занимали около
  3 ГиБ dedicated VRAM (`dwm` 981 МиБ, Edge 520, VS Code 457 и прочие).
  При `-ts 1,0.8` свободный запас Vulkan0 упал с 1778 до 1373 МиБ и clean
  local обрушился до 198.46 t/s (с `LLAMA_UBATCH_TIMING` — 225.64; первый
  убатч 562 мс, следующие 4.3-5.0 с). Перенос 1-2 слоёв на недисплейную
  карту восстановил резидентность: 4K A/B `1,0.8` = 200.43, `1,0.65` =
  1257.94, `1,0.72` = 1270.76 t/s. На `1,0.72` свободно 1990/2044 МиБ и
  полный 14K local = **1490.67 t/s**, decode 15.70.
2. **Актуальный RPC baseline:** `-ts 1,0.8,0.625`, selective drain и
  `GGML_RPC_ASYNC_GRAPH=1` = **1183.44 t/s**, decode 14.47. Реальный разрыв
  с устойчивым local = 20.6%, а не наблюдавшиеся в paging-кармане 5-7x.
  Третья карта одновременно снимает давление VRAM, поэтому RPC оставался
  около 1027 t/s на коротком 4K контроле, когда local `1,0.8` был 200 t/s.
3. **Поля split `copy=433/391 мс` не являются чистым временем сети.** Они
  включают ожидание предыдущего Vulkan-графа. С принудительным разнесением
  видно: VK1 GPU ~326-345 мс, VK0 ~286-297 мс, фактическая F16-передача
  `l_out` на RPC ~25-35 мс. Split/op-логирование сильно искажает local TPS;
  использовать его только качественно, а скорость подтверждать clean A/B.
4. **Selective server drain реализован и закрыт timeline-доказательством.**
  `SET_TENSOR` больше не вызывает `graph_compute_wait`; drain оставлен для
  mask/GRAPH/GET/COPY/CLEAR/FREE. Ранний A/B: 1187.1 против 1158.6 t/s
  (+2.5%). В последнем серверном срезе 498 команд суммарный flush = 127.8
  мс, весь в девяти `GET_TENSOR`; все 28 `SET_TENSOR_MASK` имели flush=0.
  Остаточной сериализации prefill на маске нет.
5. **F8 E4M3 transport для `l_out` реализован, но отклонён как speed lever.**
  Env `GGML_RPC_ACT_F8=1` уменьшает payload 20 МиБ F32 → 5 МиБ F8 вместо
  10 МиБ F16; `result_output` остаётся F16. 14K: **1169.51 t/s** против
  F16 **1183.44** (−1.2%, шум/CPU-конвертация). WikiText 2×8192:
  **PPL 5.7381 ± 0.15760 в обоих режимах**. Режим безопасен по качеству,
  но по умолчанию выключен и не является принятым ускорением этой lane.
6. Новые `SET_TENSOR_MASK_NPAST`, async graph и F8 payload несовместимы со
  старым wire protocol; major повышен до **RPC v5**. Клиент и сервер должны
  обновляться парой. На RTX 3080 развёрнут v5, scheduled task возвращён на
  обычный `start-rpc-srv.bat` без timeline-логирования.
7. **20.6% RPC-gap локализован: синхронный fallback VK0→RPC уничтожает
  межубатчевый overlap двух локальных GPU.** Лёгкий
  `GGML_SCHED_SPLIT_SUMMARY=1` (без op-query и принудительного sync), adjacent
  14K/19 707 токенов: local `-ts 1,0.72` = **1485.71 t/s**, RPC
  `-ts 1,0.8,0.625` = **1150.68 t/s**. Для 19 полных ubatch медианы:
  local split-sum **642.1 мс** (Vulkan1 72.0; Vulkan0 556.1), RPC
  **846.5 мс** (Vulkan1 73.3; Vulkan0 386.8; RPC0-copy 380.7); необъяснённый
  остаток одинаков — 7.1/7.2 мс. Следовательно весь gap находится внутри
  scheduler split path, а не в graph build, inputs, mask, серверном drain или
  декодере.
8. **Точная точка сериализации:** RPC backend оставляет
  `cpy_tensor_async = NULL`; поэтому `ggml_backend_sched_compute_splits()`
  попадает в fallback `ggml_backend_synchronize(input_backend)` перед
  `ggml_backend_tensor_copy(input, input_cpy)`. В local Vulkan0 — последний
  split: его граф ubatch N остаётся в очереди и перекрывается с Vulkan1
  ubatch N+1. В RPC Vulkan0 становится промежуточным split: перед отправкой
  `l_out` планировщик ждёт его завершения на основном потоке. Серверный
  `GRAPH_COMPUTE_ASYNC` отправляется за 0.7-1.2 мс и RTX 3080 уже скрыта;
  потеря — последовательные ~380 мс Vulkan1 и ~380 мс Vulkan0.

Следующий практический рычаг — **реальный async VK0→RPC outbound lane**:
worker должен по событию дождаться Vulkan0, передать `l_out` и поставить
RPC graph в том же порядке, пока основной поток начинает Vulkan1 следующего
ubatch. Одного `GRAPH_COMPUTE_ASYNC`, event-capability или четырёх scheduler
copies недостаточно: без `cpy_tensor_async` fallback всё равно синхронизирует
VK0. Split/VRAM-баланс и дальнейшее сжатие `l_out` эту сериализацию не лечат.
9. **Попытка async outbound конвейера (2026-08-25) — опровергнута как
   отдельный механизм; откат на sync copy.** Реализовано: per-socket
   упорядоченная очередь команд (`rpc_send_queue`), fire-and-forget
   `GRAPH_COMPUTE_ASYNC`, событийные фенсы (`GRAPH_WAIT`/no-op barrier),
   `cpy_tensor_async` для GPU→RPC split-входа. Измерения (14K/19 707,
   тот же контракт): sync copy + sync-фенс на каждую копию `GRAPH_WAIT` =
   **1044 t/s** (−12%); sync copy + no-op `synchronize` = **1192 t/s**
   (эквивалент 1183 base); async copy + no-op `synchronize` = **падение**
   (`GGML_ASSERT "tensor buffer not set"`, worker переживает scheduler-буфер,
   VK0 читает `l_out` без события, следующий ubatch переиспользует буфер).
   Для честного конвейера нужны per-device события + ping-pong `l_out` на
   VK0, владеемый scheduler'ом — это не уровень `cpy_tensor_async`.
10. **Detached-thread async split-copy (`GGML_SCHED_ASYNC_SPLIT_COPY`,
    ggml-backend.cpp) — racy, НЕ включать.** Async-копия на отдельном
    потоке: 2/2 чистых прогона зависли (14K); c `GGML_RPC_ASYNC_DEBUG=1`
    прошёл (1382 t/s, но печати меняют тайминги). Причина: detached worker
    не участвует в жизни буферов/событий scheduler. Оставлен в коде только
    для диагностики, по умолчанию env-гейт выключен.
11. **Принятый рычаг сессии: отказ от блокирующего `RECOMPUTE` в
    async-режиме.** В `GGML_RPC_ASYNC_GRAPH=1` клиент не делает
    синхронный recompute-хвост, а отправляет полный граф fire-and-forget.
    A/B на 14K (q8_0 KV, MTP n=4, `-ts 1,0.8,1.2`, идентичный контракт,
    чистые прогоны): control без env = **1278.55 t/s** (decode 31.45);
    async graph = **1384.98 t/s** (decode 28.76) = **+8.3%**;
    независимые подтверждения: 1378.34 (без инструментации), 1382.25
    (debug-инструментация). Стабильно 3/3 без зависаний. ВАЖНО:
    сравнивать только с этим control (`-ts 1,0.8,1.2`, q8_0 KV); база
    1149-1183 t/s была на другом балансе `-ts 1,0.8,0.625`.
12. **94K-лейн подтверждён (2026-08-25, async-контур, без PPL).**
    `rpc3080-94k-asyncgraph` (ctx 98304, 58 185 ток, q8_0 KV, MTP n=4,
    `-ts 1,1,0.8` после alloc-fail на 1,2): **977.32 t/s**, decode 24.13
    (66.56/64.27 с на 128 ток). Локальный 98K (2×9070XT, spec=none) =
    1355.34 t/s → RPC-gap −28%: серверная 3080 на ≥33K снова видима
    (~576 мс/уб MMQ) и на 94K не полностью скрыта.
    **Баланс-свип `-ts` на 94K:** `1,1,0.8` = 977.32 → `1,1,1.0` =
    1054.31 (+7.9%) → `1,1,1.1` = **1066.81 (+9.2%, плато)**; decode при
    этом падает 24.13 → 23.16 → 19.61 t/s (серверные слои в decode-пути
    дороже). VRAM-потолок 3080 на 96K: 1.1 даёт 8569/10267 MiB, 1.2 не
    влезает. Рекомендованный пресет 94K-лейна: `-ts 1,1,1.1` +
    `GGML_RPC_ASYNC_GRAPH=1` (1066.81 t/s).
    **Фиксы прекондиций:** `-ts 1,1,0.8` (RPC0 KV при 96K требует
    `-ts` не выше ~1.1; 1.2 → alloc fail compute pp buffers);
    `--task-hard-timeout 180` обязателен (дефолт 45 с режет ~58K-промпт).
13. **Диагноз из split-timing (14K, `-ts 1,1,0.8`, 19 ubatch): каскад
    одного ubatch.** Медианы: split VK1 copy=3.08 compute=52.98 →
    VK0 copy=**297.17** compute=49.95 → RPC0 copy=**322.00** compute=0.69;
    wall ~731 мс/уб. Копия split3 = ожидание VK1-GPU (~300 мс), копия
    split4 = ожидание VK0-GPU + передача (~322 мс). **85% стены =
    последовательные GPU двух локальных карт**; RPC0-сервер скрыт
    (compute_call 0.7 мс). Единственный следующий рычаг — межubatch
    конвейер: submit VK1(ubatch N+1), пока VK0(N)+RPC0(N) работают
    (потенциал wall ~731 → ~400-450 мс/уб, т.е. −40% prefill).
    Реализация требует 2-буферного run-ahead на уровне llama-context
    (двойные буферы входов/активаций + события), не уровня compute_splits:
    внутри одного ubatch каскад слоёв VK1→VK0→RPC0 неустраним.
14. **Async outbound copy через RPC-очередь реализован и стабилизирован,
    но НЕ даёт прироста — честное A/B (2026-08-25).**
    Код: `cpy_tensor_async` для RPC-бэкенда включён (интерфейс не NULL,
    set→graph через per-socket очередь), в
    `ggml_backend_rpc_synchronize` честный блокирующий барьер (ждёт
    pending копии/графы) — лечит «tensor buffer not set» при пересоздании
    локальных буферов (переход prefill→decode/reserve). Env-гейты для
    диагностики: `GGML_RPC_NO_ASYNC_COPY=1` (обход → старый sync-copy),
    `GGML_RPC_BARRIER_DISABLE=1` (no-op synchronize). Парные A/B в одной
    сессии (q8_0 KV, MTP n=4, `GGML_RPC_ASYNC_GRAPH=1`, без
    инструментации):
    - 14K (`-ts 1,0.8,1.2`): base (NO_ASYNC_COPY+BARRIER_DISABLE)
      **1378.02 t/s** vs async **1371.17** (−0.5%, шум);
    - 94K (`-ts 1,1,1.1`): base **1065.99** vs async **1065.82** (паритет).
    Split-timing (14K): `l_out-39` VK0→RPC0 20 MiB копия = **0.038 мс**
    (было 322 мс синхронного ожидания VK0 через fallback у copy) — копия
    реально перестала блокировать планировщик. Но wall не изменился:
    следующий ubatch и так перекрывался; а стена определяется каскадом
    **локальных** GPU: split3 `l_out-21` VK1→VK0 20 MiB =
    255-277 мс ОЖИДАНИЕ VK1-GPU (не сеть, не сервер; local Vulkan
    sync-copy) + split2/3 VK0 compute 270 мс. Плюс на ≥33K серверная 3080
    сама выходит в ~576 мс/уб (MM ~382 мс константа). ВЫВОД: RPC-путь
    снят с критического пути; оставшиеся рычаги — (а) межubatch конвейер
    VK1(N+1)||VK0(N)+RPC0(N) (п.13, большая работа), (б) локальная
    асинхронная меж-устройство копия VK1→VK0 в ggml-vulkan (убрать
    sync-wait 255-277 мс) — это код Vulkan, не RPC.
15. **Точная механика сериализации RPC-входов (2026-08-25, измерено
    по строкам split input timing).** Почему async-копия не дала выигрыша:
    сериализация находится НЕ в самом копировании, а в двух барьерах
    `ggml_backend_synchronize(split_backend)` внутри
    `ggml_backend_sched_compute_splits` (ggml-backend.cpp:1698-1725):
    - **не-INPUT вход** (rs_s_copy 4 байта, CPU→RPC0): перед каждым
      не-INPUT входом планировщик ждёт dst-бэкенд; при RPC это
      `rpc_wait_pending_copies` = барьер на всю per-socket очередь. Первый
      же такой вход (идущий сразу после async-копии `l_out-39` 20 MiB,
      которая сама ждёт VK0-GPU) блокирует main на **259.4 мс**
      (вход 2/10, copy=259.38; именно это раньше выглядело как «split4
      copy=261→271 мс»). После убирания барьера (экспериментальная правка)
      вход 2 = 0.007 мс, split4 total 262 → **8.7 мс** — копия быстрая;
    - **INPUT-вход** (leaf_56 16 КБ, CPU→RPC0, вход 4/10): барьер в
      INPUT-ветке ждёт ту же незавершённую async-копию l_out:
      **244.7-252.0 мс** (bytes 16384; при малых bytes 32-80 =
      7-9.5 мс чистого RTT). Убирание барьера и здесь даёт падение
      `GGML_ASSERT "tensor buffer not set"` (ggml-backend.cpp:345): worker
      держит dst-тензор от прошлого графа, а планировщик после build
      следующего ubatch пересоздаёт буферы — известная причина, ради
      которой барьеры и были введены (п.9/п.14).
    Парные A/B после частичных фиксов (14K `-ts 1,0.8,1.2` и 94K
    `-ts 1,1,1.1`, q8_0 KV, MTP n=4, `GGML_RPC_ASYNC_GRAPH=1`): 14K
    1380.38 vs 1381.03 (паритет, аб2); 94K **1070.72** vs base 1065.99
    (+0.4%, аб4). `LLAMA_VK_MTP_PIPELINE_PARALLEL=1` (events + n_copies=2,
    штатный параллельный scheduler): 14K 1377.65 — паритет. ВЫВОД: пока в
    планировщике стоит хотя бы один полный barrier на RPC-входе, wall
    упёрся в [VK1-GPU 280 + VK0-GPU 270 + дожим сервера]; освободить main
    может только двойной буфер l_out/входов на уровне llama-context
    (run-ahead ubatch) ИЛИ snapshot host-входов в RPC-очереди + защита
    dst-буферов (worker → raw SET по server buffer id, без живого
    ggml_tensor). Оба пути — следующая сессия; текущее состояние кода не
    изменено (барьеры на месте, стабильно, 2/2+2/2+94K без падений).

### Исторический список кандидатов (частично реализованы; см. итог выше)

1. **Перекрытие (2-stage pipeline)** — скрыть серверный проход под
   локальным: ping-pong l_out + двойной RS-буфер SSM, сервер с worker+io
   потоками. Потенциал ~−20-30% prefill (116 → 85-90 с на 94K). Средняя/
   большая работа, риск корректности. **Уточнение (94K): серверная 3080
   и так скрыта на 14K, но на 94K видима; реальная цель теперь —
   межubatch конвейер VK1(N+1)∥VK0(N) (см. п.13).**
2. **Split-баланс (сократить серверные слои)** — **ПРОВЕРЕН (см. п.12+):
   `-ts 1,1,0.8` → `-ts 1,1,1.0` = 977.32 → 1054.31 t/s (+7.9%) на 94K.**
   Сред/большая работа против... простая модель «−0.84 с на слой»
   уточнена: серверный слой стоит лишь тогда, пока RPC0 не стал
   критичным; потолок VRAM 3080 на 96K `-ts` ≈1.1.
3. **Серверная генерация causal-маски** — клиент шлёт n_past (4 байта) вместо
   2048×K; убирает 566 МБ (2.4%) и рост объёмов; также сокращает wire в
   decode.
4. **Оптимизация NV-путей на сервере** — 3080 поддерживает NV_coopmat2;
   проверить фактический FATTN/MMQ route в серверном проходе.
5. F8-активации l_out — малый ROI (сеть не bottleneck), требует валидации.

Следующий диагностический шаг: per-node op-trace на rpc-server
(GGML_VK_PERF_LOGGER + лог в файл), чтобы подтвердить долю FATTN/FFN в
серверном проходе до выбора механизма.

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

## Рефактор-проба 2026-08-26: полный async → дедлок (откачена, чекпоинт 20e2630a6)

Попытка «настоящего конвейера» в один заход: (1) снятие входных барьеров
для RPC в compute_splits (INPUT + не-INPUT), (2) snapshot host-входов в
rpc_async_copy_submit, (3) граничный drain в ggml_backend_sched_reset.
РЕЗУЛЬТАТ: split4 (RPC0) total **262 → 1.4 мс** (копии реально перестали
блокировать; l_out-21 VK1→VK0 тоже 9-22 мс) — но клиент **завис** на
резерве графа (после PP→TG switch, 4-й декодный ubatch): процесс жив,
лог не растёт, GPU idle при полной VRAM. Причина (дизайн): worker-задачи
per-socket очереди держат живые указатели на ggml-тензоры/буферы клиента;
при отложенном исполнении на весь prefill они теряют валидность при
пересоздании буферов, а drain на границе ждёт worker, который в свою
очередь ждёт доступ к сокету/серверу (цепочка не сходится).
ПРАВИЛЬНЫЙ ДИЗАЙН ПОДТВЕРЖДЁН 2026-08-26 (rf5-rf7, код в работе): снэпшот
данных делается В ПОТОКЕ ПЛАНИРОВЩИКА (ggml_backend_synchronize +
tensor_get вызываются из main, а не из RPC-работника), а per-socket worker
только шлёт захваченный payload по сети. Причина дедлока rf1/rf3:
ggml_vk_synchronize (vk_backend_execution.inc:2) не потокобезопасна
(общий compute_ctx/fence/submit_pending) и worker-поток вызывал её
параллельно с graph_compute того же VK-бэкенда; детерминированное
зависание на 3-4-м декодном ubatch. После фикса: RF5/6/7 без падений.
Сняты bar barriers: RPC split4 copy 288→0.16-0.4 мс, префилл проходит.
Тайминги rf6: l_out-39 20МБ = sync 241-266 мс + get 4.9-6.7 мс — т.е.
остаток = ожидание самого VK0-графа (не RPC worker!); split3 copy
(VK1→VK0) = 268-286 мс (та же локальная стена). 14K: rf6 1372.3
(debug-печать), rf7 1363.1/27.68 (clean) — паритет с base 1323-1381
(шум ±4%), т.е. входные барьеры не были лимитом 14K.
СЛЕДУЮЩАЯ СТЕНА: локальные копии VK1→VK0 (host-путь 33-74 МБ/с, 268 мс
на 20 МБ) — код ggml-vulkan (buffer cpy → read+staging+sync), не RPC.
Ожидание 94K: снятые входные барьеры дадут прирост, верхняя граница =
стена VK1→VK0 + сервер.
Откат-точка: коммит 20e2630a6.

Хронология всех изменений RPC-контура (коммиты, эксперименты, откаты,
результаты по датам) — см. [RPC_CHANGES.md](RPC_CHANGES.md).

