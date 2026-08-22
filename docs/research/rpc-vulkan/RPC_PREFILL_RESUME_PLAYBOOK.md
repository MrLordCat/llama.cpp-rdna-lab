# RPC / Nvidia-prefill: RESUME PLAYBOOK (handoff)

Дата: 2026-08-22. Ветка: `rpc-vulkan`. Задача: ускорить prompt eval в RPC-схеме
(Qwen3.8-27B Q4_K_M, 16k, клиент = 2×RX 9070 XT Vulkan, сервер = RTX 3080 по сети).

## Статус одним абзацем

RPC-оффлоад работает (16k: 540-571 ptps, decode 23-25 tok/s; 160k decode 25.8 tok/s).
Prompt eval медленный НЕ из-за слабого compute 3080 (q4_K MMQ = 76 TFLOPS FP16-экв,
~64% пика FP16 tensor, FA cm2 = 62), а из-за суммы: локальная часть (~0.65 с/батч)
+ сеть (~0.45 с: F32-активации 20МБ туда + 20МБ обратно + МАСКА обратно 2×(19→167 мс))
+ накладные протокола. Последняя правка B3 (убрать маску из FA, src[3]=NULL)
СЛОМАЛА клиента: ассерт при старте в RPC-режиме. Локализовать → заменить B3 на
вариант C2 (маска остаётся в графе, но назначается на локальный бэкенд).

## Текущее сломанное состояние (первое, что чинить)

```
src/llama-graph.cpp  — B3-правка в build_attn_mha: для имён "attn_inp_kq_mask" и
  "attn_inp_kq_mask (copy)" передаёт kq_mask=nullptr в ggml_flash_attn_ext.
  РЕЗУЛЬТАТ: llama-server в RPC-режиме падает при старте:
  ggml/src/ggml-backend.cpp:186: GGML_ASSERT(buffer) failed
  (exit code 3221226505 = 0xC0000409). Без --rpc НЕ падает (проверено loopback).
```

Механика ассерта: в `ggml_backend_sched` цикл по split->inputs вызывает
`ggml_backend_buffer_get_usage(input->buffer)` (ggml-backend.cpp:1588 и 1610) для
тензора с NULL-буфером. Гипотеза: после B3 маска/каст `(copy)` не попадают в
граф (не аллоцируются), но остаются src какой-то ноды в сплите (потребитель
вне build_attn_mha: возможно, GDN-код qwen35 или MTP-граф берёт get_kq_mask()).
`GGML_SCHED_SPLIT_TIMING=1` НЕ помогает: fprintf в :1588 вычисляет get_usage
в аргументах ДО печати → ассерт раньше вывода.

Шаг 1 (диагностика): в ggml-backend.cpp перед :1588 (в цикле по split->inputs,
под `trace_split_timing` или безусловно) добавить:
```cpp
fprintf(stderr, "[split-input] name='%s' buffer=%s flags=%d\n",
        input->name, input->buffer ? "OK" : "NULL", input->flags);
```
Пересобрать ggml+llama-server, воспроизвести (см. команды ниже), увидеть имя.

Шаг 2 (фикс, вариант C2 — приоритетный): откатить B3 (вернуть kq_mask в FA),
вместо этого после построения графа (llama-context.cpp, место reserve) для
тензоров с именами "attn_inp_kq_mask" и "attn_inp_kq_mask (copy)" вызвать
`ggml_backend_sched_set_tensor_backend` на первый НЕ-RPC бэкенд (Vulkan1/Vulkan0).
Тогда: fill_mask+cast вычисляются локально; на RPC0 маска не живёт; в сеть не
идёт ни set (клиентский serialize_tensor уже пропускает CAUSAL_MASK-имена),
ни get (маска не на RPC0). Сервер продолжает генерировать маску сам
(fill_mask в его графе, src[3]=NULL в FA — уже работает, проверено).
Ожидание: 16k prefill 540 → ~600-650 ptps (это убирает 2×(19→167) мс/батч
и на 49k/160k даст кратно больше — маска обратно была n_kv×1024×4Б×2 копии).

## Разложение prefill-батча 1.9 с (измерено, GGML_RPC_DEBUG=1, 16k, ubatch 1024)

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
