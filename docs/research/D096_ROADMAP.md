# D096 ROADMAP — полная нативность fp8 + платформенные хвосты

Статус на 2026-08-09. Цель: последовательно закрыть все пункты (время не ограничено).

## Архитектурный разворот 2026-08-09 — один канонический FA, узкая fp8-дельта

Ручной полный клон `flash_attn_cm1.comp` в `fp8_fa_cm1.spvasm` больше не является
production-кандидатом. Диагностика v80-v132 показала, что копия дрейфует сразу в
маске, shared-memory layout, softmax, P/V и эпилоге; поиск каждой точки по отдельности
не масштабируется и не даёт надёжного соответствия каноническому Vulkan FA.

Новый порядок работы:

1. **P1 — canonical direct-f8.** Использовать существующий генерируемый вариант
  `flash_attn_cm1.comp` с `DATA_A_F8_E4M3`: K/V остаются f8 в VRAM и деквантуются
  тайлами внутри канонического шейдера. Маска, softmax, P/V и output остаются ровно
  теми же, что у проверенного cm1. Сначала закрыть корректность и сравнить с текущим
  preconvert-f16 fallback; только затем измерять производительность.
2. **P2 — native fp8 WMMA.** Не поддерживать второй полный FA-кернел. Генерировать
  SPIR-V из канонического cm1 и применять узкий fail-closed transform только к
  S-стадии K x Q: добавить float8 cooperative-matrix типы/загрузки и нужный Q-scale.
  Все якоря transform проверяются; при изменении upstream cm1 сборка должна падать,
  а не молча производить несовместимый шейдер.
3. `fp8_fa_cm1.spvasm` и vN-дампы сохраняются как диагностический архив и источник
  аппаратных фактов, но не как база дальнейшей production-разработки.

Acceptance gate P1: отдельный opt-in маршрут; осмысленный ответ без NaN/Inf; соседний
контроль с тем же f8 KV через preconvert-f16; подтверждённый route trace; после этого
короткий A/B и только затем длинный benchmark. Acceptance gate P2 дополнительно требует
автоматическое сравнение структуры неизменённых секций с каноническим SPIR-V.

P1 gate на RX 9070 XT / Qwen3.6-27B закрыт как **REJECTED 2026-08-09**:
`path=coopmat1`, `k/v=f8_e4m3`, `preconvert=0`, `shmem_staging=0` был подтверждён,
но prompt evaluation дал повреждённые активации и HTTP 500; соседний preconvert-f16
контроль остался связным. Маршрут сохраняется только как opt-in диагностический стенд.
Следующий production-кандидат — P2, а не дальнейшее разрастание P1 или ручного клона.

P2 foundation **DONE 2026-08-09**: `scripts/research/d096_fp8_fa_spirv_audit.py`
дизассемблирует реальный canonical cm1 `.spv`, независимо от числовых ID находит
границы S и PV и fail-closed проверяет их типы/геометрию/порядок. Transform получает
право менять только диапазон S-stage; PV и всё после первого Store остаются canonical.

## A. Нативный fp8 WMMA (SPIR-V asm) — ГЛАВНЫЙ БЛОКЕР БЫЛ GLSL fp8-тип

### A1 [DONE 2026-08-07] Кодировка fp8 coopmat в SPIR-V доказана
- **Рецепт** (валидирован spirv-as + spirv-val + РЕАЛЬНЫЙ драйвер RX 9070 XT):
  ```
  OpCapability Float8EXT                  (4212, SPV_EXT_float8)
  OpCapability CooperativeMatrixKHR       (6022)
  OpCapability Float8CooperativeMatrixEXT (4213, требует Float8EXT+CooperativeMatrixKHR)
  OpExtension "SPV_EXT_float8"
  OpExtension "SPV_KHR_cooperative_matrix"
  %fp8 = OpTypeFloat 8 Float8E4M3EXT     (FPEncoding 4214; E5M2 = 4215)
  OpTypeCooperativeMatrixKHR %fp8 %scope_subgroup(3) %rows %cols %use(A=0,B=1,Acc=2)
  OpCooperativeMatrixLoadKHR %t %r %ptr %int_0(row-major)/%int_1(col-major) %stride_elems
  OpCooperativeMatrixMulAddKHR %t %r %A %B %C
  ```
- **Ограничение**: LocalSize >= subgroup size (64) обязателен (при 1 потоке записывается
  только 1 элемент матрицы).
- **E5M2**: работает, но AMD декодирует с BIAS 16 (не стандартный 15) — эмпирически:
  байт 0x38 (E4M3=1.0) → E5M2 = 0.25, 0x40 (2.0) → 1.0. Для KV используем E4M3 (bias 7,
  проверен ТОЧНО, max_err=0).
- Инфраструктура: `examples/vk_fp8_probe/` (probe.cpp + fp8_mul*.spvasm) — автономный
  Vulkan-зонд: создаёт пайплайн, диспатчит, сверяет C=A*B. Сборка:
  `g++ -std=c++17 -O2 -I"$SDK/Include" probe.cpp /c/Windows/System32/vulkan-1.dll -o probe.exe`
  (SDK = C:/VulkanSDK/1.4.350.0; spirv-as/spirv-val из того же SDK).
- Driver properties (KHR query): fp8 записи A=0x3BA247FA(E4M3)/0x3BA247FB(E5M2), B то же,
  C=0x1(f32), 16x16x16, scope=subgroup. Смешанных fp8xf16 НЕТ (A и B должны быть оба fp8).

### A2 [NEXT] Порт FA-ядра cm1 на fp8 SPIR-V asm
Дизайн (обоснован в D096-A):
- S-стадия: A=K (fp8 из KV-буфера, stride=k_stride, row-major), B=Q (f32->fp8 в шейдере
  с БЛОЧНЫМ scale s_q в shmem, чтобы вернуть точность Q ~f16); acc f32; после mul:
  S *= s_q (K остаётся как есть — E4M3 в VRAM, та же точность что у текущего f8 KV).
- V-стадия: вариант (a) полный fp8: P->fp8 со scale, V fp8 raw; вариант (b) f16: P f16,
  V деквант->f16 (текущий путь). Начать с (b) для O-выхода (минимальный риск точности).
- Остальное ядро (init_indices, softmax, маски, ALiBi) — перенос структуры cm1.
- Интеграция: новый .spvasm → сборка через spirv-as в vulkan-shaders-gen (шаг как glslc),
  эмбед как нового пайплайна; роутинг: K/V=f8_e4m3 && coopmat1 -> fp8-пайплайн
  (preconvert для f8 можно отключить — fp8 читает KV напрямую, 1 байт/элемент вместо 2).
- Выигрыш: чтение KV ровно вдвое меньше чем preconvert-f16 + нет dequant-инструкций
  во внутреннем цикле. Цель по FA: +10-20% (реалистично, НЕ 2x — D094-c4/c5).
- Оценка объёма: ~1500-2500 строк asm. Долгая работа.

### A3 [NEXT] A/B бенч fp8 FA vs f16 coopmat+preconvert
- Метрики: prefill ptps (12k lane), decode tps, KV память, качество (Qwen3.6-27B, seed 42).
- Проверка точности: f8 native vs f8 preconvert — logits diff (llama-perplexity).

## Pause checkpoint — 2026-08-08, D096 native fp8 FA quality

Пауза поставлена по просьбе пользователя. Активных `llama-server` нет. Диагностическая
lane: Vulkan, `Qwen3.6-27B-Q4_K_M.gguf`, `K/V=f8_e4m3`, `GGML_VK_FA_F8_NATIVE=1`,
короткий chat prompt (`N=4`/`NQ>1` в FA), один сервер на порту 8099. Сравнивать только
с тем же запуском без `GGML_VK_FA_F8_NATIVE` (стандартный f8 preconvert FA).

Подтверждено аппаратными контролями:

- cooperative-matrix Store, MulAdd, P-load и V-load исправны: константные контроли
  v73/v75/v76/v77 дали соответственно точные `1.0` и `64.0` во всех ожидаемых слотах;
- исправлены база V-load (`sg*4`), двойной `viv*256` в V staging и logical row mapping
  (`row_tid*4` для P/O/L/M);
- P после softmax конечен и лежит в `[0,1]` (v80); O/L/M до эпилога конечны (v81);
- после `subgroupAdd`, sink-ветки и нормализации все значения O конечны
  (`-1.97..3.16`), L строго положителен (`1.77..13`), `1/L` конечен (v82);
- стандартный f8 preconvert FA на том же запросе генерирует нормальный текст, а native
  v84/v85 продолжает `////////////////////////////////`;
- принудительный shared K staging (v85) коллапс не устранил, поэтому direct global
  fp8 K coopmat-load не является причиной. Этот контроль резко ускорил короткий prompt
  (`~97 ptok/s` против `~1-3 ptok/s`) и остаётся отдельным perf-наблюдением;
- broadcast-индексы K/V приведены к формулам `neq/nek`, `neq/nev`, затем `iq/r` из
  `flash_attn_base.glsl`; на текущей lane качество не изменилось;
- Q->E4M3 encoder приведён к bias 7 и нормализации мантиссы через `2^E`, как в
  `types.glsl`; отдельная quality-проверка всё ещё дала слэши.

Текущее состояние исходника намеренно диагностическое:

- `fp8_fa_cm1.spvasm` содержит v80/v81/v82 debug dumps;
- v85 временно форсирует K через shared staging (после следующего контроля вернуть
  `kbc = (Clamp != 0)`);
- v86 добавляет GPU round-trip dump пар `(normalized Q, encoded E4M3 byte)` по адресу
  `dbg[73728 + q2_i*8 + 0..7]` для workgroup `(0,0,0)`;
- v86 собран и прошёл `spirv-as`/`spirv-val`, но ещё не запускался.

Артефакты:

- `build_logs/agent-workload/native-v82-ndbg.log` и текущий `fa_kdump.bin` — здоровый
  нормализованный эпилог;
- `build_logs/agent-workload/native-v84-e4m3-quality.log` — native slash-collapse;
- `build_logs/agent-workload/native-v85-kstage-quality.log` — staged-K slash-collapse;
- `build_logs/agent-workload/baseline-f8-quality.log` — корректный fallback-текст.

Первый шаг при возобновлении:

1. Убедиться, что `llama-server` не запущен.
2. Запустить уже собранный v86 с `GGML_VK_FA_F8_NATIVE=1`,
   `GGML_VK_FA_F8_DUMP=1`, `GGML_VK_FA8_HOSTDUMP=1` и тем же коротким prompt.
3. Декодировать пары из зоны 73728, сравнить GPU-байты с `f32_to_fp8_e4m3` и решить:
   если round-trip чистый — искать семантическую раскладку Q/K/S; если нет — чинить
   encoder. После контроля обязательно убрать qdbg, старые dumps и временный forced K staging.


## Pause checkpoint update — 2026-08-08, v96 real-V localization

### Current state

- No `llama-server` is running. The v96 server was stopped while idle.
- The real native fp8 path still collapses to slash output; the production fix
  is not complete.
- The worktree is intentionally diagnostic. Do not benchmark or accept its
  performance until the temporary dumps and forced staging paths are removed.

### Confirmed progress

- The manual SPIR-V now passes both ordinary validation and
  `spirv-val --target-env vulkan1.2 --allow-localsizeid`. The investigation
  fixed real storage-buffer Block/Offset ABI violations and invalid
  Workgroup-array `ArrayStride` decorations.
- Debug capture now arms only on real prefill (`neq1 > 4`) rather than the NQ=4
  warmup, and overlapping debug zones were separated.
- The real Q path is healthy: v90 normalized Q is finite, and the v92 E4M3
  round trip has 3072/3072 finite values, no fp8 NaN codes, MAE 0.003315 and
  max error 0.0625.
- Adjacent output capture isolated the symptom: among the 7168 positions filled
  by the fallback, native had 5636 NaNs while fallback had none.
- The v93 control that replaced only the cooperative V matrix with ones removed
  all output NaNs (7168/7168 finite) and stopped the slash collapse. Therefore
  P/L/OA/output-store alone are not the failing path.
- Host V fp8 bytes contain no E4M3 NaN codes. The v94 staged-V dump is finite
  for all four HSV tiles (16384 values total).
- Cooperative V load/store round trips are finite for subgroup 0 (v95) and
  subgroup 3 (v96), across all 16 (HSV tile, Bc chunk) combinations.

### Narrow open hypothesis

The first demonstrated NaN boundary is now the interaction of real finite P
with real finite cooperative V in MulAdd/accumulation, unless an unmeasured
P wave/tile is already non-finite. This is substantially narrower than the
original whole-kernel/output failure.

### Diagnostic state to preserve on resume

- Keep the real SPIR-V ABI fixes.
- Restore the temporary forced K staging and remove v80-v96 dump blocks only
  after the root cause is fixed.
- The v96 cooperative-V probe currently selects subgroup 3 despite the
  historical `vcm_sg0` identifier.
- Useful artifacts:
  `native-v90-real-q-clean.log`, `native-v92-qroundtrip.log`,
  `fa_out_native-v91.bin`, `fa_out_fallback.bin`,
  `native-v93-vones.log`, `native-v94-vstage.log`,
  `native-v95-vcm-roundtrip.log`, and `native-v96-vcm-sg3.log`.

### Exact next step

Do not start a server first. Decode the existing v96 dump's real-P zone
offline and classify all 384 P slots:

```powershell
python -c "import struct,math,pathlib; b=pathlib.Path(r'build_logs/agent-workload/fa_kdump.bin').read_bytes(); f=struct.unpack('<%df'%(len(b)//4),b); v=f[2048:2048+384*4]; fin=[x for x in v if math.isfinite(x)]; print('finite',len(fin),'nan',sum(math.isnan(x) for x in v),'inf',sum(math.isinf(x) for x in v),'zero',sum(x==0 for x in v),'range',((min(fin),max(fin)) if fin else None))"
```

If P is fully finite, probe subgroup 1/2 V round trips or capture the
cooperative MulAdd result per subgroup/tile. If P is not finite, map the first
bad P wave/tile and fix the P storage/layout before touching accumulation.
After the root fix, remove diagnostics, restore the normal path, then run an
adjacent fallback/native quality comparison.

## Checkpoint 2026-08-09 — попарное A/B native↔fallback (v107–v110), layout-анализ S-стадии

### Методология попарного сравнения (исправила прошлые ошибки)

- v101–v106 (same-call outdump, cross-process слои) невалидны: промежуточные
  операции перезаписывают dst, число warmup-вызовов плавает между запусками.
- v107: `GGML_VK_FA_HALF_CMP=8` (первые 8 prefill-вызовов native, остальные
  fallback, один процесс) — из 8 native только 2 полных выхода (4 и 6);
- v108: выбор по запросу `(seq/16)%2` в `vk_dispatch.inc` — запрос 1 всегда
  native, запрос 2 fallback, идентичные входы, один процесс. Это рабочий
  инструмент: outdump-файлы `fa_out_native_<n>.bin` / `fa_out_fallback_<n>.bin`
  (73728 f32 = 24 головы × 3072, слойная нумерация плавает между запусками —
  сопоставлять по содержимому, не по номеру).
- Метрика: mean |diff| по конечным значениям; «пара» = минимальный diff.
  Fallback = стандартный f16-coopmat (эталон правильного внимания).

### Результаты

| Версия | Изменение | 4-головочные (draft/nextn) | Полные слои (24 головы) |
|---|---|---|---|
| v108 | базовое (fp8 Q/K) | пара N↔F ≈ 0.078 (шум f8) | расходятся: min 0.47–1.48, слой 1: native rms 0.94 «острый» vs fallback 0.04 «мягкий» |
| v109 | V-layout fix + K direct | 0.078 (совпадает) | расходятся: min 0.33–0.64 |
| v110 | Q,K→f16 (sh_qq/sh_k16), S-scale=pscale | — | расходятся: min 0.34–0.47; N0=нули, N31=все NaN |

Выводы:

- 4-головочные вызовы (KV-малые S → плоский softmax) совпадают с fallback при
  fp8 Q/K — значит S-секция для них численно корректна (порядок операндов,
  stride, маска, масштаб — ок в этом режиме).
- Полные слои расходятся всегда; гипотеза «fp8-квант Q (субнормальные → 50%+)»
  НЕ подтвердилась (v110 f16 не сходится).
- O_epi-дамп (v107): Q читается одинаково для групп по 6 голов = корректная GQA
  для draft (4 KV-головы × 6 Q-голов), не баг.
- fallback-вызовы иногда содержат NaN-зоны (F22 в v110) — при сравнении
  отфильтровываются; native-вызовы с NaN (N31 v110) — артефакт L=0/маски.

### Layout-анализ S-стадии (ключевой результат, 2026-08-09)

Модель распределения coopmat 16×16 (подтверждена по рабочему V-пути и
S-Store `sg*16*sfshstride`):

- S-тайл (j-чанк): 4 подгруппы × ПОЛНЫЕ 16×16: sg хранит kv-строки
  `sg*16..sg*16+15`, токены 0..15 (S[64 kv][16 tok] = 1024 f32, зоны sg*256);
- K (A-операнд): per-sg 16×16 зона `sg*256` f16, элемент (r,c) = sh_k16[sg*256 + r*16 + c],
  строки кэша `j*Bc + sg*16 + r`, колонки `dv*16 + c`; staging-индекс = ksv
  (ksv = sg*256 + r*16 + c, ks_r = ksv/16 уже включает sg*16+r — НЕ добавлять sg!);
- Q (B-операнд): полный 16×16 реплицируется на все sg: элемент (token, d) =
  sh_qq[dv*256 + r*16 + c] (dv-зона 256 f16), запись: индекс
  `dv*256 + q2_r*16 + (q2_d%4)*4 + k` (q2_r = i/64, токен; d = q2_d*4+k);
- v110 баги раскладки (найдены 2026-08-09, исправляются):
  1. Q-запись: было `(dv*16+sg)*4 + r%4 + d%4` → надо `dv*256 + r*16 + (d%4)*4`;
  2. K staging: было `sg*64 + ks_r*16 + ks_c` + двойной sg в ks_vr → надо индекс
     `ksv` и `ks_vr = jc + ks_r` (без sg*16);
  3. kmat1s base: `sg*64` → `sg*256`; qmat1 base: `dv*64` → `dv*256`.

### Следующие шаги

1. Применить layout-фиксы 1–3 (v111), spirv-as+val, пересборка, тест двумя
   запросами (native/fallback), сравнение полных слоёв по головам.
2. Если снова расходятся — следующие подозреваемые: P-layout/маск-add,
   L-нормализация, oa/выходные строки (row_tid*4), O-эпилог (L=0 → NaN).
3. После сходимости: quality-прогон (не `////`), удаление диагностики
   (маркеры/outdump/vcm-остатки), финальная верификация.
4. ВАЖНО: тестовые серверы запускать на второй GPU: `-dev Vulkan1`
   (GPU0 занята дисплеем; правило пользователя от 2026-08-09).

## Checkpoint 2026-08-09 вечер — S-диагностика дампами (S РАБОТАЕТ, маска сломана)

Инструменты: `GGML_VK_FA_F8_DUMP=1` (kdump 524288 Б), зоны: S-после-маски
dbg[0..1023], маска mc0_p dbg[1024+tid*4], S-до-маски dbg[2048+tid*4];
`GGML_VK_FA8_HOSTDUMP=1` (hkv_k/q/v/mask.bin). См. D096_VERSIONS.md v120-v126.

### Прорыв (v124): S-вычисление КОРРЕКТНО
- S-до-маски (tid=0): [1.0, 1.0, 0.935, 0.942] — реальные Σ K×Q (kv0, t0..3).
- Ранние «S пуст / S[0][0]=1.0» — ошибки чтения дампов (маркеры/не та зона).
- K (f8), Q (f32), V из HOSTDUMP: нормальные, NaN=0. NaN в S[12] — из маски.

### БАГ: mask-add читает токен us_c вместо 4us_c (kv/t перепутаны)
- Раскладка S-слота (подтверждена дампами): (us_c, us_r) → S[4us_c][4us_r+k]
  (kv = 4us_c, токены 4us_r+k), а маска читалась m[4us_r+k][us_c]
  (kv = 4us_r+k, токен us_c) → паттерн [0,1,1,1] (валидны только 0,4,8,12).
- v126 (собрано, НЕ тестировано): ml_row = 4us_c, ml_c2 = 4us_r+k (шаг 1),
  slope-вектор sh_slope[4us_c+k], частичные ветки шаг 1.

### Открытые вопросы (после v126)
1. [ОБНОВЛЕНО 2026-08-09, v126b] Тест v126 ПРОВЕДЁН (14:39): ответ — слэши;
   v126b (14:56) снял kdump с NQ=13: маска НЕ causal (-inf в mc0_p даже для
   kv0..3, ровно 9 -inf среди 1024; S-после: 429 NaN/33 inf). Детали и
   декодирование — D096_VERSIONS.md v126b.
2. [ОБНОВЛЕНО] hkv_mask (v126b, NQ=4) = 0/-0.0 — m_buf ЧИСТЫЙ, «мусор» v122
   был артефактом того вызова. Источник -inf в mc0_p — внутри шейдера
   (ml-секция), не хост. ВНИМАНИЕ: v126 развернул kv/t-семантику против v124
   (v124: kv=4us_r+k, t=4us_c; v126: kv=4us_c, t=4us_r+k) — проверить по дампу.
3. S[1][1..3] = 0 при реальных K[1]/Q[1..3] — подозрение q2_gr2 = c - dv*16
   (отрицательный uint при dv≥1) → Q-колонки загружаются только для dv=0.

### Следующие шаги
1. Остановить висящий после v126b сервер (порт 8099) graceful (POST /exit).
2. Найти источник -inf в mc0_p (ml_row = iBr + 4*ml_c, ветки ml_3f/ml_2f,
   очистка var_mc) + устранить противоречие kv/t-семантики v124 vs v126.
3. Q-загрузка (q2_gr2), если S-нули останутся после causal-маски.
4. Финальная верификация: осмысленный ответ, quality-прогон, удаление
   диагностики (маркеры/outdump/kdump-код), docs/BUGS.md + CHANGELOG.md.

## Checkpoint 2026-08-09 — canonical-first P1/P2 завершены

- Полный ручной клон `fp8_fa_cm1.spvasm` снят с производственного пути. Он
  остаётся диагностическим архивом; mask/softmax/PV/output берутся только из
  канонического `flash_attn_cm1.comp`.
- P1 direct-f8 после исправления raw-load guards корректен, но в соседней Q4
  49K паре уступил preconvert по prompt eval: `1320.25` против `1451.58 pt/s`.
- P2 реализован как generated SPIR-V transform, а не копия ядра: отдельный
  Q→E4M3 prepass + узкое преобразование только S-stage cooperative loads.
- P2 прошёл structural audit, `spirv-as`, `spirv-val`, полную Vulkan-сборку и
  hardware quality gate на RX 9070 XT. Route: `path=f8_p2`, `preconvert=0`.
- Свежая соседняя пара на текущем бинарнике: control `1442.46 pt/s, 26.99 t/s`;
  P2 `1267.29 pt/s, 28.02 t/s`. Decode `+3.8%`, prompt `-12.1%`.
- Решение: P2 оставить opt-in proof-of-path, не включать по умолчанию. Следующий
  performance gate P3 — fused Q quantization/scale внутри FA workgroup, чтобы
  убрать отдельный global prepass, не возвращаясь к ручному клону полного FA.
- Канонические артефакты: `d096-p2-q4-f8-preconvert-r1.diagnostics.md`,
  `d096-p2-q4-f8-native-r1.diagnostics.md`, P2 server log и audit scripts.

## Checkpoint 2026-08-09 — P3 fused-Q завершён

- P3 устранил отдельный Q-prepass: Q→E4M3 и dynamic tile scale выполняются в
  Workgroup shared memory канонического `flash_attn_cm1.comp`; transform меняет
  только S-stage K/Q cooperative loads.
- Fail-closed contract расширен профилями `p3-base`/`fp8-p3` и named anchors
  `Q8f`/`Qf_p3_dummy`. Финальный SHA:
  `cd18c170f4af4b46c0ae02d41449ee16efb4bc4c3427010fcd55fbe7989ab52b`.
- P3 hardware quality PASS: длинный 8601-token prompt, обычные и mask-opt
  чанки, связный response preview, ошибок/NaN/slash-collapse нет.
- Performance: P3 `1337.89 pt/s, 28.18 t/s`; P2 `1267.29, 28.02`; control
  `1442.46, 26.99`. P3 вернул `5.6%` prompt относительно P2 и улучшил decode,
  но остаётся на `7.3%` ниже control по prompt.
- Vec4 loads, `dont_unroll`, fixed scale 1024 и bitwise encoder проверены и
  отклонены; активная сборка возвращена к dynamic scalar P3v1.
- Следующий gate: гибрид P3 с V-only f8→f16 preconvert. Это изолирует стоимость
  tile-local V dequant и не требует копии полного FA или fp8 P×V redesign.

## Checkpoint 2026-08-10 — P4 V-only preconvert: НОВЫЙ ЛУЧШИЙ ПУТЬ (+7.6% prompt)

- P4 реализован: fused-Q S-stage transform (как P3) + V из плотного f16
  preconvert-буфера (V-only), K остаётся raw f8. PV-стадия базы получила третью
  V-load ветку (global direct f16) — audit расширен профилями `p4-base`/`fp8-p4`
  (PV: 1 A + 3 B, f16). Финальный SHA:
  `ca27c9af9107776149acdfed6c39a41db60d22c4c71381a896449f24ae5fab3d`.
- SPIR-V: `spirv-as`/`spirv-val`/audit PASS; runtime 7 bindings;
  route `path=f8_p4|preconvert=0` подтверждён; quality связный, ошибок нет.
- Первый замер (`1157.78 pt/s`) выполнен при активной игре — НЕВАЛИДЕН.
- Чистая сессия (2026-08-10, тот же бинарник, 49K lane, соседние прогоны):
  P4 `1498.32/1493.10 pt/s` (mean 1496, разброс 0.2%) vs control (f8 preconvert
  K+V) `1349.15/1428.94/1392.25` (mean 1390, разброс 5.9%) = **+7.6% prompt**,
  decode паритет (28.2 vs 28.0 t/s). P4 стабильнее control по разбросу.
- Вывод: tile-local f8 V dequant был дороже отдельного V-preconvert pass;
  P4 — новый лучший корректный путь (opt-in, не default).
- Следующие шаги: P4 на 12K/98K lanes (проверка масштаба), затем MTP-прогон;
  при подтверждении — решение о default.

## Checkpoint 2026-08-10 — P4 MTP-гейт PASS (рабочий сценарий)

- Чистая MTP-пара (draft-mtp n=2, 128 токенов, 49K lane, r1+r2):
  control prompt `1392/1409`, decode `38/39`, acc 45% (60/133);
  P4 prompt `1463/1455`, decode `41/43`, acc 56% (67/119).
- P4 устойчиво быстрее и по prefill (+4.3..5.0%), и по MTP decode
  (+8.4..11%). Acceptance выше у P4 при меньшем draft_n — первый принятый
  токен другой из-за fp8/f16 префилла; оба пути деградируют одинаково на
  triage_diff @ temp0 (артефакт промпта, не fp8).
- Статус: P4 подтверждён под рабочим сценарием. Осталось:
  1) 12K lane, 2) 98K lane (масштаб gates), 3) решение о default.

## B. Платформенные хвосты (не fp8)

### B1 [OPEN] MTP acceptance gap Vulkan vs ROCm (D094-c8k..c8o)
- eh_proj (q8_0, K=10240) в NextN-графе: N=1 vec-путь ошибка 4e-1 (мусор),
  N>=2 MMQ 1.5e-2 (накопление порядка K-chunks). Последний подозреваемый: BK (порядок
  K-chunks) в MMQ. Цель: acceptance 0.40 -> 0.53+ (паритет с ROCm).
  N>=2 MMQ 1.5e-2 (накопление порядка K-chunks). Последний подозреваемый: BK (порядок
  K-chunks) в MMQ. Цель: acceptance 0.40 -> 0.53+ (паритет с ROCm).

### B2 [OPEN] MTP device handoff для Vulkan (D094-c8j)
- h_nextn гоняется через host (D2H+H2D) каждый MTP-шаг, ROCm — device-to-device.
  Perf-only (acceptance не влияет). Быстрый выигрыш ~5-10% к MTP decode.

### B3 [ACTIVE] D098 native ROCm FP8 KV
- План и gate ladder: `major-topology/D098_Q4KM_ROCM_FP8_KICKOFF.md`.
- G1 byte-compatible HIP copy завершён (`3/3`), G2 default-off reference
  f8->f16 FA завершён (`2/2`), G3a native FP8 KQ и G3b native FP8 V rocWMMA
  завершены (Qwen D256 prefill/decode `2/2` каждая). Следующий этап — G4
  server smoke и соседние q8/F8 speed gates.
- До G4 нет speed/default claim: Vulkan, GUI и публичные пресеты не меняются.

### B4 [OPEN] Апстрим-синк
- fp8 изменения форк-локальные; текущий upstream shared CUDA/HIP layer не
  содержит `GGML_TYPE_F8_E4M3`. Адаптация потребуется при следующем мерже.

## Ресурсы
- Инструменты: C:/VulkanSDK/1.4.350.0 (glslc, spirv-as, spirv-dis, spirv-val).
- Ссылки: RESULTS_LOG D095-D2/D3, D094-c4/c5/c6, examples/vk_fp8_probe/.

## Checkpoint 2026-08-10 — P5 fp8 PV (raw f8 V): transform+runtime готовы, гейты PASS

- P5 = P4 + fp8 P*V: A=P fp8 (Psh8, dense [col*Br+row], bitcast encoder),
  B=V raw f8 из исходного кэша (без preconvert), acc f32. Transform
  `--pv-f8`: P-load→Psh8; V-load→data_v (f16 placeholder, элементы без /4);
  KMat/QMat holder-ретайп в fp8 (вместо bypass → без dominance-ошибок);
  мёртвый f16 kvsh store удалён. Audit fp8-p5 + spirv-val PASS.
  Runtime: FA_F8_P5 (env GGML_VK_FA_F8_P5, приоритет над P4/P3/P2),
  7 bindings, v_stride = nev0 (f8 байты).
- Замеры (49K lane, одна сессия, соседние пары):
  spec=none: P5 mean 1344 vs control mean 1286 = **+4.5% prompt** (2/2);
  MTP n=2: prompt +4.5%, decode 39.7 vs 34.3 = **+15.9%**, acc 53% vs 38% (2/2).
- Качество: идентично P4 (Paris/391/Eiffel связно, 0 HTTP 500).
- ВАЖНО: тепловой дрейф ~3% за серию прогонов (control r2 1295 → r3 1259) —
  P4 session-замер (1249) невалиден; чистый P5-vs-P4 A/B отложен на
  отдельную сессию. Оба пути в сессии на ~7% ниже вчерашних абсолютов.
- Следующие шаги: (1) чистый A/B P5 vs P4 (2 прогона каждого, свежая
  сессия, перемежающиеся), (2) 12K/98K масштаб для победителя,
  (3) решение о default.

## Checkpoint 2026-08-10 — P5 vs P4 A/B (undervolt): P5 +6.0% prompt — прогресс подтверждён

- Чистая перемежающаяся пара (P5 r1, P4 r1, P5 r2, P4 r2), свежая сессия,
  андервольт: P5 mean 1365.5 (1352.27/1378.68) vs P4 mean 1288.6
  (1264.41/1312.73) = **+6.0%** (обе пары в одну сторону), decode паритет.
- Вывод: полный нативный fp8 (S + PV) быстрее P4 (fp8 S + f16 PV);
  fp8 PV приносит ~6% поверх P4 на 49K lane. P5 — новый лучший путь.
- Расследование 2026-08-10: вчерашние "49K" замеры (D096-B/C) на деле были
  12K-масштабом (prompt_n=8601, ctx 49152, ub 256, tasks=quick, --no-reuse
  --no-mmap — проверено по jsonl/diagnostics) — "1496" = короткий префилл,
  не тот же лейн, что сегодня (40650 токенов, ctx 131072, ub 128). Загадка
  "андервольт не воспроизвёл 1496" закрыта: разница в масштабе/конфиге, не
  в железе. Относительные A/B (один лейн, перемежающиеся) валидны.
- Следующие шаги: (1) 12K/98K масштаб P5, (2) MTP A/B P5-vs-P4,
  (3) решение о default.

## Checkpoint 2026-08-10 — P5 на 12K (вчерашний конфиг): +11.5% vs вчерашний P4

- Вчерашний конфиг воспроизведён (tasks quick, ctx 49152, ub 256,
  --no-reuse --no-mmap, triage_diff 8601 токенов): P4 сегодня 1487.83 ≈
  вчера 1495.7 (воспроизводимость условий подтверждена).
- P5: 1671.49/1664.34 (mean 1667.9, разброс 0.4%) = **+11.5% vs P4-вчера**,
  +20.0% vs control-вчера. Масштаб-эффект: 12K +12% vs 49K +6%.
- Вывод: прогресс подтверждён и против вчерашних абсолютов (не только
  внутрисессионно). Осталось: 98K масштаб, MTP A/B, default-решение.

## Checkpoint 2026-08-10 — P5 vs q8_0 (дефолт): +14.0% prompt, KV −5.9%

- 12K, triage_diff 8601 токенов, перемежающиеся пары: P5 mean 1633.7
  (разброс 0.05%) vs q8 mean 1432.5 = **+14.0%** (2/2 в одну сторону).
- Память (ctx 49152): KV f8 1536 MiB vs q8 1632 MiB (−96 MiB, −5.9%);
  P5 дополнительно не держит f16 preconvert-буферы.
- D095-D3 был ПАРИТЕТ (1663 vs 1658) — P5 превратил его в +14%.
- Цепочка 12K: control-f8pre 1390 < q8 1432 < P4 1496 < P5 1634.
- Осталось: MTP A/B P5-vs-q8, 98K масштаб, default-решение.

## Checkpoint 2026-08-10 — Масштаб P5 vs q8: +14% (12K) → +43% (49K) → +45.5% (98K)

- Все лейны, spec=none, triage_diff, перемежающиеся пары, все в одну сторону.
- q8 деградирует с ростом KV сильнее P5 (preconvert q8->f16 на весь KV +
  dequant; P5 raw f8 без preconvert). Route trace q8 98K: coopmat1
  preconvert=1, split_k=4.
- MTP 12K: P5 decode +8.9% vs q8 (35.4 vs 32.5), acc 35% vs 31%.
- KV память: f8 на 96 MiB меньше (ctx 49152).
- Осталось: quality-гейт P5 vs q8 (длинная генерация/перплексия),
  затем default-решение. P5-код в рабочей копии не закоммичен.

## Checkpoint 2026-08-10 — GUI-доступ к P5 (сервер + автотюн)

- Чекбоксы GGML_VK_FA_F8_P5 добавлены: Server tab (Vulkan panel),
  Single Bench, Autotune; настройки сохраняются. KV f8_e4m3 был доступен
  ранее. Пользователь может гонять fp8-сервер и fp8-автотюн из GUI.
- Осталось: quality-гейт P5 vs q8, default-решение, коммит.

## Checkpoint 2026-08-10 — Decode: fp8-шаг = f16, MTP-разница = acceptance

- spec=none: 28.62 vs 28.82 t/s (паритет). MTP: f16 52.6/82% ≫
  P4 42.5/56% > ctl 38.3/45% > P5 31.6-39.1/27-45% ≈ q8 32.5/31%.
- Квантование KV ломает nextn-acceptance (независимо от пути FA).
- Решение для продукта: f16 KV для интерактивного MTP; f8 P5 для
  prefill-интенсивных/памяти. Open: f16 KV на слое nextn (пер-слойный
  тип) — кандидат на следующий эксперимент.

## Checkpoint 2026-08-10 — D096-K: гибридный KV закрывает decode-разрыв

- Реализовано: `LLAMA_VK_MTP_KV_LAST_F16=N` (f16 для последних N KV-слоёв).
- Кривая acceptance: 27% (f8) → 52% (1) → 67% (4) → 75% (8) → 82% (16=f16).
- Decode: 30.5 → 41.0 → 46.8 → 48.3 → 54.0 t/s. LF8: +58% vs f8, -8% vs f16,
  KV 2304 vs 3072 MiB (-25%). Рекомендация: N=8 для MTP-интерактива (или N=4 при жёстком VRAM).
- Open: N=2/3 точка, пер-слойная точность V-only (K f8 + V f16), GUI-опция,
  дефолт для MTP-режима сервера.

## Checkpoint 2026-08-10 — D096-K/L: дефолт закреплён, энкодер оптимизирован

- DONE: гибридный KV = дефолт при MTP+f8 (v153); битовый E4M3-энкодер (v154).
- Свежий MTP-дефолт: dec 51.7 t/s, acc 80% (vs f8-чистый 30.5/27%).
- 49K prefill P5: 1399 pt/s — впервые выше f16-прогона при 2x меньшем KV.
- Open: GUI-настройка N (сейчас env/дефолт), N=2/3 точка, V-only гибрид
  (K f8 + V f16), долгие генерации fp8 vs f16 (perplexity), P2/P5 empty-content
  с finish=length (thinking-модель тратит лимит на reasoning).

## Checkpoint 2026-08-10 — D096-M: GUI готов (P5 без чекбоксов)

- DONE: чекбоксы P5 убраны; f8 KV в GUI (Server и Bench/Autotune) включает
  нативный путь автоматически. Сервер через GUI поднимается (smoke PASS),
  prefill 1215 pt/s на GUI-дефолтах (8K ctx, Vulkan0,Vulkan1), decode ~31.
- Open: порядок устройств GUI по умолчанию Vulkan0,Vulkan1 (рекомендация
  Vulkan1,Vulkan0 — настройка в панели), GUI-настройка N гибридного KV.

## Checkpoint 2026-08-10 — D096-P: гибрид расширен на q8_0 (регрессия закрыта)

- DONE: LLAMA_VK_MTP_KV_LAST_F16 теперь работает для f8_e4m3 И q8_0
  (K=V). q8+MTP: acceptance 31% → 74%, decode 30.5 → 51.7 t/s (GUI-конфиг).
- DONE: q8/f8/f16 «мусор/пустые ответы» диагностированы как артефакт
  thinking-модели + лимита (n_predict 128→1024 в GUI), не KV-баг (D096-N/O).
- Open: прежние пункты (N=2/3, V-only гибрид, GUI-настройка N, perplexity).

## План (долгий ран) — цель: fp8 быстрее f16 в decode и prefill

### База (свежие замеры 2026-08-10, честный A/B)
- **A/B 12K-лейн, MTP n=2, соседние прогоны, одна сессия**
  (d096-ab-mtp-f8hyb-vs-f16 / d096-ab-f16-mtp-ctl):
  | | f8-гибрид N=8 | f16 | зазор |
  |---|---|---|---|
  | prefill | 1618.7 pt/s | 1673.5 | **-3.3%** |
  | decode | 51.7 t/s | 58.1 | **-11%** |
  | acc (сегменты) | 70%/80% | 82%/98% | 75% vs 90% |
- Зазор decode раскладывается: ~8% = acceptance (75% vs 90%; дравт-голова
  зависит от скрытого состояния ВСЕХ слоёв — f8-шум ранних слоёв снижает
  качество дравта, хотя KV последнего слоя f16); ~2-3% = сами f8-слои
  (декант без аппаратных dot).
- Decode spec=none (один token; Qwen GQA remap до tuning `N=6`, scalar FA,
  короткий промпт): f8 30.4 vs f16 29.5
  vs q8 30.2 — ПАРИТЕТ (декод-путь f8 не хуже f16 без MTP).
- Prefill spec=none (короткий): f8 171 vs f16 160 pt/s (+7%, шумно).
- «Prefill отстаёт» в GUI-ощущениях = image-запросы (180 t/s с 1872 image-
  ток + 2150 ms энкодер) и/или MTP windowed-prefill; чистый текст ~паритет.
- Default decode скалярный; аудит route trace 2026-08-13 уточнил причину:
  Qwen GQA remap даёт `N=6`, а staged f8 cm1 не проходит LDS gate и падает
  в scalar. f8-декант — конверсия fp8→f32 без шкалы (raw f8), q8-декант —
  int8→f32 × block-scale.

### Факты 2026-08-12, переопределившие план
- **f8_e4m3 KV здесь — RAW fp8 без блочных шкал** (`ggml.c`: blck_size=1,
  is_quantized=false; `dequant_funcs.glsl`: dequantize4 = чистый
  fp8_e4m3_to_f32, get_dm=1.0). Стоимость хранения **1.0 B/элемент** vs
  q8_0 1.125 (−11% чтений K/V) — это и есть источник выигрыша P5 на
  префилле (+14/43/45% на 12K/49K/98K).
- Обратная сторона: **без нормализации теряются малые KV-значения**
  (e4m3 min normal 0.0156) → точность хуже q8 → низкий acceptance дравта
  (f8-last 27% vs q8-last 31%) и 85.5% vs ~90.6% при гибриде N=8.
  R9 (per-(token,kv_head) f16-scale на 256 значений) исправляет точность K
  (offline logit MSE −20.52%, метаданные +0.78%, суммарно 1.0078 B/элемент —
  всё ещё −10.4% vs q8) — при этом шкала факторизуется из P*V (умножение
  score-column после Q*K).
- **D1 закрыт статически**: RDNA4 (gfx1201) не имеет скалярных аппаратных
  dot-инструкций (V_DOT2_F32_F16 — CDNA; VK_KHR_shader_integer_dot_product
  на AMD RDNA — нет → MMQ int8-путь на этом железе не компилируется; q8-декод
  идёт тем же скалярным dequant-путём). Аппаратная fp8-математика на RDNA4 —
  только через coopmat/WMMA. Замена D1 — R9-K (D4.3) и coopmat (D4.2).
- **D2/D3/D4.1 пересмотрены 2026-08-13 после полного reboot и аудита env**.
  Бенч-раннер наследовал, но не сохранял в diagnostics экспериментальные
  `GGML_VK_FA_F8_NATIVE=1` и `GGML_VK_FA_F8_NATIVE_DECODE=1`.
  Поэтому серии 12K/49K с результатом ~13.4 tps фактически измеряли
  opt-in `f8_native` cooperative-matrix decode, а не default scalar/direct.
  Изменения load-width/LUT/staging/Bc в этих сериях не были причинным тестом
  scalar-кернела; вывод «raw-f8 reads в 2 раза медленнее» отозван.
- **Свежий adjacent A/B после reboot, Q4_K_M, Vulkan1,Vulkan0, spec=none,
  ctx49K, b512/ub512, 30.2K prompt tokens**:
  default f16 = 27.18/27.11 decode t/s; default f8 scalar = 25.45/25.03
  decode t/s (−6.4…−7.7%, а не −51%). Принудительный
  `GGML_VK_FA_F8_NATIVE_DECODE=1` воспроизводит 13.28 t/s (−47.8% к
  default f8) и route-trace подтверждает `path=f8_native`, `Br=16`,
  grouped-query `N=6`. Узкое место исторической серии — неподходящая
  cooperative-matrix геометрия для single-token decode.
- Ранний preconvert f8→f16 результат 26.2 t/s не доказывал дорогие raw-f8
  чтения: он просто выводил decode из случайно принудительного native-path.
  Default scalar f8 уже близок к f16 без полноразмерной f16-копии KV.
- Бенч-раннер теперь печатает и сохраняет активные GGML/LLAMA/HSA performance
  env (секретные значения редактируются), а runtime один раз предупреждает
  при включении diagnostic native-f8 decode.
- **Свежая соседняя пара default f8 vs q8 закрыта** (`s5r5/s5r6`, один
  `triage_diff`, 30 187 prompt + 256 decode): q8 = 1445.81 pt/s, 27.08 t/s;
  default f8 = 1445.58 pt/s, 25.56 t/s. Prefill в паритете, f8 decode
  отстаёт всего на 5.6%. Route trace подтверждает default f8
  `path=scalar, N=6, Br=8, shmem_staging=1`.

### Гипотезы и направления (приоритет сверху вниз, ред. 2026-08-12)
1. **D4 — нативный f8-декод и точность K** (заменяет старый D4 «N-кривая»;
   D4-шаги, каждый с gate):
   - **D4.1 (исправлен, 2026-08-13)**: default scalar f8 на 49K даёт
     25.45/25.03 t/s против f16 27.18/27.11. Исторические ~13.4 t/s —
     forced-native decode; вывод о 2× стоимости scalar raw-f8 отозван.
    Свежий speed-gate закрыт: default f8 25.56 против q8 27.08 t/s (−5.6%),
    при практически идентичном prefill 1445.58/1445.81 pt/s.
   - **D4.2 (закрыт для текущей геометрии)**: forced native coopmat decode
     даёт 13.28 t/s против 25.45 scalar на 49K. Не продолжать 98K sweep,
     пока нет нового small-row cooperative дизайна, устраняющего
     `Br=16`/grouped-query `N=6` недоиспользование.
   - **D4.3 (код, крупный)**: R9 runtime sidecar — per-(token,kv_head)
     f16-scale на 256 для K (V остаётся raw f8): (а) KV-жизненный цикл:
     энкодер (max-reduce 256 + деление + запись шкалы), SET_ROWS/copy/views
     (2-буферный K или scale-хвост), MTP-окно; (б) FA: S×scale после Q*K —
     декод-петля (1 FMUL на пару вместо 64 per-элементных масштабов) и
     P5-префилл (score-column multiply до softmax); (в) A/B: точность
     (acceptance 12K MTP vs q8-гибрид). Gate: acceptance(12K MTP, N-хвост)
    ≥ q8-гибрид. R9 остаётся направлением точности/acceptance; прежнее
    обоснование через «2× стоимость scalar raw-f8» отозвано (см. D4.1).

### D4.3 R9 — дизайн реализации (2026-08-12, без GPU)

Формат: K-блок = 256 значений f8 (нормализованы на блок) + 1×f16 scale.
Стоимость: 258/256 = 1.0078 B/элемент (−10.4% чтений K vs q8_0 1.125).
Qwen3.6-27B: n_embd_k_gqa = 1024 → 4 блока по 256 на строку (токен, все
kv-головы). Шкала применяется к score-column ПОСЛЕ Q*K (факторизуется из
P*V): S_raw = Q·K_norm, затем S = S_raw × scale[j] перед softmax; в декоде
это 1 FMUL на (q,k)-пару вместо 64 per-элементных масштабов dequantize4 —
одновременно дешевле И точнее (bias-смещение убирается).

Точки изменения (порядок реализации):
1. **Хранение**: K-буфер остаётся 1 байт/элемент (f8-данные НЕ включают
   шкалу); scale-буфер — отдельный ggml-тензор-спутник рядом с cache_k
   (layout: n_embd_k_gqa/256 × kv_size × n_stream, f16). Все операции
   lifecycle (SET_ROWS, copy, batch-views, MTP-окно, host-KV) дублируются
   на scale-тензор — это основная трудоёмкость (2-буферный K).
2. **Энкодер** (KV-запись f8): добавить max-reduce на 256-блок и запись
   шкалы; данные пишутся как x/scale (нормализация) вместо raw. Замена
   per-32-логики квантования f8-эка (f8 не квантованный — encoder пока
   просто кастует f32→fp8; добавить normalize+scale).
3. **FA-декод (скалярный)**: K-декант без шкалы (уже так), после KQ-цикла
   Sf[r][c] *= scale[j]; scale-чтение: 1 f16-загрузка на (c)-блок. Новый
   дескриптор BINDING (scale_k) во всех FA-путях, которые используют R9-K
   (или переиспользовать свободный слот дескрипторов — проверить маску
   на декоде: на декоде mask-дескриптор свободен).
4. **P5-префилл (cm1/fp8_fa_cm1)**: после KQ-коопмата S *= scale перед
   max/softmax; шкалы грузятся в shared один раз на CTA.
5. **Dispatch/pipeline**: новый tuning-флаг (path R9) или env
   GGML_VK_F8_K_SCALE=1; assert типов; дескрипторный сет (7→8).
6. **Включение**: opt-in env `LLAMA_VK_F8_K_SCALE` (кэш) + `GGML_VK_FA_F8_K_SCALE`
   (FA) — до полного A/B; после gate — дефолт для f8-слоёв.

Риски: (1) дескрипторный сет FA расширяется (7→8) — затронет все
FA-пути; (2) 2-буферный lifecycle — источники утечек рассинхрона
(шкалы без данных и наоборот); (3) гибридные слои (f16-хвост) не имеют
шкал — dispatch должен знать тип слоя; (4) префилл-энкодер получает
доп. проход — +0.3-0.5% времени prefill (приемлемо).
Решение по пункту 3 в момент реализации: R9-формат включается ТОЛЬКО
для чисто-f8 слоёв (не для f16-хвоста), шкалы существуют всегда при
f8-типе K (не плодить 3-й режим).
2. **D5 — гибриды MTP: какая компонента диктует acceptance** (расширенный;
   выполняется ПОСЛЕ D4.3 — V-only без точного K бессмыслен):
   - **D5.1**: отладка существующего V-only WIP (K f8 + V f16 последние N;
     сейчас 0% acceptance — вероятен шейдерный баг vf16-пути или K-шум):
     (а) spec=none smoke с V16=8 — связность ответов (баг vs шум);
     (б) с K+R9 (после D4.3) — MTP-замер acceptance. Ожидание: если K
     точен (R9), V f8 vs V f16 на last-слое мало влияет → V-only
     отклоняется в пользу полного гибрида или K-only.
   - **D5.2**: K-only гибрид (K f16 + V f8 последние N) — зеркальный к
     V-only: тип V f8 сохраняет bandwidth-выигрыш хвоста; реализация:
     kv-cache (type_k_il=f16, type_v_il=f8) + FA-пути: скалярный декод
     (K f16-чтение существует, V f8-dequant существует, расширить assert
     на (f16,f8)) и coopmat-префилл (новый cm1-вариант: K f16 + V raw f8,
     по образцу P5). A/B acceptance vs full-hybrid N=8.
   - **D5.3 (бенч)**: N-кривая лучшего гибрида (N=0/4/8/12, 12K MTP n=2):
     acceptance + decode; цель N минимальное при acc ≥ q8-уровень.
   - **D5.4 (бенч, финал)**: итоговый f8-стек vs q8-стек на 12K/49K/98K,
     spec=none и MTP n=2: P5 + D4-декод + R9-K + лучший гибрид vs
     q8-полный стек. Критерий закрытия «fp8 нативный и > q8»: prefill > q8
     (уже есть), decode ≥ q8 (цель: >, от D4.3 bandwidth), acc ≥ q8-гибрид.
3. **D6 — дравт-префилл f8 (nextn-слой)** — переопределён: статические
   f8-типы на nextn-слое ломают acceptance (27-56%); осмыслен только
   динамический bridge (f8 во время prefill → конверсия в f16 перед
   декодом) — отдельный проект после D5.4, не входит в текущий спринт.
4. **D7 — prefill-мелочи** — после D5.4: KV-запись одним проходом
   (copy_to_quant) и wg-размеры P5 на 49K/98K; target +3-5%.

### Процедура A/B (правила perf-workspace)
- Свежая сессия, соседние прогоны, f16-контроль рядом; 49K-лейн
  (147456 chars), 12K-лейн для быстрых итераций; MTP-замеры с draft
  acceptance; thermal drift — не гнать серии >4 прогонов.
- Метрики: prompt_per_second, predicted_per_second, draft_n_accepted/draft_n,
  KV MiB; запись в RESULTS_LOG/BENCH_RUNS.csv.

### Долгие хвосты (не скорость)
- Perplexity fp8 vs f16 на длинных генерациях (качество, вне speed-плана).
- P5 на 98K с новым энкодером (перемерить при следующем 98K-ранe).

## Checkpoint 2026-08-11 — D095 R1-R5: дешёвые FP8/MTP пути исчерпаны

- Direct scalar f8, raw-f8 small-N coopmat1 и альтернативное размещение f16
  слоёв не дали выигрыша; все default-off probes удалены.
- Robust two-task sweep: N=8 = 1605.79/55.49, 85.5%, 2304 MiB; полный f16 =
  1627.67/58.81, 89.6%, 3072 MiB. Реальный остаток: -1.3% prefill и -5.6%
  decode при -25% KV, а не ранние -3.3%/-11% из single-task пары.
- MTP n=3/n=4 отклонены: acceptance 49.0%/58.6%, decode 45.13/48.81;
  n=2 стабилен 55.62/54.84 и остаётся дефолтом.
- Следующий активный gate: D095 R6 block-scaled E4M3. Сначала offline-анализ
  реальных K/V блоков; только затем новый формат/шейдер. Цель — поднять draft
  acceptance >=85.5% при <=1.03125 bytes/value и превысить 55.49 tok/s.

## Checkpoint 2026-08-11 — D095 R6: block-scaled E4M3 отклонён

- Новый standalone scout снял полные K/V слоёв 3/7/11 на двух реальных
  prompt (8570/8551 токенов); целостность capture PASS.
- Weighted attention-logit MSE: raw E4M3 `0.0030764424`, block-32 E4M3
  `0.0030764146`: улучшение лишь 0.0009% при +3.125% памяти. Причина:
  power-of-two scale сдвигает exponent, но не увеличивает 3-bit mantissa.
- q8_0 при 1.0625 bytes/value дал `0.0001028896` (-96.66%), поэтому следующий
  gate R7 — block-floating int8 с B32/int8 exponent (1.03125 bytes/value).
  Сначала offline precision; runtime допустим только с raw tile dequant без
  q8 whole-KV preconvert, закрытого D096-H как длинно-контекстная регрессия.

## Checkpoint 2026-08-11 — D095 R7: BFP8 точен, но runtime закрыт

- Precision PASS: BFP8 B32 weighted logit MSE `0.0002514458` (-91.83%),
  worst pair ratio `0.0939`, storage 1.03125 bytes/value.
- Runtime admission FAIL: q8_0 в 2.44x точнее при +3.03% storage; raw
  int8/q8 scalar+coopmat уже был медленнее 22-25% из-за int32 VGPR/occupancy,
  а q8 preconvert улучшил 131K `54.1s -> 47.8s` (D094 cycle 5/6).
- BFP8 type/shader не создаётся. R8 проверяет последний offline upper bound:
  E4M3 с general f16 scale `max/240` на B16/B32/B64; gate >=25% logit-MSE.

## Checkpoint 2026-08-11 — D095 R8: precision PASS, symmetric runtime FAIL

- General-scale E4M3: B16 -45.30% weighted logit MSE, B32 -36.66%, B64
  -29.07%; numerical upper bound подтверждён.
- Runtime закрыт: K scale можно применить к partial score, но V scale меняется
  внутри P*V reduction; она требует повторной P-квантизации на D-block или
  f16 V preconvert и теряет механизм P5.
- R9 проверяет factorable K-only B256: одна f16 scale на token/head,
  V=raw E4M3, средний overhead K+V 0.390625%; gate >=15% logit-MSE.

## Checkpoint 2026-08-11 — D095 R9: factorable precision PASS

- K-only general-scale B256: weighted logit MSE `0.003076 -> 0.002445`
  (`-20.52%`), worst pair ratio `0.8737`; gate 15% пройден без локальных
  регрессий. Metadata K 0.78125%, средний K+V overhead 0.390625%.
- V остаётся raw P5 E4M3; scale применяется после Q*K к готовому score column,
  поэтому R8 P*V blocker отсутствует.
- Runtime sidecar не смешивается с текущим benchmark refresh. Следующий
  отдельный prototype должен реализовать полный KV lifecycle и превысить N=8
  `1605.79/55.49`, acceptance 85.5%; до этого speed/default claim запрещён.
- D095 R1-R9 prebuild-план завершён. Профиль до 49K: P5 + hybrid last-8 f16
  + MTP n=2; D097 отдельно исправляет 98K.

## Checkpoint 2026-08-11 — D097: 98K acceptance восстановлен

- Причина не в регрессии P5: raw E4M3 на реальных KV имеет logit MSE в 29.9x
  выше q8_0. q8_0 точнее благодаря int8 + scale/32, несмотря на имя FP8.
- Длинные N8 controls детерминированы: f8 `140/230=60.87%`, q8
  `151/208=72.60%`. N12 возвращает `152/206=73.79%`.
- Финальный bracket: q8-center `1422.71/41.96/5.4736`; f8 N12
  `1510.95/41.79/5.7618` — prompt +6.20%, decode -0.41%, aggregate +5.27%,
  acceptance +1.19pp. Цена: 5376 vs 4704 MiB KV.
- Дефолт: f8+MTP при ctx>=98304 автоматически N12; q8 и короткий f8 остаются
  N8. Env `LLAMA_VK_MTP_KV_LAST_F16` полностью переопределяет политику.
- Q8 bridge M6 даёт 90.61%/47.05 t/s при 4680 MiB, но prompt лишь +0.33%; он
  сохранён default-off как generation-heavy research profile.

## Checkpoint 2026-08-12 — инвентарь нативности fp8 (Vulkan) + decode ROCm vs Vulkan

### Что НЕ нативно (открытые пути, по приоритету)

1. **Default f8 decode — FA_SCALAR после GQA remap `N=6`**. Аудит
  2026-08-13 показал `Br=8, shmem_staging=1`; прежнее описание `n_rows==1`
  и «без переиспользования» было неверным для Qwen. Свежий 49K default:
  25.56 t/s против q8 27.08.
2. **D3 (закрыт 2026-08-13)**: forced native coopmat decode `N=6, Br=16`
  дал 13.28 t/s, на 48.0% медленнее default scalar. Оставлен diagnostic-only.
3. **D6 (open)**: MTP KV (слой nextn) всегда f16; дравт-префилл f8 не сделан.
4. **D5 (open)**: V-only гибрид (K f8 + V f16 последние N слоёв).
  Production hardening 2026-08-13: диапазоны `V_F16` и `LAST_F16`
  вычисляются независимо, зажимаются числом реальных KV-слоёв и безопасно
  комбинируются с Q8 bridge; режим остаётся default-off до quality/perf gate.
5. **R9 (open)**: factorable K-only B256 prebuild PASS (logit MSE −20.52%),
   runtime sidecar (полный KV lifecycle) не реализован; gate: превысить N=8
   1605.79/55.49, acc 85.5%.

### Лишняя работа (кандидаты)

- Default Qwen GQA decode после remap идёт как scalar `N=6, Br=8,
  shmem_staging=1`; повторное использование между шестью Q-головами делает
  тезис «staging при rows=1 бесполезен» неприменимым к этой форме.
- `GGML_VK_FA_F8_DIRECT=1` — широкий route-probe, а не чистый scalar A/B:
  на decode он открыл cm1 `N=6, Br=16, shmem_staging=0`. Свежий 49K замер
  дал 25.83 t/s (паритет с default 25.56), но prefill упал
  1445.58→1192.90 pt/s (−17.5%). Не включать по умолчанию.
- Preconvert-f16 fallback (dequant_f8_e4m3.comp) — нужен как fallback, не лишний.
- fp8_fa_cm1.spvasm (ручной клон) — диагностический архив, не production.

### Decode ROCm vs Vulkan (Qwen3.6-27B-Q4_K_M, spec=none, 12K lane)

| backend | f16 | q8_0 | f8_e4m3 |
|---|---|---|---|
| Vulkan | 29.5 t/s | 30.2 t/s | 30.4 t/s |
| ROCm   | 24.2-24.5 | 23.3-23.6 | 22.0-22.1 |

ROCm decode на 20-25% ниже Vulkan (f16: 24.2 vs 29.5). Причины (гипотезы по
коду): (1) ROCm декод идёт через wmma-кернел с ncols=16 — для 1 строки Q
15/16 вычислений впустую + полные KQ/V циклы с барьерами; (2) Vulkan Qwen
GQA remap использует scalar `N=6, Br=8` (d_split, staging).
ROCm MTP decode: свежие числа отсутствуют (последние MTP-замеры — Vulkan
49K n=2: 51.7-55.5 t/s; 98K n=12: 41.8-42.8).

## Checkpoint 2026-08-12 — D2/D3 результаты отозваны аудитом 2026-08-13

Коммит `e2121d64f`. Исходная таблица сохранена как историческая, но не должна
использоваться для выводов: persistent terminal наследовал
`GGML_VK_FA_F8_NATIVE=1` и `GGML_VK_FA_F8_NATIVE_DECODE=1`, а раннер тогда
не сохранял active env в diagnostics.

| lane | direct (D2) | staging (контроль) | Δ decode |
|---|---|---|---|
| 12K r1 | 1685.1 ptps / 20.14 tps | — | — |
| 12K r2 | — | 1703.7 / 19.39 | — |
| 12K r3 | 1735.2 / 20.12 | — | **+3.8%** (стабильно, 2 прогона) |
| 49K r4 | 1550.8 / 13.15 | — | — |
| 49K r5 | — | 1566.6 / 13.26 | паритет (−0.8%, шум) |

**D2 — старый вывод отозван.** На Qwen single-token GQA исходный `neq1=1`
до окончательного tuning remap превращается в `N=6`; чистый default trace
показывает scalar `Br=8, shmem_staging=1`. `GGML_VK_FA_F8_DIRECT=1` влияет
не только на scalar staging: он отключает preconvert на prefill и делает
cm1 `Br=16, shmem_staging=0` допустимым на decode. Свежий s5r6/s5r7:
default 1445.58 pt/s, 25.56 t/s; direct 1192.90 pt/s, 25.83 t/s. Decode
паритет, prefill −17.5%; route-probe остаётся default-off.

**D3 — coopmat1 f8-native на decode окончательно отклонён**: чистый forced
native s5r4 = 13.28 t/s против default scalar s5r6 = 25.56 t/s (−48.0%).
Route trace: native `N=6, Br=16`; default scalar `N=6, Br=8`. Старое
«паритет 13.27 vs 13.15» сравнивало два прогона с одним и тем же скрытым
native route. Opt-in оставлен только для диагностики с runtime-warning.

Закрытие аудита:
- не продолжать D2/D3 sweep на 12K/98K с текущей геометрией;
- раннер сохраняет active GGML/LLAMA/HSA/ROCm/HIPBLASLT env и редактирует
  возможные credentials/endpoints/account identifiers;
- следующий содержательный speed-вопрос — только новый small-row cooperative
  дизайн; R9 остаётся отдельной гипотезой точности/acceptance.
