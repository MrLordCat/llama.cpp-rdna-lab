# D096 — карта версий fp8_fa_cm1.spvasm (vN)

Журнал изменений и результатов по версиям. Обновлять ПОСЛЕ каждого vN
(правка → валидация → сборка → тест → запись). Дата: 2026-08-09.

Тестовая lane: Vulkan, Qwen3.6-27B-Q4_K_M, K/V=f8_e4m3,
`GGML_VK_FA_F8_NATIVE=1 GGML_VK_FA_HALF_CMP=8 GGML_VK_FA_OUTDUMP=1`,
2 запроса ("Say hello in one sentence."), сравнение outdump-файлов
(`fa_out_native_<n>.bin` / `fa_out_fallback_<n>.bin`, 73728 f32 = 24 головы × 3072).
Сервер: `-dev Vulkan1,Vulkan0` (основная — вторая GPU, правило пользователя).

---

## v107 (2026-08-08)
- HALF_CMP=8: первые 8 prefill-вызовов native, остальные fallback (один процесс).
- Результат: только 2 из 8 native-вызовов дали полные выходы (4 и 6) — мало данных.
- Урок: нужен чередующийся выбор по запросу, а не по числу вызовов.

## v108 (2026-08-08)
- Диспетчер: `(seq/16)%2` в `vk_dispatch.inc` — запрос 1 native, запрос 2 fallback.
- Результат: полные слои расходятся (min diff 0.47–1.48); слой 1: native rms 0.94
  («острый») vs fallback 0.04 («мягкий»); 4-головочные вызовы совпадают (≈0.078).
- Вывод: S-секция для 4-головочных численно корректна; полные — нет.

## v109 (2026-08-08)
- Исправлена V-раскладка (kvsh): зоны `(vs_r>>4)*256 + htv*64 + (vs_r%16)*4 + vs_c/4`,
  V-load база `bcv*256 + htv*64 + sg*16`; K переведён на direct global load (kbc=false).
- Результат: 4-головочные по-прежнему совпадают (0.078); полные расходятся
  (min 0.33–0.64).

## v110 (2026-08-09)
- Q/K → f16 (sh_qq/sh_k16), S-scale = pscale (убрана maxQ-нормализация).
- Результат: полные расходятся (min 0.34–0.47); N0 = нули; N31 = все NaN.
- Урок: гипотеза «fp8-квант Q виноват» отвергнута.

## v111 (2026-08-09)
- Layout-фиксы по модели (A) (16×16 на СГ): Q-запись `dv*256+r*16+dm4*4`,
  K-индекс ksv, базы kmat1s sg*256 / qmat1 dv*256 (f16-индексы).
- Результат: расходятся (мин 0.255, N9–F21). По головам: 9 голов БИТ-В-БИТ
  совпадают (h0, h1, h6, h7, h12, h13, h17, h18, h23), остальные — нет.
- ВЫВОД (важный): совпавшие головы — плоское внимание (нечувствительны к K);
  у расходящихся F-строки 6..11 часто = 0 (недописанные fallback-слоты).

## v112 (2026-08-09)
- Расширен pcg-дамп push-констант (129024+): nek2, nev2, nb12, nb22, nb11, nb21.
- Push-константы полного prefill: neq2=24, nek2=4, nev2=4, gqa=1, nb11=1024,
  nb12=256 → KV-индексация ПРАВИЛЬНАЯ (модель имеет 4 KV-головы: rk2=6, rv2=6).
- S-зона kdump (первый тайл): ТОЛЬКО S[0][0]=1.0, всё остальное 0.0, M=0,
  PVSH/KVSH частично ненулевые (V-стаг работает). K-загрузка/умножение сломаны.
- Диагноз: f16-коопмат не читает мои sh_k16/sh_qq (схлопнут до [0][0]).

## v113 (2026-08-09)
- sh_qq/sh_k16 → v4f16-массивы (1024), коопмат-load через p_sh_v4f16
  (как в работающем V-пути), записи двухиндексные (слот, компонент).
- Результат: S-зона БЕЗ ИЗМЕНЕНИЙ (S[0][0]=1.0, остальное 0).

## v114 (2026-08-09) — контроль muladd
- K=1.0, Q=1.0 константами (вместо загрузок). Правильный S = 16.0 везде.
- Результат: S[0][0]=1.0, остальное 0 → muladd/раскладка ломаются при
  загрузке, а не данные.

## v115 (2026-08-09) — ЕДИНАЯ РАСКЛАДКА 4×16-СГ (по работающему V-пути)
- Ключевой вывод: f16-коопмат на RX 9070 XT: СГ владеет 4×16 (64 f16),
  зона СГ = 64 f16 (база sg*16 v4-слотов), НЕ 16×16 (256 f16).
  V-путь (bc_q/bc_k) подтверждает: база `bcv*256 + htv*64 + sg*16` (v4),
  элемент (r,c) = слот sg*16 + r*4 + c/4 (r=0..3).
- Правки:
  1. kmat1s base: sg*64 → sg*16 (v4-слоты); элемент (r,c): sg*16 + r*4 + c/4.
  2. qmat1 base: dv*64 → dv*16 (v4-слоты).
  3. K-запись: слот ksv/4, компонент ksv%4 (f16-индекс ksv, kv-строка jc+ksv/16) — уже так.
  4. Q-запись: f16-индекс = dv*64 + (t/4)*64 + (t%4)*16 + (d%4)*4 + k
     (t = q2_r = i/64: токен; СГ = t/4; r = t%4); слот = индекс/4.
  5. Q-zero: та же формула.
  6. S-STOR: st_sg = sg*4 (было sg*16) → st_o = sg*64 f32 (4×16-подтайл на СГ);
     softmax-чтение (us_c*16+us_r) совпадает: kv = sg*4 + r.
  7. Возвращены реальные загрузки K (LUT f8→f16) и Q (FConvert f32→f16),
     контрольные константы v114 убраны.
- Сборка: spirv-as+val → NO-LLAMA. Тест: kdump S-зона должна иметь ненулевые
  колонки 0..11 (kv-строки 0..11), маска -inf в колонках 12..63.

## v115 (2026-08-09) — раскладка 4×16-СГ (по работающему V-пути)
- Гипотеза: f16-коопмат: СГ владеет 4×16 (64 f16), зона sg*16 v4-слотов.
- Правки: базы kmat1s/qmat1 sg*16/dv*16 (v4), Q-запись (t/4)*64+(t%4)*16,
  S-STOR sg*64. ВОЗВРАЩЕНЫ реальные K/Q (убран v114-контроль).
- Результат: S БЕЗ ИЗМЕНЕНИЙ (S[0][0]=1.0, остальное 0) → гипотеза неверна.

## v116 (2026-08-09) — K column-major (как bc_k: %int_1)
- Правки: kmat1s/kd → ColumnMajor; K-запись col-major-слоты (d%16)*16+(kv%4).
- Результат: S БЕЗ ИЗМЕНЕНИЙ (S[0][0]=1.0) → не layout-вопрос K.

## v117 (2026-08-09) — Q d-мажорная раскладка (строки B = d)
- Правки: Q-запись: слот dv*16 + (d%4)*16 + k*4 + t/4, комп t%4.
  Сборка: 2 фикса дублей id (q2_s3 → sb3, s1/s2 → z1/z2), метка q2_zero.
- Результат: S БЕЗ ИЗМЕНЕНИЙ (S[0][0]=1.0) → не layout-вопрос Q.

## КЛЮЧЕВОЕ ОТКРЫТИЕ (после v117)
S[0][0]=1.0 = 16×0.0625 — это ПРАВИЛЬНО масштабированный (us_sq) результат
свёртки при K=Q=1: muladd для [0][0] работает, остальные элементы = 0.
Сравнение с GLSL-эталоном (flash_attn_cm1.comp): S-стадия использует
kvsh_stride=6, Qf-раскладку [токен*66+d], K: RowMajor offset sg*96 stride 6,
Q: ColumnMajor offset d*4 stride 66, S-Store sg*256 — НЕСОВМЕСТИМО с
подтверждённой V-семантикой (r*16+c)/4. Вывод: НА ЭТОМ ДРАЙВЕРЕ fp8- и
f16-коопмат-загрузки имеют РАЗНЫЕ семантики раскладок; f16-путь (v110+)
принципиально не сходится. РАБОЧИЙ ПУТЬ = fp8-коопматы (v57: 4-головочные
давали 0.078!).

## v118 (2026-08-09) — ВОЗВРАТ fp8-пути v57 (рабочие загрузки)
- Правки (S-стадия целиком по v57): ks-цикл → kstg[ksv] (fp8, без LUT);
  kmat1s: cm_f8a, sh_kstg, base sg*16, stride 16, RowMajor; k_direct: fp8 из
  global (kd_row = jc+sg*16, stride k_stride); qmat1: cm_f8b, sh_qf8, base
  dv*16, stride qstride8(264), ColumnMajor; q2-запись: qf8[r*qstride8+d*4+k]
  (битовый f32→f8-энкодер v57); S-STOR: sg*256 (RowMajor, stride 16).
- Сборка: VALID, NO-LLAMA. Результат: S ПО-ПРЕЖНЕМУ S[0][0]=1.0, остальное 0,
  M=0. → ПРОБЛЕМА НЕ В РАСКЛАДКЕ ЗАГРУЗОК.

## v119 (2026-08-09) — kbc=false → k_direct (рабочий путь v108-109)
- Найдено: у v57 `kbc = (sc_Clamp != 0)` → для FA без Clamp = false (прямые
  глобальные fp8-загрузки K, `k_direct`); у меня было жёстко true (kstg-путь).
- Правка: `%kbc = OpCopyObject %bool %bool_false`. Сборка VALID, NO-LLAMA.
- Результат: S-зона БЕЗ ИЗМЕНЕНИЙ (S[0][0]=1.0, остальное 0, -inf: 0, M=0).
- Наблюдение: -inf в S-зоне НЕТ ВООБЩЕ — маска (kv≥N → -inf) не видна в
  дампе; либо дамп снимается ДО маск-add, либо маска не применяется.
  Маск-add-секция (ml) пишет ALiBi-маску из m_buf в mc0_p; -inf-маскирование
  kv≥KV выполняется в `ma_body` (ма_селект по ma_mf) и в softmax
  (`sm_sk = kbc && (j*Bc+col_tid >= KV)` — при kbc=false ВЫКЛЮЧЕНО; зато
  ma_селект сработает, если mask_opt записал -inf в mc0_p).

## ОТКРЫТЫЕ ВОПРОСЫ (пауза по просьбе пользователя 2026-08-09)
1. S[0][0]=1.0 (16×0.0625) при любых раскладках (f16 v112-117, fp8 kstg v118,
   fp8 k_direct v119) — свёртка для [0][0] работает, остальные 0.
   Гипотезы: (а) coopmat-load читает только первую строку/первый слот;
   (б) muladd-редукция/распределение ланов неверно; (в) дамп-зона читает
   не тот слот S-массива.
2. Куда именно пишется dbg1 (до/после маск-add) — проверить порядок в
   us-секции; -inf=0 намекает, что маска либо не применяется, либо дамп
   раньше неё.
3. НЕ проверено сравнение с fallback-ожиданием: посчитать ожидаемую S-зону
   в Python из hkv-дампов (K/Q реальные) и сравнить с kdump — это даст
   однозначный ответ, читаются ли K/Q правильно.
4. v108-109 (fp8, kbc=false) давал 4-головочные 0.078 — но полные расходились
   (min 0.47). После восстановления S-загрузок решать (Б): маска/чанки/
   M/L/PV-рескейлинг при KV=256 (16 чанков).
5. Проверить `var_sf` инициализацию cmz_f32 (есть, строка 1495) и
   mstride/маск-opt флаги (GGML_VK_FA8_HOSTDUMP, mask_opt: -inf в mc0_p).

## Снапшоты/артефакты
- Рабочий эталон S-стадии (fp8): build_logs/agent-workload/fp8_fa_cm1_EXP_v57.spvasm.
- GLSL-эталон: ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp (S: 292-311).
- Текущий файл: ggml/src/ggml-vulkan/vulkan-shaders/fp8_fa_cm1.spvasm (v126).

## История v107-v114 — см. выше (записи от 2026-08-08/09).

---

# Сессия 2026-08-09 вечер — диагностика S через дампы (S РАБОТАЕТ)

Инструментарий сессии: `GGML_VK_FA_F8_DUMP=1` + kdump-зоны:
- S после mask-add: dbg[tid*4..+3] (0..1023);
- маска-вектор mc0_p всех tid: dbg[1024 + tid*4..+3] (v122+);
- S ДО mask-add: dbg[2048 + tid*4..+3] (v124+).
`GGML_VK_FA8_HOSTDUMP=1`: прямые K/Q/V/mask (hkv_*.bin).
Чтение kdump: python, `d[2048+tid*4]`, tid = 16*us_c + us_r.

## v120 (2026-08-09) — S-дамп v121-инструментарий (S-зона 0..1023)
- Ошибка чтения ранних дампов: dbg[0]=1 был маркером, не S. Добавлен
  настоящий S-дамп (sh_sfsh[col_tid*64+row_tid] после маск-add).
- Результат: S-зона = 1024 слота: 922 ненулевых, реальные значения
  (S[0][0]=1.0, t4≈131.7, t8≈92.9), causal-маска (kv≤tok) работает,
  НО валидны только колонки 0,4,8,12 — паттерн [0,1,1,1] на группу из 4.
  Колонка 12 = NaN во всех kv-строках. → маска-секция читает НЕ тот слот.

## v121 (2026-08-09) — маска-вектор mc0_p в дамп
- Наблюдение: mc0_p (tid=0) = [0,0,0,0] — правильный causal-вектор для
  (kv0..3, t0)... но S[0][1..3] уже -inf → -inf не из mc0_p tid=0.

## v122 (2026-08-09) — HOSTDUMP (K/Q/mask прямые) + дамп mc0_p всех tid
- hkv_q.bin: Q f32 нормальный (tok0..3, NaN=0). hkv_k.bin: K f8 нормальный
  (kv0..12, NaN=0). hkv_mask.bin: НЕ 0/-inf, а мусор (0.42, 135.5, 17 NaN)!
- Маска в этом вызове (NQ=4 warmup) — мусорный буфер; в 13-вызове (ARMED)
  дамп не снялся (static fa8_host_dumped).

## v123 (2026-08-09) — mc0_p ВСЕХ tid (dbg[1024+tid*4])
- mc0_p: tid=0: [0,0,0,0]; tid=16: [-inf,0,0,0]; tid=80: [-inf,0,0,0].
  Паттерн mask[0] = -inf для многих tid — НЕ causal-треугольник.

## v124 (2026-08-09) — S ДО mask-add (dbg[2048+tid*4]) — ПРОРЫВ
- S-до-маски: tid=0: [1.0, 1.0, 0.935, 0.942] — РЕАЛЬНЫЕ значения (kv0, t0..3).
  tid=1: [0.95, 0.45, 1.0, 1.0]. → S-вычисление (K×Q fp8-coopmat) РАБОТАЕТ,
  -inf появляется ТОЛЬКО в mask-add.
- Раскладка S-слота подтверждена: (us_c, us_r) → v4 = S[4*us_r+k][4*us_c]
  (kv-строки 4us_r+k, колонка-группа 4us_c) — НЕ (kv=4us_c, t=4us_r+k).
- БАГ: mask-add применяет mask[k] = m[4us_r+k][us_c] (kv: 4us_r+k, t: us_c),
  а S-слот = (kv: 4us_r+k, t: 4us_c): токен в маске НЕ умножен на 4 →
  для групп us_c>0 читается колонка маски us_c вместо 4us_c → [0,1,1,1]-паттерн.

## v125 (2026-08-09) — ПРАВКА: ml_c2 = jv*Bc + 4*us_c (неполная)
- Результат: kv0 видит t0..15 (все реальные) — направление верное, НО
  появились новые ошибки: kv1..3, kv5..7, kv9..11 -inf везде; kv4: t0..3
  реальные (мусор slope×mask[0.42...]), kv12: t0..3 реальные (должны быть
  замаскированы) → ml_row (kv) и slope тоже перепутаны.

## v126 (2026-08-09) — ПОЛНАЯ ПРАВКА МАСКИ (собрано, НЕ протестировано)
- Итоговая семантика (подтверждена дампами v124/v125): S-слот (us_c, us_r)
  = S[4us_c][4us_r+k] — kv = 4us_c (строки!), t = 4us_r+k (колонки).
- Правки в mask-секции (ml):
  1. ml_row = iBr + 4*ml_c (kv: 4us_c; было 4us_r);
  2. ml_c2 = jv*Bc + 4*ml_r (t: 4us_r+k; было us_c / 4us_c);
  3. вектор маски: шаг 1 (ml_ab + 1, +2, +3) — по токенам, не по kv (ml_ms);
  4. slope-вектор в mask-add: sh_slope[4*us_c + k] (было 4*us_r + k);
  5. частичные ветки (ml_3f/ml_2f): шаг загрузок +1 (было +ml_ms).
- kv-границы (ml_fok: row+3 < nem1) и ml_kvok (t < pKV) остались.
- ВТОРОЙ баг (не в маске): S[1][1..3] = 0 в S-до-маски при реальных
  K[1]/Q[1..3] — Q-загрузка читает только токен 0? Гипотеза: q2_gr2 =
  (c - dv*16) — отрицательный uint при dv≥1 → ok2=false → qf8[..]=0.
- ТРЕТИЙ вопрос: hkv_mask (4,256) = мусор (0.42/135.5/NaN) вместо 0/-inf;
  вероятно, буфер mask не заполняется хостой для FA_F8_NATIVE, либо дамп
  снят не с того вызова (README: дамп с NQ=4, а не с NQ=13; fa8_host_dumped
  одноразовый static). Проверить хост (vk_dispatch.inc, mask-путь).

## v126b (2026-08-09, 14:37 сборка / 14:54-14:56 тест) — v126 протестирован с дампами
- Сборка v126 завершена в 14:37 (build-vulkan/bin/llama-server.exe). Тест 14:39:
  ответ сервера — слэш-коллапс (`reasoning_content` = 24 слэша, finish_reason=length).
- v126b: перезапуск с `GGML_VK_FA_F8_DUMP=1 GGML_VK_FA8_HOSTDUMP=1`, запрос
  "Say hello in one sentence.". Дамп kdump снят с РЕАЛЬНОГО prefill:
  NQ=13 (13 токенов), N=13, KV=256, gqa=1, HSK=256 (dbg0=1, ARMED→SKIP×8→READ).
  HOSTDUMP снят с NQ=4 warmup-вызова (fa8_host_dumped одноразовый).
- Декодирование fa_kdump.bin (14:56):
  - S-после-маски (dbg 0..1023): 562 finite / 429 NaN / 33 inf / 459 zero,
    диапазон -234.3..128.7 — НЕ causal-треугольник;
  - S-до-маски (dbg 2048..): 996 finite / 28 NaN / 600 zero; tid0 = [0,0,0,0]
    (в v124 было [1.0, 1.0, 0.935, 0.942]!);
  - mc0_p (dbg 1024..): 1015 finite (=0) + ровно 9 значений -inf; паттерн:
    -inf в ПЕРВОМ компоненте у tid0,4,8,12 и ещё нескольких tid — не causal
    (для kv0, t0..3 маска должна быть 0);
  - S-после tid0 = [1.0, -inf, -inf, -inf] — -inf применён к kv0-строкам.
- hkv-дампы (NQ=4): Q (24576 f32) все конечные; K (131072 f8) 13304 ненулевых;
  mask = 0/-0.0 — m_buf ЧИСТЫЙ (в отличие от «мусора» v122). Значит -inf в
  mc0_p приходит НЕ из m_buf.
- Маск-add-секция v126 (spvasm 1638-1699): ma_o = us_c*sfshstride+us_r,
  slope = sh_slope[4*us_c+k], ma_pr = slope*mask, select(mask==-inf → -inf).
- Выводы:
  1. Маска в v126 НЕ работает: -inf в mc0_p есть даже там, где её быть не
     должно. Источник — внутри шейдера (ml-секция: ml_row = iBr + 4*ml_c,
     частичные ветки ml_3f/ml_2f; проверить iBr и var_mc очистку), а не хост.
  2. ВНИМАНИЕ: v126 развернул kv/t-семантику относительно v124 (v124: слот
     (us_c,us_r) → S[4us_r+k][4us_c], kv=4us_r+k; v126: kv=4us_c, t=4us_r+k).
     Одна из интерпретаций неверна — проверить по v124-дампу (там значения
     S[0..3][0] были реальными) ПЕРЕД дальнейшими правками маски.
  3. NaN в S-до-маски (28) — «dirty fp8 cache rows» (комментарий в коде);
     429 NaN в S-после — softmax с -inf (exp(-inf - (-inf)) = NaN) — следствие
     п.1, а не отдельный баг.
  4. Ответ: слэш-коллапс сохраняется (как и ожидалось при сломанной маске).
- ЧИСТОТА: сервер v126b НЕ был остановлен агентом — висит на порту 8099
  (PID 17616, старт 14:54). Остановить graceful (POST /exit) перед любыми
  дальнейшими GPU-тестами.

## Следующие шаги
1. Остановить висящий сервер (порт 8099) graceful-способом.
2. Найти источник -inf в mc0_p: ml-секция (ml_row = iBr + 4*ml_c), частичные
   ветки ml_3f/ml_2f, очистка var_mc; сравнить с GLSL-эталоном
   (flash_attn_cm1.comp, MASK_ENABLE-секция).
3. Устранить противоречие kv/t-семантики v124 vs v126 (см. выше) — без этого
   маска не станет causal.
4. Баг Q-загрузки (q2_gr2 = c - dv*16 отрицательный при dv≥1) — если после
   фикса маски S[1][1..3] останутся нулями.
5. После causal-маски: тест ответа сервера, затем удаление диагностики
   (маркеры D096-F/G, kdump/outdump-код), quality-сравнение fallback↔native.

## v133 (2026-08-09) — архитектурный stop-loss и переход к canonical-first

- Ручное восстановление полного `fp8_fa_cm1.spvasm` остановлено как основной путь:
  v80-v132 накопили пересекающиеся дампы и независимый дрейф S/mask/P/V/output.
- Производственная база теперь только `flash_attn_cm1.comp` и его генерируемый
  `DATA_A_F8_E4M3` вариант. Первый gate — прямой f8 KV без глобального preconvert,
  с неизменными каноническими mask/softmax/PV/output секциями.
- Нативный fp8 WMMA переносится во второй этап: узкий автоматический SPIR-V transform
  S-стадии с проверяемыми якорями и fail-closed поведением при upstream drift.
- `fp8_fa_cm1.spvasm` остаётся диагностическим архивом. Новые исправления полного
  ручного клона не принимаются без отдельного аппаратного эксперимента.
- Следующая проверка: adjacent quality A/B canonical direct-f8 против f8 preconvert-f16,
  затем route trace и только после корректности короткий performance run.

## v134 (2026-08-09) — canonical direct-f8 P1 REJECTED

- Добавлен opt-in `GGML_VK_FA_F8_DIRECT=1`: он отключает глобальный f8→f16
  preconvert и выбирает существующий `DATA_A_F8_E4M3` вариант
  `flash_attn_cm1.comp` с tile-local K/V dequant и `SHMEM_STAGING=0`.
- Первый запуск ошибочно попал в scalar из-за правки одноимённого tuning-блока;
  route trace это сразу выявил. Правка перенесена строго в coopmat1.
- Валидный trace: `path=coopmat1`, `k=f8_e4m3`, `v=f8_e4m3`, `Br=16`, `Bc=64`,
  `shmem_staging=0`, `preconvert=0`, `f8_direct=1`.
- Quality gate провален: модель выдала повреждённый многоязычный поток, сервер
  завершил запрос HTTP 500. Соседний контроль с теми же f8 KV и
  `preconvert=1`, `path=coopmat1`, `k_type_eff=f16` дал связное рассуждение.
- Вывод: full-FA control flow больше не подозревается; граница сужена до
  canonical f8 cm1 tile staging / cooperative S-PV interaction. P1 не включать
  по умолчанию и не бенчмаркать. Перейти к P2 — узкому fail-closed transform.
- Артефакты: `d096-p1-direct-cm1-quality.server.log`,
  `d096-p1-preconvert-control.server.log` и соответствующие response JSON.

## v135 (2026-08-09) — P2 canonical SPIR-V structural audit

- Добавлен `scripts/research/d096_fp8_fa_spirv_audit.py`; принимает `.spv` или
  `.spvasm`, при бинарном входе вызывает `spirv-dis`.
- Аудит не зависит от числовых result ID: восстанавливает float widths, constants,
  cooperative types и границы двух MulAdd/Store-стадий.
- Fail-closed contract canonical cm1:
  - ровно четыре 16x16 coopmat-типа: f16 A/B/Acc и f32 Acc;
  - S: 2 A-load + 1 B-load → f32 MulAdd → первый Store;
  - PV: 1 A-load + 2 B-load → f16 MulAdd → второй Store;
  - любой structural drift завершает аудит ненулевым кодом.
- Проверено на реально собранном
  `flash_attn_f32_f16_f8_e4m3_cm1.spv`: 4496 строк, S 2244..2316,
  PV 3062..3141, SHA-256
  `6241e140b0658551069f5051e46a180722c0300354dfe1da5463f82c14bfa6d3`.
- Следующий шаг P2: transform заменяет только S-stage cooperative A/B типы и
  загрузки на E4M3; первый Store, mask/softmax/PV/output должны остаться
  побитово каноническими после перенумерации ID.

## v136 (2026-08-09) — P1 direct-f8 исправлен и измерен

- Причина quality-провала v134 найдена в семи compile-time ветках
  `flash_attn_cm1.comp`: `BLOCK_SIZE == 1` выбирал raw f16 load и для
  `DATA_A_F8_E4M3`, то есть пары fp8-байтов интерпретировались как f16.
- Guards сужены до `BLOCK_SIZE == 1 && !defined(DATA_A_F8_E4M3)`; после этого
  `GGML_VK_FA_F8_DIRECT=1` даёт связный ответ без `////`, NaN и HTTP 500.
- Соседняя Q4 49K пара (один бинарник, 8601 prompt tokens, b512/ub256,
  `Vulkan1,Vulkan0`, f8 K/V, `spec=none`):
  - preconvert: `1451.58 pt/s`, `27.10 t/s`;
  - direct-f8 P1: `1320.25 pt/s`, `27.80 t/s`.
- P1 корректен, но prompt eval на `9.0%` медленнее. Он остаётся диагностическим
  opt-in и не становится default; результат подтвердил, что одной отмены
  глобального preconvert недостаточно.

## v137 (2026-08-09) — P2 generated fp8 WMMA без ручного клона FA

- Добавлен `fa_q_f32_f8.comp`: отдельный prepass квантует Q по тайлам
  `16xHSK` в E4M3 и сохраняет `attention_scale / quant_scale` для S-стадии.
- `flash_attn_cm1.comp` остаётся единственным источником mask/softmax/PV/output;
  под `D096_FP8_S_BASE` добавлены только узкие descriptor/scale hooks.
- `d096_fp8_fa_spirv_transform.py` fail-closed преобразует собранный P2-base:
  находит S-loads по bindings, добавляет `SPV_EXT_float8`, переводит только
  S A/B cooperative types и loads в E4M3 и затем запускает structural audit,
  `spirv-as` и `spirv-val`.
- В runtime добавлен отдельный opt-in `FA_F8_P2` (`GGML_VK_FA_F8_P2=1`),
  Q8/scale prealloc buffers и отдельный 9-binding pipeline. Fallback и обычный
  preconvert не изменены.
- Офлайн-контракт прошёл: S = fp8 A/B + f32 accumulator; PV = f16 A/B/f16
  accumulator; transformed SHA-256
  `d724e9b9dd7e773076226c57e7e871cdc08412860f6171f34f093e401521763a`.

## v138 (2026-08-09) — P2 hardware quality PASS, performance REJECT

- На RX 9070 XT route trace подтвердил реальный
  `flash_attn_f32_f16_f8_p2|path=f8_p2`, `preconvert=0`; проверены обычные и
  mask-opt чанки KV. Короткий quality-run дал связное английское reasoning без
  slash-collapse, NaN и HTTP 500.
- Свежая соседняя Q4 49K пара на текущем бинарнике:
  - `d096-p2-q4-f8-preconvert-r1`: `1442.46 pt/s`, `26.99 t/s`;
  - `d096-p2-q4-f8-native-r1`: `1267.29 pt/s`, `28.02 t/s`.
- P2 decode выше на `3.8%`, но prompt eval ниже на `12.1%`; aggregate wall TPS
  ниже на `10.7%`. В текущем виде P2 не принимается как performance path.
- Главный следующий эксперимент — P3: перенести Q→E4M3 и scale внутрь
  канонического FA workgroup (узкий GLSL/SPIR-V hook), чтобы убрать отдельный
  global prepass, сохранив generated-transform подход и канонические хвосты.

## v139 (2026-08-09) — P3 fused-Q generated path, quality PASS

- Добавлен отдельный opt-in `FA_F8_P3` (`GGML_VK_FA_F8_P3=1`). Он использует
  тот же canonical-first transform, но Q→E4M3 и tile scale выполняются внутри
  FA workgroup; глобальные Q8/scale buffers и отдельный dispatch не нужны.
- P3 base хранит 4096 encoded Q bytes в named Workgroup array `Q8f`; transform
  fail-closed перенаправляет ровно один named dummy Q-load и биткастит ровно
  один encoder store. K остаётся binding 1, downstream PV/mask/softmax/output
  остаются каноническими.
- Runtime shape gate: f8 K/V, f32acc, `HSK=256`, `Br=16`, aligned strides,
  ordinary prefill `N>8`; pipeline имеет стандартные 7 bindings.
- Встроенный SPIR-V прошёл `p3-base`/`fp8-p3` audits, `spirv-as` и `spirv-val`.
  Финальный dynamic-scale P3 SHA-256:
  `cd18c170f4af4b46c0ae02d41449ee16efb4bc4c3427010fcd55fbe7989ab52b`.
- Hardware route: `flash_attn_f32_f16_f8_p3|path=f8_p3|preconvert=0`;
  проверены обычные и mask-opt KV чанки. 8601-token workload завершён без
  ошибок; response preview связный (`Thinking Process...`).
- Результат `d096-p3-q4-f8-native-r1`: `1337.89 pt/s`, `28.18 t/s`.
  Это `+5.6%` prompt против P2, но всё ещё `-7.3%` против соседнего preconvert
  control (`1442.46 pt/s`); decode выше control на `4.4%`.
- P3 остаётся корректным opt-in research path, не default.

## v140 (2026-08-09) — P3 микро-варианты REJECTED

- P3v2, vec4 Q loads: `1290.76 pt/s`, `27.83 t/s`; больше SPIR-V/register
  pressure, чем у scalar P3.
- P3v3, `dont_unroll` Q loops: `1302.90 pt/s`, `27.91 t/s`; хуже P3v1.
- P3v4, fixed Q scale 1024: prompt `1330.62 pt/s`, но quality gate завершился
  HTTP 500. Один hardware dump (`max_abs=0.0957`) не покрывает межслойный
  диапазон Q; dynamic tile scale обязателен.
- P3v5, bitwise E4M3 encoder: `1328.70 pt/s`, `27.70 t/s`; корректен, но не
  быстрее штатного encoder. Все rejected варианты сняты из активной сборки.
- Следующий чистый performance-вопрос: V-стадия всё ещё делает tile-local
  f8→f16 dequant. Проверить гибрид `native fp8 K×Q + V-only preconvert` прежде,
  чем проектировать полный fp8 P×V с отдельным scale.

## v141 (2026-08-10) — P4 V-only preconvert: НЕВАЛИДНЫЙ замер (игра на фоне)

- Реализован opt-in `GGML_VK_FA_F8_P4`: тот же fused-Q S-stage transform, но V
  читается из плотного f16 preconvert-буфера (V-only), K остаётся raw f8.
  PV-стадия в базе получила третью V-load ветку (global direct f16) — audit
  расширен профилями `p4-base`/`fp8-p4` (PV: 1 A + 3 B, f16).
- SPIR-V контракт подтверждён: `spirv-as`/`spirv-val`/audit прошли; runtime
  dispatch 7 bindings; route trace: `flash_attn_f32_f16_f8_p4|path=f8_p4`.
  Финальный SHA: `ca27c9af9107776149acdfed6c39a41db60d22c4c71381a896449f24ae5fab3d`.
- Первый замер `d096-p4-q4-f8-v16-r1` (`1157.78 pt/s`, `26.11 t/s`) ВЫПОЛНЕН
  ПРИ АКТИВНОЙ ИГРЕ НА GPU — цифра не сопоставима с соседними P2/P3/control
  (все на чистой машине). Quality при этом корректный (связный ответ, ошибок
  нет).
- Статус: P4 «не подтверждено/под нагрузкой». Требуется повторный замер на
  чистой машине в паре с соседним control (протокол AGENTS.md: adjacent
  baseline под одинаковой нагрузкой), прежде чем делать вывод о V-preconvert.

## v142 (2026-08-10) — P4 ЧИСТЫЙ ЗАМЕР: НОВЫЙ ЛУЧШИЙ ПУТЬ (+7.6% prompt)

- Чистая сессия на том же бинарнике (49K Q4 lane, соседние прогоны, игра
  завершена):
  - P4 `d096-p4-q4-f8-v16-clean-r1/r2`: `1498.32/1493.10 pt/s`, `28.35/28.01 t/s`
    (mean 1496, разброс 0.2%);
  - control `d096-p4-control-preconvert-clean-r1/r2/r3`: `1349.15/1428.94/
    1392.25 pt/s`, `28.97/28.09/26.95 t/s` (mean 1390, разброс 5.9%).
- Итог: P4 +7.6% prompt, decode паритет; P4 стабильнее по разбросу.
  Замер под игрой (v141, 1157.78) подтверждён как артефакт нагрузки.
- Вывод: tile-local f8 V dequant дороже отдельного V-preconvert pass;
  K raw f8 + fused Q + V f16 — лучшая комбинация на этой линии.
- Следующие шаги: P4 на 12K/98K lanes, MTP-прогон, затем решение о default.

## v143 (2026-08-10) — P4 MTP-гейт PASS (рабочий сценарий)

- Чистая MTP-пара (spec=draft-mtp n=2, 128 токенов, 49K lane, 2 прогона
  каждый):
  - control: prompt `1392.46/1409.16`, decode `38.19/38.81`, acc `60/133=45%`;
  - P4: prompt `1462.68/1454.54`, decode `41.41/43.14`, acc `67/119=56%`.
- Устойчиво 2/2: P4 prompt `+4.3..5.0%`, decode `+8.4..11%`;
  acceptance 56% vs 45% при меньшем draft_n (119 vs 133) — стартовая точка
  первого принятого токена другая из-за fp8/f16 префилла.
- Оба пути дают одинаковый MTP-коллапс triage_diff @ temp0 (известный
  артефакт, D095) — fp8-путь качеством не хуже control.
- Вывод: P4 работает и под рабочим MTP-сценарием. Осталось: 12K/98K масштаб,
  затем решение о default (пока opt-in).


## v144 (2026-08-10) — P5: fp8 PV (raw f8 V, f32 acc) — transform + runtime готовы

- P5 = P4 + fp8 P*V: PV-стадия переведена на fp8 coopmat (A=P fp8, B=V raw fp8
  из исходного кэша, acc f32); f16 preconvert V-пасс НЕ нужен.
- SPIR-V (transform `--pv-f8`): P-load → Psh8 (uint8 Workgroup array, dense
  [col*Br+row]) с bitcast-encoder store'ами; V-load → `data_v` placeholder
  (полуэлементы без /4), retype переменной в fp8 block; Function-holder'ы
  KMat/QMat ретайпнуты в fp8 (вместо bypass — устраняет dominance-ошибки);
  мёртвый f16 kvsh store удалён; f16 kvsh fallback-ветка не тронута.
  Audit fp8-p5: S 1A+1B fp8 acc f32; PV 1A fp8 + 2B (fp8 direct + f16 kvsh)
  acc f32; 5 coop-типов (без f16 acc); spirv-as/val PASS.
- Runtime: `FA_F8_P5` enum/route (env `GGML_VK_FA_F8_P5`), dispatch 7 bindings,
  v_stride = nev0 (f8 байты), без preconvert; P5 имеет приоритет над P4/P3/P2.
- Качество (сервер, Q4_K_M + f8_e4m3 KV, Vulkan dual): связные ответы
  (Paris/391/Eiffel), 0 HTTP 500; идентичное P4 поведение (seed42-длинное
  thinking = артефакт модели, не пути). 5/5 стабильных ответов @ temp0.
- Статус: бенч 49K lane spec=none в процессе (r1/r2) — цифры в RESULTS_LOG.

## v145 (2026-08-10) — P5 замеры: prompt +4.5%, MTP decode +15.9% vs control

- 49K lane, spec=none (147456 chars, max 16 tok), соседние пары, одна сессия:
  - P5: `1338.06 / 1350.77` (mean 1344), decode `25.1/25.6`;
  - control (f8 preconvert K+V): `1277.53 / 1295.15` (mean 1286), decode `26.1/24.9`;
  - **P5 +4.3..4.7% prompt (2/2), decode паритет**.
- MTP gate (draft-mtp n=2, 128 токенов, 49K lane, 2/2):
  - P5: prompt `1349.04/1348.50`, decode `39.69/39.72`, acc `65/123 = 52.8%`;
  - control: prompt `1294.12/1286.47`, decode `34.34/34.15`, acc `55/143 = 38.5%`;
  - **P5 prompt +4.5%, decode +15.9%, acc +14pp** (детерминировано seed 42).
- P4 session-замер (1248.94) НЕВАЛИДЕН: 8-й прогон подряд → тепловой дрейф
  (control r3 = 1258.67 vs r2 = 1295.15 подтвердил дрейф ~3% за 4 прогона).
  Чистый A/B P5-vs-P4 — следующий шаг (отдельная сессия).
- Оба пути в этой сессии на ~7% ниже вчерашних абсолютов (сессионный фон,
  не бинарник: control 1286 vs 1390 вчера).

## v146 (2026-08-10) — P5 vs P4 прямой A/B: +6.0% prompt (fp8 PV = прогресс)

- Свежая сессия, андервольт включён, перемежающиеся прогоны (P5 r1, P4 r1,
  P5 r2, P4 r2), 49K lane (147456 chars, spec=none, max 16):
  - P5: `1352.27 / 1378.68` (mean **1365.5**), decode `24.71/24.81`;
  - P4: `1264.41 / 1312.73` (mean **1288.6**), decode `25.28/23.84`.
- Обе пары в одну сторону: r1 +6.9%, r2 +5.0% → **P5 +6.0% mean**, decode
  паритет. P5 (полный fp8: S+PV) стабильно быстрее P4 (fp8 S + f16 PV) —
  fp8 PV приносит реальный выигрыш поверх P4.
- Абсолюты vs вчера НЕСОПОСТАВИМЫ: вчерашние замеры (D096-B/C) были на
  12K-масштабе (prompt_n=8601, ctx 49152, ub 256, tasks=quick, --no-reuse
  --no-mmap) — "1496" относится к короткому префиллу, не к 49K lane.
  Сегодня 40650 токенов (ctx 131072, ub 128, полный tasks). Причина
  расхождения найдена при расследовании 2026-08-10 (jsonl/diagnostics).
- Следующие шаги: 12K/98K масштаб для P5, MTP A/B P5-vs-P4, решение о default.

## v147 (2026-08-10) — P5 на вчерашнем 12K-конфиге: 1667.9 vs 1495.7 = +11.5%

- Повтор вчерашнего конфига ДЛЯ P5 (tasks quick, ctx 49152, ub 256,
  --no-reuse --no-mmap, triage_diff, 8601 токенов):
  - P5: `1671.49 / 1664.34` (mean **1667.9**, разброс 0.4%);
  - P4 сегодня: `1487.83` — вчерашние условия воспроизведены (вчера
    1498.32/1493.10, mean 1495.7; отклонение -0.7%);
  - control вчера: mean 1390.1 (1349/1429/1392).
- P5 vs P4-вчера **+11.5%**, vs P4-сегодня (одна сессия) +12.3%,
  vs control-вчера +20.0%. Decode 28.8 vs 27.8-28.4 t/s.
- review_bug-пара (8582 токенов): P5 1658.8 vs P4 1574.7 = +5.3%.
- Масштаб-эффект: 12K +12%, 49K +6% — на коротком префилле FA-доля времени
  выше, поэтому fp8-PV-выигрыш больше.
- Итог: прогресс против вчерашних абсолютов подтверждён (а не только
  внутрисессионный A/B). Дальше: 98K + MTP A/B P5-vs-P4, default-решение.

## v148 (2026-08-10) — P5 (f8) vs q8_0: +14.0% prompt, KV −96 MiB

- Перемежающиеся прогоны (q8 r1, P5 r1, P5 r2, q8 r2), 12K конфиг
  (triage_diff 8601 ток, ctx 49152, ub 256):
  - P5 (f8_e4m3 KV): `1632.87 / 1634.45` (mean **1633.7**, разброс 0.05%);
  - q8_0 KV: `1448.81 / 1416.16` (mean **1432.5**, разброс 2.3%);
  - **+14.0% mean** (пары: +12.7%/+15.4%).
- Память (server.log, ctx 49152): KV q8 = 1632 MiB (2×816), KV f8 = 1536 MiB
  (2×768) = **−96 MiB (−5.9%)**; модель не отличается
  (8207.6+7455.3+682.0 MiB). Доп. выигрыш P5: нет f16 preconvert-буферов.
- Контекст: D095-D3 (coopmat1+preconvert, до P1-P5) давал f8/q8 ПАРИТЕТ
  (1663.6 vs 1657.8); теперь P5 = +14% vs q8 на 12K.
- Цепочка 12K: control-f8pre 1390.1 < q8 1432.5 < P4 1495.7 < P5 1633.7.
- Осталось: MTP A/B P5-vs-q8, 98K масштаб, default-решение.

## v149 (2026-08-10) — Масштабная кривая P5 vs q8: 12K +14%, 49K +43%, 98K +45.5%

- spec=none, triage_diff, перемежающиеся пары:
  - 12K (8601 ток): P5 mean 1633.7 vs q8 mean 1432.5 = **+14.0%**;
  - 49K (40668 ток): P5 mean 1365.5 vs q8 mean 957.0 (968.1/945.9) = **+42.7%**;
  - 98K (58583 ток): P5 mean 1230.7 (1245.0/1216.5) vs q8 mean 845.8
    (837.6/854.0) = **+45.5%**. Все пары в одну сторону.
- Причина масштаб-эффекта: q8-путь (coopmat1, preconvert=1 по route trace)
  платит q8->f16 preconvert на весь KV каждый слой + dequant; P5 читает raw
  f8 без preconvert — накладные не растут с KV так же.
- MTP (12K, draft-mtp n=2, 128 ток): P5 prompt 1483.1 vs q8 1402.3
  (+5.8%), decode 35.4 vs 32.5 (+8.9%), acc 35% vs 31%.
- Память (ctx 49152): KV f8 1536 vs q8 1632 MiB (-96 MiB, -5.9%).
- Итог: P5 кандидат в default; осталось quality-гейт (длинная генерация
  f8 vs q8) и решение о default.

## v150 (2026-08-10) — GUI: P5 доступен в сервере и бенчмарке

- Server tab (Vulkan-панель): чекбокс "Native FP8 attention (P5 path)" →
  env GGML_VK_FA_F8_P5=1 (KV f8_e4m3 + FA обязательны, тултип с цифрами);
  сохранение в settings (fp8_p5).
- Benchmark tab: чекбокс "FP8 native FA (P5)" в Single Bench и
  "FP8 native FA (P5) for f8_e4m3 runs" в Autotune (env идёт в
  bench-процесс; лог "Env overrides"). KV f8_e4m3 уже был в комбо/чипах.
- Smoke: env включается только при выборе Vulkan backend; settings
  load/save round-trip OK; compileall/import OK.

## v151 (2026-08-10) — Decode-диагноз: шаг паритет, MTP-отставание = acceptance

- spec=none decode (256 ток, 12K): f16 28.82 vs P5 28.62 — ПАРИТЕТ.
  Скорость decode-шага fp8 НЕ отстаёт от f16 (нативность подтверждена).
- MTP n=2 (128 ток, 12K, соседние): f16 dec 52.60 acc 82%; P4 dec 42.46
  acc 56%; control dec 38.28 acc 45%; P5 dec 31.6-39.1 acc 27-45%;
  q8 dec 32.5 acc 31%.
- Причина: draft (nextn) чувствителен к KV-качеству: ЛЮБОЕ квантование KV
  роняет acceptance (82% → 27-56%), даже с f16-preconvert (control 45%).
  V-качество внутри fp8 тоже влияет: P4 (V f16) 56% > P5 (V f8) 27-45%.
- Prefill: 12K P5 1633.7 vs f16 ~1593 (+2.5%); 49K P5 1365.5 vs f16 1376.1
  (-0.8%, паритет); q8 на 49K 957 — сильно хуже обоих.
- Вывод: fp8 быстрее f16 на prefill (короткий) / паритет (длинный);
  MTP-интерактив выигрывает у f16 KV из-за acceptance. Открыто:
  пер-слойный f16 KV для nextn-слоя (вернёт acceptance при f8-хранении).

## v152 (2026-08-10) — D096-K: гибридный KV для MTP (последние N слоёв f16)

- Правка: src/llama-kv-cache.cpp — пер-слойный тип K/V: env
  LLAMA_VK_MTP_KV_LAST_F16=N (только при K/V f8_e4m3) держит последние N
  KV-слоёв (filter-aware) в f16. Лог: "size = ... (last N layers f16)".
- Кривая (одна свежая сессия, 12K, MTP n=2, 128 ток):
  N=0: dec 30.48 acc 27% KV 1536 MiB
  N=1: dec 41.04 acc 52% KV 1632 MiB
  N=4: dec 46.78 acc 67% KV 1920 MiB
  N=8: dec 48.27 acc 75% KV 2304 MiB
  N=16: dec 53.98 acc 82% KV 3072 MiB (= f16, повторяет вчерашнее 52.6/82)
- Вывод: draft-голова чувствительна к f8-KV последних слоёв; N=4..8 —
  сладкое пятно (90%+ f16-декода при -25..-37% памяти vs f16). Prefill
  не меняется (1602-1609 ptps во всех точках).
- Qwen3.6-27B: гибридная (16 attention-KV слоёв + 48 recurrent + 1 nextn);
  MTP-контекст имеет свой KV слоя 64 (всегда f16, 192 MiB).

## v153 (2026-08-10) — D096-K: гибридный KV включён по умолчанию

- common/common.cpp (common_context_params_to_llama): если cache_type f8_e4m3
  (K и V) + speculative type draft-mtp + env не задан -> setenv
  LLAMA_VK_MTP_KV_LAST_F16=8 (Windows _putenv_s / POSIX setenv).
  Отключение: LLAMA_VK_MTP_KV_LAST_F16=0; тонкая настройка: любое N.
- Свежий замер без env (d096-dec-mtp-default-r1): dec 51.68 t/s, acc 80/97
  (82%... 80%), ptps 1602.8 — 96% от f16-скорости при 25% экономии KV-памяти (2304 vs 3072 MiB).
- Проверено: лог "MTP + f8 KV: enabling hybrid KV cache (last 8 layers f16)".

## v154 (2026-08-10) — D096-L: битовый E4M3-энкодер

- ggml/src/ggml-vulkan/vulkan-shaders/types.glsl: f32_to_fp8_e4m3 переписан
  без frexp/ldexp/FDiv: exp_field = exp32 - 120 (bias-сдвиг), mantissa =
  ((bits & 0x7FFFFF) + 0x80000) >> 20 (round-half-up + carry в экспоненту),
  f >= 240 -> clamp (exp 14, man 7). Subnormal-зона (exp32 < 121) без изменений.
- Эмуляция (300K значений, Q/P-домен): 96.8% бит-в-бит; 3.2% отличаются на
  1 ulp мантиссы (новый точнее: корректный carry вместо min-clamp).
- Замеры: 49K P5 1399.0 pt/s (вчера: P5 1365.5, f16 1376.1 — вечерний прогон
  после серии, так что прирост реальный); 12K P5 1621.1; quality: 391+2=393,
  длинный thinking осмысленный.
- Влияет на все f8-пути (P2/P3/P4/P5, fa_q_f32_f8, copy_to_quant).

## v155 (2026-08-10) — D096-M: GUI упрощён (P5 автоматом при f8 KV)

- gui/server_backend_panels.py: удалён чекбокс "Native FP8 attention (P5 path)";
  _VulkanPanel.env() всегда отдаёт GGML_VK_FA_F8_P5=1 (kernel игнорирует для
  не-f8 KV). to_settings/from_settings: ключ fp8_p5 удалён (legacy игнор).
- gui/benchmark_tab.py: удалены чекбоксы "FP8 native FA (P5)" (single) и
  "FP8 native FA (P5) for f8_e4m3 runs" (autotune); _bench_env_overrides()
  всегда ставит GGML_VK_FA_F8_P5=1 для Vulkan.
- Smoke-тест через GUI-код (build_logs/agent-workload/gui_server_smoke.py):
  ServerTabWidget._compose_server_command() -> запуск, health OK, 3 замера,
  graceful stop (CTRL_BREAK). GUI-дефолты (ctx 8192, b 512/ub 256,
  -dev Vulkan0,Vulkan1 — порядок по умолчанию GUI): prefill 2006 ток =
  1215.5 pt/s, decode ~31 t/s. Примечание: bench-конфигурация
  (Vulkan1,Vulkan0, ub 256, длинный промпт) даёт 1621 — разница от порядка
  устройств и длины промпта.

## v156 (2026-08-10) — D096-N: диагностика качества q8/fp8 (не баг) + GUI-лимит

- Жалоба "q8 отупевшее / fp8 думает и не выдаёт" — НЕ деградация KV:
  - q8_0 KV (P5=1): 391+2=393 (finish=stop, 188 ток), Python sum-of-squares
    корректен (обрыв только на лимите 600).
  - f8_e4m3+P5: 393 ok; при max_tokens=1500 — finish=stop, 986 токенов,
    полный корректный код; при 600 — finish=length, пустой content.
- Причина: thinking-модель тратит reasoning 150-600+ токенов; GUI-дефолт
  n_predict=128 (gui/inference_tab.py) всегда обрезал до ответа.
- Фикс: дефолт n_predict 128 -> 1024 (create_ui + load_settings fallback).

## v157 (2026-08-10) — D096-O: «мусор q8» не воспроизводится; GUI-конфиг 1-в-1

- Жалоба: q8_0 KV выдает мусор в thinking («force_majeure_agent», иероглифы).
- Воспроизведение точной GUI-команды (ctx 49152, b 8192/ub 1024, -t 16, ngl 999,
  --mmproj, --spec-type draft-mtp, --ctx-checkpoints 4, -dev Vulkan0,Vulkan1,
  LLAMA_OUTPUT_DEVICE=Vulkan1, P5=1): 20+ запусков q8_0/f8_e4m3/f16 ×
  {MTP, none} × {temp 0, 0.7} × лимиты 300-512 — МУСОРА НЕТ ни разу.
- Наблюдаемые артефакты (у ВСЕХ KV, включая f16): finish=length с пустым
  content при малых лимитах (thinking-модель тратит 150-600+ токенов на
  reasoning; даже "привет" = 180-260 ток); длинные «thinking process»-тексты
  в content; Q8-MTP msg2: "Here's a thinking process... What is the capital
  of France?" — галлюцинация thinking про чужой запрос.
- Доп. факт: MTP + q8_0: известный коллапс acceptance (31%, D096-K-диагноз) —
  дравт при квантованном KV бесполезен; fp8+гибрид работает (80%).
- Вывод: код q8_0-пути не менялся (все шейдерные правки только для
  DATA_A_F8_E4M3); вероятная причина пользовательского мусора — лимит 128
  (старый GUI-дефолт n_predict, исправлен в v156) + обрыв reasoning, и/или
  грязный драйвер после taskkill //F (были хард-киллы в тот же день).
- Рекомендация GUI: Tokens >= 1024; для q8_0 KV выключить MTP (spec none);
  после хард-киллов — перезапуск/пауза.

## v158 (2026-08-10) — D096-P: гибридный KV для q8_0 (фикс регрессии q8+MTP)

- Причина регрессии: LLAMA_VK_MTP_KV_LAST_F16 был жёстко ограничен
  f8_e4m3 (src/llama-kv-cache.cpp) — q8+MTP после D094-преконвертных
  оптимизаций давал acceptance 31% / decode 30.5 t/s. Раньше (до
  оптимизаций) q8+MTP работал нормально — это подтверждённая регрессия.
- Фикс: условие kv_q8 (K и V = q8_0) добавлено к kv_f8; common.cpp дефолт
  расширен ("MTP + quantized KV: enabling hybrid KV cache").
- Проверено на GUI-конфигурации (ctx 49152, b8192/ub1024, mmproj, MTP n=2,
  Vulkan0,Vulkan1, temp 0.7): acceptance 178/242 = 74% (было 31%),
  decode 51.7 t/s (было 30.5), ответ нормальный. KV: 8 последних слоёв f16,
  остальные q8.

## v159 (2026-08-11) — D095 R1-R5: FP8/MTP polish и корректность env=0

- Исправлено отключение гибридного KV: `LLAMA_VK_MTP_KV_LAST_F16=0` теперь
  действительно задаёт N=0, а не принудительно N=1.
- Исправлен lifetime аргумента variadic-лога гибридного KV: строка суффикса
  хранится в локальном `std::string` до вызова `LOG_INF`.
- R1/R3 kernel-probes (direct scalar reads и raw-f8 small-N coopmat1)
  активировались, но не ускорили wall time; код удалён.
- Robust N sweep: N=8 = 1605.79/55.49, acceptance 85.5%, KV 2304 MiB;
  f16/N=16 = 1627.67/58.81, 89.6%, 3072 MiB. Остаток decode = 5.6% при
  экономии 25% KV; N=9/10 хуже, дефолт N=8 сохранён.
- MTP-depth bracket: n=2 = 55.62/54.84 t/s; n=3 = 45.13, n=4 = 48.81.
  Дефолт n=2 подтверждён. Следующий gate — block-scaled E4M3 (D095 R6).

## v160 (2026-08-11) — D095 R6: real-KV precision scout

- Добавлен diagnostic-only `llama-kv-precision-scout`: callback снимает
  post-RoPE Q/K/V ранних full-attention слоёв без изменения штатного графа;
  Python-оркестратор воспроизводит `triage_diff,review_bug` и сохраняет
  CSV/Markdown.
- Capture целостен: слои 3/7/11, 8570/8551 prompt tokens, K/V token counts
  совпадают на всех шести task/layer парах.
- R6 block-scaled E4M3 отклонён: weighted logit MSE raw `0.0030764424`, B32
  `0.0030764146` (только -0.0009%). Power-of-two scale не добавляет мантиссу.
- q8_0-контроль дал `0.0001028896` (-96.66%) при 1.0625 bytes/value. Следующий
  prebuild gate R7 — BFP8: int8 payload + power-of-two exponent, B32 = 1.03125
  bytes/value; до результата новый GGML type запрещён.

## v161 (2026-08-11) — D095 R7: BFP8 precision PASS, runtime gate FAIL

- BFP8 P2/B32: weighted logit MSE `0.0002514458`, -91.83% против raw E4M3;
  worst task/layer ratio `0.0939`; storage 1.03125 bytes/value.
- Новый тип отклонён: q8_0 точнее в 2.44x при +3.03% storage, а его raw
  int8/cooperative path уже проиграл 22-25% (D094 cycle 5). Принятый q8
  preconvert улучшал 131K `54.1s -> 47.8s`; BFP8 повторяет закрытый механизм.
- Runtime-кода нет. Следующий offline gate R8: general-scale E4M3
  (`scale=max/240`) B16/B32/B64; новый тип запрещён до precision+WMMA gates.

## v162 (2026-08-11) — D095 R8: general-scale E4M3 upper bound

- Precision PASS: weighted logit MSE B16 `0.0016826834` (-45.30%), B32
  `0.0019487026` (-36.66%), B64 `0.0021820721` (-29.07%); все пары лучше raw.
- Symmetric K/V runtime FAIL: V scale меняется внутри Bc reduction P*V и не
  выносится после WMMA. Нужна отдельная P-квантизация на каждый D-block либо
  полный f16 V dequant; оба варианта уничтожают P5 work-volume win.
- R9 — последний factorable gate: general-scale B256 только для K, одна f16
  scale на token/head; V остаётся raw E4M3, средний KV overhead 0.390625%.

## v163 (2026-08-11) — D095 R9: K-only per-token scale precision PASS

- General-scale E4M3 B256 только для K: weighted attention-logit MSE
  `0.003076 -> 0.002445` (`-20.52%`); worst task/layer ratio `0.8737`, то есть
  локальных регрессий на двух prompt и слоях 3/7/11 нет.
- Metadata K = `2/256 = 0.78125%`; средний overhead K+V = `0.390625%`.
  V остаётся byte-identical raw E4M3 P5, поэтому scale не входит в P*V.
- Offline gate PASS, но runtime-кода нет: нужен отдельный per-layer/token/head
  sidecar с SET_ROWS/copy/view/sequence lifecycle и scale load + score multiply
  в Q*K. Это отдельный default-off prototype после стабильного Q4 refresh.
- D095 R1-R9 prebuild-программа закрыта; D097 меняет только long-context
  hybrid-policy, P5 shader остаётся прежним.

## v164 (2026-08-11) — D097: FP8 98K MTP acceptance recovery

- Root cause: E4M3 имеет 3 mantissa bits и без block scale на captured KV даёт
  attention-logit MSE в 29.9x выше q8_0. Поэтому предположение "f8 точнее q8"
  для этих конкретных форматов неверно.
- 256-token controls: f8 N8 acceptance `140/230=60.87%`, q8 N8
  `151/208=72.60%`, результаты повторяются бит-в-бит по draft counts.
- Принят N12 для f8+MTP при ctx>=98304: `1510.95/41.79/5.7618`, acceptance
  73.79%, против q8-center `1422.71/41.96/5.4736`, 72.60%. KV 5376 vs
  4704 MiB — trade-off документирован.
- `common.cpp` выбирает N12 только для long f8; q8/short f8 используют N8.
  Explicit `LLAMA_VK_MTP_KV_LAST_F16` (включая 0) остаётся rollback.
- Default-off `LLAMA_VK_MTP_KV_Q8_BEFORE_F16=6` даёт 90.61% acceptance и
  47.05 decode t/s при 4680 MiB, но не сохраняет prompt advantage, поэтому не
  является общим default.
