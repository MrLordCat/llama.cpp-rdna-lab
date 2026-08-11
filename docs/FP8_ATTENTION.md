# FP8 E4M3: KV cache, attention paths and defaults

Актуальное состояние локальной реализации fp8 (D095/D096, 2026-08-10).
Это сводный документ «как реализовано сейчас»; историю версий см. в
`docs/research/D096_VERSIONS.md`, результаты — в `docs/research/RESULTS_LOG.md`,
план — в `docs/research/D096_ROADMAP.md`.

## 1. Обзор

fp8 в этом форке состоит из двух независимых уровней:

1. **KV-кэш типа `f8_e4m3`** (данные): K/V хранятся в fp8 E4M3 вместо f16.
   Квантование выполняется битовым энкодером в Vulkan-шейдерах.
   Включается `--cache-type-k f8_e4m3 --cache-type-v f8_e4m3`.
2. **Пути Flash Attention** (вычисления): как FA-кернел потребляет f8 KV —
   через f16-преконверт или напрямую (нативные fp8-пути).

Оба уровня включаются независимо: KV f8 работает и с преконвертом, и с
нативными путями; пути P2–P5 активны только при f8 KV.

## 2. Пути Flash Attention (приоритет сверху вниз)

Выбор пути — в `ggml/src/ggml-vulkan/runtime/vk_dispatch.inc`
(`ggml_vk_flash_attn`). Пути fail-closed: при невыполнении условий
следующий ниже по списку.

| Путь | Env-включение | Условия | Что делает | Статус |
| --- | --- | --- | --- | --- |
| **P5** (D096-F) | `GGML_VK_FA_F8_P5=1` | f8 K/V, HSK=256, neq1>8, f32acc, KV%64==0, выравнивания 8, coopmat1 + coopmat 16×16×16 f32acc | V остаётся raw f8; fp8 P×V с f32-аккумулятором; raw-fp8 S | **default для Vulkan** (D096-M) |
| P4 | `GGML_VK_FA_F8_P4=1` | как P5 | dense f16 V-preconvert буфер (V → f16 перед P×V) | опционально |
| P3 | `GGML_VK_FA_F8_P3=1` | как P4, HSK=256, Br=16 | P3 + фикс shmem Q8 ABI (4096 байт) | опционально |
| P2 | `GGML_VK_FA_F8_P2=1` | f8 K/V, neq1>8, f32acc, KV%64==0, HSK%16==0 | узкий fail-closed transform S-стадии (K×Q fp8 coopmat) | опционально |
| native | `GGML_VK_FA_F8_NATIVE=1` | f8 K/V, neq1>1, f32acc | полный native fp8 WMMA (экспериментальный диагностический) | эксперимент |
| P1 direct | `GGML_VK_FA_F8_DIRECT=1` | f8 K/V, neq1>1 | canonical cm1 с `DATA_A_F8_E4M3`: только загрузка тайлов K/V отличается | эксперимент |
| preconvert | (default; откл. `GGML_VK_FA_NO_PRECONVERT=1`) | q8_0 или f8_e4m3 K/V, neq1>1 | K/V → f16 отдельным проходом, FA идёт по чистому f16-пути | default fallback |

Диагностика A/B native vs fallback внутри процесса:
`GGML_VK_FA_HALF_CMP=N` (первые N prefill-вызовов native, далее через запрос
по чётности) — для попарного сравнения outdump-файлов.

## 3. Квантование KV в f8: битовый энкодер (D096-L)

`ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`, `f32_to_fp8_e4m3`:

- программный вариант (Log2/Floor/Ldexp/FDiv/Round, ~40 инструкций на
  компонент) **отклонён** (D096) — ел преимущество fp8 по памяти;
- битовый вариант (текущий, ~12 инструкций):
  - `exp_field = exp32 − 120` (сдвиг bias), `mantissa =
    ((bits & 0x7FFFFF) + 0x80000) >> 20` (round-half-up с переносом в
    экспоненту), `f >= 240 → clamp` (exp 14, man 7);
  - subnormal-зона (exp32 < 121) без изменений.
- Точность (эмуляция на 300K значений, Q/P-домен): 96.8% бит-в-бит совпадает
  со старым; 3.2% отличаются на 1 ulp мантиссы — новый точнее (корректный
  carry вместо clamp после округления).
- Затрагивает все f8-пути: KV-квантование, P2/P3/P4/P5, `fa_q_f32_f8`,
  `copy_to_quant`.

## 4. Гибридный KV-кэш (D096-K) и дефолты

Проблема: MTP-дравт читает KV только последнего слоя внимания. fp8-шум в
последних слоях ломает распределение дравта: acceptance падает с 82% до 27%
(f8-чистый), decode 54 → 30 t/s.

Решение (`src/llama-kv-cache.cpp`): env `LLAMA_VK_MTP_KV_LAST_F16=N` —
последние N KV-слоёв (фильтр-сознательно, `last_kv_il`/`kv_f16_start`)
хранятся в f16, остальные — в квантованном виде (f8_e4m3 ИЛИ q8_0;
D096-P). Лог при запуске:
`MTP + quantized KV: enabling hybrid KV cache (last N layers f16)`.

**Дефолт N=8** (`common/common.cpp`, `common_context_params_to_llama`):
при `--spec-type draft-mtp` + K/V в `f8_e4m3` или `q8_0` без явного env
автоматически выставляется `LLAMA_VK_MTP_KV_LAST_F16=8`.
Отключение: `LLAMA_VK_MTP_KV_LAST_F16=0`; другое N — любым значением.

### Кривая N (49K-лейн, 49152 ctx, MTP draft n=2, P5)

| N | KV MiB | decode t/s | acceptance | примечание |
| --- | --- | --- | --- | --- |
| 0 (f8 чистый) | 1536 | 30.5 | 27% | базовая точка |
| 1 | 1632 | 41.0 | 52% | |
| 4 | 1920 | 46.8 | 67% | −37.5% памяти vs f16 |
| **8 (дефолт)** | **2304** | **51.7*** | **80%*** | *свежий прогон default-r1; lf8-r1: 48.3/75% |
| 16 (=f16) | 3072 | 54.0 | 82% | контроль |

f16-слой стоит **+96 MiB** (K+V) относительно f8-слоя — учитывать при
оценке памяти. N=8: −25% KV-памяти vs f16 при −4% decode; префилл не
затрагивается (1602–1609 pt/s во всех точках).

## 5. Производительность (сводно, Qwen3.6-27B-Q4_K_M, dual Vulkan)

| Конфигурация | Лейн | pt/s | decode t/s |
| --- | --- | --- | --- |
| f16 KV (контроль) | 12K | ~1606 | — |
| q8_0 KV (preconvert) | 12K | 1432.5 | — |
| **f8 + P5 (битовый энкодер)** | 12K | **1621.1** | — |
| f16 KV (контроль) | 49K | ~1376 | — |
| **f8 + P5** | 49K | **1399.0** | — |
| f8 + P5 + MTP n=2 (гибрид N=8, дефолт) | 12K MTP | 1602.8 | **51.7** (acc 80%) |
| f16 + MTP n=2 (контроль) | 12K MTP | 1602.4 | 54.0 (acc 82%) |

Правила замеров: свежая сессия, A/B соседними прогонами, не более ~6
прогонов до валидного замера (тепловой дрейф −3..5%), GPU free-проверка
перед стартом (`tasklist | grep llama` пуст). Первый прогон после сборки —
аномалия (r2 подтверждает).

## 6. GUI и бенчмарки (D096-M)

- Чекбоксы P5 **убраны** (`gui/server_backend_panels.py`,
  `gui/benchmark_tab.py`): `GGML_VK_FA_F8_P5=1` выставляется всегда для
  Vulkan (ядро игнорирует его при не-f8 KV — для выбора «просто f8 KV
  кэш» без дополнительных опций). Legacy-настройка `fp8_p5` из старых
  сессий игнорируется.
- Smoke-тест `build_logs/agent-workload/gui_server_smoke.py` поднимает
  сервер командой из `ServerTabWidget._compose_server_command()`: health OK,
  замеры через `body["timings"]` ответа (не `/slots` — там timings пустые
  после release), graceful stop CTRL_BREAK.
- GUI-дефолты (ctx 8192, b 512/ub 256, `-dev Vulkan0,Vulkan1` — порядок
  GUI по умолчанию): prefill 2006 ток ≈ 1215 pt/s, decode ~31 t/s.
  Рекомендация для максимальной скорости: в Backend Settings выбрать
  порядок `Vulkan1,Vulkan0` (вторая GPU первой, правило машины).

## 7. Известные ограничения и ловушки

- **Пустой `content` при `finish=length`** — артефакт thinking-модели
  (весь max_tokens уходит на reasoning), НЕ баг fp8: q8_0 и f8+P5 дают
  идентичное качество ("391+2"=393; Python-код корректен при достаточном
  лимите; fp8: finish=stop на 986 токенах при max_tokens=1500). GUI-дефолт
  `n_predict` поднят со 128 до 1024 (D096-N) — старый дефолт всегда обрезал
  thinking-ответы; для содержательного ответа нужно `max_tokens >= 400`
  (лучше 1000+).
- **P5 accuracy**: P после softmax в [0,1], E4M3 имеет 3-битную мантиссу;
  per-row scale митигирует; acc — f32 (больше регистров). Качество-гейт
  для коротких ответов пройден («393» корректно).
- **Квантование не является причиной коллапса MTP-декода** — причина в
  чтении дравтом только последнего слоя KV; гибрид решает именно это.
  Гибрид работает для f8_e4m3 И q8_0 (D096-P): q8+MTP был регрессией
  (acceptance 31%) после D094 — теперь 74%/51.7 t/s (тот же гибрид).
- **«Мусор в ответах q8» не воспроизводится** (D096-O, 20+ прогонов
  точной GUI-команды: q8/f8/f16 × MTP/none × temp 0/0.7): наблюдаются
  только finish=length с пустым content при малых лимитах (thinking
  съедает лимит) и длинные thinking-процессы в тексте — у всех KV,
  включая f16. Требуется max_tokens >= 1024 для коротких ответов.
- **Драйвер AMD**: не хард-киллить активный GPU-сервер (`taskkill //F`) —
  грязный драйвер, segfault 139 при следующей инициализации. Graceful stop
  (CTRL_BREAK), после hard-kill — пауза перед следующими GPU-тестами.
- **Сборка шейдеров**: правки `.spvasm`/`.comp` не подхватываются при
  равенстве mtime (точность ninja — секунды): `touch` файла и проверка
  шага `Generate vulkan shaders` в выводе сборки. MinGW-линкованные
  утилиты (vulkan-shaders-gen) требуют `C:\Strawberry\c\bin` в PATH.

## 8. Как включить

Сервер (рекомендуемая production-конфигурация):

```bash
build-vulkan/bin/llama-server.exe -m models/Qwen3.6-27B-Q4_K_M.gguf \
  --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -dev Vulkan1,Vulkan0 -sm layer -ts 1,1 -ngl 99 --no-mmap -fit off
# гибридный KV (последние 8 слоёв f16) включится автоматически
```

Bench (12K-лейн):

```bash
export GGML_VK_FA_F8_P5=1
python scripts/agent_workload_bench.py --server-bin build-vulkan/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q4_K_M.gguf --ctx-size 49152 --ubatch-size 256 \
  --cache-type-k f8_e4m3 --cache-type-v f8_e4m3 --task-ids triage_diff --no-reuse \
  --real-context-mode repo-snapshot --real-context-chars 24576 \
  --server-extra "-dev Vulkan1,Vulkan0 --spec-type none --no-mmap -fit off" \
  --label d096-main-p5-12k
```

Контрольные конфигурации: f16 KV (те же флаги, `--cache-type-k f16
--cache-type-v f16`); MTP-контроль — `--spec-type draft-mtp` + f16 KV.
