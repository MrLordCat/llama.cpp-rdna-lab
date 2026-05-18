# Qwen Speed Research

Дата среза: 2026-05-07.

## ОБНОВЛЕНИЕ 2026-05-18: текущий acceleration cycle заархивирован

- Активная no-spec cold-first lane остановлена на `Qwen3.6-27B-Q3_K_S`, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, thinking on, no reuse.
- Последние reference controls: E045 `11.6534 TPS`, E053 `11.7681 TPS`, E056 `11.6726 TPS`, E058 `11.6132 TPS`.
- E053-E059 закрыли простые продолжения: broad/shape MMQ для `Q3_K ne11=2048`, compute16, hipBLASLt/Stream-K env sweeps, GDN chunking, Q3_K 128-thread/half2/unroll4 conversion variants.
- Главный архивный документ: `docs/research/PERFORMANCE_ARCHIVE_2026-05-18.md`.
- Возобновлять работу только при новом upstream/RDNA4 сигнале, MTP-enabled GGUF, изменившемся route mix или новой high-ceiling design gate идее.

## ОБНОВЛЕНИЕ 2026-05-10: стартовая точка <16k теперь главный performance target

- После prompt-heavy context-wall проверки активный фокус смещён на стартовую точку ниже `16k` (текущий reference `ctx=12288`).
- Причина: даже на `16k-32k` при большом входящем prompt throughput резко деградирует, поэтому 64k уже не является ближайшей «точкой входа» для оптимизации.
- Новая цель: `25-27 TPS` на стартовой prompt-heavy точке через изменения кода llama.cpp/ggml.

## ОБНОВЛЕНИЕ 2026-05-12: native `ub1024` cliff на RDNA4/ROCm исправлен

- Симптом: после снятия старого guard/cap полный `PP reserve outputs 904/1024 -> 1` мог падать в prompt-prefill slow pocket (`~300 tok/s`) при тех же FATTN/GDN/MMQ routes.
- Диагностика: full trace показал одинаковые node counts и kernel route classes, но широкое замедление memory-heavy ops (GLU/RMS_NORM/ADD/SSM_CONV/MUL_MAT/FATTN).
- Root cause: один крупный ROCm graph compute vbuffer allocation попадал в плохой RDNA4/Windows residency/placement pocket.
- Фикс: `ggml/src/ggml-alloc.c` теперь по умолчанию режет ROCm graph compute vbuffer на backend chunks максимум `256 MiB`.
- Контроль: `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` возвращает старый slow path и используется как negative control.
- Результат: `ctx=32768,b=5120,ub=1024,q4_0/q4_0,ngram-mod` даёт `PP reserve outputs 1024 -> 1`, prompt `~1114 tok/s`, decode `~25 tok/s` на practical `max-tokens=120`.

## ОБНОВЛЕНИЕ 2026-05-08: MTP УЖЕ РЕАЛИЗОВАНА

Полный поиск кодовой базы показал, что `common_speculative_state_mtp` уже реализована в `common/speculative.cpp` (строка 604+) и `llama_set_mtp()` существует в `src/llama-context.cpp`. Текущий `llama-server --help` показывает `--spec-type mtp` как доступный вариант.

**Что работает**:
- MTP enum (`COMMON_SPECULATIVE_TYPE_MTP`) + string mapping
- Draft token generation из MTP контекста
- Recurrent state rollback для rejected tokens
- Server integration (loading MTP head, slot management)

**Что нужно**:
- MTP-enabled GGUF для Qwen3.6 (HuggingFace или self-convert)
- Бенчмарк на RX 9070 XT для baseline (сравнить с ngram-mod 26.52 TPS)
- Профилирование memory overhead (MTP может требовать больше VRAM)

**Практический план**:
1. Найти или сконвертировать Qwen3.6-27B-MTP GGUF
2. Запустить: `llama-server -m ...mtp.gguf --spec-type mtp --spec-draft-n-max 3`
3. Сравнить TG (token generation) с baseline
4. Если стабильно выше на 50%+ → сделать experimental profile в GUI

## Вывод

Да, MTP можно внедрять в этот форк самостоятельно, примерно как раньше добавлялся TurboQuant, но это крупнее и рискованнее, чем просто добавить новый quant type. MTP затрагивает model loader, model architecture registry, graph builder, context hooks, recurrent-state rollback, server batching и converter. Самый быстрый путь — не писать с нуля, а портировать PR `ggml-org/llama.cpp#22673` в отдельную ветку и затем локально стабилизировать под RX 9070 XT / ROCm.

Для Qwen-моделей самые перспективные ускорения:

1. MTP для Qwen3.6 MTP-enabled GGUF.
2. `ngram-mod` для coding-agent задач с повторяющимся контекстом.
3. Draft-model speculative decoding для non-MTP моделей.
4. KV cache compression: `q8_0`, `q4_0`, и локальный TurboQuant `tq3_0`, если качество/стабильность приемлемы.
5. ROCm/Vulkan-specific tuning: `-np 1`, `-ngl 999`, `--flash-attn on`, правильный `-b/-ub`, HIP SDK 7.1, `gfx1201`.
6. GUI-level auto-detection: включать MTP только если binary реально поддерживает `--spec-type mtp` и модель похожа на MTP GGUF.

## MTP: что нужно портировать

PR #22673 меняет 40+ файлов. Основные зоны:

| Зона | Файлы | Зачем |
| --- | --- | --- |
| CLI args/spec type | `common/arg.cpp`, `common/common.h`, `common/speculative.cpp` | добавить `COMMON_SPECULATIVE_TYPE_MTP` и `--spec-type mtp` |
| MTP speculative state | `common/speculative.cpp` | генерировать draft tokens из MTP context |
| Context hook | `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-mtp.h` | передавать hidden states trunk-модели в MTP head |
| Model arch registry | `src/llama-arch.*`, `src/llama-model.cpp`, `src/models/models.h` | добавить `qwen35_mtp`, `qwen35moe_mtp` |
| Qwen MTP graph | `src/models/qwen35_mtp.cpp`, `src/models/qwen35moe_mtp.cpp` | построить graph MTP блока |
| Recurrent rollback | `src/llama-memory-recurrent.*`, `src/llama-memory-hybrid*`, `src/llama-cparams.h`, `include/llama.h` | partial rollback для rejected draft tokens |
| Server integration | `tools/server/server-context.cpp` | загрузить MTP head из того же GGUF, отключить несовместимые пути |
| Converter/GGUF metadata | `convert_hf_to_gguf.py`, `gguf-py/gguf/constants.py` | конвертировать HF MTP tensors в GGUF |
| Backend fixes | `ggml/src/ggml-*` | gated delta net / recurrent state / backend ops support |

## Почему это не просто GUI-флаг

Текущий форк уже умеет speculative decoding, но не умеет MTP. Если просто передать:

```text
--spec-type mtp --spec-draft-n-max 3
```

локальный `llama-server` отклонит `mtp`, потому что `COMMON_SPECULATIVE_TYPE_MTP` отсутствует. Даже если добавить enum/arg, сервер должен уметь:

- загрузить MTP head из того же GGUF через `override_arch`;
- получить hidden state `t_h_pre_norm` от trunk-модели;
- декодировать MTP context;
- откатывать recurrent state после rejected draft tokens;
- не ломать server slot accounting;
- отключать `ctx_shift`, `cache_reuse`, `n_parallel > 1` для MTP.

## Текущие риски PR #22673

Из обсуждения PR:

- PR открыт и draft, не merged.
- Есть reports о большом recurrent state memory overhead, который уже частично чинится автором.
- MTP + multimodal/mmproj воспроизводимо падает у пользователей; text-only работает.
- `n_parallel > 1` в PR запрещён.
- На RX 9070 XT + RX 6600 через Vulkan один пользователь видел TG 22 -> 42 t/s, но PP 750 -> 296 t/s, и coding workload стал примерно на 20% медленнее из-за prefill regression.
- Для multi-GPU AMD может быть важен порядок `--device` / `--spec-device-draft`.

Для нашего форка это означает: MTP должен быть отдельным experimental profile, не default.

## Ожидаемый профит на этой машине

### Qwen3.6-35B-A3B / Qwen3.6-27B MTP

Реалистичная гипотеза для RX 9070 XT:

- TG: возможно +50-100% при text-only generation-heavy задачах.
- PP: может не измениться или просесть, особенно если branch затрагивает prompt processing / recurrent state.
- VRAM: может вырасти из-за MTP context и recurrent rollback.
- Лучший старт: `--spec-draft-n-max 3`, не 5.

### Qwen coding-agent workload

Для задач, где агент много читает/переписывает повторяющийся код, `ngram-mod` может быть выгоднее MTP:

```text
--spec-type ngram-mod
--spec-ngram-mod-n-match 24
--spec-draft-n-min 48
--spec-draft-n-max 64
```

MTP ускоряет novel-token generation, а ngram ускоряет повторы из уже увиденного контекста. Идея совмещать MTP + ngram обсуждается в PR, но в текущей реализации это не готовый режим.

## План внедрения MTP в наш форк

### Этап 0: baseline

Перед портированием замерить текущий master:

```powershell
build-rocm\bin\Release\llama-server.exe -m models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf -ngl 999 --flash-attn on -np 1 -c 32768 --cache-type-k q8_0 --cache-type-v q8_0 --port 8081
```

Снять:

- prompt processing tok/s;
- generation tok/s;
- VRAM;
- latency first token;
- стабильность длинной генерации.

### Этап 1: отдельная ветка

```powershell
git switch -c feature/qwen-mtp
git fetch upstream pull/22673/head:upstream-pr-22673
git merge --no-ff upstream-pr-22673
```

Если конфликтует с TurboQuant или GUI, не править GUI на этом этапе. Сначала поднять core.

### Этап 2: минимальная сборка

```powershell
cmake -B build-rocm-mtp -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-mtp --config Release -j 4 --target llama-server llama-cli
```

### Этап 3: binary capability detection

```powershell
build-rocm-mtp\bin\llama-server.exe --help | Select-String -Pattern "spec-type|mtp"
```

GUI должен включать MTP controls только если этот тест успешен.

### Этап 4: text-only MTP smoke test

```powershell
build-rocm-mtp\bin\llama-server.exe `
  -m models\Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf `
  -ngl 999 `
  --flash-attn on `
  -np 1 `
  -c 32768 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  --spec-type mtp `
  --spec-draft-n-max 3 `
  --port 8081
```

Не включать `--mmproj` на первом проходе.

### Этап 5: GUI integration

Только после smoke test:

- добавить `Speculative Decoding` group в Launch Server;
- варианты: `none`, `ngram-mod`, `mtp`, `draft model`;
- для MTP показывать warning:
  - requires MTP-enabled GGUF;
  - forces `-np 1`;
  - incompatible with Vision/MMProj until fixed;
  - disables ctx shift/cache reuse where applicable;
- проверять binary help перед стартом.

## Другие ускорения кроме MTP

### 1. Qwen preset tuning

Добавить в `gui/model_presets.json` отдельные профили:

- `Qwen3.6 MTP text-only`;
- `Qwen3.6 coding ngram-mod`;
- `Qwen3.6 VLM no-MTP`;
- `Qwen3.5 9B low-latency`.

### 2. Speculative draft model

Для non-MTP Qwen3.6 можно тестировать:

```text
-hfd unsloth/Qwen3.5-0.8B-GGUF:Q8_0
--spec-draft-n-max 16
--spec-draft-p-min 0.75
```

Риск: дополнительная VRAM/RAM и tokenizer/model compatibility.

### 3. KV cache strategy

Для 16 GB VRAM:

- short coding: `q8_0` KV;
- long context: `q4_0` KV;
- experimental: `tq3_0` KV with flash attention, если текущий build стабилен.

### 4. Backend A/B runner

Сделать маленький GUI/CLI benchmark wrapper:

- одна модель;
- один prompt set;
- ROCm vs Vulkan;
- MTP vs no MTP;
- q8_0 vs q4_0 vs tq3_0 KV;
- вывод CSV в `build_logs/bench-qwen-*.csv`.

Это даст больше пользы, чем угадывать параметры на глаз.

## Решение

Рекомендуемый порядок:

1. Дочистить repo profile/docs/ignore.
2. Сделать `feature/qwen-mtp`.
3. Подтянуть PR #22673 как merge, не cherry-pick по одному файлу.
4. Починить конфликты с TurboQuant.
5. Собрать `build-rocm-mtp`.
6. Сравнить Qwen3.6-35B-A3B text-only baseline vs MTP.
7. Если TG растёт без PP/VRAM катастрофы, добавить GUI controls.

## Источники

- `ggml-org/llama.cpp#22673`: https://github.com/ggml-org/llama.cpp/pull/22673
- Upstream speculative decoding docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- Qwen3.6 MTP GGUF notes: https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF
- ROCm llama.cpp docs: https://rocm.docs.amd.com/projects/llama-cpp/en/docs-25.09/
