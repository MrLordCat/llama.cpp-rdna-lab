# MTP / Multi-Token Prediction Notes

Дата среза: 2026-05-07.

## Что такое MTP

MTP, Multi-Token Prediction, это вариант speculative decoding, где модель имеет встроенные головы/модули для предсказания нескольких следующих токенов. В отличие от классического speculative decoding, отдельная маленькая draft-модель не обязательна: MTP-enabled GGUF может сам генерировать draft-токены, а основная модель затем проверяет и принимает удачные токены пачкой.

Практический смысл: если acceptance rate высокий, generation throughput может вырасти примерно в 1.5-2x или выше. Prompt processing обычно почти не меняется.

## Статус в llama.cpp

На момент проверки MTP поддержка не находится в текущем локальном дереве форка как стабильная функция.

Актуальная upstream работа:

- PR: `https://github.com/ggml-org/llama.cpp/pull/22673`
- Название: `llama + spec: MTP Support`
- Автор: `am17an`
- Статус на GitHub при проверке: Draft
- Дата PR: 2026-05-04

PR заявляет поддержку MTP heads и тесты на Qwen3.6 27B / Qwen3.6 35B-A3B. В описании PR показан steady-state acceptance около 72-75% с 3 draft tokens и wall time лучше baseline примерно в 2x на опубликованном benchmark.

## Статус в этом форке

Локальный поиск показал:

- `common/speculative.cpp` поддерживает `draft`, `eagle3`, `ngram_simple`, `ngram_map_k`, `ngram_map_k4v`, `ngram_mod`, `ngram_cache`.
- `common/arg.cpp` не содержит `mtp` в списке `--spec-type`.
- `src/llama-model.cpp` содержит комментарии про NextN/MTP tensors как сохранённые, но не используемые runtime-логикой.
- GUI имеет поле `Extra Arguments`, но передача `--spec-type mtp` в текущую локальную сборку приведёт к ошибке, пока не подтянут код PR/commit с MTP.

## Как понять, что MTP уже можно включать

После обновления core llama.cpp проверить:

```powershell
rg -n "COMMON_SPECULATIVE_TYPE_MTP|spec.*mtp|--spec-type.*mtp|MTP Support" common src include tools examples docs
```

И проверить help:

```powershell
build-rocm\bin\Release\llama-server.exe --help | Select-String -Pattern "spec-type|mtp"
```

MTP можно считать доступным только если help явно показывает `mtp` как допустимый `--spec-type`, а сервер стартует с MTP-enabled GGUF.

## Пример команды после подтягивания MTP

```powershell
build-rocm\bin\Release\llama-server.exe `
  -m models\Qwen3.6-27B-MTP-Q4_K_M.gguf `
  --spec-type mtp `
  --spec-draft-n-max 3 `
  --flash-attn on `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  -ngl 999 `
  -c 32768 `
  -np 1 `
  --port 8081
```

Через текущий GUI это временно можно будет проверить в `Launch Server -> Extra Arguments`:

```text
--spec-type mtp --spec-draft-n-max 3
```

Только после того, как локальный `llama-server --help` подтвердит поддержку.

## Модели

MTP не включается на любой GGUF. Нужен GGUF, в котором сохранены MTP/NextN tensors и модельная архитектура поддерживает такой режим. На момент исследования чаще всего упоминаются Qwen3.6 MTP variants:

- Qwen3.6-27B MTP GGUF.
- Qwen3.6-35B-A3B MTP GGUF.

Для RX 9070 XT 16 GB начинать лучше с Q4/IQ4 quant, `-np 1`, `-c 32768`, `--cache-type-k q8_0 --cache-type-v q8_0`. Если VRAM тесно, снижать context и переходить на `q4_0` KV.

## Риски

- PR был draft при проверке, значит API/аргументы могут измениться.
- MTP может конфликтовать с multimodal/vision путём. В community notes встречались предупреждения, что vision + MTP может падать на конкретных сборках PR.
- Ускорение зависит от acceptance rate. Для некоторых задач и prompt-стилей прирост может быть меньше.
- Для ROCm/RDNA4 обязательно проверять не только старт сервера, но и длинную генерацию без memory faults.

## План внедрения в форк

1. Не включать MTP UI как стабильный режим, пока upstream PR не будет либо смержен, либо локально протестирован.
2. Подтянуть MTP в отдельную ветку, не смешивая с GUI-документацией.
3. Собрать ROCm и CPU sanity build.
4. Проверить `llama-server --help` и запуск без MTP.
5. Проверить запуск с MTP-enabled GGUF и `--spec-type mtp --spec-draft-n-max 3`.
6. Сравнить tokens/sec на одинаковом prompt с MTP и без MTP.
7. Только после успешного теста добавить GUI-переключатель:
   - `Speculative Type: none / draft / ngram / mtp`
   - `MTP draft tokens`
   - warning, если выбран multimodal/vision
   - warning, если server binary не показывает `mtp` в `--help`

Более подробная карта портирования и Qwen speed roadmap: [QWEN_SPEED_RESEARCH.md](QWEN_SPEED_RESEARCH.md).

## Источники

- GitHub PR `ggml-org/llama.cpp#22673`: `https://github.com/ggml-org/llama.cpp/pull/22673`
- Upstream speculative decoding docs: `https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md`
- NVIDIA Megatron MTP overview: `https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/multi-token-prediction.html`
