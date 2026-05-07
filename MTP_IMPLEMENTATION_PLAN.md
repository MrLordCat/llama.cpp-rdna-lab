# MTP Implementation Plan

Фокус: ROCm build на RX 9070 XT, text-only Qwen3.6 сначала, GUI после core validation.

Статус на 2026-05-07:

- core/runtime часть PR `ggml-org/llama.cpp#22673` уже влита в текущий `master`;
- текущий `build\bin\llama-server.exe --help` показывает `mtp` и `ngram-mod` в `--spec-type`;
- `ngram-mod` уже проверен benchmark runner и даёт measurable gain на обеих целевых Qwen3.6 моделях;
- полноценный MTP benchmark ещё не сделан, потому что нужен MTP-enabled GGUF и отдельный text-only smoke test.

## Phase 1: Baseline

1. Собрать/подтвердить текущий ROCm build:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --config Release -j 4 --target llama-server llama-cli
```

1. Прогнать короткий агентный benchmark:

```powershell
python scripts\agent_workload_bench.py --label rocm-baseline
```

1. Сохранить CSV/JSONL/server log из `build_logs/agent-workload/`.

Rollback point: commit baseline benchmark tool and results metadata, not model files.

## Phase 2: Current Integration State

Отдельная ветка больше не требуется: MTP core уже находится в текущем `master`.

Что уже считается сделанным:

- speculative runtime знает `mtp` и `ngram-mod`;
- loader/arch/runtime path для Qwen MTP уже в дереве;
- protected local docs/GUI слой сохранён поверх merge;
- baseline и `ngram-mod` benchmark уже сняты.

Что ещё не считается завершённым:

- MTP-enabled Qwen3.6 GGUF ещё не проверен локально;
- нет text-only smoke benchmark с `--spec-type mtp`;
- нет GUI controls и GUI benchmark mode.

## Phase 3: ROCm Build

1. Базовый GUI/ROCm configure на текущем `master` уже проходит.

1. Для изолированных MTP smoke tests допустимо собирать отдельно:

```powershell
cmake -B build-rocm-mtp -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-mtp --config Release -j 4 --target llama-server llama-cli
```

1. Проверить capability:

```powershell
build-rocm-mtp\bin\llama-server.exe --help | Select-String -Pattern "spec-type|mtp"
```

Gate: без `mtp` в help дальше не идти.

## Phase 4: Text-Only Smoke Test

1. Получить MTP-enabled Qwen3.6 GGUF.
1. Запустить без GUI и без mmproj:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-mtp-draft3 `
  --server-bin build-rocm-mtp\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf `
  --server-extra "--spec-type mtp --spec-draft-n-max 3"
```

1. Сравнить с `rocm-baseline`.

Gate:

- нет crash;
- VRAM не уходит в OOM;
- `completion_tps_wall` лучше baseline на большинстве задач;
- server log показывает acceptance rate;
- PP regression не делает весь workload медленнее.

## Phase 5: ngram-mod Comparison

Статус: уже проверено на текущем build.

Полученный clean snapshot:

- `Qwen3.6-35B-A3B-UD-IQ3_XXS`: `37.454 -> 41.007` aggregate completion TPS, примерно `+9.5%`;
- `Qwen3.6-27B-Q3_K_S`: `12.055 -> 13.547` aggregate completion TPS, примерно `+12.4%`.

Команда для повторного прогона:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-ngram-mod `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-draft-n-min 48 --spec-draft-n-max 64"
```

Gate: `ngram-mod` уже заслуживает отдельного GUI preset для coding tasks.

## Phase 6: GUI Integration

Добавлять только после успешного ROCm text-only теста.

GUI изменения:

- группа `Speculative Decoding`;
- `none`, `ngram-mod`, `mtp`;
- MTP draft tokens spinbox, default `3`;
- `ngram-mod` quick preset для coding-agent workloads;
- disable/warn MTP when Vision/MMProj enabled;
- force/warn `parallel=1` for MTP;
- detect binary support by `llama-server --help`;
- show warning if selected model filename does not contain `MTP`/`NextN`.
- отдельный benchmark mode/tab/dialog, чтобы не перегружать основной Launch Server экран.

## Phase 7: Benchmarked Defaults

После нескольких прогонов записать в `gui/model_presets.json`:

- `Qwen3.6 text ROCm baseline`;
- `Qwen3.6 text MTP draft3`;
- `Qwen3.6 coding ngram-mod`;
- `Qwen3.6 VLM no-MTP`.

Дополнительно:

- хранить build commit вместе с benchmark metadata;
- сохранять отдельный md-summary по билдам, чтобы сравнение между будущими ROCm builds было явным.

## Do Not Do Yet

- Не смешивать MTP и `--mmproj`.
- Не включать MTP по умолчанию.
- Не оптимизировать CUDA/Metal раньше ROCm.
- Не переписывать speculative pipeline с нуля, пока PR #22673 можно портировать.
