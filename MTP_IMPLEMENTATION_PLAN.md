# MTP Implementation Plan

Фокус: ROCm build на RX 9070 XT, text-only Qwen3.6 сначала, GUI после core validation.

## Phase 1: Baseline

1. Собрать/подтвердить текущий ROCm build:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --config Release -j 4 --target llama-server llama-cli
```

2. Прогнать короткий агентный benchmark:

```powershell
python scripts\agent_workload_bench.py --label rocm-baseline
```

3. Сохранить CSV/JSONL/server log из `build_logs/agent-workload/`.

Rollback point: commit baseline benchmark tool and results metadata, not model files.

## Phase 2: MTP Branch

1. Создать отдельную ветку:

```powershell
git switch -c feature/qwen-mtp
git fetch upstream pull/22673/head:upstream-pr-22673
git merge --no-ff upstream-pr-22673
```

2. При конфликтах приоритет:

- сохранить TurboQuant changes;
- сохранить GUI layer;
- принять upstream MTP core changes в `common/`, `src/`, `include/`, `ggml/`, `tools/server/`, `convert_hf_to_gguf.py`;
- не импортировать upstream `.github/**`, `docs/**`, root docs.

3. Не добавлять GUI controls на этом этапе.

Rollback point: если merge слишком грязный, удалить ветку и попробовать cherry-pick только MTP commits после анализа `git show --stat`.

## Phase 3: ROCm Build

1. Собирать отдельно:

```powershell
cmake -B build-rocm-mtp -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-mtp --config Release -j 4 --target llama-server llama-cli
```

2. Проверить capability:

```powershell
build-rocm-mtp\bin\llama-server.exe --help | Select-String -Pattern "spec-type|mtp"
```

Gate: без `mtp` в help дальше не идти.

## Phase 4: Text-Only Smoke Test

1. Получить MTP-enabled Qwen3.6 GGUF.
2. Запустить без GUI и без mmproj:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-mtp-draft3 `
  --server-bin build-rocm-mtp\bin\llama-server.exe `
  --model models\Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf `
  --server-extra "--spec-type mtp --spec-draft-n-max 3"
```

3. Сравнить с `rocm-baseline`.

Gate:

- нет crash;
- VRAM не уходит в OOM;
- `completion_tps_wall` лучше baseline на большинстве задач;
- server log показывает acceptance rate;
- PP regression не делает весь workload медленнее.

## Phase 5: ngram-mod Comparison

На текущем или MTP branch проверить:

```powershell
python scripts\agent_workload_bench.py `
  --label rocm-ngram-mod `
  --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-draft-n-min 48 --spec-draft-n-max 64"
```

Gate: если agent workload лучше с `ngram-mod`, сделать его отдельным GUI preset для coding tasks.

## Phase 6: GUI Integration

Добавлять только после успешного ROCm text-only теста.

GUI изменения:

- группа `Speculative Decoding`;
- `none`, `ngram-mod`, `mtp`;
- MTP draft tokens spinbox, default `3`;
- disable/warn MTP when Vision/MMProj enabled;
- force/warn `parallel=1` for MTP;
- detect binary support by `llama-server --help`;
- show warning if selected model filename does not contain `MTP`/`NextN`.

## Phase 7: Benchmarked Defaults

После нескольких прогонов записать в `gui/model_presets.json`:

- `Qwen3.6 text ROCm baseline`;
- `Qwen3.6 text MTP draft3`;
- `Qwen3.6 coding ngram-mod`;
- `Qwen3.6 VLM no-MTP`.

## Do Not Do Yet

- Не смешивать MTP и `--mmproj`.
- Не включать MTP по умолчанию.
- Не оптимизировать CUDA/Metal раньше ROCm.
- Не переписывать speculative pipeline с нуля, пока PR #22673 можно портировать.
