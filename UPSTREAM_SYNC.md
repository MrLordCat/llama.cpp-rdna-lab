# Upstream Sync Policy

Цель: догонять `ggml-org/llama.cpp` по важным core/runtime изменениям, не импортируя upstream Actions, upstream documentation и чужие агентские инструкции поверх локального форка.

## Почему это нужно

Этот репозиторий не является чистым mirror upstream. Здесь есть:

- PyQt6 GUI;
- ROCm/RDNA4 workflow под RX 9070 XT;
- TurboQuant изменения;
- локальная документация;
- локальные агентские инструкции;
- собственные GitHub/project файлы.

Слепой `git merge upstream/master` может принести полезный runtime-код, но одновременно перезаписать `.github`, `docs`, `README.md` и instruction-файлы.

## Защищённые зоны

Не импортировать из upstream без отдельного решения:

```text
.github/**
docs/**
README.md
AGENTS.md
CLAUDE.md
MTP.md
UPSTREAM_SYNC.md
gui/README.md
gui/QUICKSTART.md
.gemini/**
.devops/**
```

Эти правила также записаны в `.gitattributes` через `merge=ours`.

## Одноразовая локальная настройка merge driver

В каждом clone нужно один раз включить driver `ours`:

```powershell
git config merge.ours.driver true
```

Проверка:

```powershell
git config --get merge.ours.driver
```

Ожидаемый вывод:

```text
true
```

## Безопасная процедура

1. Проверить рабочее дерево:

```powershell
git status --short --branch
```

2. Получить upstream:

```powershell
git fetch upstream
```

3. Посмотреть масштаб изменений:

```powershell
git log --oneline --left-right --cherry-pick HEAD...upstream/master
git diff --stat HEAD...upstream/master
```

4. Если есть локальные незакоммиченные GUI/ROCm/TurboQuant изменения, не начинать merge, пока не понятно, как их сохранить.

5. Для обычного догоняющего merge:

```powershell
git merge --no-ff upstream/master
```

6. Если Git всё равно принёс изменения в защищённые зоны, вернуть локальную сторону:

```powershell
git checkout --ours README.md AGENTS.md CLAUDE.md MTP.md UPSTREAM_SYNC.md
git checkout --ours .github docs gui/README.md gui/QUICKSTART.md
git add README.md AGENTS.md CLAUDE.md MTP.md UPSTREAM_SYNC.md .github docs gui/README.md gui/QUICKSTART.md
```

7. Разрешить конфликты только в core/runtime коде.

8. Проверить:

```powershell
python -m py_compile run.py gui\main_window.py gui\server_tab.py gui\benchmark_tab.py gui\build_tab.py gui\build_manager.py gui\dependency_checker.py gui\hardware_detector.py
git diff --check
```

9. Для ROCm-sensitive изменений сначала configure:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
```

## Что импортировать из upstream в первую очередь

Обычно полезно брать:

- `common/`
- `src/`
- `include/`
- `ggml/`
- `tools/`
- `examples/`
- `CMakeLists.txt`
- `cmake/`
- model conversion scripts, если они нужны для новых GGUF/model architectures

С осторожностью:

- `requirements/`
- `pyproject.toml`
- `scripts/`
- `tests/`
- `vendor/`

Не брать автоматически:

- `.github/`
- `docs/`
- root README;
- upstream agent instruction files;
- чужие issue templates/actions.

## MTP sync branch

Для MTP лучше делать отдельную ветку:

```powershell
git switch -c feature/mtp-sync
git fetch upstream pull/22673/head:upstream-pr-22673
git merge --no-ff upstream-pr-22673
```

После этого проверить:

```powershell
rg -n "mtp|MTP|COMMON_SPECULATIVE_TYPE" common src include tools examples
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --config Release -j 4 --target llama-server
```

Не переносить MTP в `master`, пока:

- сервер без MTP запускается как раньше;
- `--spec-type mtp` есть в `llama-server --help`;
- MTP-enabled GGUF реально генерирует быстрее baseline;
- multimodal/vision путь не ломается или явно отключает MTP.

## Если нужно перенести только core без merge

Иногда безопаснее создать временную ветку и cherry-pick конкретные commits upstream, чем делать большой merge. Для MTP или ROCm fixes это предпочтительно, если PR маленький и зоны изменений понятны.

Перед cherry-pick:

```powershell
git show --stat <commit>
git show --name-only <commit>
```

Не cherry-pick commit, который в основном меняет `.github`, docs или release automation, если цель была core/runtime.
