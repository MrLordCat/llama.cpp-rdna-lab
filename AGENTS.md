# AGENTS.md

Инструкции для AI-агентов, работающих в этом репозитории.

## Идентичность проекта

Это `llama.cpp-with-GUI`: форк `ggml-org/llama.cpp` с PyQt6 GUI, ROCm/Vulkan workflow под AMD Radeon RX 9070 XT и локальными TurboQuant экспериментами. Не относись к нему как к чистому upstream llama.cpp.

## Локальное железо

- OS: Windows 11 Pro build 26200.
- CPU: AMD Ryzen 7 5800X3D, 8 cores / 16 threads.
- RAM: 64 GB.
- GPU: AMD Radeon RX 9070 XT, target `gfx1201`.
- Preferred GPU backend: ROCm/HIP SDK 7.1.
- Fallback backend: Vulkan.
- ROCm builds on Windows must use Ninja and ROCm clang/clang++, not Visual Studio generator.

## Главные цели форка

- Сохранять работоспособный GUI в `gui/`.
- Не ломать ROCm/RDNA4 workflow.
- Сохранять TurboQuant типы и GUI-интеграцию KV cache.
- Догонять upstream по core/runtime, но не импортировать upstream docs/actions/instructions поверх локальных.
- Готовить MTP поддержку только после проверки конкретного upstream PR/commit и совместимого MTP GGUF.
- Для Qwen performance work сначала читать `PROJECT_PROFILE.md`, `MTP.md` и `QWEN_SPEED_RESEARCH.md`.

## Защищённые файлы и директории

При синхронизации с upstream не импортировать без явного запроса пользователя:

- `.github/**`
- `docs/**`
- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `MTP.md`
- `PROJECT_PROFILE.md`
- `QWEN_SPEED_RESEARCH.md`
- `UPSTREAM_SYNC.md`
- `gui/README.md`
- `gui/QUICKSTART.md`
- `.gemini/**`
- `.devops/**`

Если upstream меняет эти пути, сохраняй локальную версию и вручную переноси только реально нужную техническую информацию.

## Git hygiene

- Рабочее дерево может быть грязным. Не откатывай чужие изменения.
- Перед правками смотри `git status --short --branch`.
- Не используй `git reset --hard`, `git checkout -- <path>` или массовое удаление без прямого запроса пользователя.
- Для ручных правок используй `apply_patch`.
- Перед merge/cherry-pick проверь, какие локальные файлы уже изменены.

## Upstream sync policy

Основной документ: `UPSTREAM_SYNC.md`.

Короткая версия:

1. Fetch upstream.
2. Сначала оценить diff/stat и конфликтные зоны.
3. Импортировать core llama.cpp изменения: `common/`, `src/`, `include/`, `ggml/`, `tools/`, `examples/`, `CMakeLists.txt`, scripts/converters по необходимости.
4. Не импортировать upstream `.github`, `docs`, root README и agent instruction files.
5. После merge проверить GUI launch path, ROCm configure/build path и server command generation.

## MTP policy

MTP означает Multi-Token Prediction. На 2026-05-07 upstream работа отслеживается через `ggml-org/llama.cpp#22673`, draft PR `llama + spec: MTP Support`.

В текущем локальном дереве нет полноценного `--spec-type mtp`; есть speculative decoding без MTP (`draft`, `eagle3`, `ngram-*`) и NextN/MTP tensor metadata preservation. Не добавляй GUI-переключатель MTP как будто он уже работает. Допустимо:

- документировать MTP;
- добавлять guarded/experimental UI только если сервер реально поддерживает `--spec-type mtp`;
- использовать Extra Arguments для ручного теста после подтягивания нужного PR;
- предупреждать, что MTP требует MTP-enabled GGUF.

## Проверки после изменений

Минимум:

```powershell
python -m py_compile gui\llama_gui.py gui\build_manager.py gui\dependency_checker.py gui\hardware_detector.py
git diff --check
```

Для build-path изменений:

```powershell
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release -j
```

Для ROCm-path изменений проверять configure отдельно и не запускать долгую сборку без причины:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
```
