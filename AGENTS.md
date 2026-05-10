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
- Для performance work считать текущей практической целью ускорение `Qwen3.6-27B` на стартовой prompt-heavy точке ниже `16k` (текущий reference `ctx=12288`) до `25-27 TPS`.
- Сохранять TurboQuant типы и GUI-интеграцию KV cache.
- Догонять upstream по core/runtime, но не импортировать upstream docs/actions/instructions поверх локальных.
- Готовить MTP поддержку только после проверки конкретного upstream PR/commit и совместимого MTP GGUF.
- Для Qwen performance work сначала читать `PROJECT_PROFILE.md`, `BENCHMARKS.md`, `MTP.md`, `MTP_IMPLEMENTATION_PLAN.md` и `QWEN_SPEED_RESEARCH.md`.

## Защищённые файлы и директории

При синхронизации с upstream не импортировать без явного запроса пользователя:

- `.github/**`
- `docs/**`
- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `MTP.md`
- `MTP_IMPLEMENTATION_PLAN.md`
- `PROJECT_PROFILE.md`
- `QWEN_SPEED_RESEARCH.md`
- `BENCHMARKS.md`
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
- Не используй `cmd.exe`/`cmd` для длинных build/benchmark/run сценариев: в этом репозитории они склонны зависать. Предпочитай прямой запуск из `bash`/PowerShell через `run_in_terminal` и избегай лишней cmd-обвязки.

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

## Copilot On-Track Protocol (Performance R&D)

Для всех kernel/perf-экспериментов придерживаться одного цикла:

1. Перед запуском бенчей очищать override-окружение (`HSA_OVERRIDE_GFX_VERSION`) и убеждаться, что нет фонового `llama-server`.
2. Для быстрых итераций по prompt-heavy lane по умолчанию использовать `--runs 1`; до `3 runs` повышать только финальное подтверждение пограничных или реально promising дельт.
3. Любой speed claim соотносить с текущей стартовой точкой `ctx=12288` (или ближайшей <16k) в `scripts/agent_workload_bench.py --real-context-mode repo-snapshot`, а не со старыми 64k/128k headline.
4. Для измерений «чистой» стартовой точки отключать reuse (`--cache-ram 0 --ctx-checkpoints 0`) и фиксировать этот факт в label.
5. Если новая идея не бьёт baseline или даёт нестабильность, откатывать экспериментальные правки до чистого дерева.
6. Если идея подтверждена, фиксировать одновременно: код + запись в `BENCHMARKS.md` + артефакты в `build_logs/agent-workload/`.
7. Главный трек поиска ускорений: кодовые изменения в llama.cpp/ggml prefill/runtime path (`ggml/src`, `src`, `common`), а не только перебор server flags.
8. Для benchmark-режима всегда держать thinking включённым (использовать `--no-disable-thinking`), чтобы результаты были сопоставимы между сессиями.

Цель: избегать «шума» и держать только воспроизводимые ускорения в `master`.
