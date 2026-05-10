# llama.cpp-with-GUI

Локальный форк `ggml-org/llama.cpp`, заточенный под Windows-машину владельца и AMD ROCm/Vulkan workflow. Главная цель форка: удобный PyQt6 GUI поверх `llama.cpp`, быстрые локальные сборки под конкретное железо, управление моделями GGUF и сохранение собственных настроек при догоне upstream.

Это не чистый upstream README. Оригинальная документация llama.cpp остаётся полезной как справочник по API и форматам, но корневая документация этого форка описывает именно локальную сборку `llama.cpp-with-GUI`.

## Текущий фокус

- PyQt6 GUI для запуска `llama-server`, CLI-инференса, скачивания моделей и сборки проекта.
- Оптимизация под AMD Radeon RX 9070 XT через ROCm/HIP SDK 7.1, с Vulkan как fallback.
- Реалистичная производительность Qwen3.6 в prompt-heavy режиме со стартовой точкой ниже `16k` (`ctx=12288` как текущий reference) и целевой метрикой `25-27 TPS` на этой стартовой точке.
- TurboQuant KV/weight experiments: `tbq3_0`, `tbq4_0`, `tq3_0`.
- Защита локальных GUI, документации и агентских инструкций от случайного перетирания при `upstream/master` merge.
- Подготовка к MTP/Multi-Token Prediction, когда поддержка станет достаточно стабильной для форка.

## Performance Focus Update (2026-05-10)

- Активные 128k прогоны временно остановлены.
- Также остановлен старый `64k`-центричный lane как основной, потому что новый prompt-heavy benchmark показал раннюю деградацию уже ниже `16k`.
- Новый базовый сценарий для всех performance claims: `scripts/agent_workload_bench.py` с `--real-context-mode repo-snapshot` и большим входящим prompt.
- Текущая стартовая точка: `ctx=12288` (no-reuse, prompt-heavy) с фактическим уровнем около `9.24 TPS`.
- Новая инженерная цель: поднять стартовую точку до `25-27 TPS` через изменения в кодовой базе llama.cpp/ggml (prefill/runtime path), а не только server args.

## Статус спринтов (2026-05-07)

Выполнены три практических спринта с контрольными checkpoint-бенчами и smoke-запуском GUI между этапами.

- Sprint 1 (Launch Server): добавлены и сохранены через QSettings runtime-контролы `Parallel`, `HTTP Threads`, `Flash Attention`, `--no-warmup`, `-fit on/off`; для `--spec-type mtp` теперь принудительно ставится `--parallel 1`.
- Sprint 2 (ROCm build path): в configure-path усилены ROCm флаги `GGML_HIP`, `GGML_HIP_MMQ_MFMA`, `GGML_HIP_NO_VMM`, `AMDGPU_TARGETS`; после configure добавлена валидация `CMakeCache.txt` с предупреждением в GUI при расхождениях.
- Sprint 3 (GUI benchmark mode): во вкладку Build добавлена кнопка `Quick ROCm Bench`, которая запускает `scripts/agent_workload_bench.py` и пишет результаты в `build_logs/agent-workload/`.

Контрольные quick-бенчи (одинаковый профиль) показали:

- baseline: `14.72 TPS`
- после Sprint 1: `13.00 TPS`
- после Sprint 2: `13.01 TPS`
- после Sprint 3: `13.02 TPS`

Это короткий smoke-профиль, поэтому значения следует трактовать как контроль стабильности пайплайна, а не как финальный performance verdict.

## Железо и окружение

Слепок снят 2026-05-07 на рабочей машине:

| Компонент | Значение |
| --- | --- |
| OS | Microsoft Windows 11 Pro, 64-bit, build 26200 |
| Motherboard | Gigabyte B550 GAMING X V2 |
| CPU | AMD Ryzen 7 5800X3D, 8 cores / 16 threads |
| RAM | 64 GB DDR4 |
| GPU | AMD Radeon RX 9070 XT |
| GPU driver | 32.0.23033.1002 |
| ROCm/HIP | `C:\Program Files\AMD\ROCm\7.1` |
| Python | 3.13.4 |
| CMake | 3.29.2 |
| Git | 2.49.0.windows.1 |
| Ninja | 1.13.0 |
| Main disks | `C:` 1.5 TB, `D:` 1.0 TB |

ROCm build target for RX 9070 XT / RDNA4 is expected to be `gfx1201`. With HIP SDK 7.1 the old `HSA_OVERRIDE_GFX_VERSION` workaround should not be needed.

## Запуск GUI

```powershell
python run.py
```

Альтернативы:

```powershell
run.bat
start-gui.bat
python gui\llama_gui.py
```

`run.py` добавляет `gui/` в `PYTHONPATH`, проверяет Python-зависимости через `dependency_checker.py` и запускает `gui/llama_gui.py`.
Текущая точка входа GUI использует модульную структуру и запускает `gui/main_window.py`.

## Возможности GUI

GUI находится в `gui/` и состоит из шести основных вкладок:

| Вкладка | Назначение |
| --- | --- |
| Launch Server | Запуск `llama-server`, выбор модели, backend/build, GPU layers, context, batch, `parallel`, `threads-http`, spec-type, KV cache, Extra Arguments |
| Inference | Запуск `llama-cli` для одиночного prompt/inference |
| Download Models | Поиск и скачивание GGUF с Hugging Face, сортировка и выбор файлов |
| Build & Setup | Проверка зависимостей, CMake configure/build, выбор CPU/CUDA/Metal/Vulkan/SYCL/ROCm |
| Installed Builds | Просмотр, переименование и удаление нескольких build-директорий |
| System Info | Определение CPU/GPU/RAM и рекомендации backend |

В GUI уже есть локальные улучшения: multimodal/vision controls с `--mmproj`, экспорт build log, ROCm-aware сборка, KV cache presets и модельные пресеты для Qwen, Llama, Mistral, DeepSeek, Phi и embedding-моделей.

## Рекомендуемая сборка для этой машины

ROCm:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --config Release -j 4
```

Vulkan fallback:

```powershell
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j
```

CPU sanity build:

```powershell
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release -j
```

ROCm на Windows должен использовать Ninja/clang из HIP SDK, а не Visual Studio generator.

## MTP / Multi-Token Prediction

MTP сейчас рассматривается как перспективный путь ускорения generation. На 2026-05-07 актуальная работа в upstream идёт в draft PR `ggml-org/llama.cpp#22673` (`llama + spec: MTP Support`). В текущем локальном дереве полноценного runtime MTP ещё нет: есть speculative decoding (`draft`, `eagle3`, `ngram-*`) и сохранение NextN/MTP tensor metadata, но `--spec-type mtp` локально пока не поддержан.

Подробный план и ссылки вынесены в [MTP.md](MTP.md). Коротко: MTP имеет смысл тестировать только с MTP-enabled GGUF, например Qwen3.6 MTP variants, и только после подтягивания PR/commit, где `llama-server` знает `--spec-type mtp`.

## TurboQuant и KV cache

Форк содержит TurboQuant-интеграцию:

| Тип | Назначение | Бэкенд |
| --- | --- | --- |
| `tbq3_0` | CPU-only TurboQuant 3-bit | CPU |
| `tbq4_0` | CPU-only TurboQuant 4-bit | CPU |
| `tq3_0` | GPU TurboQuant 3-bit / ~3.5 bpw | CUDA/HIP |

GUI предупреждает, что TurboQuant KV-типы требуют flash attention. Для `tbq*` GUI форсирует CPU/no GPU offload, потому что эти варианты CPU-only.

## Модели

Локальные модели лежат в `models/`. Большие `.gguf` файлы не стоит коммитить. Для RX 9070 XT 16 GB практичные стартовые точки:

- 7B-14B Q8/Q6/Q5 для скорости и качества.
- 27B-35B Q4/IQ4 с `parallel=1`, flash attention и сжатым KV cache.
- MoE модели с малым active parameter count, например Qwen3/Qwen3.6 A3B variants.
- Для длинного контекста уменьшать `parallel`, KV cache переводить в `q8_0`, `q4_0` или экспериментальный `tq3_0`.

## Правила синхронизации upstream

Не делать слепой `git merge upstream/master`, если цель только догнать ядро llama.cpp. Этот форк защищает собственные файлы:

- `.github/**`
- `docs/**`
- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `MTP.md`
- `UPSTREAM_SYNC.md`
- GUI-документацию `gui/README.md`, `gui/QUICKSTART.md`

Процедура описана в [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md). Идея простая: импортировать core/runtime изменения из upstream, но не забирать upstream Actions, upstream docs и upstream агентские инструкции.

## Агентам

Перед правками читать [AGENTS.md](AGENTS.md). Там указаны локальные приоритеты форка, защищённые пути и правило: не перетирать пользовательские GUI/ROCm/TurboQuant изменения ради чистоты upstream.

## Локальный профиль и Qwen speed work

- [PROJECT_PROFILE.md](PROJECT_PROFILE.md) — железо, окружение, модели, remotes и локальные defaults.
- [QWEN_SPEED_RESEARCH.md](QWEN_SPEED_RESEARCH.md) — исследование MTP, speculative decoding, KV cache и ROCm/Vulkan tuning для Qwen.
- [BENCHMARKS.md](BENCHMARKS.md) — короткий agent-workload benchmark для baseline/progression.
- [MTP_IMPLEMENTATION_PLAN.md](MTP_IMPLEMENTATION_PLAN.md) — пошаговый план внедрения MTP.

## Полезные файлы

| Файл | Что это |
| --- | --- |
| `run.py` | Основной launcher GUI |
| `gui/main_window.py` | Главное модульное PyQt6 окно |
| `gui/server_tab.py` | Launch Server вкладка (runtime-параметры и запуск сервера) |
| `gui/build_tab.py` | Build/Configure вкладка и quick benchmark |
| `gui/build_manager.py` | CMake/ROCm/Vulkan orchestration |
| `gui/hardware_detector.py` | Определение железа |
| `gui/model_downloader.py` | Hugging Face модели |
| `gui/model_presets.json` | Рекомендованные параметры моделей |
| `BUILD_CHEATSHEET.txt` | Быстрая шпаргалка по сборке |
| `MSVC_FIX.md` | Заметки по MSVC detection |
| `PROJECT_PROFILE.md` | Персональный профиль форка |
| `MTP.md` | Исследование и план внедрения MTP |
| `MTP_IMPLEMENTATION_PLAN.md` | Детальный план работ по MTP |
| `QWEN_SPEED_RESEARCH.md` | План ускорений Qwen |
| `BENCHMARKS.md` | Как мерить baseline и прогресс |
| `UPSTREAM_SYNC.md` | Как догонять upstream без импорта лишнего |
| `AGENTS.md` | Инструкции для AI-агентов |
