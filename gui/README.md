# GUI

PyQt6-интерфейс для локального `llama.cpp-with-GUI` на Windows.

## Запуск

Из корня репозитория:

```powershell
python run.py
```

Python-зависимости:

```powershell
python -m pip install -r requirements.txt
```

## Поддерживаемые сборки

GUI показывает только три backend-варианта:

- `CPU`;
- `Vulkan`;
- `ROCm/HIP`.

Build & Setup создаёт отдельную директорию для каждого варианта, сохраняет
запись в `gui/build_versions.json` и проверяет CMake cache после ROCm configure.

## Основные вкладки

| Вкладка | Назначение |
| --- | --- |
| Launch Server | модель, build, context, KV, split, MTP/DFlash, vision, запуск и остановка сервера |
| Bench and Autotune | prompt/decode benchmark, sweep batch/ubatch/KV/spec и live progress |
| Download Models | поиск и загрузка GGUF |
| Build & Setup | configure/build CPU, Vulkan и ROCm |
| Installed Builds | реестр и состояние локальных builds |
| System Info | CPU, RAM, GPU и доступность SDK |

## Vision

В Launch Server включите Vision и выберите совместимый `mmproj-*.gguf`. GUI
добавит `--mmproj`. Projector должен быть выпущен для той же модели, что и
основной GGUF.

## Безопасная остановка

Кнопки остановки сначала отправляют серверу мягкое завершение и ждут выхода.
Принудительное завершение является последним fallback и не должно применяться
во время GPU benchmark без явного решения пользователя.

## Autotune artifacts

Результаты и server logs находятся в `build_logs/agent-workload/`. Главные
сводки: `BENCH_RUNS.csv`, `BENCH_RECENT.md` и `BENCH_LANES.md`.
