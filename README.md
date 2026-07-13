# llama.cpp-with-GUI

Локальный форк `ggml-org/llama.cpp` для Windows, двух AMD Radeon RX 9070 XT и
agent-heavy работы с большими контекстами. В репозитории объединены `llama.cpp`,
PyQt6 GUI, автотюн, MTP/DFlash и локальные оптимизации ROCm/Vulkan.

## Поддерживаемые бэкенды

| Бэкенд | Назначение | Статус |
| --- | --- | --- |
| CPU | sanity, fallback, конвертация и тесты | поддерживается |
| Vulkan | основной prompt-eval и универсальный AMD runtime | поддерживается |
| ROCm/HIP | AMD runtime, MTP и kernel research | поддерживается |

Нативные CUDA, Metal, SYCL, OpenCL, CANN и остальные upstream-бэкенды удалены.
ROCm продолжает компилировать общий HIP/CUDA-compatible kernel layer из
`ggml/src/ggml-cuda`; это внутренняя зависимость HIP, а не поддержка NVIDIA.
Подробности: [Supported backends](docs/SUPPORTED_BACKENDS.md).

## Быстрый запуск

```powershell
python run.py
```

Также доступны `run.bat` и `start-gui.bat`. GUI умеет:

- собирать и выбирать CPU, Vulkan и ROCm builds;
- запускать `llama-server` с сохранёнными параметрами;
- настраивать dual-GPU layer split и output device;
- запускать benchmark/autotune с live prompt progress;
- выбирать MTP, DFlash и обычный decode;
- подключать vision projector через `--mmproj`;
- скачивать и учитывать локальные GGUF-модели.

## Сборка

CPU:

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4 --target llama-server
```

Vulkan:

```powershell
cmake -S . -B build-vulkan -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 4 --target llama-server
```

ROCm/HIP SDK 7.1 для RDNA4:

```powershell
cmake -S . -B build-rocm -G Ninja `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS=gfx1201 `
  -DGGML_HIP_MMQ_MFMA=ON `
  -DGGML_HIP_NO_VMM=ON `
  -DGGML_OPENMP=OFF `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm -j 4 --target llama-server
```

На Windows ROCm должен использовать clang из HIP SDK и Ninja. GUI выставляет
этот toolchain автоматически. При этом host-link требует MSVC Build Tools с
workload `Desktop development with C++` и Windows SDK: без них ROCm clang не
найдёт `kernel32.lib`, `msvcrtd.lib` и остальные системные библиотеки.

## Рабочий профиль

Основная машина:

- Windows 11;
- AMD Ryzen 7 5800X3D, 64 GB RAM;
- две Radeon RX 9070 XT 16 GB, `gfx1201`;
- ROCm/HIP SDK 7.1;
- Qwen3.6-27B Q3_K_S MTP как основная длинноконтекстная модель.

Для dual Vulkan обычно используется `-dev Vulkan1,Vulkan0 -sm layer`; GPU1
выбран первым, поскольку GPU0 обслуживает дисплей и системную нагрузку. Точный
`-ts` нужно брать из актуального autotune, а не считать постоянным.

## MTP

MTP полностью интегрирован через upstream NextN pipeline. Для MTP-enabled GGUF:

```text
--spec-type draft-mtp --spec-draft-n-max 3
```

После Vulkan warm-cache и safe rows 5-8 split оптимизаций MTP в проверенных
dual-GPU lane давал примерно `1.29-1.42x` к decode baseline. Это не ускорение
prompt evaluation: длинный prefill всегда сравнивается отдельно с `spec=none`.
`n_max=3` и `n_max=4` следует выбирать по acceptance и длине ответа.

Подробные измерения находятся в [BENCHMARKS.md](BENCHMARKS.md) и
`docs/research/major-topology/`.

## Большой контекст

Для prompt-heavy измерений используйте один и тот же backend, модель, KV,
`ctx`, batch/ubatch, split, cache policy и фоновую нагрузку. Канонический runner:

```powershell
python scripts/agent_workload_bench.py --help
```

История GUI/autotune хранится в:

- `build_logs/agent-workload/BENCH_RUNS.csv`;
- `build_logs/agent-workload/BENCH_RECENT.md`;
- `build_logs/agent-workload/BENCH_LANES.md`.

## Vision

Для Qwen3.6-27B включите Vision в Launch Server и выберите соответствующий
`models/mmproj-F16.gguf`. Projector должен совпадать с архитектурой и embedding
dimension текстовой модели. Для первичной проверки image-запроса используйте
`Spec: None`, чтобы отделить vision pipeline от speculative decode.

## Структура

| Путь | Назначение |
| --- | --- |
| `gui/` | PyQt6 приложение |
| `src/`, `common/`, `include/` | llama runtime |
| `ggml/src/ggml-cpu/` | CPU backend |
| `ggml/src/ggml-vulkan/` | Vulkan backend и shaders |
| `ggml/src/ggml-hip/` | ROCm build orchestration |
| `ggml/src/ggml-cuda/` | внутренние HIP-compatible kernels |
| `scripts/agent_workload_bench.py` | benchmark/autotune runner |
| `docs/research/` | воспроизводимые performance-эксперименты |

## Разработка

Перед изменениями прочитайте [AGENTS.md](AGENTS.md). Политика переноса upstream:
[UPSTREAM_SYNC.md](UPSTREAM_SYNC.md). Этот форк переносит нужные core/runtime
изменения вручную и не возвращает удалённые бэкенды автоматически.
