# llama.cpp-with-GUI — Инструкции для AI-моделей

> **Это форк [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** с добавлением PyQt6 GUI и Google TurboQuant квантизации.
> Владелец: [MrLordCat](https://github.com/MrLordCat)

---

## Структура репозитория

```
llama.cpp-with-GUI/
├── gui/                    # PyQt6 GUI (весь GUI-код здесь)
│   ├── llama_gui.py        # Главное приложение (~2150 строк, 6 вкладок)
│   ├── build_manager.py    # CMake оркестрация сборки (~850 строк)
│   ├── build_manager_v2.py # Альтернативный build manager (~400 строк)
│   ├── dependency_checker.py    # Проверка зависимостей
│   ├── dependency_installer.py  # Автоустановка (MSVC, CMake и т.д.)
│   ├── hardware_detector.py     # Определение CPU/GPU/RAM
│   ├── hardware_detector_v2.py  # Альтернативный детектор
│   ├── model_downloader.py      # HuggingFace интеграция (~600 строк)
│   ├── requirements-gui.txt     # PyQt6, huggingface-hub, psutil, requests, tqdm
│   ├── README.md           # Документация GUI
│   └── QUICKSTART.md       # Быстрый старт
├── ggml/src/
│   ├── ggml-turboq.c       # TurboQuant CPU квантизация/деквантизация
│   ├── ggml-turboq.h       # TurboQuant хелперы (Householder QR, вращения)
│   └── ggml-turboq-tables.h # Lloyd-Max кодовые книги (2-bit, 3-bit, 4-bit)
├── run.py                  # Точка входа GUI (авто-зависимости + запуск)
├── run.bat / run.sh        # Лаунчеры для Windows / Linux+macOS
├── start-gui.bat           # Windows лаунчер с проверкой зависимостей
├── build_exe.py            # PyInstaller сборка .exe
├── build-gui-exe.bat       # Батник для сборки .exe
├── LLaMA-GUI.spec          # PyInstaller спецификация
├── translate_ui.py         # Русский→Английский перевод строк GUI
├── BUILD_CHEATSHEET.txt    # Шпаргалка по сборке (RU)
├── MSVC_FIX.md             # Документация фикса обнаружения MSVC
├── ISSUE_FIXED_FILE_SIZES.md # Документация фикса размеров файлов HF
├── test_*.py               # Тесты GUI (auto_installer, features, file_listing, file_sizes, rocm_check)
└── CLAUDE.md               # ← ВЫ ЗДЕСЬ. Инструкции для AI-моделей
```

---

## Git Remotes

| Remote | URL | Назначение |
|--------|-----|------------|
| `origin` | `MrLordCat/llama.cpp-with-GUI.git` | Форк (основной) |
| `upstream` | `ggml-org/llama.cpp.git` | Оригинальный llama.cpp |
| `animehacker` | `animehacker/llama-turboquant.git` | Источник TurboQuant |
| `turboquant` | `elusznik/llama.cpp.git` | Альтернативный TQ remote |

**Синхронизация с upstream**: GUI имеет встроенную кнопку "Обновить из Upstream" (вкладка Build).
Конфликты в `.github/workflows/` разрешаются автоматически (берётся upstream версия).

---

## TurboQuant (TBQ) — Кастомные типы квантизации

### Типы и ID

| Тип | GGML Type ID | LLAMA_FTYPE | Размер блока | bpw | Бэкенд |
|-----|-------------|-------------|-------------|-----|--------|
| **TBQ3_0** | 42 | 41 | 256 (QK_K) | ~3.06 | CPU only |
| **TBQ4_0** | 43 | 42 | 256 (QK_K) | ~4.06 | CPU only |
| **TQ3_0** | 44 | — | 32 (QK_TQ3_0) | ~3.5 | CUDA/HIP |

### Алгоритмы

- **TBQ3_0 / TBQ4_0** (CPU, 256-элементные блоки):
  1. Случайная ортогональная матрица Q через Householder QR из seed-Гауссовой матрицы
  2. Применение Q^T к входному вектору (сферическое вращение)
  3. Извлечение нормы как scale `d`
  4. Lloyd-Max квантизация повёрнутого единичного вектора

- **TQ3_0** (GPU, 32-элементные блоки):
  1. Walsh-Hadamard Transform (WHT32) на GPU
  2. Polar Quantization: 3-bit (8 уровней) Lloyd-Max + 1-bit MSB
  3. Скалярный scale gamma (fp16) на блок

### Структуры данных

```c
// TBQ3_0: CPU 3-bit
typedef struct {
    uint8_t qs[3*QK_K/8];  // 96 байт квантованных значений
    ggml_half d;            // 2 байта scale
} block_tbq3_0;

// TBQ4_0: CPU 4-bit
typedef struct {
    uint8_t qs[QK_K/2];    // 128 байт
    ggml_half d;            // 2 байта scale
} block_tbq4_0;

// TQ3_0: GPU 3-bit (PolarQuant + WHT)
typedef struct {
    uint8_t qs[QK_TQ3_0/4]; // 8 байт: 2-bit кодовые индексы
    uint8_t qr[QK_TQ3_0/8]; // 4 байта: верхний 1 бит
    ggml_half gamma;         // 2 байта: scale на блок
} block_tq3_0;
```

### Матрица поддержки бэкендов

| Бэкенд | TBQ3_0 | TBQ4_0 | TQ3_0 |
|---------|--------|--------|-------|
| CPU | ✅ | ✅ | ❌ |
| CUDA | ❌ | ❌ | ✅ |
| HIP (ROCm) | ❌ | ❌ | ✅ |
| Metal | ❌ | ❌ | ❌ |

### Ключевые файлы TBQ

- Определения типов: `ggml/include/ggml.h` (строки ~432-434)
- Структуры блоков: `ggml/src/ggml-common.h` (строки ~280-311)
- CPU квантизация: `ggml/src/ggml-turboq.c`
- Кодовые книги: `ggml/src/ggml-turboq-tables.h`
- CPU vec_dot: `ggml/src/ggml-cpu/quants.c` (строки ~550-595)
- CUDA деквантизация TQ3_0: `ggml/src/ggml-cuda/convert.cu` (строки ~489-710)
- CUDA WHT32: `ggml/src/ggml-cuda/cpy-utils.cuh` (строки ~214-300)
- LLAMA_FTYPE: `include/llama.h` (строки ~155-158)
- Квантизация pipeline: `src/llama-quant.cpp`
- CLI квантизатор: `tools/quantize/quantize.cpp` (строки ~46-47)
- Бенчмарки: `tools/llama-bench/llama-bench.cpp` (строки ~494-501)
- CLI аргументы: `common/arg.cpp` (строки ~393-395)
- GUI интеграция: `gui/llama_gui.py` (строки ~821-829, GUI ставит `--ngl 0` для TBQ)

---

## GUI — Архитектура

### Главные классы (gui/llama_gui.py)

| Класс | Назначение |
|-------|-----------|
| `LlamaCppGUI(QMainWindow)` | Главное окно, 6 вкладок, QSettings |
| `ServerThread(QThread)` | Запуск llama-server в фоне |
| `InferenceThread(QThread)` | Запуск llama-cli для инференса |
| `UpdateForkThread(QThread)` | Git операции синхронизации с upstream |

### 6 вкладок GUI

1. **🚀 Launch Server** — Запуск llama-server (модель, контекст, GPU layers, threads, batch, CORS, API key, KV cache quant, ROCm targets)
2. **⚡ Inference** — CLI инференс (prompt, temperature, top-p/k, context, threads)
3. **📥 Download Models** — Поиск/скачивание с HuggingFace (сортировка, фильтры, resume)
4. **🔧 Build & Setup** — Сборка (update from upstream, выбор бэкенда, зависимости, CMake configure + build)
5. **📋 Installed Builds** — Управление несколькими сборками (переименование, удаление, список исполняемых файлов)
6. **💻 System Info** — Информация о железе и рекомендации

### Бэкенды сборки

GUI поддерживает: **CPU, CUDA, Metal, Vulkan, SYCL, ROCm**

Для ROCm:
- Автоопределение GPU targets (gfx1100, gfx1201 и т.д.)
- RDNA4 workaround для ROCm < 6.4 (`HSA_OVERRIDE_GFX_VERSION`)
- Используется Ninja (не Visual Studio) для ROCm на Windows
- Ограничение параллельных jobs для экономии VRAM

### Зависимости GUI

```
PyQt6>=6.6.0
PyQt6-Qt6>=6.6.0
huggingface_hub>=0.20.0
requests>=2.31.0
psutil>=5.9.0
tqdm>=4.66.0
```

---

## Сборка проекта

### Быстрая сборка (CMake)

```bash
# CPU
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# CUDA
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release -j

# ROCm (AMD)
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --config Release -j

# Vulkan
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j
```

### Сборка GUI .exe (PyInstaller)

```bash
python build_exe.py        # или
build-gui-exe.bat
```

### Запуск GUI

```bash
python run.py              # или
run.bat                    # Windows
run.sh                     # Linux/macOS
start-gui.bat              # Windows (с проверкой зависимостей)
```

---

## ROCm — Специфика

- **GPU**: AMD Radeon RX 9070 XT (gfx1201, RDNA 4, 16 GB VRAM, 32 CU, 2460 MHz)
- **HIP SDK**: 7.1 (clang 21.0.0) — основной; также установлены 6.2 и 6.4
- **Компилятор**: `C:/Program Files/AMD/ROCm/7.1/bin/clang++.exe`
- **Линкер**: `C:/Program Files/AMD/ROCm/7.1/bin/lld-link.exe`
- **Ключевые флаги**: `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1201`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_NO_VMM=ON`
- **Директория сборки**: `build-rocm/`
- ROCm сборки МЕДЛЕННЫЕ (много template instantiations для mmq/mmf/fattn)
- **RDNA4 workaround**: `HSA_OVERRIDE_GFX_VERSION=11.0.0` нужен ТОЛЬКО для ROCm < 6.4. HIP SDK 7.1 поддерживает gfx1201 нативно — workaround НЕ нужен
- Генератор: Ninja (обязательно для ROCm на Windows, Visual Studio не поддерживается)

---

## Тесты

| Файл | Назначение |
|------|-----------|
| `test_auto_installer.py` | Тест автоустановки зависимостей |
| `test_features.py` | Проверка импортов model_downloader |
| `test_file_listing.py` | Тест листинга файлов HuggingFace |
| `test_file_sizes.py` | Валидация размеров файлов через HfFileSystem |
| `test_rocm_check.py` | Проверка обнаружения ROCm |
| `tests/test-quantize-fns.cpp` | C++ тесты квантизации (включая TBQ пороги ошибок) |

---

## Важные замечания

1. **GUI целиком на русском** (есть `translate_ui.py` для перевода на английский)
2. **TBQ типы — CPU-only**, GUI автоматически ставит `--ngl 0` при их выборе
3. **TQ3_0 — GPU-only** (CUDA/HIP), нет CPU fallback
4. При merge из upstream возможны конфликты в `.github/workflows/` — GUI разрешает их автоматически
5. `build_manager_v2.py` и `hardware_detector_v2.py` — альтернативные/legacy версии, основные — без суффикса `_v2`
6. Файл `AGENTS.md` из upstream не используется в этом форке (в `.gitignore`)
7. Сборка ROCm требует Ninja (не Visual Studio generator)
8. **MSVC detection** имеет 3-уровневую стратегию: PATH → filesystem → Registry

---

## Среда разработки (актуально на апрель 2026)

| Компонент | Версия |
| --------- | ------ |
| ОС | Windows 11 (build 26200) |
| CPU | AMD Ryzen 7 5800X3D (8 ядер / 16 потоков, 3.4 GHz) |
| RAM | 64 GB DDR4 |
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA 4, 16 GB VRAM) |
| HIP SDK | 7.1 (clang 21.0.0); также 6.2, 6.4 |
| CMake | 4.0.2 |
| Ninja | 1.13.0 |
| MSVC | 14.44.35207 (VS 2022 Build Tools, не в PATH) |
| Python | 3.13.4 |
| Git | 2.49.0 |
| Vulkan SDK | 1.4.313.1 |
| CUDA | не установлена |

---

## Qwen3.5 — Заметки

- Гибридная архитектура: Gated DeltaNet + Gated Attention
- Требует GATED_DELTA_NET op (добавлен в CUDA/HIP в марте 2026)
- 9B модель: thinking ВЫКЛЮЧЕН по умолчанию, включать: `--chat-template-kwargs "{\"enable_thinking\":true}"`
- Рекомендация: `--cache-type-k bf16 --cache-type-v bf16`
