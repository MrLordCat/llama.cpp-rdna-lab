# GUI Модулизация: Полный План

**Текущее состояние:** llama_gui.py сокращена с 4260 → 4055 строк. Потоки перемещены в `threads.py`.

**Цель:** Разделить llama_gui.py на модули ≤500 строк каждый для улучшения поддерживаемости.

---

## 📋 Структура модулей (итого ~3000 строк кода)

### ✅ Уже сделано

| Модуль | Строк | Статус | Описание |
|--------|-------|--------|---------|
| `gui/threads.py` | 260 | ✅ | ServerThread, InferenceThread, UpdateForkThread |

### ⏳ Осталось создать

| Модуль | Строк | Содержимое | Зависит от |
|--------|-------|-----------|-----------|
| `gui/project_utils.py` | 300 | ProjectManager класс (поиск проекта, пути) | - |
| `gui/server_tab.py` | 450 | ServerTabWidget + 14 методов настроек | project_utils.py |
| `gui/server_control.py` | 380 | ServerController (запуск/управление) | threads.py, server_tab.py |
| `gui/inference_tab.py` | 160 | InferenceTabWidget + методы инференса | project_utils.py |
| `gui/download_tab.py` | 200 | DownloadTabWidget + методы загрузки | project_utils.py |
| `gui/build_tab.py` | 460 | BuildTabWidget + configure + build | project_utils.py |
| `gui/builds_info_tab.py` | 360 | BuildsInfoTabWidget + управление | project_utils.py |
| `gui/hardware_tab.py` | 90 | HardwareTabWidget + детектирование | project_utils.py |
| `gui/settings.py` | 100 | SettingsManager (load/save/close) | - |
| `gui/main_window.py` | 250 | LlamaCppGUI coordinator (init, events) | Все остальные |

---

## 🔧 Инструкции по модулизации

### 1. Создать `gui/project_utils.py`

**Извлечь из llama_gui.py:**
- `_find_or_select_project_root()` (67 строк)
- `_is_valid_llama_cpp_repo()` (15 строк)
- `_get_common_repo_locations()` (20 строк)
- `_ask_user_for_repo_path()` (12 строк)
- `get_build_dir_for_backend()` (18 строк)
- `_get_build_version_info()` (28 строк)
- `get_available_builds()` (45 строк)
- `_detect_build_backend()` (32 строк)

**Создать класс ProjectManager:**
```python
class ProjectManager:
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or self.find_or_select()
    
    # Методы выше
```

**Импорты:**
```python
from pathlib import Path
from typing import Optional, Dict
from PyQt6.QtWidgets import QFileDialog, QMessageBox
```

---

### 2. Создать `gui/server_tab.py`

**Извлечь из llama_gui.py:**
- `create_server_tab()` - полностью (~461 строк)

**Вспомогательные методы:**
```
browse_server_model()
browse_server_mmproj()
on_server_model_selected()
_set_server_vision_controls_enabled()
_get_server_speculative_type()
_set_server_speculative_controls_enabled()
_looks_like_mtp_model()
_looks_like_vision_model()
_find_mmproj_for_model()
refresh_server_vision_controls()
_load_model_presets()
_find_matching_preset()
apply_model_preset()
open_presets_file()
```

**Создать класс ServerTabWidget:**
```python
class ServerTabWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.create_ui()
    
    def create_ui(self) -> QWidget:
        # Весь код create_server_tab()
```

---

### 3. Создать `gui/server_control.py`

**Извлечь из llama_gui.py:**
```
start_server()
stop_server()
on_server_ready()
open_web_interface()
append_server_output()
server_finished()
server_error()
_build_server_command()
```

**Класс ServerController:**
```python
class ServerController:
    def __init__(self, parent, project_root):
        self.parent = parent
        self.project_root = project_root
        self.server_thread = None
```

---

### 4. Создать `gui/inference_tab.py`

**Содержит:**
- `create_inference_tab()` (~159 строк)
- `run_inference()`
- `stop_inference()`
- Callbacks: `append_output()`, `inference_finished()`, `inference_error()`

**Класс InferenceTabWidget**

---

### 5. Создать `gui/download_tab.py`

**Содержит:**
- `create_download_tab()` (~143 строки)
- `search_hf_models()`, `display_search_results()`
- `download_model()`, `cancel_download()`
- Callbacks для ListFilesThread и DownloadThread

**Класс DownloadTabWidget**

---

### 6. Создать `gui/build_tab.py`

**Содержит:**
- `create_build_tab()` (~145 строк)
- `configure_build()`
- `build_project()`
- `install_dependencies()`
- `check_for_updates()`, `update_fork()`
- Все callbacks для BuildThread, ConfigureThread

**Класс BuildTabWidget**

---

### 7. Создать `gui/builds_info_tab.py`

**Содержит:**
- `create_builds_info_tab()` (~353 строки)
- `rename_build_folder()`
- `delete_selected_build()`
- `refresh_builds_info()`
- `update_vscode_config()`

**Класс BuildsInfoTabWidget**

---

### 8. Создать `gui/hardware_tab.py`

**Содержит:**
- `create_hardware_tab()` (~31 строка)
- `detect_hardware()`
- `update_rocm()`

**Класс HardwareTabWidget**

---

### 9. Создать `gui/settings.py`

**Содержит:**
- `load_settings()` - восстановление параметров
- `save_settings()` - сохранение параметров
- `closeEvent()` - обработка закрытия окна

**Класс SettingsManager:**
```python
class SettingsManager:
    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.settings = QSettings("LlamaCpp", "GUI")
    
    def load(self):
        # Код load_settings()
    
    def save(self):
        # Код save_settings()
```

---

### 10. Обновить `gui/llama_gui.py` → `gui/main_window.py`

**Содержит только:**
- `LlamaCppGUI(QMainWindow)` класс
- `__init__()` - инициализация и загрузка табов
- Event handlers высокого уровня
- Координация между табами

**Новый импорт:**
```python
from threads import ServerThread, InferenceThread, UpdateForkThread
from project_utils import ProjectManager
from server_tab import ServerTabWidget
from server_control import ServerController
from inference_tab import InferenceTabWidget
from download_tab import DownloadTabWidget
from build_tab import BuildTabWidget
from builds_info_tab import BuildsInfoTabWidget
from hardware_tab import HardwareTabWidget
from settings import SettingsManager
```

**Переименовать файл:**
```bash
mv gui/llama_gui.py gui/main_window.py
```

**Обновить run.py:**
```python
from gui.main_window import LlamaCppGUI  # было: from gui.llama_gui import LlamaCppGUI
```

---

## ✨ Порядок создания

1. **Сначала утилиты (без зависимостей):**
   - `project_utils.py` ← используется всеми табами
   - `settings.py` ← используется main_window

2. **Затем табы (независимые друг от друга):**
   - `server_tab.py`
   - `inference_tab.py`
   - `download_tab.py`
   - `build_tab.py`
   - `builds_info_tab.py`
   - `hardware_tab.py`

3. **Контроллеры:**
   - `server_control.py` ← зависит от server_tab.py и threads.py

4. **Главное окно:**
   - `main_window.py` ← импортирует все остальные

---

## 🧪 Тестирование на каждом этапе

После каждого модуля:
```bash
# 1. Проверить синтаксис всех Python файлов
python -m py_compile gui/*.py

# 2. Протестировать импорты
python -c "from gui.threads import ServerThread; print('✓ threads.py OK')"
python -c "from gui.project_utils import ProjectManager; print('✓ project_utils.py OK')"
# ... и т.д.

# 3. Запустить GUI
python run.py
```

---

## 📝 Контрольный список

- [ ] Создан `gui/project_utils.py`
- [ ] Создан `gui/server_tab.py`
- [ ] Создан `gui/server_control.py`
- [ ] Создан `gui/inference_tab.py`
- [ ] Создан `gui/download_tab.py`
- [ ] Создан `gui/build_tab.py`
- [ ] Создан `gui/builds_info_tab.py`
- [ ] Создан `gui/hardware_tab.py`
- [ ] Создан `gui/settings.py`
- [ ] Переименован `llama_gui.py` → `main_window.py`
- [ ] Обновлены импорты в `main_window.py`
- [ ] Обновлены импорты в `run.py`
- [ ] Все синтаксис проверены
- [ ] GUI успешно запускается

---

## 🎯 Результат

После завершения:
- ✅ 12 модулей вместо 1 гигантского файла
- ✅ Каждый модуль ≤500 строк (легче поддерживать)
- ✅ Логическое разделение (по функциональности)
- ✅ Вся функциональность сохранена

**Текущий прогресс:** 1 из 12 модулей готово (threads.py) ✅
