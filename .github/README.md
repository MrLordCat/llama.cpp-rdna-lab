# Repository Automation

Этот форк сохраняет `.github/**` как локальный слой. Upstream workflows из `ggml-org/llama.cpp` не должны автоматически перетирать локальные настройки.

Для обычной разработки на машине владельца главный путь проверки:

```powershell
python -m py_compile run.py gui\main_window.py gui\server_tab.py gui\benchmark_tab.py gui\build_tab.py gui\build_manager.py gui\dependency_checker.py gui\hardware_detector.py
git diff --check
```

Для ROCm-sensitive изменений:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
```

Если upstream добавляет полезный workflow, переносить его вручную и только после проверки, что он не ломает GUI/TurboQuant/ROCm assumptions этого форка.
