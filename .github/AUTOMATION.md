# Repository Automation

Agent orchestration, BYOK model routing, ownership, and handoff rules are
defined in `AGENT_WORKFLOW.md`. VS Code custom roles and reusable entry points
live under `.github/agents/` and `.github/prompts/`.

This fork keeps `.github/**` as a local integration layer. Upstream workflows
from `ggml-org/llama.cpp` must not automatically overwrite the fork's local
automation and hardware assumptions.

For normal development on the reference machine, use the focused validation
path:

```powershell
python -m py_compile run.py gui\main_window.py gui\server_tab.py gui\benchmark_tab.py gui\build_tab.py gui\build_manager.py gui\dependency_checker.py gui\hardware_detector.py
git diff --check
```

For ROCm-sensitive changes, configure an RDNA4 HIP build:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
```

If upstream adds a useful workflow, port it manually and validate that it does
not break the GUI, TurboQuant, ROCm, or local Windows assumptions before
enabling it in this fork.
