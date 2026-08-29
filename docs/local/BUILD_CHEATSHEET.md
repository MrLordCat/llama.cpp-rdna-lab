# llama.cpp-rdna-lab build cheatsheet

Supported backends: CPU, Vulkan, ROCm/HIP. Run commands from the repository
root on Windows.

## CPU

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4 --target llama-server
```

## Vulkan

```powershell
cmake -S . -B build-vulkan -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 4 --target llama-server
```

## ROCm/HIP 7.1, RX 9070 XT

Prerequisite: MSVC Build Tools Desktop C++ workload + Windows SDK.

```powershell
cmake -S . -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm -j 4 --target llama-server
```

## GUI

```powershell
python run.py
```

## Validation

```powershell
python -m compileall -q gui scripts run.py
ctest --test-dir build-cpu -L main --output-on-failure
```

## Driver safety

Do not run hipMemGetInfo probes.
Do not run bash scripts/stage-vulkan-dlls.sh.
Do not use llama-server --version/--help as a post-build probe.
Stop llama-server gracefully; avoid hard termination while a GPU backend is active.
