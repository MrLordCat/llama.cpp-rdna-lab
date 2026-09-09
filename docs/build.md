# Build Guide

This fork supports CPU, Vulkan and ROCm/HIP builds. The reference builds are
Windows x64 builds.

## Requirements

- Git and 64-bit Python 3.11 or newer with `pip`;
- CMake 3.14 or newer and Ninja (tested with CMake 3.29 and Ninja 1.12);
- Visual Studio Build Tools 2022 with **Desktop development with C++**, the
  MSVC v143 toolset, and a Windows 10 or 11 SDK;
- the current AMD display driver, including the Vulkan runtime;
- full LunarG Vulkan SDK with `glslc`, `spirv-as`, `spirv-dis`, and
  `spirv-val` for Vulkan/FP8 shader builds;
- AMD ROCm/HIP SDK 7.1 for Windows for ROCm builds;
- Strawberry Perl for Windows ROCm configuration and the reference MinGW
  Vulkan toolchain;
- OpenSSL development files. HTTPS is enabled by default; use
  `-DLLAMA_OPENSSL=OFF` only when HTTPS/model downloads are not required.

The tested Vulkan build uses the GCC 13.2 MinGW-w64 toolchain bundled with
Strawberry Perl. A MinGW executable also needs `libgcc_s_seh-1.dll`,
`libstdc++-6.dll`, and `libwinpthread-1.dll` either beside the executable or on
`PATH`. The GUI launch environment handles the configured toolchain; for a
manual launch, put `C:\Strawberry\c\bin` before other MinGW installations on
`PATH` to avoid loading incompatible runtime DLLs.

The tested ROCm build uses `clang.exe` and `clang++.exe` from HIP SDK 7.1, not
MSVC as the compiler, but still links against MSVC v143 and Windows SDK host
libraries. Strawberry Perl is also required. A full HIP compilation is memory
intensive; 64 GB RAM and `-j 4` are recommended for this fork. Allow roughly
30 GB of free disk space for source, two build trees, and one local model.

Install the Python side and verify the native tools before opening the GUI:

```powershell
python -m pip install --upgrade pip
python -m pip install -r gui/requirements-gui.txt
cmake --version
ninja --version
glslc --version
spirv-as --version
spirv-dis --version
spirv-val --version
```

The GUI's **Build & Setup** tab checks the configured dependencies and creates
backend-specific build directories. Manual equivalents are shown below.

## CPU

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4 --target llama-server
```

## Vulkan

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\<version>"
$env:PATH = "$env:VULKAN_SDK\Bin;C:\Strawberry\c\bin;$env:PATH"

cmake -S . -B build-vulkan -G Ninja `
  -DGGML_VULKAN=ON `
  -DCMAKE_C_COMPILER=C:\Strawberry\c\bin\gcc.exe `
  -DCMAKE_CXX_COMPILER=C:\Strawberry\c\bin\g++.exe `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 4 --target llama-server
```

## ROCm/HIP on Windows RDNA4

```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.1"
$env:ROCM_PATH = $env:HIP_PATH
$env:CMAKE_PREFIX_PATH = "$env:HIP_PATH\lib\cmake"
$env:PATH = "$env:HIP_PATH\bin;C:\Strawberry\perl\bin;C:\Strawberry\c\bin;$env:PATH"

cmake -S . -B build-rocm -G Ninja `
  -DGGML_HIP=ON `
  -DAMDGPU_TARGETS=gfx1201 `
  -DGGML_HIP_MMQ_MFMA=ON `
  -DGGML_HIP_ROCWMMA_FATTN=ON `
  -DGGML_HIP_NO_VMM=ON `
  -DGGML_OPENMP=OFF `
  -DCMAKE_C_COMPILER="$env:HIP_PATH\bin\clang.exe" `
  -DCMAKE_CXX_COMPILER="$env:HIP_PATH\bin\clang++.exe" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm -j 4 --target llama-server
```

ROCm uses clang from the HIP SDK but still needs the Windows SDK and MSVC host
libraries. Missing `kernel32.lib`, `msvcrtd.lib`, or similar files indicates an
incomplete Build Tools environment. The fork includes rocWMMA 7.1 headers under
`third_party/rocwmma`; no separate rocWMMA SDK install is required for the
command above.

## Focused targets

```powershell
cmake --build build-vulkan -j 4 --target llama-server
cmake --build build-vulkan -j 4 --target test-backend-ops
```

## Validation safety

Do not use `llama-server --version`/`--help`, `hipMemGetInfo`, hard process
termination or `bash scripts/stage-vulkan-dlls.sh` as build checks. Prefer CMake
target success, Python syntax checks, `ctest` and a user-approved server run.

See [Supported Backends](SUPPORTED_BACKENDS.md) for the backend allowlist.
