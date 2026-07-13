#!/bin/bash
# Post-build DLL staging for Vulkan MinGW builds.
# Git Bash puts /mingw64/bin first in PATH, causing incompatible DLL loading.
# This script copies the correct Strawberry GCC runtime DLLs next to the exe.
# Run from repo root: bash scripts/stage-vulkan-dlls.sh

set -euo pipefail

VULKAN_BIN="build-vulkan/bin"
if [ -d "/c/Strawberry/c/bin" ]; then
    STRAWBERRY_BIN="/c/Strawberry/c/bin"
elif [ -d "/mnt/c/Strawberry/c/bin" ]; then
    STRAWBERRY_BIN="/mnt/c/Strawberry/c/bin"
else
    echo "ERROR: Strawberry GCC runtime DLL directory not found."
    exit 1
fi

if [ ! -f "$VULKAN_BIN/llama-server.exe" ]; then
    echo "ERROR: $VULKAN_BIN/llama-server.exe not found. Build Vulkan first."
    exit 1
fi

echo "Staging MinGW runtime DLLs for Vulkan build..."

# Keep each dependency family from the same Strawberry installation. In
# particular, libcrypto needs zlib1__ and libgomp needs libdl; omitting either
# makes Windows terminate llama-server with 0xC0000135 before startup logging.
RUNTIME_DLLS=(
    libgcc_s_seh-1.dll
    libstdc++-6.dll
    libwinpthread-1.dll
    libgomp-1.dll
    libdl.dll
    libcrypto-3-x64__.dll
    libssl-3-x64__.dll
    zlib1__.dll
)

for dll in "${RUNTIME_DLLS[@]}"; do
    if [ ! -f "$STRAWBERRY_BIN/$dll" ]; then
        echo "ERROR: required runtime DLL not found: $STRAWBERRY_BIN/$dll"
        exit 1
    fi
    cp -f "$STRAWBERRY_BIN/$dll" "$VULKAN_BIN/"
done

echo "Done. Skipped llama-server.exe --version to avoid touching GPU drivers after staging."
echo "Set STAGE_VULKAN_VERIFY=1 to run the old post-stage version check manually."
if [ "${STAGE_VULKAN_VERIFY:-0}" = "1" ]; then
    "$VULKAN_BIN/llama-server.exe" --version
fi
