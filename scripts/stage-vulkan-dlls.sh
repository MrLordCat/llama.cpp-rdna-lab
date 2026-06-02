#!/bin/bash
# Post-build DLL staging for Vulkan MinGW builds.
# Git Bash puts /mingw64/bin first in PATH, causing incompatible DLL loading.
# This script copies the correct Strawberry GCC runtime DLLs next to the exe.
# Run from repo root: bash scripts/stage-vulkan-dlls.sh

set -euo pipefail

VULKAN_BIN="build-vulkan/bin"
STRAWBERRY_BIN="/c/Strawberry/c/bin"

if [ ! -f "$VULKAN_BIN/llama-server.exe" ]; then
    echo "ERROR: $VULKAN_BIN/llama-server.exe not found. Build Vulkan first."
    exit 1
fi

echo "Staging MinGW runtime DLLs for Vulkan build..."

cp -u "$STRAWBERRY_BIN/libgcc_s_seh-1.dll" "$VULKAN_BIN/"
cp -u "$STRAWBERRY_BIN/libstdc++-6.dll"    "$VULKAN_BIN/"
cp -u "$STRAWBERRY_BIN/libwinpthread-1.dll" "$VULKAN_BIN/"

# Optional: OpenSSL DLLs (only needed if Vulkan links against them)
if [ -f "$STRAWBERRY_BIN/libcrypto-3-x64__.dll" ]; then
    cp -u "$STRAWBERRY_BIN/libcrypto-3-x64__.dll" "$VULKAN_BIN/"
fi
if [ -f "$STRAWBERRY_BIN/libssl-3-x64__.dll" ]; then
    cp -u "$STRAWBERRY_BIN/libssl-3-x64__.dll" "$VULKAN_BIN/"
fi
if [ -f "$STRAWBERRY_BIN/libgomp-1.dll" ]; then
    cp -u "$STRAWBERRY_BIN/libgomp-1.dll" "$VULKAN_BIN/"
fi

echo "Done. Verify: ./$VULKAN_BIN/llama-server.exe --version"
"$VULKAN_BIN/llama-server.exe" --version
