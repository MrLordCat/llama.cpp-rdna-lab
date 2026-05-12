#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build_upstream_rocm_stock.sh [options]

Build latest upstream llama.cpp ROCm llama-server and import runtime artifacts into
this fork as build-rocm-upstream-stock.

Options:
  --upstream-dir <path>      Upstream clone directory (default: ../llama.cpp-upstream-stock)
  --import-dir <path>        Import directory inside fork (default: <fork>/build-rocm-upstream-stock)
  --rocm-root <path>         ROCm root (default: C:/Program Files/AMD/ROCm/7.1)
  --amdgpu-targets <list>    AMDGPU targets (default: gfx1201)
  --ggml-openmp <ON|OFF>     GGML OpenMP toggle (default: OFF on Windows ROCm)
  --build-subdir <name>      Upstream build subdir name (default: build-rocm)
  --jobs <n>                 Build parallel jobs (default: 16)
  -h, --help                 Show this help

Examples:
  scripts/build_upstream_rocm_stock.sh
  scripts/build_upstream_rocm_stock.sh --amdgpu-targets gfx1201 --jobs 24
EOF
}

FORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEFAULT_UPSTREAM_DIR="$(cd "$FORK_ROOT/.." && pwd -P)/llama.cpp-upstream-stock"

UPSTREAM_DIR="$DEFAULT_UPSTREAM_DIR"
IMPORT_DIR="$FORK_ROOT/build-rocm-upstream-stock"
ROCM_ROOT="${ROCM_ROOT:-C:/Program Files/AMD/ROCm/7.1}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1201}"
GGML_OPENMP="${GGML_OPENMP:-OFF}"
BUILD_SUBDIR="build-rocm"
JOBS="${JOBS:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-dir)
      UPSTREAM_DIR="$2"
      shift 2
      ;;
    --import-dir)
      IMPORT_DIR="$2"
      shift 2
      ;;
    --rocm-root)
      ROCM_ROOT="$2"
      shift 2
      ;;
    --amdgpu-targets)
      AMDGPU_TARGETS="$2"
      shift 2
      ;;
    --ggml-openmp)
      GGML_OPENMP="$2"
      shift 2
      ;;
    --build-subdir)
      BUILD_SUBDIR="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

GGML_OPENMP_UPPER="$(echo "$GGML_OPENMP" | tr '[:lower:]' '[:upper:]')"
if [[ "$GGML_OPENMP_UPPER" != "ON" && "$GGML_OPENMP_UPPER" != "OFF" ]]; then
  echo "Invalid --ggml-openmp value: $GGML_OPENMP (expected ON or OFF)" >&2
  exit 2
fi
GGML_OPENMP="$GGML_OPENMP_UPPER"

for tool in git cmake; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required tool not found in PATH: $tool" >&2
    exit 1
  fi
done

ROCM_CLANG="$ROCM_ROOT/bin/clang++.exe"
ROCM_CC="$ROCM_ROOT/bin/clang.exe"
ROCM_LLD="$ROCM_ROOT/bin/lld-link.exe"

if [[ ! -f "$ROCM_CLANG" ]]; then
  echo "ROCm compiler not found: $ROCM_CLANG" >&2
  echo "Pass --rocm-root or set ROCM_ROOT." >&2
  exit 1
fi

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  echo "[stock-rocm] Cloning upstream into: $UPSTREAM_DIR"
  git clone https://github.com/ggml-org/llama.cpp.git "$UPSTREAM_DIR"
fi

echo "[stock-rocm] Updating upstream master (ff-only)"
git -C "$UPSTREAM_DIR" fetch origin
git -C "$UPSTREAM_DIR" checkout master
git -C "$UPSTREAM_DIR" pull --ff-only origin master

UPSTREAM_COMMIT="$(git -C "$UPSTREAM_DIR" rev-parse --short=12 HEAD)"
UPSTREAM_BUILD_DIR="$UPSTREAM_DIR/$BUILD_SUBDIR"

echo "[stock-rocm] Configuring ROCm build in: $UPSTREAM_BUILD_DIR"
cmake -S "$UPSTREAM_DIR" -B "$UPSTREAM_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_OPENMP="$GGML_OPENMP" \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_C_COMPILER="$ROCM_CC" \
  -DCMAKE_CXX_COMPILER="$ROCM_CLANG" \
  -DCMAKE_LINKER="$ROCM_LLD"

echo "[stock-rocm] Building llama-server"
cmake --build "$UPSTREAM_BUILD_DIR" --target llama-server -j "$JOBS"

IMPORT_BIN="$IMPORT_DIR/bin"
mkdir -p "$IMPORT_BIN"

echo "[stock-rocm] Importing runtime artifacts to: $IMPORT_DIR"
shopt -s nullglob

for f in \
  "$UPSTREAM_BUILD_DIR/bin/llama-server.exe" \
  "$UPSTREAM_BUILD_DIR/bin/llama-server"; do
  if [[ -f "$f" ]]; then
    cp -f "$f" "$IMPORT_BIN/"
  fi
done

for pattern in "llama*.dll" "ggml*.dll" "libllama*.so*" "libggml*.so*" "libllama*.dylib" "libggml*.dylib"; do
  for f in "$UPSTREAM_BUILD_DIR/bin"/$pattern; do
    cp -f "$f" "$IMPORT_BIN/"
  done
done

if [[ -f "$UPSTREAM_BUILD_DIR/CMakeCache.txt" ]]; then
  cp -f "$UPSTREAM_BUILD_DIR/CMakeCache.txt" "$IMPORT_DIR/CMakeCache.txt"
fi

if [[ -f "$UPSTREAM_BUILD_DIR/llama-version.cmake" ]]; then
  cp -f "$UPSTREAM_BUILD_DIR/llama-version.cmake" "$IMPORT_DIR/llama-version.cmake"
fi

NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$IMPORT_DIR/UPSTREAM_STOCK_BUILD_INFO.txt" <<EOF
source_repo=https://github.com/ggml-org/llama.cpp.git
source_dir=$UPSTREAM_DIR
source_commit=$UPSTREAM_COMMIT
built_utc=$NOW_UTC
upstream_build_dir=$UPSTREAM_BUILD_DIR
import_dir=$IMPORT_DIR
amdgpu_targets=$AMDGPU_TARGETS
ggml_openmp=$GGML_OPENMP
rocm_root=$ROCM_ROOT
EOF

echo "[stock-rocm] Done. Imported upstream stock ROCm binary commit: $UPSTREAM_COMMIT"
echo "[stock-rocm] Binary path: $IMPORT_BIN"
