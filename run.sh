#!/bin/bash
# Quick start for GUI 2.0 (the local web UI) on Linux/macOS

cd "$(dirname "$0")" || exit 1

PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    if command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "Python not found. Install Python 3.9+ first."
        exit 1
    fi
fi

if ! $PYTHON_CMD -c "import fasthtml, uvicorn" 2>/dev/null; then
    echo "Installing GUI 2.0 dependencies..."
    PIP_ARGS="--user"
    if [ -f "$($PYTHON_CMD -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')/EXTERNALLY-MANAGED" ]; then
        PIP_ARGS="--user --break-system-packages"
    fi
    $PYTHON_CMD -m pip install -r gui2/requirements.txt $PIP_ARGS
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        exit 1
    fi
fi

exec $PYTHON_CMD -m gui2 "$@"
