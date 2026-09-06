#!/usr/bin/env bash
# Launch GUI 2.0 - the local web UI (http://127.0.0.1:8770 by default).
# Linux/macOS counterpart of start-gui.bat: one venv in $HOME, dependencies
# installed on first run, then `python -m gui2` from this directory.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${GUI2_VENV:-$HOME/.local/share/gui2-venv}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[INFO] Creating the GUI 2.0 environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR" || { echo "[ERROR] python3 -m venv failed"; exit 1; }
fi

if ! "$VENV_DIR/bin/python" -c "import fasthtml, uvicorn" >/dev/null 2>&1; then
    echo "[INFO] Installing GUI 2.0 dependencies..."
    "$VENV_DIR/bin/pip" install --disable-pip-version-check -r "$REPO_DIR/gui2/requirements.txt" \
        || { echo "[ERROR] Could not install dependencies"; exit 1; }
fi

cd "$REPO_DIR"
exec "$VENV_DIR/bin/python" -m gui2 "$@"
