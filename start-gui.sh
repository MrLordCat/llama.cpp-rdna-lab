#!/usr/bin/env bash
# Launch GUI 2.0 - the local web UI (http://127.0.0.1:8770 by default).
# Linux/macOS counterpart of start-gui.bat: one venv in $HOME, dependencies
# installed on first run, then `python -m gui2` from this directory.
#
# Started from the desktop shortcut, this script inherits a terminal that closes
# the moment it exits, so every failure has to say what happened and wait to be
# read. Two such failures are real on this machine:
#   - the worktree lives on a second NTFS NVMe that is mounted on demand, not at
#     boot, so a click right after a reboot can find no repository at all (the
#     home-side wrapper in ~/.local/bin/llama-gui2 mounts it; this waits anyway);
#   - a second click while the GUI is already up must join that instance instead
#     of dying on the port bind.
set -u

REPO_DIR="${GUI2_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_DIR="${GUI2_VENV:-$HOME/.local/share/gui2-venv}"
GUI2_HOST="${GUI2_HOST:-127.0.0.1}"
GUI2_PORT="${GUI2_PORT:-8770}"
MOUNT_WAIT="${GUI2_MOUNT_WAIT:-30}"

say() { printf '%s\n' "$*"; }

fatal() {
    say "[ERROR] $*"
    if [ -t 0 ] || [ -t 1 ]; then
        printf 'Press Enter to close...'
        read -r _ || true
    fi
    exit 1
}

# Open a URL without stealing the terminal: GUI2_OPEN_CMD makes this testable.
open_url() {
    local url="$1"
    if [ -n "${GUI2_OPEN_CMD:-}" ]; then
        # shellcheck disable=SC2086
        $GUI2_OPEN_CMD "$url" >/dev/null 2>&1 && return 0
    fi
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 && return 0
    fi
    if command -v open >/dev/null 2>&1; then
        open "$url" >/dev/null 2>&1 && return 0
    fi
    return 1
}

# /run/media/<user>/<uuid>/... - the uuid names the volume udisks would mount.
volume_uuid() {
    printf '%s' "$1" | sed -n 's|^/run/media/[^/]*/\([^/]*\).*|\1|p'
}

mount_volume() {
    local uuid dev
    uuid="$(volume_uuid "$REPO_DIR")"
    [ -n "$uuid" ] || return 1
    command -v udisksctl >/dev/null 2>&1 || return 1
    dev="/dev/disk/by-uuid/$uuid"
    [ -e "$dev" ] || return 1
    say "[INFO] Mounting $dev ..."
    udisksctl mount -b "$dev" >/dev/null 2>&1 && return 0
    udisksctl mount -b "$dev" --no-user-interaction >/dev/null 2>&1 && return 0
    return 1
}

if [ ! -d "$REPO_DIR" ]; then
    say "[INFO] $REPO_DIR is not there yet - the volume holding the worktree is not mounted."
    mount_volume || say "[INFO] Could not ask udisks to mount it; waiting anyway."
    for _ in $(seq 1 "$MOUNT_WAIT"); do
        [ -d "$REPO_DIR" ] && break
        sleep 1
    done
fi
[ -d "$REPO_DIR" ] || fatal "the worktree is still missing at $REPO_DIR - mount the NTFS volume (Dolphin: the 931 GB disk) and start again"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    say "[INFO] Creating the GUI 2.0 environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR" || fatal "python3 -m venv failed"
fi

if ! "$VENV_DIR/bin/python" -c "import fasthtml, uvicorn" >/dev/null 2>&1; then
    say "[INFO] Installing GUI 2.0 dependencies..."
    "$VENV_DIR/bin/pip" install --disable-pip-version-check -r "$REPO_DIR/gui2/requirements.txt" \
        || fatal "could not install dependencies"
fi

# A second click must not fight the first instance for the port (bash only;
# without /dev/tcp support the check simply does not fire).
if (exec 3<>"/dev/tcp/$GUI2_HOST/$GUI2_PORT") 2>/dev/null; then
    say "[INFO] GUI 2.0 already answers on http://$GUI2_HOST:$GUI2_PORT/ - opening that instance."
    open_url "http://$GUI2_HOST:$GUI2_PORT/history" || say "[INFO] Open http://$GUI2_HOST:$GUI2_PORT/ manually."
    exit 0
fi

cd "$REPO_DIR" || fatal "could not enter $REPO_DIR"
exec "$VENV_DIR/bin/python" -m gui2 "$@"
