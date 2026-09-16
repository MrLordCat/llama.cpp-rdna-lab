"""Entry point: python -m gui2 [--host H] [--port N] [--data-root PATH]"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser
from dataclasses import replace
from pathlib import Path

import uvicorn

from gui2.config import AppConfig
from gui2.web.app import create_app


def probe_host(host: str) -> str:
    """The address a browser or a probe can actually connect to.

    A wildcard bind means "every interface", which is not a destination: both the
    readiness probe and the URL handed to the browser use loopback instead.
    """
    return "127.0.0.1" if host in ("", "0.0.0.0", "::", "[::]") else host


def port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """True when something already accepts connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((probe_host(host), port)) == 0


def wait_for_port(host: str, port: int, timeout: float = 20.0, interval: float = 0.2) -> bool:
    """Wait until host:port accepts connections; False when it never does."""
    deadline = time.monotonic() + timeout
    while True:
        if port_in_use(host, port, timeout=min(interval, timeout) or interval):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def open_when_ready(url: str, host: str, port: int, timeout: float = 20.0) -> threading.Thread:
    """Open `url` in a browser once the server is really listening.

    Opening it before uvicorn binds shows a connection error even though the GUI
    is about to come up, which reads as "the GUI did not start".
    """
    def worker() -> None:
        if wait_for_port(host, port, timeout):
            webbrowser.open(url)

    thread = threading.Thread(target=worker, daemon=True, name="gui2-open-browser")
    thread.start()
    return thread


def main() -> int:
    config = AppConfig.load()

    parser = argparse.ArgumentParser(prog="gui2", description="llama.cpp RDNA lab GUI 2.0")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--data-root", default=str(config.data_root),
                        help="worktree that owns build_logs/agent-workload")
    parser.add_argument("--open", action="store_true", help="open the UI in a browser")
    args = parser.parse_args()

    config = replace(config, host=args.host, port=args.port, data_root=Path(args.data_root))
    url = f"http://{probe_host(config.host)}:{config.port}/history"

    # A second launcher - a second click on the desktop shortcut - joins the
    # instance that already holds the port instead of dying on the bind.
    if port_in_use(config.host, config.port):
        print(f"[INFO] GUI 2.0 is already serving {url} - opening that instance")
        if args.open:
            webbrowser.open(url)
        return 0

    if args.open:
        open_when_ready(url, config.host, config.port)

    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
