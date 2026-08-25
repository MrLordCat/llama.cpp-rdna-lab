"""Entry point: python -m gui2 [--host H] [--port N] [--data-root PATH]"""

from __future__ import annotations

import argparse
import webbrowser
from dataclasses import replace
from pathlib import Path

import uvicorn

from gui2.config import AppConfig
from gui2.web.app import create_app


def main() -> None:
    config = AppConfig.load()

    parser = argparse.ArgumentParser(prog="gui2", description="llama.cpp RDNA lab GUI 2.0")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--data-root", default=str(config.data_root),
                        help="worktree that owns build_logs/agent-workload")
    parser.add_argument("--open", action="store_true", help="open the UI in a browser")
    args = parser.parse_args()

    config = replace(config, host=args.host, port=args.port, data_root=Path(args.data_root))

    if args.open:
        webbrowser.open(f"http://{config.host}:{config.port}/history")

    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
