"""Runtime configuration.

Model, build and artifact locations are configuration, not constants: the
benchmark history lives in whichever llama.cpp worktree owns the lab, which is
not necessarily the worktree GUI 2.0 runs from.

Resolution order: explicit argument -> JSON config file -> environment -> repo
defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = REPO_ROOT / "gui2.config.json"

ENV_CONFIG = "GUI2_CONFIG"
ENV_DATA_ROOT = "GUI2_DATA_ROOT"
ENV_MODELS_DIR = "GUI2_MODELS_DIR"
ENV_BUILDS_ROOT = "GUI2_BUILDS_ROOT"
ENV_HOST = "GUI2_HOST"
ENV_PORT = "GUI2_PORT"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Everything GUI 2.0 needs to know about the machine it runs on."""

    data_root: Path = REPO_ROOT
    models_dir: Path | None = None
    builds_root: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8770

    @property
    def artifacts_dir(self) -> Path:
        return self.data_root / "build_logs" / "agent-workload"

    @property
    def history_csv(self) -> Path:
        return self.artifacts_dir / "BENCH_RUNS.csv"

    @property
    def models(self) -> Path:
        return self.models_dir or self.data_root / "models"

    @property
    def builds(self) -> Path:
        return self.builds_root or self.data_root

    @property
    def bench_script(self) -> Path:
        return REPO_ROOT / "scripts" / "agent_workload_bench.py"

    @property
    def state_dir(self) -> Path:
        """What the GUI itself learns, kept apart from what the lab produces."""
        return self.data_root / "build_logs" / "gui2"

    @property
    def memory_json(self) -> Path:
        return self.state_dir / "memory.json"

    @classmethod
    def load(cls, config_file: Path | None = None) -> "AppConfig":
        path = config_file or Path(os.environ.get(ENV_CONFIG, "") or DEFAULT_CONFIG_FILE)
        data: dict[str, object] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                data = loaded

        def directory(env: str, key: str) -> Path | None:
            value = os.environ.get(env) or data.get(key)
            return Path(str(value)).expanduser() if value else None

        data_root = directory(ENV_DATA_ROOT, "data_root") or REPO_ROOT
        host = os.environ.get(ENV_HOST) or str(data.get("host") or "127.0.0.1")
        port_text = os.environ.get(ENV_PORT) or str(data.get("port") or 8770)
        try:
            port = int(port_text)
        except ValueError:
            port = 8770

        return cls(
            data_root=data_root,
            models_dir=directory(ENV_MODELS_DIR, "models_dir"),
            builds_root=directory(ENV_BUILDS_ROOT, "builds_root"),
            host=host,
            port=port,
        )
