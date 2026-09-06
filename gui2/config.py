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
import re
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
ENV_DISPLAY_DEVICES = "GUI2_DISPLAY_DEVICES"

_WINDOWS_ABS = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _portable_path(value: str) -> Path:
    """A configured path that works from both Windows and this worktree.

    gui2.config.json is written with Windows paths (D:/GitHub/...) because the
    lab lives on a Windows machine; on Linux the same paths only differ in
    where the drive is mounted. An absolute path or one that already exists is
    used as-is; a Windows-absolute D:/GitHub/... path that is missing here is
    retried as REPO_ROOT.parent / <the part after GitHub>, which is where the
    sibling lab checkout sits on this machine.
    """
    path = Path(str(value)).expanduser()
    if path.is_absolute() or path.exists():
        return path
    match = _WINDOWS_ABS.match(str(value))
    if not match:
        return path
    rest = Path(match.group(2))
    if rest.parts[:1] != ("GitHub",):
        return path
    sibling = REPO_ROOT.parent.joinpath(*rest.parts[1:])
    return sibling if sibling.exists() else path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Everything GUI 2.0 needs to know about the machine it runs on."""

    data_root: Path = REPO_ROOT
    models_dir: Path | None = None
    builds_root: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8770
    display_devices: tuple[str, ...] = ()

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
        """bench2, preferably the copy belonging to the builds being measured.

        bench2 reads its level tables and writes its results relative to its
        own location, so running the lab's copy is what puts the numbers where
        the rest of the lab's numbers are, and what makes the commit it records
        the commit that was actually built.
        """
        lab = self.builds / "scripts" / "bench2.py"
        return lab if lab.is_file() else REPO_ROOT / "scripts" / "bench2.py"

    @property
    def bench_results(self) -> Path:
        return self.bench_script.parent.parent / "build_logs" / "bench"

    @property
    def state_dir(self) -> Path:
        """What the GUI itself learns, kept apart from what the lab produces."""
        return self.data_root / "build_logs" / "gui2"

    @property
    def memory_json(self) -> Path:
        return self.state_dir / "memory.json"

    @property
    def autotune_state_json(self) -> Path:
        return self.state_dir / "autotune-state.json"

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
            return _portable_path(str(value)) if value else None

        data_root = directory(ENV_DATA_ROOT, "data_root") or REPO_ROOT
        host = os.environ.get(ENV_HOST) or str(data.get("host") or "127.0.0.1")
        port_text = os.environ.get(ENV_PORT) or str(data.get("port") or 8770)
        try:
            port = int(port_text)
        except ValueError:
            port = 8770

        display_value = os.environ.get(ENV_DISPLAY_DEVICES) or data.get("display_devices")
        if isinstance(display_value, (list, tuple)):
            display_devices = tuple(str(value).strip() for value in display_value
                                    if str(value).strip())
        elif display_value:
            display_devices = tuple(str(display_value).replace(",", " ").split())
        else:
            display_devices = ()

        return cls(
            data_root=data_root,
            models_dir=directory(ENV_MODELS_DIR, "models_dir"),
            builds_root=directory(ENV_BUILDS_ROOT, "builds_root"),
            host=host,
            port=port,
            display_devices=display_devices,
        )
