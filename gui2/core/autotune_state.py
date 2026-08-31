"""The last Autotune form, kept as the same safe query used in its URL."""

from __future__ import annotations

import json
import threading
from pathlib import Path

SCHEMA = 1


class AutotuneStateStore:
    """Persist the last submitted Autotune page across GUI restarts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def query(self) -> str:
        with self._lock:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return ""
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            return ""
        query = payload.get("query")
        return query if isinstance(query, str) else ""

    def remember(self, query: str) -> None:
        if not query:
            return
        payload = json.dumps({"schema": SCHEMA, "query": query}, indent=1)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(payload, encoding="utf-8")
                temporary.replace(self.path)
            except OSError:
                pass


__all__ = ["AutotuneStateStore"]