"""The last form a page was left with, kept beside the GUI's other state.

A form is remembered as the query string that reproduces it -- the same text a
link to that page would carry -- so what is written down is read back by the
same code that reads a URL, and "the page as I left it" and "the page as this
link describes it" stay one mechanism instead of two.

Two things follow from that and are deliberate:

* the file holds no secrets. The Server page builds the string with
  ``state_query``, which drops the API key, because a remembered key would
  reappear in the address bar as soon as someone followed a link;
* a missing, unreadable or older-schema file simply means "nothing remembered
  yet". Remembering a preference must never be the reason a page fails to
  open, so every failure to read or write is swallowed.

The path comes from :class:`gui2.config.AppConfig`, so the same code keeps the
file in ``build_logs/gui2`` on Windows and on Linux.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

#: bumped when the shape of the file changes; an older file is ignored rather
#: than guessed at
SCHEMA = 1

class FormStateStore:
    """Persist the last submitted form of one page across GUI restarts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def query(self) -> str:
        """The remembered form as a query string; "" when there is none."""
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
        """Write the form down. An empty form is nothing to remember."""
        if not query:
            return
        payload = json.dumps({"schema": SCHEMA, "query": query}, indent=1)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # written beside the target and renamed, so a crash mid-write
                # cannot leave a half-form behind for the next start to read
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(payload, encoding="utf-8")
                temporary.replace(self.path)
            except OSError:
                pass  # remembering is a convenience, not a reason to fail a page

__all__ = ["FormStateStore", "SCHEMA"]
