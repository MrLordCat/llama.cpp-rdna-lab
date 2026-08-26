"""One supervised child process and its log.

A job owns exactly one `subprocess.Popen` plus the thread draining its output.
Nothing here knows about the web layer: the UI polls `snapshot()` and
`log_since()`, so a dropped connection or a page reload cannot disturb a run.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from gui2.core.measured import Measurement, Reader
from gui2.proc import hidden

Status = Literal["starting", "running", "stopping", "exited", "failed"]

LIVE: frozenset[Status] = frozenset({"starting", "running", "stopping"})


class LogBuffer:
    """Bounded log with stable line numbers.

    Line numbers keep counting after old lines are dropped, so a poller that
    went away and came back learns it missed something instead of silently
    replaying the tail.
    """

    def __init__(self, capacity: int = 4000) -> None:
        self._lines: deque[str] = deque(maxlen=capacity)
        self._dropped = 0
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            if len(self._lines) == self._lines.maxlen:
                self._dropped += 1
            self._lines.append(line.rstrip("\r\n"))

    @property
    def total(self) -> int:
        """Number of lines ever written, including dropped ones."""
        with self._lock:
            return self._dropped + len(self._lines)

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def since(self, cursor: int) -> tuple[int, list[str]]:
        """Lines after `cursor`, and the cursor to pass in next time."""
        with self._lock:
            total = self._dropped + len(self._lines)
            start = min(max(cursor, self._dropped), total)
            return total, list(self._lines)[start - self._dropped:]

    def tail(self, count: int) -> list[str]:
        with self._lock:
            return list(self._lines)[-count:] if count > 0 else []

    def text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable view of a job, safe to render from any thread."""

    kind: str
    label: str
    status: Status
    argv: tuple[str, ...]
    pid: int | None = None
    started_at: float | None = None
    ended_at: float | None = None
    returncode: int | None = None
    error: str = ""
    log_total: int = 0

    @property
    def alive(self) -> bool:
        return self.status in LIVE

    @property
    def runtime(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.ended_at or time.time()) - self.started_at

    @property
    def runtime_text(self) -> str:
        seconds = int(self.runtime)
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

    @property
    def outcome(self) -> str:
        if self.error:
            return self.error
        if self.status in LIVE:
            return self.status
        if self.returncode == 0:
            return "finished"
        return f"exited with {self.returncode}"


@dataclass(frozen=True, slots=True)
class JobSpec:
    """Everything needed to start a child, without starting it."""

    kind: str
    label: str
    argv: tuple[str, ...]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)


class Job:
    """A running (or finished) child process."""

    def __init__(self, spec: JobSpec, capacity: int = 4000) -> None:
        self.spec = spec
        self.log = LogBuffer(capacity)
        #: the buffer above forgets; this remembers what the run said it took
        self._memory = Reader()
        self._lock = threading.Lock()
        self._popen: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._status: Status = "starting"
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._error = ""
        self._stop_sent = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn the child. Failure is recorded, not raised."""
        environment = os.environ.copy()
        environment.update(self.spec.env)
        self._started_at = time.time()
        try:
            self._popen = subprocess.Popen(
                list(self.spec.argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(self.spec.cwd) if self.spec.cwd else None,
                env=environment,
                **hidden.spawn_options(),
            )
        except OSError as exc:
            with self._lock:
                self._status = "failed"
                self._error = str(exc)
                self._ended_at = time.time()
            self.log.append(f"[gui2] failed to start: {exc}")
            return

        with self._lock:
            self._status = "running"
        self.log.append("[gui2] " + " ".join(self.spec.argv))
        self._reader = threading.Thread(target=self._drain, name=f"gui2-{self.spec.kind}", daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        popen = self._popen
        assert popen is not None and popen.stdout is not None
        try:
            for line in popen.stdout:
                self.log.append(line)
                self._memory.feed(line)
        except Exception as exc:  # pragma: no cover - pipe teardown races
            self.log.append(f"[gui2] log stream ended: {exc}")
        finally:
            popen.wait()
            with self._lock:
                self._status = "exited"
                self._ended_at = time.time()

    def request_stop(self) -> bool:
        """Ask for a graceful shutdown. Returns False if already finished."""
        popen = self._popen
        if popen is None or popen.poll() is not None:
            return False
        with self._lock:
            already_sent = self._stop_sent
            self._stop_sent = True
            self._status = "stopping"
        if already_sent:
            return True
        try:
            hidden.send_break(popen.pid)
            self.log.append("[gui2] graceful stop requested")
        except Exception as exc:
            self.log.append(f"[gui2] graceful stop failed: {exc}")
        return True

    def force_stop(self) -> bool:
        """Hard-kill the tree. Only for a child that ignored the break."""
        popen = self._popen
        if popen is None or popen.poll() is not None:
            return False
        self.log.append(f"[gui2] force-killing pid {popen.pid}")
        try:
            hidden.kill_tree(popen.pid)
        except Exception as exc:  # pragma: no cover - process died in between
            self.log.append(f"[gui2] force stop failed: {exc}")
        return True

    def wait(self, timeout: float | None = None) -> int | None:
        """Wait for exit; returns the code, or None on timeout."""
        popen = self._popen
        if popen is None:
            return None
        try:
            code = popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        return code

    # -- observation -------------------------------------------------------

    def snapshot(self) -> Snapshot:
        popen = self._popen
        with self._lock:
            status = self._status
            started_at = self._started_at
            ended_at = self._ended_at
            error = self._error
        return Snapshot(
            kind=self.spec.kind,
            label=self.spec.label,
            status=status,
            argv=self.spec.argv,
            pid=popen.pid if popen else None,
            started_at=started_at,
            ended_at=ended_at,
            returncode=popen.poll() if popen else None,
            error=error,
            log_total=self.log.total,
        )

    def log_since(self, cursor: int) -> tuple[int, list[str]]:
        return self.log.since(cursor)

    def measurement(self) -> Measurement:
        """The memory this run reported, as far as it has got."""
        return self._memory.result()


def job_spec(kind: str, label: str, argv: Sequence[str], cwd: Path | None = None,
             env: dict[str, str] | None = None) -> JobSpec:
    return JobSpec(kind=kind, label=label, argv=tuple(str(token) for token in argv),
                   cwd=cwd, env=dict(env or {}))
