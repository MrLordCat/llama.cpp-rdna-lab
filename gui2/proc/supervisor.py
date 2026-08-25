"""The single owner of child processes.

The GPUs are one resource. A server and a benchmark competing for them do not
merely give bad numbers, they can wedge the driver, so every GPU job goes
through one slot: starting a second one while the first lives is refused, not
queued. The finished job stays visible until the next start, so the exit code
and the tail of the log survive the process that produced them.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from gui2.proc.runner import Job, Snapshot, job_spec


class Busy(RuntimeError):
    """Raised when the GPU slot is taken."""

    def __init__(self, current: Snapshot) -> None:
        super().__init__(f"{current.label} is still {current.status}")
        self.current = current


class Supervisor:
    """Serialises access to the GPU slot."""

    def __init__(self, capacity: int = 4000) -> None:
        self._capacity = capacity
        self._lock = threading.RLock()
        self._job: Job | None = None

    # -- state -------------------------------------------------------------

    @property
    def job(self) -> Job | None:
        with self._lock:
            return self._job

    def snapshot(self) -> Snapshot | None:
        job = self.job
        return job.snapshot() if job else None

    def is_busy(self) -> bool:
        snapshot = self.snapshot()
        return bool(snapshot and snapshot.alive)

    def log_since(self, cursor: int) -> tuple[int, list[str]]:
        job = self.job
        return job.log_since(cursor) if job else (cursor, [])

    # -- control -----------------------------------------------------------

    def start(self, kind: str, label: str, argv: Sequence[str],
              cwd: Path | None = None, env: dict[str, str] | None = None) -> Job:
        """Start a GPU job, or raise `Busy` if one is already running."""
        with self._lock:
            current = self._job.snapshot() if self._job else None
            if current is not None and current.alive:
                raise Busy(current)
            job = Job(job_spec(kind, label, argv, cwd, env), capacity=self._capacity)
            self._job = job
        job.start()
        return job

    def request_stop(self) -> bool:
        job = self.job
        return job.request_stop() if job else False

    def force_stop(self) -> bool:
        job = self.job
        return job.force_stop() if job else False

    def wait(self, timeout: float | None = None) -> int | None:
        job = self.job
        return job.wait(timeout) if job else None

    def shutdown(self, timeout: float = 30.0) -> None:
        """Ask a live job to stop; used when the GUI itself goes away.

        The job is left running if it will not stop in time -- an unfinished
        benchmark is worth more than a tidy exit.
        """
        job = self.job
        if job is None or not job.snapshot().alive:
            return
        job.request_stop()
        job.wait(timeout)
