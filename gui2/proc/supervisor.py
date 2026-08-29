"""The single owner of child processes.

The GPUs are one resource. A server and a benchmark competing for them do not
merely give bad numbers, they can wedge the driver, so every GPU job goes
through one slot: starting a second one while the first lives is refused. The
finished job stays visible until the next start, so the exit code and the tail
of the log survive the process that produced them.

One thing is allowed to queue, and only because it is one thought: a search
over server settings is several benchmark runs that must not overlap. They are
handed over as a list and taken from it one at a time, and stopping one
abandons the rest -- somebody who stops a search means the search.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Sequence

from gui2.core.measured import Measurement
from gui2.proc.runner import Job, JobSpec, Snapshot, job_spec


class Busy(RuntimeError):
    """Raised when the GPU slot is taken."""

    def __init__(self, current: Snapshot) -> None:
        super().__init__(f"{current.label} is still {current.status}")
        self.current = current


class Supervisor:
    """Serialises access to the GPU slot."""

    def __init__(self, capacity: int = 4000,
                 on_finish: "Callable[[Job], None] | None" = None) -> None:
        self._capacity = capacity
        self._on_finish = on_finish
        self._lock = threading.RLock()
        self._job: Job | None = None
        self._pending: list[JobSpec] = []

    # -- state -------------------------------------------------------------

    @property
    def job(self) -> Job | None:
        with self._lock:
            return self._job

    @property
    def pending(self) -> int:
        """Runs still queued behind the one on the slot."""
        with self._lock:
            return len(self._pending)

    def snapshot(self) -> Snapshot | None:
        job = self.job
        return job.snapshot() if job else None

    def is_busy(self) -> bool:
        snapshot = self.snapshot()
        return bool(snapshot and snapshot.alive)

    def log_since(self, cursor: int) -> tuple[int, list[str]]:
        job = self.job
        return job.log_since(cursor) if job else (cursor, [])

    def measurement(self) -> Measurement:
        """What the job on the slot reported about its own memory.

        Read off the output that is already being captured, so nothing is
        started and nothing is asked of a driver. Meaningful from the moment
        the model finishes loading, and complete once the process has exited.
        """
        job = self.job
        return job.measurement() if job else Measurement()

    # -- control -----------------------------------------------------------

    def start(self, kind: str, label: str, argv: Sequence[str],
              cwd: Path | None = None, env: dict[str, str] | None = None) -> Job:
        """Start a GPU job, or raise `Busy` if one is already running."""
        return self.start_all(kind, [(label, argv)], cwd, env)

    def start_all(self, kind: str, runs: Sequence[tuple[str, Sequence[str]]],
                  cwd: Path | None = None, env: dict[str, str] | None = None) -> Job:
        """Start the first of a chain and queue the rest behind it."""
        if not runs:
            raise ValueError("nothing to start")
        specs = [job_spec(kind, label, argv, cwd, env) for label, argv in runs]
        with self._lock:
            current = self._job.snapshot() if self._job else None
            if current is not None and current.alive:
                raise Busy(current)
            self._pending = specs[1:]
        return self._spawn(specs[0])

    def _spawn(self, spec: JobSpec) -> Job:
        with self._lock:
            job = Job(spec, capacity=self._capacity, on_finish=self._finished)
            self._job = job
        job.start()
        if job.snapshot().status == "failed":
            # nothing was spawned, so nothing will report the end of it
            self._finished(job)
        return job

    def _finished(self, job: Job) -> None:
        """Record the run, then take the next one off the queue."""
        if self._on_finish is not None:
            self._on_finish(job)
        with self._lock:
            following = self._pending.pop(0) if self._pending else None
        if following is not None:
            self._spawn(following)

    def request_stop(self) -> bool:
        self._abandon_queue()
        job = self.job
        return job.request_stop() if job else False

    def force_stop(self) -> bool:
        self._abandon_queue()
        job = self.job
        return job.force_stop() if job else False

    def _abandon_queue(self) -> None:
        with self._lock:
            self._pending = []

    def wait(self, timeout: float | None = None) -> int | None:
        job = self.job
        return job.wait(timeout) if job else None

    def shutdown(self, timeout: float = 30.0) -> None:
        """Ask a live job to stop; used when the GUI itself goes away.

        The job is left running if it will not stop in time -- an unfinished
        benchmark is worth more than a tidy exit.
        """
        self._abandon_queue()
        job = self.job
        if job is None or not job.snapshot().alive:
            return
        job.request_stop()
        job.wait(timeout)
