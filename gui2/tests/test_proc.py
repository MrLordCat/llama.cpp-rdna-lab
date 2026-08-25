"""Supervisor tests.

Deliberately harmless children only: a short python snippet, never
llama-server, a benchmark or anything that touches a GPU.
"""

from __future__ import annotations

import sys
import time

import pytest

from gui2.proc import Busy, LogBuffer, Supervisor

SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]
GREETER = [sys.executable, "-c", "print('hello'); print('world')"]


def wait_until(predicate, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_log_buffer_keeps_stable_line_numbers():
    log = LogBuffer(capacity=3)
    for index in range(5):
        log.append(f"line {index}\n")

    assert log.total == 5
    assert log.dropped == 2
    assert log.tail(2) == ["line 3", "line 4"]

    cursor, lines = log.since(0)
    assert cursor == 5
    # the first two lines are gone; the reader is not silently fed the tail twice
    assert lines == ["line 2", "line 3", "line 4"]

    cursor, lines = log.since(cursor)
    assert lines == []


def test_job_captures_output_and_exit_code():
    supervisor = Supervisor()
    job = supervisor.start("test", "greeter", GREETER)
    assert job.wait(timeout=30) == 0

    snapshot = supervisor.snapshot()
    assert snapshot is not None
    assert not snapshot.alive
    assert snapshot.returncode == 0
    assert snapshot.outcome == "finished"
    assert ["hello", "world"] == [line for line in job.log.tail(2)]


def test_missing_binary_is_recorded_not_raised():
    supervisor = Supervisor()
    supervisor.start("test", "nope", ["gui2-no-such-binary-xyz"])

    snapshot = supervisor.snapshot()
    assert snapshot is not None
    assert snapshot.status == "failed"
    assert snapshot.error
    assert not snapshot.alive


def test_second_gpu_job_is_refused_while_the_first_lives():
    supervisor = Supervisor()
    supervisor.start("server", "sleeper", SLEEPER)
    try:
        assert wait_until(supervisor.is_busy)
        with pytest.raises(Busy) as raised:
            supervisor.start("bench", "other", GREETER)
        assert raised.value.current.label == "sleeper"
    finally:
        supervisor.force_stop()
        supervisor.wait(timeout=30)

    # the slot frees up once the child is gone
    assert wait_until(lambda: not supervisor.is_busy())
    supervisor.start("bench", "other", GREETER)
    assert supervisor.wait(timeout=30) == 0


def test_graceful_stop_ends_the_child():
    supervisor = Supervisor()
    supervisor.start("server", "sleeper", SLEEPER)
    assert wait_until(supervisor.is_busy)

    assert supervisor.request_stop() is True
    stopped = wait_until(lambda: not supervisor.is_busy(), timeout=30)
    if not stopped:
        supervisor.force_stop()
        supervisor.wait(timeout=30)
        pytest.fail("child ignored the graceful stop signal")

    snapshot = supervisor.snapshot()
    assert snapshot is not None
    assert snapshot.returncode != 0 or snapshot.status == "exited"
    assert "graceful stop requested" in supervisor.job.log.text()
