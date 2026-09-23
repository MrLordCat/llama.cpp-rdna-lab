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


# -- what a request cost, as the log says ----------------------------------

from gui2.core.logstats import TurnTracker  # noqa: E402

PROMPT_LINE = "prompt eval time = 3217.81 ms /  55 tokens (  58.51 ms per token,   17.09 tokens per second)"
DECODE_LINE = "   eval time =   950.79 ms /  43 tokens (  22.11 ms per token,   45.23 tokens per second)"


def _request(tracker, prompt_ms, prompt_tokens, decode_ms, decode_tokens):
    tracker.feed(f"prompt eval time = {prompt_ms} ms / {prompt_tokens} tokens")
    tracker.feed(f"   eval time = {decode_ms} ms / {decode_tokens} tokens")


def test_the_tracker_keeps_only_the_last_three_requests():
    tracker = TurnTracker()
    for index in range(1, 5):  # four requests; the first one must fall out
        _request(tracker, 1000 * index, 100 * index, 1000, 50)

    # prompt: 100, 100, 100, 100 t/s over the last three; decode: 50 t/s
    assert tracker.summary() == "last 3: prompt 100.0 t/s · decode 50.0 t/s"


def test_the_tracker_reads_the_real_server_lines():
    tracker = TurnTracker()
    tracker.feed("slot print_timing: id  | task  | ")
    tracker.feed(PROMPT_LINE)
    tracker.feed(DECODE_LINE)
    tracker.feed("  total time =  4168.59 ms /  98 tokens")

    # 55 tokens in 3.21781 s and 43 in 0.95079 s, straight from the two lines
    assert tracker.summary() == "last 1: prompt 17.1 t/s · decode 45.2 t/s"


def test_an_indented_eval_line_is_not_confused_with_the_prompt_line():
    tracker = TurnTracker()
    tracker.feed(PROMPT_LINE)  # the prompt line alone: a prompt-only turn

    assert tracker.summary() == "last 1: prompt 17.1 t/s"
    assert "decode" not in tracker.summary()


def test_a_decode_line_without_a_prompt_still_counts():
    tracker = TurnTracker()
    tracker.feed("   eval time =  1000.00 ms /  25 tokens")

    assert tracker.summary() == "last 1: decode 25.0 t/s"


def test_nothing_finished_means_nothing_to_show():
    tracker = TurnTracker()
    tracker.feed("llama_model loaded")
    tracker.feed("slot release: id  | task  | stop processing")

    assert tracker.summary() == ""


def test_the_job_reports_the_average_of_its_last_requests():
    child = [sys.executable, "-c",
             "print('prompt eval time = 1000.00 ms / 100 tokens')\n"
             "print('   eval time = 1000.00 ms / 50 tokens')\n"
             "import time; time.sleep(2)"]
    supervisor = Supervisor()
    supervisor.start("server", "timing child", child)
    try:
        assert wait_until(lambda: supervisor.turns_summary()
                          == "last 1: prompt 100.0 t/s · decode 50.0 t/s")
    finally:
        supervisor.force_stop()
        supervisor.wait(timeout=30)
