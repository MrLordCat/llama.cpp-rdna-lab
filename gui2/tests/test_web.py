"""Route-level tests.

The GPU slot is exercised with a harmless python child, never llama-server:
these tests must stay safe to run while the real GPUs are busy.
"""

from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient

from gui2.config import AppConfig
from gui2.web.app import create_app

GREETER = [sys.executable, "-c", "print('hello from the child')"]


@pytest.fixture()
def client(tmp_path):
    app = create_app(AppConfig(data_root=tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def test_server_page_renders_the_form_and_an_idle_process_panel(client):
    html = client.get("/server").text
    assert 'name="ctx_size"' in html
    assert "Start server" in html
    assert 'id="runstate"' in html
    assert "Nothing has been started from this GUI." in html
    # nothing alive means nothing polls
    assert "/server/status" not in html


def test_start_refuses_an_incomplete_spec_without_spawning(client):
    response = client.post("/server/start", data={"host": "127.0.0.1", "port": "8080"})
    assert response.status_code == 200
    assert "Select a model file" in response.text
    assert client.app.state.supervisor.snapshot() is None


def test_status_and_log_partials_follow_a_job(client):
    supervisor = client.app.state.supervisor
    supervisor.start("test", "greeter", GREETER)
    assert supervisor.wait(timeout=30) == 0

    status = client.get("/server/status").text
    assert "greeter" in status
    # the job is finished, so the panel must not ask to be polled again
    assert "every 2s" not in status

    first = client.get("/server/log?cursor=0").text
    assert "hello from the child" in first
    assert 'id="logtail"' in first

    total = supervisor.snapshot().log_total
    tail = client.get(f"/server/log?cursor={total}").text
    assert "hello from the child" not in tail
    # caught up on a dead job: the poller stops asking
    assert "hx-trigger" not in tail


def test_stop_on_an_idle_slot_is_harmless(client):
    assert "Nothing to stop" in client.post("/server/stop").text
