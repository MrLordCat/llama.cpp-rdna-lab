"""Route-level tests.

The GPU slot is exercised with a harmless python child, never llama-server:
these tests must stay safe to run while the real GPUs are busy.
"""

from __future__ import annotations

import re
import sys

import pytest
from starlette.testclient import TestClient

from gui2.config import AppConfig
from gui2.tests.fixtures import QWEN35_27B, write_gguf
from gui2.web.app import create_app

GREETER = [sys.executable, "-c", "print('hello from the child')"]


@pytest.fixture()
def client(tmp_path):
    app = create_app(AppConfig(data_root=tmp_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def models(tmp_path):
    """Two models with different trained context lengths."""
    folder = tmp_path / "models"
    folder.mkdir(exist_ok=True)
    return {
        "short": str(write_gguf(folder / "short.gguf", context=40960, layers=48)),
        "long": str(write_gguf(folder / "long.gguf", context=262144, layers=64,
                               embedding=5120, hparams=QWEN35_27B)),
    }


def test_server_page_renders_the_form_and_an_idle_process_panel(client):
    html = client.get("/server").text
    assert 'name="ctx_size"' in html
    assert "Start server" in html
    assert 'id="runstate"' in html
    assert "Nothing has been started from this GUI." in html
    # nothing alive means nothing polls
    assert "/server/status" not in html


def _ctx_box(html: str) -> dict[str, str]:
    """Attributes of the number box that submits the context value."""
    tag = re.search(r'<input[^>]*name="ctx_size"[^>]*>', html)
    assert tag, "the context field is missing from the response"
    return dict(re.findall(r'(\w+)="([^"]*)"', tag.group(0)))


def test_context_slider_tops_out_at_the_selected_model(client, models):
    html = client.get("/server", params={"model": models["short"]}).text
    box = _ctx_box(html)

    assert box["max"] == "40960"
    assert int(box["value"]) <= 40960
    assert "what this model was trained for" in html


def test_switching_models_reaims_a_context_left_at_the_old_ceiling(client, models):
    # was at the maximum a 40K model allows; a 256K model should lift it
    response = client.post("/server/bounds", data={
        "model": models["long"], "ctx_size": "40960", "_ceiling": "ctx_size:40960"})
    box = _ctx_box(response.text)
    assert box["max"] == "262144"
    assert box["value"] == "131072", "growth stops at the default, not at the model's ceiling"

    # and the other way round: 256K no longer fits a 40K model
    response = client.post("/server/bounds", data={
        "model": models["short"], "ctx_size": "262144", "_ceiling": "ctx_size:262144"})
    box = _ctx_box(response.text)
    assert box["max"] == box["value"] == "40960"


def test_a_deliberate_context_survives_a_model_change(client, models):
    response = client.post("/server/bounds", data={
        "model": models["long"], "ctx_size": "8192", "_ceiling": "ctx_size:40960"})
    assert _ctx_box(response.text)["value"] == "8192"


def test_choosing_a_model_refreshes_the_command_in_one_request(client, models):
    response = client.post("/server/bounds", data={
        "model": models["short"], "ctx_size": "262144", "_ceiling": "ctx_size:262144"})

    assert 'hx-swap-oob="true"' in response.text, "the preview has to come back with the bounds"
    assert "-c 40960" in response.text
    assert "ctx_size=40960" in response.headers.get("HX-Push-Url", "")


def test_the_memory_panel_prices_a_run_before_it_starts(client, models):
    html = client.post("/server/preview", data={
        "model": models["long"], "ctx_size": "32768",
        "cache_type_k": "f16", "cache_type_v": "f16"}).text

    assert "Estimated total" in html
    # the cost of the context belongs next to the slider that sets it, and the
    # cheaper cache types have to be priced or the choice means nothing
    assert 'id="kvline"' in html
    assert "at q8_0" in html and "at q4_0" in html


def test_a_context_change_repriced_the_cache(client, models):
    def kv_line(ctx: str) -> str:
        html = client.post("/server/preview", data={"model": models["long"], "ctx_size": ctx}).text
        return re.search(r'id="kvline"[^>]*>([^<]*)<', html).group(1)

    assert kv_line("32768") != kv_line("65536")
    # no model, nothing to say: the line stays empty rather than guessing
    empty = client.post("/server/preview", data={"ctx_size": "32768"}).text
    assert re.search(r'id="kvline"[^>]*>\s*<', empty)


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
