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


def announce(*lines: str) -> list[str]:
    """A child that says what llama-server says as it allocates, then exits."""
    return [sys.executable, "-c", "print(%r)" % "\n".join(lines)]


def test_a_run_of_other_settings_is_reported_but_does_not_replace_the_estimate(client, models):
    supervisor = client.app.state.supervisor
    supervisor.start("test", "pretend-server", announce(
        "load_tensors:      Vulkan0 model buffer size =  4615.84 MiB",
        "llama_kv_cache:    Vulkan0 KV buffer size =  1380.00 MiB",
    ))
    assert supervisor.wait(timeout=30) == 0

    html = client.post("/server/preview", data={
        "model": models["long"], "ctx_size": "32768"}).text

    # the numbers are real, but they belong to some other command line
    assert "Estimated total" in html
    assert "not these settings" in html
    assert "Measured, not estimated" not in html


def test_a_finished_run_reports_its_own_buffers_instead_of_the_estimate():
    from fasthtml.common import to_xml

    from gui2.core.devices import Scan
    from gui2.core.gguf import read_facts
    from gui2.core.measured import parse_text
    from gui2.core.runspec import DEFAULTS
    from gui2.web import server_page

    measurement = parse_text(
        "load_tensors:      Vulkan0 model buffer size =  4615.84 MiB\n"
        "llama_kv_cache:    Vulkan0 KV buffer size =  1380.00 MiB\n"
        "common_memory_breakdown_print: | memory breakdown [MiB]   | total   free    self"
        "   model   context   compute    unaccounted |\n"
        "common_memory_breakdown_print: |   - Vulkan0 (RX 9070 XT) | 16304 = 8817 +"
        " (6264 =  4615 +    1629 +      19) +        1221 |\n"
    )
    html = to_xml(server_page.memory_panel(
        DEFAULTS, read_facts("nothing.gguf"), Scan(), "vulkan", measurement, matches=True))

    assert "Measured, not estimated" in html
    assert "5.9 GiB" in html          # 4615.84 + 1380 + 19 of compute
    # 6014.84 of buffers plus 1221 the driver kept, against a 16304 MiB card
    assert "Card in use: Vulkan0 at 44%" in html


def test_the_same_run_from_a_different_build_directory_still_counts():
    from gui2.web.server_page import same_run

    argv = ["D:/a/build-vulkan/bin/llama-server.exe", "-m", "m.gguf", "-c", "4096"]
    moved = ["D:/b/build-vulkan/bin/llama-server.exe", "-m", "m.gguf", "-c", "4096"]
    other = ["D:/a/build-vulkan/bin/llama-server.exe", "-m", "m.gguf", "-c", "8192"]

    assert same_run(argv, moved), "rebuilding elsewhere does not change what a run costs"
    assert not same_run(argv, other)
    assert not same_run(argv, [])


def test_a_port_someone_else_holds_is_reported_with_a_free_one():
    """A plain listening socket -- nothing is launched, nothing touches a GPU."""
    import socket

    from gui2.core.runspec import DEFAULTS
    from gui2.web.server_page import _port_problems

    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        problems = _port_problems(DEFAULTS.with_values({"port": port}))
        assert len(problems) == 1 and problems[0].level == "warn"
        assert str(port) in problems[0].message
        # the advice has to be actionable, not just a complaint
        suggested = re.search(r"port (\d+) is free", problems[0].message)
        assert suggested and int(suggested.group(1)) != port
    finally:
        listener.close()

    assert _port_problems(DEFAULTS.with_values({"port": port})) == []
