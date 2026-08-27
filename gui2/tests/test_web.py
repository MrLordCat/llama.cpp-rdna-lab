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
    known = server_page.Reading(measurement, "a run of these settings 3 minutes ago")
    html = to_xml(server_page.memory_panel(
        DEFAULTS, read_facts("nothing.gguf"), Scan(), "vulkan", known))

    assert "Measured, not estimated" in html
    assert "3 minutes ago" in html, "a figure nobody can trace is worth little"
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


# -- the second machine ----------------------------------------------------


def test_the_worker_command_is_generated_from_its_own_boxes(client):
    html = client.post("/server/rpc/command", data={
        "rpc_port": "50055", "rpc_devices": "Vulkan0, Vulkan1", "rpc_open": "on"}).text

    assert "rpc-server -H 0.0.0.0 -p 50055 -d Vulkan0,Vulkan1" in html
    # generated, never executed: the command belongs to the other machine
    assert "hx-post=\"/server/rpc/command\"" in html


def test_an_unchecked_box_makes_the_worker_private_again(client):
    html = client.post("/server/rpc/command", data={"rpc_port": "50052"}).text
    assert "-H 127.0.0.1" in html


def test_workers_are_only_asked_who_they_are_when_someone_asks(client):
    page = client.get("/server?rpc_endpoints=192.168.1.9:50052").text
    # no probe on render: a worker serves one client at a time, and the page
    # must not queue behind whatever it is already doing
    assert "Press Check to ask them who they are" in page
    assert "192.168.1.9:50052" in page


def test_checking_a_worker_reports_the_version_and_the_memory(client):
    from gui2.tests.test_rpc import FakeWorker

    fake = FakeWorker()
    fake.start()
    try:
        html = client.post("/server/rpc/check", data={"rpc_endpoints": fake.endpoint}).text
    finally:
        fake.close()

    assert "RPC0" in html and fake.endpoint in html
    assert "protocol 5.0.0" in html
    assert "8.0 GiB free of 16.0 GiB" in html


def test_a_worker_that_is_not_there_is_named_as_the_one_that_is_missing(client):
    html = client.post("/server/rpc/check", data={
        "rpc_endpoints": "127.0.0.1:9,192.0.2.1:50052"}).text
    assert "RPC0" in html and "RPC1" in html
    assert "devrow bad" in html


def test_checking_a_worker_renames_the_devices_it_actually_offers(client):
    import time

    from gui2.tests.test_rpc import FakeWorker, GIB

    fake = FakeWorker(devices=((15 * GIB, 16 * GIB), (7 * GIB, 8 * GIB)))
    fake.start()
    try:
        html = client.post("/server/rpc/check", data={"rpc_endpoints": fake.endpoint}).text
        assert "RPC0" in html and "RPC1" in html, "one worker, two GPUs, two names"

        # the picker is rescanned from the same answer, off the request thread
        query = f"/server/devices?backend=vulkan&rpc_endpoints={fake.endpoint}"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            field = client.get(query).text
            if "looking for devices" not in field:
                break
            time.sleep(0.1)
    finally:
        fake.close()

    assert re.findall(r'devname">(RPC\d)', field) == ["RPC0", "RPC1"]
    assert "15.0 GiB free" in field and "7.0 GiB free" in field


def test_a_finished_server_run_is_written_down_for_the_next_one(client):
    supervisor = client.app.state.supervisor
    memory = client.app.state.memory
    argv = announce(
        "load_tensors:      Vulkan0 model buffer size =  4615.84 MiB",
        "llama_kv_cache:    Vulkan0 KV buffer size =  1280.00 MiB",
    )
    # the command carries a context, which is what makes it rescalable later
    argv += ["-c", "32768"]

    supervisor.start("server", "pretend-server", argv)
    assert supervisor.wait(timeout=30) == 0     # joins the drain thread, so the
                                                # finish callback has already run
    _measurement, record, exact = memory.recall(argv)
    assert exact and record is not None and record.context == 32768

    doubled = [*argv[:-1], "65536"]
    scaled, from_record, exact = memory.recall(doubled)
    assert not exact and from_record is record
    assert scaled.vram[0].kv_mib == 2560.00


def generated_command(html: str) -> str:
    """The llama-server line as the page shows it, not the help text around it."""
    block = re.search(r"<pre>llama-server(.*?)</pre>", html, re.S)
    assert block, "the command is missing from the response"
    return block.group(1)


def test_the_address_bar_keeps_one_value_per_field_and_every_chosen_device():
    from gui2.web.server_page import state_query

    class Submitted:
        """A form where the worker panel was serialised twice."""

        def multi_items(self):
            return [("port", "8080"), ("devices", "Vulkan0"), ("rpc_port", "50052"),
                    ("devices", "Vulkan1"), ("rpc_port", "50200"), ("api_key", "secret")]

    query = state_query(Submitted())

    assert query.count("rpc_port=") == 1 and "rpc_port=50200" in query
    assert query.count("devices=") == 2, "each ticked device is a value of its own"
    assert "secret" not in query


def test_a_link_that_names_one_setting_leaves_every_other_default_alone(client, models):
    """The Models page links here with nothing but a path."""
    command = generated_command(client.get("/server", params={"model": models["long"]}).text)

    # an unchecked box submits nothing, but a link is not a submission: reading
    # it as one turns every default on the page off
    assert "-ngl 999" in command, "'offload every layer' is on by default"
    assert "--no-mmproj-offload" not in command
    assert "--metrics" in command


def test_clearing_a_checkbox_in_the_form_still_clears_it(client, models):
    command = generated_command(client.post("/server/preview", data={
        "_form": "1", "model": models["long"], "ctx_size": "32768"}).text)

    assert "--no-mmproj-offload" in command
    assert "--metrics" not in command


def test_the_models_page_lists_what_is_on_disk_and_links_it_to_the_server(client, models):
    html = client.get("/models").text

    assert "qwen35-27b.gguf" not in html     # the fixture names them plainly
    assert "long.gguf" in html and "short.gguf" in html
    assert 'href="/server?model=' in html
    assert "per 1K" in html, "the price of context is the reason to open this page"


def test_changing_the_cache_type_reprices_every_model_in_one_request(client, models):
    response = client.get("/models/rows", params={"kv": "q4_0"})

    assert 'id="modelrows"' in response.text
    # the explanation beside the picker must not be left describing f16
    assert 'id="kvhint"' in response.text and 'hx-swap-oob="true"' in response.text
    assert "a quarter of the size" in response.text
    # the address bar has to keep rendering the same page when reloaded
    assert response.headers["hx-push-url"] == "/models?kv=q4_0"


def test_a_benchmark_is_not_filed_as_a_server_run(client):
    supervisor = client.app.state.supervisor
    memory = client.app.state.memory
    argv = announce("load_tensors:      Vulkan0 model buffer size =  4615.84 MiB")

    supervisor.start("bench", "pretend-bench", argv)
    assert supervisor.wait(timeout=30) == 0

    # the allocations belong to a llama-server this GUI did not compose
    _measurement, record, _exact = memory.recall(argv)
    assert record is None


def bench_command(html: str) -> str:
    """The autotune line as the page shows it."""
    block = re.search(r"<pre>[^<]*agent_workload_bench\.py(.*?)</pre>", html, re.S)
    assert block, "the autotune command is missing from the response"
    return block.group(1)


def test_arriving_from_the_server_page_sweeps_that_one_configuration(client, models):
    html = client.get("/autotune", params={"model": models["long"], "ctx_size": "32768",
                                           "_form": "1"}).text

    assert "long.gguf" in html
    # the run under test travels with the page but is edited in one place only
    assert 'name="ctx_size" value="32768"' in html.replace("'", '"')
    assert 'href="/server?model=' in html
    assert "--autotune-ctx-values 32768" in bench_command(html)
    assert "One configuration" in html, "a sweep of one value is a measurement"


def test_the_autotune_page_counts_the_run_before_anything_is_started(client, models):
    html = client.post("/autotune/preview", data={
        "_form": "1", "_autotune": "1", "model": models["long"], "tasks": "quick", "runs": "3",
        "task_hard_timeout": "30", "startup_timeout": "120"}).text

    assert "2 prompts × 3 repeats — 6 requests in all" in html
    assert "cannot outlast" in html


def test_the_server_pages_link_does_not_clear_the_autotune_defaults(client, models):
    """It carries `_form` for the run, which is not an autotune submission."""
    html = client.get("/autotune", params={"model": models["long"], "_form": "1"}).text
    command = bench_command(html)

    assert "--no-reuse" in command, "'start every prompt cold' is on by default"
    assert "--write-diagnostics" in command and "--no-write-diagnostics" not in command


def test_clearing_an_autotune_checkbox_in_the_form_still_clears_it(client, models):
    command = bench_command(client.post("/autotune/preview", data={
        "_form": "1", "_autotune": "1", "model": models["long"]}).text)

    assert "--reuse" in command and "--no-reuse" not in command
    assert "--no-write-diagnostics" in command


def test_a_sweep_says_how_many_configurations_it_multiplies_out_to(client, models):
    html = client.post("/autotune/preview", data={
        "_form": "1", "_autotune": "1", "model": models["long"],
        "sweep_batch": "256,512", "sweep_ubatch": "64,128"}).text

    assert "4 server configurations (1 × 2 × 2 × 1 × 1)" in html
    assert "2 prompts against each of them — 8 requests in all" in html
    # the settings the Server page chose for these are not what is measured
    assert "is replaced by the lines above" in html
    assert "--autotune" in bench_command(html)


def test_a_server_already_on_the_gpus_stops_the_run_before_it_starts(
        client, models, monkeypatch):
    """The script's own --background-server-policy fail, said in advance."""
    monkeypatch.setattr("gui2.core.machine.running_servers", lambda *a, **k: ("26924",))

    html = client.post("/autotune/preview", data={
        "_form": "1", "_autotune": "1", "model": models["long"]}).text
    assert "already running (pid 26924)" in html
    assert "will not start" in html

    # and pressing start anyway is refused rather than queued behind it
    started = client.post("/autotune/start", data={
        "_form": "1", "_autotune": "1", "model": models["long"]}).text
    assert "already running" in started
    assert client.app.state.supervisor.snapshot() is None

    # told to share, it measures anyway and says the numbers include the other run
    shared = client.post("/autotune/preview", data={
        "_form": "1", "_autotune": "1", "model": models["long"],
        "background_server_policy": "warn"}).text
    assert "includes its load" in shared


def _write_sweep_history(tmp_path, model: str, best: str) -> None:
    """One finished autotune run of `model`, as the script records it."""
    from gui2.tests.test_history import HEADER

    folder = tmp_path / "build_logs" / "agent-workload"
    folder.mkdir(parents=True, exist_ok=True)
    row = ("2026-08-19 18:26:10,run-x,bld-1,build-vulkan,vulkan,autotune,earlier,"
           f"{model},0,quick,,1,12288,sweep,sweep,sweep,sweep,none,base,,1,-1,1,on,16,"
           "repo-snapshot,24576,0.88,1,0.2,0.9,13.3,13.3,1600.0,28.4,,,0,autotune,vulkan|lane,"
           f"{best},,,x.csv,x.log,0")
    (folder / "BENCH_RUNS.csv").write_text(f"{HEADER}\n{row}\n", encoding="utf-8")


def test_an_earlier_sweep_of_the_same_model_is_offered_back(client, models, tmp_path):
    """A sweep records only its winner; that is what makes the next one narrower."""
    _write_sweep_history(tmp_path, models["long"],
                         "ctx=12288 b=8192 ub=1024 kv=q4_0 spec=none")

    html = client.get("/autotune", params={"model": models["long"], "runs": "3",
                                           "_form": "1", "_autotune": "1"}).text
    assert "What 1 earlier sweep of this model chose" in html
    assert "12K context, batch 8192/1024, KV q4_0" in html
    assert "13.3 t/s overall" in html

    link = re.search(r'href="/autotune\?([^"]*)"', html)
    assert link and "sweep_ubatch=1024" in link.group(1)
    # and it carries the rest of the page, so following it changes only the five
    assert "runs=3" in link.group(1)


def test_the_autotune_page_keeps_the_whole_run_in_the_address_bar(client, models):
    response = client.post("/autotune/preview", data={
        "_form": "1", "_autotune": "1", "model": models["long"],
        "api_key": "secret", "tasks": "v2-review"})

    pushed = response.headers["hx-push-url"]
    assert pushed.startswith("/autotune?")
    assert "tasks=v2-review" in pushed
    assert "secret" not in pushed, "the key must not reach the address bar"
    # and reloading that URL renders the same page rather than the defaults
    assert "v2-review" in client.get(pushed).text
