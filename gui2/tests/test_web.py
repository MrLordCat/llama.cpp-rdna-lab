"""Route-level tests.

The GPU slot is exercised with a harmless python child, never llama-server:
these tests must stay safe to run while the real GPUs are busy.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlencode

import pytest
from fasthtml.common import to_xml
from starlette.testclient import TestClient

from gui2.config import AppConfig
from gui2.core.results import Result
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

def test_the_top_tabs_carry_the_current_state_between_server_and_autotune(client):
    """A worker typed after the page loaded has updated the URL but not the
    baked href of the tab links; the links re-read the address bar instead,
    so the round trip Server → Autotune → Server keeps the worker."""
    html = client.get("/server").text
    tab = re.search(r'<a[^>]*data-path="/autotune"[^>]*>', html)
    assert tab, "the Autotune tab is on the Server page"
    assert "location.search" in tab.group(0)

    back = client.get("/autotune").text
    tab = re.search(r'<a[^>]*data-path="/server"[^>]*>', back)
    assert tab, "the Server tab is on the Autotune page"

    # and the History/Models tabs are still plain links
    assert 'data-path' not in re.search(r'<a[^>]*href="/history"[^>]*>', html).group(0)

def test_worker_check_keeps_the_address_in_the_url(client):
    """The links to the Autotune page come from the address bar; a worker that
    was typed and checked must survive that trip."""
    from gui2.tests.test_rpc import FakeWorker

    fake = FakeWorker()
    fake.start()
    try:
        response = client.post("/server/rpc/check", data={
            "rpc_endpoints": fake.endpoint, "_form": "1"})
    finally:
        fake.close()
    pushed = response.headers.get("hx-push-url", "")
    assert pushed.startswith("/server?")
    assert fake.endpoint in unquote(pushed)

def test_the_autotune_address_box_refreshes_the_device_list(client, models):
    """A worker pasted into Autotune appears as an RPC card without a trip
    back to the Server page first."""
    html = client.get("/autotune", params={"model": models["long"], "_form": "1"}).text
    field = re.search(r'<input[^>]*name="rpc_endpoints"[^>]*>', html)
    assert field, "the worker addresses box is on the Autotune page too"
    assert 'hx-get="/server/devices"' in field.group(0)
    assert 'hx-target="#devicefield"' in field.group(0)

def test_one_file_does_the_other_machines_whole_setup(client):
    """The .bat is what a user runs there: firewall rule and rpc-server."""
    bat = client.get("/server/rpc/worker.bat",
                     params={"rpc_port": "50052", "rpc_devices": "Vulkan0",
                             "rpc_open": "on"}).text
    assert "localport=50052" in bat
    assert '"%~dp0rpc-server.exe" -H 0.0.0.0 -p 50052 -d Vulkan0' in bat
    assert "Administrator" in bat, "the firewall rule needs an elevated shell"

    panel = client.post("/server/rpc/command", data={
        "rpc_port": "50052", "rpc_devices": "Vulkan0",
        "rpc_host": "192.168.1.60", "rpc_open": "on"}).text
    assert "Download rpc-worker-50052.bat" in panel
    assert "Fill the Worker addresses box" in panel
    assert "192.168.1.60:50052" in panel, "the address the local box wants"

    # the bat prints host:port; pasting the whole thing works too
    panel = client.post("/server/rpc/command", data={
        "rpc_port": "50052",
        "rpc_host": "192.168.1.60:50052", "rpc_open": "on"}).text
    assert "192.168.1.60:50052" in panel


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


def bench_commands_shown(html: str) -> list[str]:
    """Every bench2 command the page is offering to run, in order."""
    return re.findall(r"<pre>[^<]*bench2\.py(.*?)</pre>", html, re.S)


def bench_command(html: str) -> str:
    """The bench2 line as the page shows it."""
    found = bench_commands_shown(html)
    assert found, "the bench2 command is missing from the response"
    return found[0]


def test_arriving_from_the_server_page_measures_that_one_configuration(client, models):
    html = client.get("/autotune", params={"model": models["long"], "ctx_size": "32768",
                                           "_form": "1"}).text

    assert "long.gguf" in html
    # the run under test travels with the page but is edited in one place only
    assert 'name="ctx_size" value="32768"' in html.replace("'", '"')
    assert 'href="/server?model=' in html
    # 32768 reaches level 1's 16K of context but not level 2's 48K
    assert "--level 1" in bench_command(html)
    assert "One configuration" in html, "one value per row is a measurement"


#: The six rows are tick lists: a browser sends one entry per ticked word and
#: nothing at all for a row with none ticked, so a form that omits one is
#: saying it was emptied. Tests that are not about that must send them all.
TICKED = {"levels": "1", "batch": "8192", "ubatch": "1024", "kv": "q8_0", "spec": "none"}


def autotune_form(model: str, **overrides) -> dict:
    return {"_form": "1", "_autotune": "1", "model": model, **TICKED, **overrides}


def test_the_autotune_page_counts_the_run_before_anything_is_started(client, models):
    html = client.post("/autotune/preview", data=autotune_form(
        models["long"], runs="3", health_timeout="120")).text

    assert "L1 against one server of 16K" in html
    assert "3 requests in all" in html
    assert "2 minutes to answer" in html


def test_a_session_is_measured_beside_the_single_levels(client, models):
    """The new mode: one conversation kept between turns, not one big prompt."""
    command = bench_command(client.post("/autotune/preview", data=autotune_form(
        models["long"], levels="", session_levels="1")).text)

    assert "--session-level 1" in command
    assert "--level" not in command, "nothing was ticked on the levels row"


def test_a_session_is_ten_turns_and_the_page_counts_them(client, models):
    html = client.post("/autotune/preview", data=autotune_form(
        models["long"], levels="", session_levels="1")).text

    assert "SL1 against one server of 32K" in html
    assert "10 requests in all" in html


def test_levels_and_sessions_share_the_one_server_the_largest_asks_for(client, models):
    """bench2 loads once per run, so a big session lifts the small level's server."""
    html = client.post("/autotune/preview", data=autotune_form(
        models["long"], levels="0", session_levels="1")).text

    assert "L0, SL1 against one server of 32K" in html
    assert len(bench_commands_shown(html)) == 1, "one server is one command"


def test_the_server_pages_link_does_not_clear_the_autotune_defaults(client, models):
    """It carries `_form` for the run, which is not an autotune submission."""
    command = bench_command(client.get("/autotune", params={
        "model": models["long"], "_form": "1"}).text)

    assert "--warmup-shot" in command, "'one unmeasured shot first' is on by default"
    assert "--no-warmup-shot" not in command
    # and the run's own settings seed the rows rather than emptying them
    assert "--level 4" in command, "the default 128K context reaches level 4"
    assert "--kv-k f16" in command, "which is what the Server page's KV type is"


def test_clearing_an_autotune_switch_in_the_form_still_clears_it(client, models):
    """Absent means off, but only because this really is a submission."""
    command = bench_command(client.post(
        "/autotune/preview", data=autotune_form(models["long"])).text)

    assert "--no-warmup-shot" in command


def test_a_search_says_how_many_runs_it_multiplies_out_to(client, models):
    html = client.post("/autotune/preview", data=autotune_form(
        models["long"], batch=["4096", "8192"], kv=["q8_0", "q4_0"])).text

    # the search is one compact panel: the commands are there, but collapsed
    assert "Show the 4 commands" in html
    assert "4 runs, one per combination of batch, kv" in html
    assert len(bench_commands_shown(html)) == 4, "one bench2 process per configuration"
    # and each is named after the workload and the combination it measures
    assert "run-long-l1-b4096-u1024-q4_0-none" in html


def test_a_ubatch_above_one_batch_is_skipped_rather_than_blocking(client, models):
    """llama-server refuses the pair; dropping it keeps the rest of the search."""
    html = client.get("/autotune", params=autotune_form(
        models["long"], batch=["1024", "8192"], ubatch=["128", "1024", "2048"],
        spec=["none", "mtp"], spec_n=["2"])).text

    # (1024, 2048) is the pair that cannot run: 2 of 12 combinations
    assert "2 of 12 combinations skipped" in html
    assert "Start 10 runs" in html
    assert "Show the 10 commands" in html


def test_the_results_table_follows_the_queue_bench2_is_measuring(client):
    """One row per run, filling in as bench2 records; no scraping of its log."""
    from gui2.core.bench import Configuration
    client.app.state.live_board = [
        ("run-a-b1024-u128-q8_0-none", Configuration(1024, 128, "q8_0", "none")),
        ("run-a-b8192-u1024-q8_0-none", Configuration(8192, 1024, "q8_0", "none")),
    ]
    supervisor = client.app.state.supervisor
    # a child that stays up for the assertions, like a real model load would
    sleeper = [sys.executable, "-c",
               "import time; print('loading'); time.sleep(30)"]
    supervisor.start("autotune", "bench2 · run-a-b1024-u128-q8_0-none", sleeper)
    try:
        html = client.get("/autotune/results").text
        assert 'id="results"' in html
        assert "b1024-u128-q8_0-none" in html, "the run's settings name its row"
        assert "measuring" in html and "queued" in html
        assert "no result" not in html, "a queue that just started has no dead runs yet"
    finally:
        supervisor.force_stop()
        assert supervisor.wait(timeout=30) is not None


def test_the_ticked_rows_are_buttons_rather_than_boxes_to_type_in(client, models):
    """Every value is one button holding its own word."""
    html = client.get("/autotune", params={"model": models["long"], "_form": "1",
                                           "ctx_size": "32768"}).text

    assert 'name="kv" value="q8_0"' in html, "every KV type is offered"
    assert 'name="spec" value="mtp"' in html
    assert 'name="session_levels" value="1"' in html, "sessions sit beside the levels"
    # and the level the Server page's context reaches arrives already ticked
    assert re.search(r'name="levels" value="1"[^>]*checked', html)


def test_a_scenario_the_model_cannot_reach_is_not_offered(client, models):
    """llama-server refuses it, and bench2 would spend a health timeout finding out."""
    html = client.get("/autotune", params={"model": models["short"], "_form": "1"}).text

    # short.gguf was trained to 40960: level 1 fits, level 2's 48K does not
    assert 'name="levels" value="1"' in html
    assert 'name="levels" value="2"' not in html
    assert 'name="session_levels" value="1"' in html
    assert 'name="session_levels" value="2"' not in html


def test_a_scenario_already_chosen_stays_on_the_row_whatever_the_model_says(client, models):
    """Dropping a ticked value would change the run without saying so."""
    html = client.get("/autotune", params=autotune_form(models["short"], levels="5")).text

    assert re.search(r'name="levels" value="5"[^>]*checked', html)


def test_the_address_bar_keeps_every_ticked_value_not_just_the_last(client, models):
    """Several buttons share a row's name; keeping one would narrow it on reload."""
    data = autotune_form(models["long"])
    del data["kv"]
    response = client.post("/autotune/preview", content=urlencode(
        list(data.items()) + [("kv", "q8_0"), ("kv", "q4_0")]),
        headers={"content-type": "application/x-www-form-urlencoded"})

    pushed = response.headers["hx-push-url"]
    assert "kv=q8_0&kv=q4_0" in pushed
    # and reloading it measures both again rather than only the second
    reloaded = bench_commands_shown(client.get(pushed).text)
    assert len(reloaded) == 2
    assert "--kv-k q8_0" in reloaded[0] and "--kv-k q4_0" in reloaded[1]


def test_unticking_every_value_on_a_row_is_reported_rather_than_ignored(client, models):
    """Silently falling back to the default would measure something nobody asked for."""
    data = autotune_form(models["long"])
    del data["kv"]

    html = client.post("/autotune/preview", data=data).text
    assert "KV cache types: nothing ticked" in html


def test_nothing_ticked_to_measure_is_refused_rather_than_defaulted(client, models):
    """Given neither, bench2 falls back to level 1 on its own."""
    data = autotune_form(models["long"])
    del data["levels"]

    html = client.post("/autotune/preview", data=data).text
    assert "Nothing ticked to measure" in html
    assert "falls back to level 1" in html


def test_the_workers_reach_bench2_through_the_server_arguments(client, models):
    """A search over RPC is the point of having RPC: it must survive the handover."""
    command = bench_command(client.post("/autotune/preview", data=autotune_form(
        models["long"], rpc_endpoints="10.0.0.5:50052", devices="RPC0,Vulkan0")).text)

    # bench2 has its own device list, and would otherwise pick cards from its profile
    assert "--dev RPC0,Vulkan0" in command
    # but it has no notion of a worker, so the endpoints ride along to llama-server
    assert "--rpc 10.0.0.5:50052" in command


def test_the_autotune_page_edits_the_servers_own_settings_in_place(client, models):
    """Model, build, devices and split are the point of the run; changing them
    here must not need a round trip through the Server page."""
    html = client.get("/autotune", params={"model": models["long"], "_form": "1"}).text

    assert 'name="model"' in html and 'name="build_dir"' in html
    assert 'id="devicefield"' in html, "the device picker sits on this form too"
    assert 'name="rpc_endpoints"' in html
    assert 'name="split_mode"' in html and 'name="tensor_split"' in html
    assert 'name="gpu_layers"' in html and 'name="gpu_layers_all"' in html
    assert 'name="parallel"' in html and 'name="flash_attn"' in html
    # one control per setting: no hidden twin arguing with the visible one
    assert html.count('name="model"') == 1
    assert html.count('name="build_dir"') == 1
    # a model change has to refresh the whole form, not just the preview
    assert 'hx-post="/autotune/form"' in html


def test_server_settings_changed_on_autotune_reach_bench2(client, models):
    """The point of editing them here: they must land in the command."""
    command = bench_command(client.post("/autotune/preview", data=autotune_form(
        models["long"], gpu_layers="12", parallel="2", flash_attn="off",
        split_mode="layer", tensor_split="3,2", devices="Vulkan0",
        rpc_endpoints="10.0.0.5:50052")).text)

    assert "--gpu-layers 12" in command
    assert "--parallel 2" in command
    assert "--no-flash-attn" in command
    assert "--dev Vulkan0" in command
    assert "--sm layer" in command and "--ts 3,2" in command
    assert "--rpc 10.0.0.5:50052" in command


def test_changing_the_model_rerenders_the_whole_form(client, models):
    """The level chips and the fit verdict follow the model, so the preview
    alone is not enough."""
    response = client.post("/autotune/form", data=autotune_form(models["short"]))

    assert 'id="autotuneform"' in response.text
    assert "short.gguf" in response.text
    # the preview and the measured list refresh in the same answer, out of band
    assert 'id="autotunepreview"' in response.text
    assert 'id="earlier"' in response.text and 'hx-swap-oob="true"' in response.text
    assert response.headers["hx-push-url"].startswith("/autotune?")


def test_draft_tokens_are_an_axis_of_their_own(client, models):
    """Tick 2 and 3: two runs, one per guess-ahead count."""
    html = client.post("/autotune/preview", data=autotune_form(
        models["long"], spec=["mtp"], spec_n=["2", "3"])).text

    assert "2 runs, one per combination of spec_n" in html
    commands = bench_commands_shown(html)
    assert len(commands) == 2
    assert "--spec-n 2" in commands[0] and "--spec-n 3" in commands[1]


def test_the_share_balancer_draws_one_slider_per_device():
    """1,1 and 0.8,1 become sliders; the hidden box still submits the -ts list."""
    from gui2.core.devices import Device
    from gui2.core.runspec import DEFAULTS
    from gui2.web import server_page
    devices = (Device(name="Vulkan0", total_mib=16384),
               Device(name="Vulkan1", total_mib=16384),
               Device(name="Vulkan2", total_mib=16384))
    html = to_xml(server_page.split_balancer(DEFAULTS, devices))

    assert html.count('type="range"') == 3, "one slider per card, however many there are"
    assert 'name="split_Vulkan0"' in html and 'name="split_Vulkan2"' in html
    assert 'name="tensor_split"' in html, "the face writes what the form submits"
    assert 'onclick="splitAuto(this)"' in html, "Automatic returns to llama.cpp's own split"
    assert 'onclick="splitEqual(this)"' in html, "Equal is one click away"
    # a written share like 3,2 scales onto the bars: 60 and 40 of 100
    ratio = to_xml(server_page.split_balancer(
        replace(DEFAULTS, tensor_split="3,2"), devices[:2]))
    assert 'value="60"' in ratio and 'value="40"' in ratio
    # and equal weights stay equal through the round trip: 100 each, not 34,33,33
    equal = re.findall(r'<input type="range"[^>]*value="(\d+)"', html)
    assert equal == ["100", "100", "100"]


def test_the_device_picker_renders_cards_in_the_order_they_are_chosen(client):
    """-dev names cards left to right and -ts pairs by position; drag order
    has to survive a re-render, not just the moment of dragging."""
    import time

    from gui2.tests.test_rpc import FakeWorker, GIB

    fake = FakeWorker(devices=((8 * GIB, 16 * GIB),))
    fake.start()
    try:
        client.post("/server/rpc/check", data={
            "rpc_endpoints": fake.endpoint, "_form": "1"})
        query = urlencode([("backend", "vulkan"), ("rpc_endpoints", fake.endpoint),
                           ("devices", "RPC0"), ("devices", "Vulkan0")])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            field = client.get(f"/server/devices?{query}").text
            if "looking for devices" not in field:
                break
            time.sleep(0.1)
        rows = re.findall(r'<span class="devname">([^<]+)</span>', field)
        assert rows[0] == "RPC0", f"the chosen order comes first: {rows}"
        assert "-dev RPC0,Vulkan0" in field
        handle = re.search(r'<span[^>]*class="draghandle"[^>]*>', field)
        assert handle and 'draggable="true"' in handle.group(0), \
            "a dedicated handle is draggable; the checkbox row is not"
        assert "deviceDrag" in field, "the drag script rides with the picker"
    finally:
        fake.close()


def test_selecting_one_device_keeps_the_other_cards_available():
    """Chosen cards move to the front; unchosen cards must not be filtered out."""
    from gui2.core.devices import Device
    from gui2.core.runspec import DEFAULTS
    from gui2.web.server_page import ordered_devices

    found = (Device(name="Vulkan0"), Device(name="Vulkan1"), Device(name="RPC0"))
    ordered = ordered_devices(found, replace(DEFAULTS, devices="Vulkan1"))
    assert [device.name for device in ordered] == ["Vulkan1", "Vulkan0", "RPC0"]


def _write_bench_rows(root: Path, rows: list[dict]) -> Path:
    """bench2's index.csv, with the columns this page reads."""
    folder = root / "build_logs" / "bench"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "index.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        keys = ["run_name", "type", "level", "timestamp", "backend", "model",
                "ctx", "prefill_tps", "decode_tps", "aggregate_tps",
                "mtp_draft_n", "status", "path", "session_turns"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return folder


def test_the_autotune_history_filters_and_expands(tmp_path):
    """One row per run (its best decode), filtered by lane/backend/spec/build,
    and ▸ lists the scenarios the run is made of."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    folder = tmp_path / "build_logs" / "bench"
    folder.mkdir(parents=True, exist_ok=True)
    _write_bench_rows(tmp_path, [
        dict(run_name="vk-model-b1024-u128-q8_0-none", type="single", level="1",
             timestamp="2026-08-29T10:00:00+03:00", backend="vk", model="long.gguf",
             ctx="8192", prefill_tps="900", decode_tps="30", aggregate_tps="28",
             mtp_draft_n="", status="ok", path=str(folder / "run-vk-none"),
             session_turns="0"),
        dict(run_name="vk-model-b1024-u128-q8_0-none", type="single", level="2",
             timestamp="2026-08-29T10:02:00+03:00", backend="vk", model="long.gguf",
             ctx="49152", prefill_tps="1000", decode_tps="35", aggregate_tps="32",
             mtp_draft_n="", status="ok", path=str(folder / "run-vk-none"),
             session_turns="0"),
        dict(run_name="rocm-model-b1024-u128-q8_0-mtp-n2", type="single", level="1",
             timestamp="2026-08-29T11:00:00+03:00", backend="rocm", model="long.gguf",
             ctx="8192", prefill_tps="800", decode_tps="40", aggregate_tps="36",
             mtp_draft_n="2", status="ok", path=str(folder / "run-rocm-mtp"),
             session_turns="0"),
    ])
    run_dir = folder / "run-vk-none"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "run.json").write_text(
        '{"server": {"server_bin": "D:/x/build-vulkan-gcc16/bin/llama-server.exe"}}',
        encoding="utf-8")

    app = create_app(AppConfig(data_root=tmp_path, builds_root=tmp_path))
    with TestClient(app) as owner:
        html = owner.get("/autotune/history").text
        assert html.count('class="run-row"') == 2, "one row per run, not per scenario"
        assert "35.0" in html, "the row shows the run's best decode"

        backend = owner.get("/autotune/history", params={"backend": "rocm"}).text
        assert backend.count('class="run-row"') == 1 and "vk-model" not in backend

        mtp = owner.get("/autotune/history", params={"mtp": "mtp"}).text
        assert mtp.count('class="run-row"') == 1 and "rocm-model" in mtp

        lane = owner.get("/autotune/history", params={"lane": "L1"}).text
        assert "30.0" in lane and "35.0" not in lane, \
            "the lane picks which scenario stands for the run"

        build = owner.get("/autotune/history", params={"build": "build-vulkan-gcc16"}).text
        assert build.count('class="run-row"') == 1 and "vk-model" in build

        detail = owner.get("/autotune/history/run",
                           params={"run_name": "vk-model-b1024-u128-q8_0-none"}).text
        assert "every scenario" in detail and "35.0" in detail and "30.0" in detail


def test_a_result_reads_its_speculation_from_the_name_when_the_index_is_silent():
    """bench2 declares mtp_draft_n but never fills it; the name is the record,
    whether the token ends the name or is followed by workload tokens."""
    assert Result(run_name="vk-b8192-u1024-q8_0-none").spec_mode == "none"
    assert Result(run_name="vk-b8192-u1024-q8_0-mtp").spec_mode == "mtp"
    assert Result(run_name="vk-b8192-u1024-q8_0-mtp-n2").spec_mode == "mtp"
    assert Result(run_name="x-mtp-n2-l0-l4-r3").spec_mode == "mtp"
    assert Result(run_name="x-mtp-n2-l0-l4-r3").draft_n == "2"
    assert Result(run_name="x-mtp").draft_n == ""
    # an explicit index cell still wins over the name
    assert Result(run_name="x-mtp-n2", mtp_draft_n="3").draft_n == "3"


def test_the_history_shows_mtp_even_when_the_index_column_is_empty(tmp_path):
    """The real index never records the lookahead; a run whose folder says it
    was mtp must not read as none in the history table."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    _write_bench_rows(tmp_path, [
        dict(run_name="vk-model-b8192-u1024-q8_0-none", type="single", level="1",
             timestamp="2026-08-29T10:00:00+03:00", backend="vk", model="long.gguf",
             ctx="8192", prefill_tps="900", decode_tps="30", aggregate_tps="28",
             mtp_draft_n="", status="ok", path="", session_turns="0"),
        dict(run_name="vk-model-b8192-u1024-q8_0-mtp-n2", type="single", level="1",
             timestamp="2026-08-29T11:00:00+03:00", backend="vk", model="long.gguf",
             ctx="8192", prefill_tps="800", decode_tps="40", aggregate_tps="36",
             mtp_draft_n="", status="ok", path="", session_turns="0"),
        dict(run_name="subProject_q4-mtp-n2-l0-l4-r3", type="single", level="1",
             timestamp="2026-08-29T12:00:00+03:00", backend="vk", model="long.gguf",
             ctx="8192", prefill_tps="700", decode_tps="45", aggregate_tps="40",
             mtp_draft_n="", status="ok", path="", session_turns="0"),
    ])

    app = create_app(AppConfig(data_root=tmp_path, builds_root=tmp_path))
    with TestClient(app) as owner:
        html = owner.get("/autotune/history").text
        mtp_only = owner.get("/autotune/history", params={"mtp": "mtp"}).text
        none_only = owner.get("/autotune/history", params={"mtp": "none"}).text

    assert html.count('class="run-row"') == 3
    assert "mtp n2" in html, "the lookahead is read from the run name"
    assert mtp_only.count('class="run-row"') == 2, \
        "the token counts however far it sits from the end of the name"
    assert none_only.count('class="run-row"') == 1 and "subProject_q4" not in none_only


def test_rows_written_before_this_run_are_not_shown_as_its_results(tmp_path):
    """bench2 leaves the previous search's rows in the index until the run of
    the same folder name finishes; the panel must not read them as this one's."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    _write_bench_rows(tmp_path, [
        dict(run_name="base-b1024-u128-q8_0-mtp-n2", type="single", level="1",
             timestamp="2026-08-29T10:00:00+03:00", backend="vk", model="long.gguf",
             ctx="8192", prefill_tps="900", decode_tps="30", aggregate_tps="28",
             mtp_draft_n="2", status="ok", path="", session_turns="0"),
    ])

    import gui2.web.autotune_page as autotune_page
    from gui2.config import AppConfig
    from gui2.core.bench import Configuration
    config = AppConfig(data_root=tmp_path, builds_root=tmp_path)
    board = [("base-b1024-u128-q8_0-mtp-n2", Configuration(1024, 128, "q8_0", "mtp", 2))]

    # the previous search finished at 10:00; this queue began at noon, so its
    # old row is silent until bench2 records the new run's own
    assert autotune_page.board_results(config, board, "2026-08-29T12:00:00+03:00") == {}
    found = autotune_page.board_results(config, board, "2026-08-29T09:00:00+03:00")
    assert found["base-b1024-u128-q8_0-mtp-n2"].decode_tps == 30
    # and no cut-off at all keeps working (page load without a run in progress)
    assert autotune_page.board_results(config, board)["base-b1024-u128-q8_0-mtp-n2"]


def test_the_results_panel_hides_stale_rows_while_the_new_run_is_queued(tmp_path):
    """The live panel route uses the same cut-off as the page render: an old row
    of the same run name must not read as this queue's result."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    _write_bench_rows(tmp_path, [
        dict(run_name="base-b1024-u128-q8_0-mtp-n2", type="single", level="3",
             timestamp="2026-08-29T10:00:00+03:00", backend="vk", model="long.gguf",
             ctx="8192", prefill_tps="900", decode_tps="30", aggregate_tps="28",
             mtp_draft_n="2", status="ok", path="", session_turns="0"),
        dict(run_name="base-b1024-u128-q8_0-mtp-n2", type="single", level="1",
             timestamp="2026-08-29T12:05:00+03:00", backend="vk", model="long.gguf",
             ctx="8192", prefill_tps="1500", decode_tps="41.35", aggregate_tps="39",
             mtp_draft_n="2", status="ok", path="", session_turns="0"),
    ])

    from gui2.config import AppConfig
    from gui2.core.bench import Configuration
    app = create_app(AppConfig(data_root=tmp_path, builds_root=tmp_path))
    with TestClient(app) as owner:
        owner.app.state.live_board = [
            ("base-b1024-u128-q8_0-mtp-n2", Configuration(1024, 128, "q8_0", "mtp", 2))]
        owner.app.state.live_started = "2026-08-29T12:00:00+03:00"
        html = owner.get("/autotune/results").text
    assert "41.35" in html, "the row bench2 just wrote is shown"
    assert "30.00" not in html, "the previous search's row is not read as this one's"


def test_history_and_analytics_shows_finished_autotune_runs(tmp_path):
    """History & Analytics reads BENCH_RUNS.csv; a finished autotune run lands
    there as soon as the history page is opened, resolved from bench2's index."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    run_dir = tmp_path / "build_logs" / "bench" / "run-vk"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_bench_rows(tmp_path, [
        dict(run_name="vk-qwen-b1024-u128-f8_e4m3-mtp-n2", type="single", level="1",
             timestamp="2026-08-29T10:00:00+03:00", backend="vk", model="qwen.gguf",
             ctx="8192", prefill_tps="900", decode_tps="30", aggregate_tps="28",
             mtp_draft_n="2", status="ok", path=str(run_dir), session_turns="0"),
        dict(run_name="vk-qwen-b1024-u128-f8_e4m3-mtp-n2", type="single", level="2",
             timestamp="2026-08-29T10:02:00+03:00", backend="vk", model="qwen.gguf",
             ctx="49152", prefill_tps="1000", decode_tps="35", aggregate_tps="32",
             mtp_draft_n="2", status="ok", path=str(run_dir), session_turns="0"),
    ])
    import json as _json
    (run_dir / "run.json").write_text(_json.dumps({
        "model": "D:/x/models/qwen.gguf",
        "type": "single",
        "levels": ["1", "2"],
        "server": {"batch_size": 1024, "ubatch_size": 128, "kv_k": "f8_e4m3",
                   "kv_v": "f8_e4m3", "spec": "mtp", "spec_n": 2, "gpu_layers": 64,
                   "parallel": 1, "context_source": "synthetic", "runs": 1,
                   "temperature": 0.2, "top_p": 0.9, "flash_attn": True},
    }), encoding="utf-8")

    from gui2.config import AppConfig
    app = create_app(AppConfig(data_root=tmp_path, builds_root=tmp_path))
    with TestClient(app) as owner:
        html = owner.get("/history").text
    assert "vk-qwen-b1024-u128-f8_e4m3-mtp-n2" in html, "the run is in History & Analytics"
    assert "autotune" in html
    # and the canonical CSV actually carries it, for anything else that reads it
    assert (tmp_path / "build_logs" / "agent-workload" / "BENCH_RUNS.csv").is_file()


def test_a_server_already_on_the_gpus_stops_the_run_before_it_starts(
        client, models, monkeypatch):
    """bench2's preflight refuses outright; there is no policy to soften it."""
    monkeypatch.setattr("gui2.core.machine.running_servers", lambda *a, **k: ("26924",))

    html = client.post("/autotune/preview", data=autotune_form(models["long"])).text
    assert "already running (pid 26924)" in html
    assert "refuses to start while one is" in html

    # and pressing start anyway is refused rather than queued behind it
    started = client.post("/autotune/start", data=autotune_form(models["long"])).text
    assert "already running" in started
    assert client.app.state.supervisor.snapshot() is None


def _write_bench_index(root, model: str, setup: dict | None = None) -> None:
    """One finished bench2 session of `model`, as bench2 records it.

    The index row and the run folder beside it, because the settings behind a
    measurement are written only in the folder's own run.json.
    """
    folder = root / "build_logs" / "bench"
    run = folder / "vk-long"
    run.mkdir(parents=True, exist_ok=True)
    header = ("run_name,timestamp,type,level,backend,commit,model,ctx,prefill_tps,"
              "decode_tps,aggregate_tps,decode_slope,session_turns,status,path")
    row = (f"vk-long,2026-08-19T18:26:10+03:00,session,1,vk,a1b2c3d,{Path(model).name},"
           f"32768,1600.0,28.4,31.2,-0.21,10,ok,{run}")
    (folder / "index.csv").write_text(f"{header}\n{row}\n", encoding="utf-8")
    server = {"batch": 4096, "ubatch": 512, "kv_k": "q4_0", "spec": "none"}
    (run / "run.json").write_text(json.dumps({"server": {**server, **(setup or {})}}),
                                  encoding="utf-8")


def test_what_bench2_already_measured_of_this_model_is_offered_back(models, tmp_path):
    """Each row is a combination someone tried, with the numbers it produced."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    _write_bench_index(tmp_path, models["long"])

    app = create_app(AppConfig(data_root=tmp_path, builds_root=tmp_path))
    with TestClient(app) as owner:
        html = owner.get("/autotune", params=autotune_form(models["long"])).text

    assert "What 1 earlier measurement of this model found" in html
    # one row, read across: when, what was measured, what it was set to, how fast
    assert "2026-08-19 18:26:10" in html
    assert ">a1b2c3d<" in html, "the commit bench2 recorded for that build"
    assert ">SL1<" in html
    assert ">4096<" in html and ">512<" in html and ">q4_0<" in html
    assert ">28.40<" in html
    assert ">-0.210<" in html, "how much a session slows down as it grows"


def test_an_earlier_combination_can_be_put_back_on_the_rows(models, tmp_path):
    """Trying combinations means starting from one that was already tried."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    _write_bench_index(tmp_path, models["long"])

    app = create_app(AppConfig(data_root=tmp_path, builds_root=tmp_path))
    with TestClient(app) as owner:
        html = owner.get("/autotune", params=autotune_form(models["long"])).text
        link = re.search(r'href="(/autotune\?[^"]*)"[^>]*>use<', html)
        assert link, "an earlier run whose folder is still there can be tried again"
        back = owner.get(link.group(1).replace("&amp;", "&")).text

    # the row's own settings arrive ticked, and so does the scenario it measured
    assert 'name="batch" value="4096" checked' in back
    assert 'name="ubatch" value="512" checked' in back
    assert 'name="kv" value="q4_0" checked' in back
    # the row is a session, so it comes back on that row and not as a single level
    assert 'name="session_levels" value="1" checked' in back
    assert 'name="levels" value="1" checked' not in back


def test_a_run_whose_folder_is_gone_keeps_its_numbers_and_loses_its_link(models, tmp_path):
    """The index outlives the folders; a row it cannot explain must not pretend to."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    _write_bench_index(tmp_path, models["long"])
    (tmp_path / "build_logs" / "bench" / "vk-long" / "run.json").unlink()

    app = create_app(AppConfig(data_root=tmp_path, builds_root=tmp_path))
    with TestClient(app) as owner:
        html = owner.get("/autotune", params=autotune_form(models["long"])).text

    assert ">28.40<" in html, "the measurement is still worth showing"
    assert not re.search(r'href="/autotune[^"]*"[^>]*>use<', html)


def test_measuring_the_same_thing_twice_is_a_warning_not_a_surprise(models, tmp_path):
    """The folder name carries the settings, so a clash is a repeat of that exact run."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "bench2.py").write_text("", encoding="utf-8")
    taken = tmp_path / "build_logs" / "bench" / "run-long-l1-b8192-u1024-q8_0-none"
    taken.mkdir(parents=True)

    app = create_app(AppConfig(data_root=tmp_path, builds_root=tmp_path))
    with TestClient(app) as owner:
        html = owner.post("/autotune/preview", data=autotune_form(models["long"])).text
        changed = owner.post("/autotune/preview",
                             data=autotune_form(models["long"], ubatch="512")).text

    assert "Measured before" in html and taken.name in html
    # and the next combination is a different folder, so it says nothing at all
    assert "Measured before" not in changed


def test_the_header_links_carry_what_each_page_comes_from(client, models):
    """Server → Autotune in the header must not open an empty sweep."""
    html = client.get("/server", params={"model": models["long"], "ctx_size": "32768"}).text
    link = re.search(r'href="/autotune\?([^"]*)"', html)
    assert link, "the header link to Autotune has to carry the run"
    assert "model=" in link.group(1) and "ctx_size=32768" in link.group(1)
    assert "_form=1" in link.group(1)

    html = client.get("/autotune", params={"model": models["long"], "ctx_size": "32768",
                                            "_form": "1"}).text
    link = re.search(r'href="/server\?([^"]*)"', html)
    assert link and "model=" in link.group(1) and "ctx_size=32768" in link.group(1)


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
