"""FastHTML application wiring. Page rendering lives in the *_page modules."""

from __future__ import annotations

import secrets
from pathlib import Path

from fasthtml.common import Div, HtmxResponseHeaders, Link, RedirectResponse, Script, fast_app

from gui2.config import AppConfig
from gui2.core.devices import DeviceService
from gui2.core.history import HistoryStore
from gui2.core.memstore import MemoryStore
from gui2.core.runspec import parse_rpc_endpoints
from gui2.proc import Supervisor
from gui2.proc.hidden import suppress_error_dialogs
from gui2.web import history_page, models_page, server_page

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(config: AppConfig | None = None):
    config = config or AppConfig.load()
    store = HistoryStore(config.history_csv)
    memory = MemoryStore(config.memory_json)

    def learn(job) -> None:
        """Keep what a finished server run said its memory was.

        Only server runs: a benchmark's command line is the script's, and its
        allocations belong to a llama-server this GUI did not compose.
        """
        if job.spec.kind == "server":
            memory.remember(job.spec.argv, job.measurement())

    # One supervisor per app: the GPU slot is a process-wide resource.
    supervisor = Supervisor(on_finish=learn)
    suppress_error_dialogs()

    # Device discovery reads logs and the registry only, so it is safe to run
    # at startup even while the GPUs are busy.
    devices = DeviceService([config.artifacts_dir, config.builds / "build_logs" / "agent-workload"])
    devices.start()

    app, rt = fast_app(
        pico=False,
        surreal=False,
        htmx=False,
        default_hdrs=True,
        static_path=str(STATIC_DIR),
        secret_key=secrets.token_hex(32),
        hdrs=(Script(src="/htmx.min.js"), Link(rel="stylesheet", href="/app.css")),
    )

    @rt("/", methods=["GET"])
    def index():
        return RedirectResponse("/history", status_code=303)

    @rt("/history", methods=["GET"])
    def history(req):
        return history_page.page(history_page.read_state(req.query_params), store.runs(), config)

    @rt("/history/rows", methods=["GET"])
    def history_rows(req):
        state = history_page.read_state(req.query_params)
        # keep the address bar on a URL that renders the full page when reloaded
        return (
            history_page.results(state, store.runs()),
            HtmxResponseHeaders(push_url=f"/history?{state.query()}"),
        )

    @rt("/history/run/{index}", methods=["GET"])
    def history_run(index: int):
        run = next((item for item in store.runs() if item.index == index), None)
        if run is None:
            return Div("Run not found", cls="panel muted")
        return history_page.detail(run)

    def scanned(spec):
        """Current device list, kept in step with the build and the RPC workers."""
        build = server_page.build_of(config, spec)
        backend = build.backend if build else ""
        return server_page.rescan(devices, spec, backend), backend

    @rt("/server", methods=["GET"])
    def server(req):
        spec = server_page.spec_from_params(req.query_params)
        scan, backend = scanned(spec)
        # the query string also carries the worker-setup boxes, which are not
        # llama-server flags and so are not part of the spec
        return server_page.page(config, spec, supervisor, scan, backend,
                                req.query_params, memory)

    @rt("/server/preview", methods=["POST"])
    async def server_preview(req):
        params = await req.form()
        spec = server_page.spec_from_params(params)
        scan, backend = scanned(spec)
        return (
            server_page.preview(config, spec, scan, supervisor=supervisor, store=memory),
            server_page.devices_field(spec, scan, backend, oob=True),
            # the context slider carries the price of the context: it has to
            # follow the same change that moved it
            server_page.kv_line(spec, server_page.model_facts(spec), oob=True),
            HtmxResponseHeaders(push_url="/server?" + server_page.state_query(params)),
        )

    @rt("/server/bounds", methods=["POST"])
    async def server_bounds(req):
        params = await req.form()
        spec = server_page.spec_from_params(params)
        facts = server_page.model_facts(spec)
        spec = server_page.refit(spec, facts, server_page.read_ceilings(params))
        scan, _backend = scanned(spec)
        return (
            server_page.bounded_fields(spec, facts),
            server_page.preview(config, spec, scan, oob=True, supervisor=supervisor,
                                store=memory),
            HtmxResponseHeaders(push_url="/server?" + server_page.state_query(params, spec)),
        )

    @rt("/server/rpc/command", methods=["POST"])
    async def server_rpc_command(req):
        # pure text generation: the command is for the other machine to run
        return server_page.worker_panel(await req.form())

    @rt("/server/rpc/check", methods=["POST"])
    async def server_rpc_check(req):
        spec = server_page.spec_from_params(await req.form())
        fleet = server_page.check_workers(spec)
        # only the probe knows a worker has two GPUs, and that renames every
        # RPC device after it — so the picker is redrawn from the same answer
        devices.remember(fleet)
        scan, backend = scanned(spec)
        return (
            server_page.rpc_status(spec, fleet),
            server_page.devices_field(spec, scan, backend, oob=True),
        )

    @rt("/server/devices", methods=["GET"])
    def server_devices(req):
        spec = server_page.spec_from_params(req.query_params)
        backend = req.query_params.get("backend", "")
        devices.start(parse_rpc_endpoints(spec.rpc_endpoints), backend)
        return server_page.devices_field(spec, devices.state(), backend)

    @rt("/server/devices", methods=["POST"])
    async def server_devices_rescan(req):
        spec = server_page.spec_from_params(await req.form())
        build = server_page.build_of(config, spec)
        backend = build.backend if build else ""
        devices.refresh(parse_rpc_endpoints(spec.rpc_endpoints), backend)
        return server_page.devices_field(spec, devices.state(), backend)

    @rt("/server/start", methods=["POST"])
    async def server_start(req):
        params = await req.form()
        spec = server_page.spec_from_params(params)
        return server_page.start(config, supervisor, spec, devices.state())

    @rt("/server/status", methods=["GET"])
    def server_status():
        return server_page.run_panel(supervisor)

    @rt("/server/stop", methods=["POST"])
    def server_stop():
        stopping = supervisor.request_stop()
        return server_page.run_panel(supervisor, "Graceful stop requested" if stopping else "Nothing to stop")

    @rt("/server/kill", methods=["POST"])
    def server_kill():
        killed = supervisor.force_stop()
        return server_page.run_panel(supervisor, "Force stop sent" if killed else "Nothing to stop", "error")

    @rt("/server/log", methods=["GET"])
    def server_log(cursor: int = 0):
        return server_page.log_since(supervisor, cursor)

    @rt("/models", methods=["GET"])
    def models():
        return models_page.page(config)

    # A child is deliberately *not* stopped when uvicorn exits: a restarted GUI
    # is no reason to interrupt GPU work. The supervisor is exposed so tests and
    # future pages talk to the same GPU slot.
    app.state.supervisor = supervisor
    app.state.devices = devices
    app.state.memory = memory
    return app
