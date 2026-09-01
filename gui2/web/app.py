"""FastHTML application wiring. Page rendering lives in the *_page modules."""

from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

from fasthtml.common import Div, HtmxResponseHeaders, Link, RedirectResponse, Script, fast_app
from starlette.responses import Response

from gui2.config import AppConfig
from gui2.core import rpc
from gui2.core.autotune_state import AutotuneStateStore
from gui2.core.devices import DeviceService
from gui2.core.history import HistoryStore, sync_autotune_runs
from gui2.core.memstore import MemoryStore
from gui2.core.runspec import parse_rpc_endpoints
from gui2.proc import Supervisor
from gui2.proc.hidden import suppress_error_dialogs
from gui2.web import autotune_page, bench_history, history_page, models_page, server_page

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(config: AppConfig | None = None):
    config = config or AppConfig.load()
    store = HistoryStore(config.history_csv)
    memory = MemoryStore(config.memory_json)
    autotune_state = AutotuneStateStore(config.autotune_state_json)

    def learn(job) -> None:
        """Keep what a finished run left behind, in the store it belongs to.

        A server run feeds the memory notes; an autotune run is carried into
        the canonical history CSV, because bench2 records it in its own index
        and History & Analytics reads BENCH_RUNS.csv.
        """
        if job.spec.kind == "server":
            memory.remember(job.spec.argv, job.measurement())
        elif job.spec.kind == "autotune":
            try:
                sync_autotune_runs(config.history_csv,
                                   config.bench_results / "index.csv")
            except OSError:
                pass  # a generated artifact, not a reason to fail the job

    # One supervisor per app: the GPU slot is a process-wide resource.
    supervisor = Supervisor(on_finish=learn)
    suppress_error_dialogs()

    # Device discovery reads logs and the registry only, so it is safe to run
    # at startup even while the GPUs are busy.
    devices = DeviceService(
        [config.artifacts_dir, config.builds / "build_logs" / "agent-workload"],
        config.display_devices,
    )
    devices.start()

    app, rt = fast_app(
        pico=False,
        surreal=False,
        htmx=False,
        default_hdrs=True,
        static_path=str(STATIC_DIR),
        secret_key=secrets.token_hex(32),
        # the stylesheet is served without cache headers, and a stale cache is
        # how a theme fix looks broken: version it by file mtime instead
        hdrs=(Script(src="/htmx.min.js"),
              Link(rel="stylesheet",
                   href=f"/app.css?v={(STATIC_DIR / 'app.css').stat().st_mtime_ns}")),
    )

    @rt("/", methods=["GET"])
    def index():
        return RedirectResponse("/history", status_code=303)

    @rt("/history", methods=["GET"])
    def history(req):
        # catch up: runs finished while the queue was not being watched (or
        # before this GUI started) end up in the canonical CSV the page reads
        sync_autotune_runs(config.history_csv, config.bench_results / "index.csv")
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

    def restored_autotune_query(params) -> str:
        """Saved sweep choices merged onto an explicit run from Server.

        A full Autotune URL already says everything and is left alone. A
        Server link owns the model, build and devices, while the remembered
        form supplies only Autotune's workload and sweep fields.
        """
        saved = autotune_state.query()
        if not saved or autotune_page.AUTOTUNE_MARKER in params:
            return ""
        if not params:
            return saved
        current = list(params.multi_items())
        named = {key for key, _value in current}
        bench_names = set(autotune_page.BENCH_BY_NAME)
        remembered = [(key, value) for key, value in parse_qsl(
            saved, keep_blank_values=True) if key in bench_names and key not in named]
        return urlencode([*current, *remembered,
                          (autotune_page.AUTOTUNE_MARKER, "1")])

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
            # the split bars follow the device count, which this change may have
            # moved: a third card must draw a third bar without a reload
            server_page.balancer_field(spec, scan, backend, oob=True),
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

    @rt("/server/rpc/worker.bat")
    async def server_rpc_worker_bat(req):
        """The one file to run on the other machine: firewall + worker, done."""
        plan = server_page.worker_plan(req.query_params)
        return Response(
            rpc.worker_bat(plan),
            media_type="application/octet-stream",
            headers={"Content-Disposition":
                     f'attachment; filename="rpc-worker-{plan.port}.bat"'},
        )

    @rt("/server/rpc/check", methods=["POST"])
    async def server_rpc_check(req):
        params = await req.form()
        spec = server_page.spec_from_params(params)
        fleet = server_page.check_workers(spec)
        # only the probe knows a worker has two GPUs, and that renames every
        # RPC device after it — so the picker is redrawn from the same answer
        devices.remember(fleet)
        scan, backend = scanned(spec)
        return (
            server_page.rpc_status(spec, fleet),
            server_page.devices_field(spec, scan, backend, oob=True),
            # a worker's answer can add cards; the balancer must draw them too
            server_page.balancer_field(spec, scan, backend, oob=True),
            # keep the address bar in step: the links to the Autotune page are
            # built from the URL, and a worker typed but never submitted would
            # otherwise silently not follow the user there
            HtmxResponseHeaders(push_url="/server?" + server_page.state_query(params, spec)),
        )

    @rt("/server/devices", methods=["GET"])
    def server_devices(req):
        spec = server_page.spec_from_params(req.query_params)
        build = server_page.build_of(config, spec)
        backend = req.query_params.get("backend", "") or (build.backend if build else "")
        devices.start(parse_rpc_endpoints(spec.rpc_endpoints), backend)
        return (
            server_page.devices_field(spec, devices.state(), backend),
            server_page.balancer_field(spec, devices.state(), backend, oob=True),
        )

    @rt("/server/splitbalancer", methods=["GET"])
    def server_splitbalancer(req):
        spec = server_page.spec_from_params(req.query_params)
        scan, backend = scanned(spec)
        return server_page.balancer_field(spec, scan, backend)

    @rt("/server/modelfield", methods=["GET"])
    def server_modelfield(req):
        spec = server_page.spec_from_params(req.query_params)
        scan, backend = scanned(spec)
        if "autotune" in req.query_params:
            return autotune_page.model_field(config, spec, scan, backend,
                                             server_page._options(config, spec))
        return server_page.model_field(config, spec, scan, backend)

    @rt("/server/devices", methods=["POST"])
    async def server_devices_rescan(req):
        spec = server_page.spec_from_params(await req.form())
        build = server_page.build_of(config, spec)
        backend = build.backend if build else ""
        devices.refresh(parse_rpc_endpoints(spec.rpc_endpoints), backend)
        return (
            server_page.devices_field(spec, devices.state(), backend),
            server_page.balancer_field(spec, devices.state(), backend, oob=True),
        )

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

    @rt("/autotune", methods=["GET"])
    def autotune(req):
        if restored := restored_autotune_query(req.query_params):
            return RedirectResponse("/autotune?" + restored, status_code=303)
        # the server under test arrives in the query string from the Server page;
        # a link without bench values is read as a measurement of that one run
        spec = server_page.spec_from_params(req.query_params)
        scan, backend = scanned(spec)
        return autotune_page.page(config, spec,
                                  autotune_page.autotune_from_params(req.query_params, spec),
                                  supervisor, scan, backend,
                                  autotune_page.measured(config, spec),
                                  app.state.live_board, app.state.live_started)

    @rt("/autotune/preview", methods=["POST"])
    async def autotune_preview(req):
        params = await req.form()
        query = autotune_page.state_query(params)
        autotune_state.remember(query)
        spec = server_page.spec_from_params(params)
        scan, backend = scanned(spec)
        bench = autotune_page.autotune_from_params(params, spec)
        return (
            autotune_page.preview(config, spec, bench, scan, backend),
            # the model can change under this form, and what has been measured
            # of the old one says nothing about the new one
            autotune_page.earlier_panel(autotune_page.measured(config, spec), spec, bench,
                                        oob=True),
            # the device and split bars live on the form itself: a changed
            # card count must redraw them without a reload here too
            server_page.balancer_field(spec, scan, backend, oob=True),
            HtmxResponseHeaders(push_url="/autotune?" + query),
        )

    @rt("/autotune/form", methods=["POST"])
    async def autotune_form(req):
        """The form re-rendered around a model or build change.

        Those two decide the level chips, the fit verdict and the device
        list, which the preview alone cannot refresh.
        """
        params = await req.form()
        query = autotune_page.state_query(params)
        autotune_state.remember(query)
        spec = server_page.spec_from_params(params)
        bench = autotune_page.autotune_from_params(params, spec)
        scan, backend = scanned(spec)
        results = autotune_page.measured(config, spec)
        return (
            autotune_page.form(config, spec, bench, results, scan, backend),
            autotune_page.preview(config, spec, bench, scan, backend, oob=True),
            autotune_page.earlier_panel(results, spec, bench, oob=True),
            HtmxResponseHeaders(push_url="/autotune?" + query),
        )

    @rt("/autotune/start", methods=["POST"])
    async def autotune_start(req):
        params = await req.form()
        autotune_state.remember(autotune_page.state_query(params))
        spec = server_page.spec_from_params(params)
        bench = autotune_page.autotune_from_params(params, spec)
        # the queue the Results panel will follow; replaced by the next start
        app.state.live_board = autotune_page.board_for(config, spec, bench)
        # and when the queue began: rows before this belong to the previous
        # search of the same parameters, which reuses the same folder names
        app.state.live_started = datetime.now().astimezone().isoformat(timespec="seconds")
        app.state.live_series_id = (
            f"series-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}")
        return autotune_page.start(config, supervisor, spec, bench, app.state.live_board,
                                   app.state.live_series_id)

    @rt("/autotune/results", methods=["GET"])
    def autotune_results():
        board = app.state.live_board
        return autotune_page.results_panel(
            board, autotune_page.board_results(config, board, app.state.live_started),
            supervisor)

    @rt("/autotune/history", methods=["GET"])
    def autotune_history(req):
        return bench_history.page(config, req.query_params)

    @rt("/autotune/history/run", methods=["GET"])
    def autotune_history_run(req):
        return bench_history.run_detail(config, req.query_params.get("series_id", ""))

    @rt("/autotune/history/hide", methods=["GET"])
    def autotune_history_hide(req):
        return bench_history.hidden_detail(req.query_params.get("series_id", ""))

    @rt("/models", methods=["GET"])
    def models(req):
        return models_page.page(config, devices.state(), req.query_params)

    @rt("/models/rows", methods=["GET"])
    def models_rows(req):
        view = models_page.read_state(req.query_params)
        return (
            models_page.results(config, devices.state(), view),
            # the sentence beside the picker describes the chosen type, so it
            # has to travel with the table it explains
            models_page.cache_hint(view, oob=True),
            HtmxResponseHeaders(push_url=f"/models?{view.query()}"),
        )

    # A child is deliberately *not* stopped when uvicorn exits: a restarted GUI
    # is no reason to interrupt GPU work. The supervisor is exposed so tests and
    # future pages talk to the same GPU slot.
    app.state.supervisor = supervisor
    app.state.devices = devices
    app.state.memory = memory
    app.state.autotune_state = autotune_state
    app.state.live_board = []
    app.state.live_started = ""
    app.state.live_series_id = ""
    return app
