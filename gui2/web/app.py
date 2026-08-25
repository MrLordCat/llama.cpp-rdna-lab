"""FastHTML application wiring. Page rendering lives in the *_page modules."""

from __future__ import annotations

import secrets
from pathlib import Path

from fasthtml.common import Div, HtmxResponseHeaders, Link, RedirectResponse, Script, fast_app

from gui2.config import AppConfig
from gui2.core.history import HistoryStore
from gui2.proc import Supervisor
from gui2.proc.hidden import suppress_error_dialogs
from gui2.web import history_page, models_page, server_page

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(config: AppConfig | None = None):
    config = config or AppConfig.load()
    store = HistoryStore(config.history_csv)
    # One supervisor per app: the GPU slot is a process-wide resource.
    supervisor = Supervisor()
    suppress_error_dialogs()

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

    @rt("/server", methods=["GET"])
    def server(req):
        return server_page.page(config, server_page.spec_from_params(req.query_params), supervisor)

    @rt("/server/preview", methods=["POST"])
    async def server_preview(req):
        params = await req.form()
        return (
            server_page.preview(config, server_page.spec_from_params(params)),
            HtmxResponseHeaders(push_url="/server?" + server_page.state_query(params)),
        )

    @rt("/server/start", methods=["POST"])
    async def server_start(req):
        params = await req.form()
        return server_page.start(config, supervisor, server_page.spec_from_params(params))

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
    return app
