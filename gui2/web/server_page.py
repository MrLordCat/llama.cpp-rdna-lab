"""Server launch page: the form and the command are both generated from the
parameter schema, so a new flag is one line in `gui2.core.params`.

The page previews and validates; starting processes belongs to the supervisor.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fasthtml.common import (
    A,
    Button,
    Details,
    Div,
    Form,
    H3,
    Input,
    Label,
    Option,
    Pre,
    Select,
    Span,
    Summary,
    Textarea,
)

from gui2.config import AppConfig
from gui2.core.bench import BenchSpec, to_bench_argv
from gui2.core.inventory import Build, discover_builds, discover_models, find_build
from gui2.core.params import GROUPS, SCHEMA, Param
from gui2.core.runspec import DEFAULTS, Problem, RunSpec, mask_api_key, to_argv, validate
from gui2.proc import Busy, Supervisor
from gui2.web.layout import shell

#: Never echoed into the address bar, browser history or the access log.
SECRET_PARAMS = frozenset({"api_key"})


def state_query(params) -> str:
    """Query string that reproduces the form on reload, without the secrets."""
    return urlencode([(key, params[key]) for key in params.keys() if key not in SECRET_PARAMS])


def spec_from_params(params) -> RunSpec:
    """A spec from form/query values; absent checkboxes mean off."""
    if not params:
        return DEFAULTS
    values = {key: params[key] for key in params.keys()}
    for param in SCHEMA:
        if param.kind == "bool":
            values[param.name] = param.name in params
    return DEFAULTS.with_values(values)


def _options(config: AppConfig, spec: RunSpec) -> dict[str, list[tuple[str, str]]]:
    models = [(f"{model.name}  ({model.size_text})", str(model.path))
              for model in discover_models(config.models) if not model.is_mmproj]
    if spec.model and spec.model not in {value for _label, value in models}:
        models.insert(0, (f"{Path(spec.model).name}  (not in models dir)", spec.model))

    builds = [(f"{build.name} · {build.backend}" + ("" if build.usable else " · no llama-server"), build.name)
              for build in discover_builds(config.builds)]
    return {
        "model": [("— select —", "")] + models,
        "build_dir": [("— select —", "")] + builds,
    }


def _control(param: Param, spec: RunSpec, options: dict[str, list[tuple[str, str]]]):
    value = getattr(spec, param.name)
    if param.name in options:
        return Select(*[Option(label, value=item, selected=item == value)
                        for label, item in options[param.name]], name=param.name)
    if param.kind == "bool":
        return Input(type="checkbox", name=param.name, checked=bool(value))
    if param.kind == "choice":
        return Select(*[Option(choice or "— default —", value=choice, selected=choice == value)
                        for choice in param.choices], name=param.name)
    if param.kind in {"int", "float"}:
        return Input(type="number", name=param.name, value=str(value),
                     min=param.minimum, max=param.maximum, step=param.step,
                     inputmode="numeric" if param.kind == "int" else None)
    if param.name in SECRET_PARAMS:
        return Input(type="password", name=param.name, value=str(value),
                     autocomplete="new-password")
    return Input(type="text", name=param.name, value=str(value))


def _field(param: Param, spec: RunSpec, options: dict[str, list[tuple[str, str]]]):
    hint = param.help or (f"emits {param.flag}" if param.flag else "")
    return Label(
        Span(param.label),
        _control(param, spec, options),
        Span(hint, cls="hint") if hint else None,
        cls="field inline" if param.kind == "bool" else "field",
        title=hint,
    )


def _build_field(spec: RunSpec, options: dict[str, list[tuple[str, str]]]):
    return Label(
        Span("Build"),
        Select(*[Option(label, value=item, selected=item == spec.build_dir)
                 for label, item in options["build_dir"]], name="build_dir"),
        Span("supplies llama-server; capabilities are read from CMakeCache", cls="hint"),
        cls="field",
    )


def form(config: AppConfig, spec: RunSpec) -> Form:
    options = _options(config, spec)
    panels = []
    for group in GROUPS:
        fields = [_field(param, spec, options) for param in SCHEMA if param.group == group]
        if group == "Model & build":
            fields.insert(1, _build_field(spec, options))
        panels.append(Div(H3(group), Div(*fields, cls="grid"), cls="panel"))

    panels.append(Div(
        H3("Extra arguments"),
        Textarea(spec.extra_args, name="extra_args", rows=3,
                 placeholder="--spec-type draft-mtp --spec-draft-n-max 3"),
        Span("Anything here wins over the generated flag with the same name.", cls="hint"),
        cls="panel",
    ))

    panels.append(Div(
        # type=button: the form never submits natively, htmx owns every request
        Button("Start server", type="button", cls="primary",
               hx_post="/server/start", hx_target="#runstate", hx_swap="outerHTML"),
        Span("Runs exactly the command shown on the right.", cls="hint"),
        cls="panel runbar",
    ))

    return Form(
        *panels,
        cls="paramform",
        # Validation lives in core.runspec.validate(), which reports problems in the
        # preview panel.  Browser validation would instead abort the htmx request
        # silently (a number outside min/max/step stops the preview from updating).
        novalidate=True,
        # POST keeps the API key in the request body: out of the address bar,
        # out of browser history and out of the uvicorn access log.
        enctype="application/x-www-form-urlencoded",
        hx_post="/server/preview",
        hx_target="#preview",
        # preview() renders its own #preview wrapper, so replace the node itself
        # instead of nesting a second element with the same id.
        hx_swap="outerHTML",
        hx_trigger="change, keyup changed delay:400ms",
    )


def _command_lines(argv: list[str]) -> str:
    lines = [Path(argv[0]).name]
    current = ""
    for token in argv[1:]:
        if token.startswith("-"):
            if current:
                lines.append(current)
            current = "  " + token
        else:
            current += f" {token}"
    if current:
        lines.append(current)
    return "\n".join(lines)


def preview(config: AppConfig, spec: RunSpec) -> Div:
    build: Build | None = find_build(discover_builds(config.builds), spec.build_dir)
    backend = build.backend if build else ""
    binary = build.server_bin if build and build.server_bin else Path("llama-server")

    problems = list(validate(spec, backend=backend, supports_rpc=build.supports_rpc if build else None))
    if build is not None and not build.usable:
        problems.insert(0, Problem("error", f"{build.name} has no llama-server binary"))

    argv = to_argv(spec, binary)
    bench_argv = to_bench_argv(spec, BenchSpec(), config.bench_script, binary)

    messages = [
        Div(("⚠ " if problem.level == "error" else "note: ") + problem.message,
            cls="problem err" if problem.level == "error" else "problem muted")
        for problem in problems
    ]

    return Div(
        Div(
            H3("Command"),
            *messages,
            Pre(f"# build: {build.name} ({build.backend})" if build else "# build: not selected"),
            Pre(_command_lines(mask_api_key(argv))),
            cls="panel",
        ),
        Details(
            Summary("Benchmark command from the same spec"),
            Pre(_command_lines(mask_api_key(bench_argv))),
            cls="panel",
        ),
        id="preview",
    )


def _endpoint(argv: tuple[str, ...]) -> str:
    """The URL a running server listens on, read back from its own argv."""
    host, port = "127.0.0.1", ""
    for flag, value in zip(argv, argv[1:]):
        if flag == "--host":
            host = "127.0.0.1" if value in {"0.0.0.0", "::"} else value
        elif flag == "--port":
            port = value
    return f"http://{host}:{port}" if port else ""


def run_panel(supervisor: Supervisor, message: str = "", level: str = "note") -> Div:
    snapshot = supervisor.snapshot()
    alive = bool(snapshot and snapshot.alive)

    body: list = []
    if message:
        body.append(Div(("⚠ " if level == "error" else "") + message,
                        cls="problem err" if level == "error" else "problem muted"))

    if snapshot is None:
        body.append(Div("Nothing has been started from this GUI.", cls="muted"))
    else:
        body.append(Div(
            Span(snapshot.status, cls=f"badge {snapshot.status}"),
            Span(snapshot.label, cls="label"),
            Span(f"pid {snapshot.pid}" if snapshot.pid else "", cls="muted"),
            Span(snapshot.runtime_text, cls="muted"),
            cls="runline",
        ))
        if not alive:
            body.append(Div(snapshot.outcome, cls="muted"))

    controls: list = []
    if alive and snapshot is not None:
        controls.append(Button("Stop", type="button", hx_post="/server/stop",
                               hx_target="#runstate", hx_swap="outerHTML"))
        endpoint = _endpoint(snapshot.argv)
        if endpoint:
            controls.append(A("Open web UI", href=endpoint, target="_blank", cls="button"))
        controls.append(Details(
            Summary("force stop"),
            Div("A hard kill interrupts GPU work mid-flight and can leave the driver "
                "unhappy. Use it only when a graceful stop was ignored.", cls="hint"),
            Button("Force stop", type="button", cls="danger", hx_post="/server/kill",
                   hx_target="#runstate", hx_swap="outerHTML"),
            cls="inline-details",
        ))

    return Div(
        H3("Process"),
        *body,
        Div(*controls, cls="row") if controls else None,
        id="runstate",
        cls="panel",
        # Poll only while something is alive; the last response has no trigger,
        # so an idle page makes no requests at all.
        hx_get="/server/status" if alive else None,
        hx_trigger="every 2s" if alive else None,
        hx_swap="outerHTML" if alive else None,
    )


def _poller(supervisor: Supervisor, cursor: int):
    """Self-replacing tail marker: new lines land in front of it."""
    snapshot = supervisor.snapshot()
    alive = bool(snapshot and snapshot.alive)
    unread = bool(snapshot and cursor < snapshot.log_total)
    trigger = "every 1s" if alive else ("load delay:300ms" if unread else None)
    return Span(
        # the marker carries the placeholder, so the first real line replaces it
        Div("no output yet", cls="muted") if not snapshot or not snapshot.log_total else None,
        id="logtail",
        hx_get=f"/server/log?cursor={cursor}",
        hx_trigger=trigger,
        hx_swap="outerHTML",
    )


def log_since(supervisor: Supervisor, cursor: int):
    cursor, lines = supervisor.log_since(cursor)
    return (*[Div(line, cls="logline") for line in lines], _poller(supervisor, cursor))


def log_panel(supervisor: Supervisor, oob: bool = False) -> Div:
    cursor, lines = supervisor.log_since(0)
    return Div(
        H3("Log"),
        Div(
            *[Div(line, cls="logline") for line in lines],
            _poller(supervisor, cursor),
            id="log",
            cls="logbox",
            # htmx's own event hook, not a JS dependency: keep the tail in view
            **{"hx-on::after-settle": "this.scrollTop = this.scrollHeight"},
        ),
        id="logpanel",
        cls="panel",
        hx_swap_oob="true" if oob else None,
    )


def start(config: AppConfig, supervisor: Supervisor, spec: RunSpec):
    """Validate, then hand the command to the supervisor."""
    build = find_build(discover_builds(config.builds), spec.build_dir)
    blocking = [problem.message for problem
                in validate(spec, backend=build.backend if build else "",
                            supports_rpc=build.supports_rpc if build else None)
                if problem.level == "error"]
    if build is None:
        blocking.append("Select a build")
    elif not build.usable:
        blocking.append(f"{build.name} has no llama-server binary")
    if blocking:
        return run_panel(supervisor, "; ".join(dict.fromkeys(blocking)), "error")

    assert build is not None and build.server_bin is not None
    label = f"llama-server · {Path(spec.model).name} · {build.name}"
    try:
        supervisor.start("server", label, to_argv(spec, build.server_bin), cwd=build.path)
    except Busy as busy:
        return run_panel(supervisor, f"{busy.current.label} is still running", "error")
    return run_panel(supervisor), log_panel(supervisor, oob=True)


def page(config: AppConfig, spec: RunSpec, supervisor: Supervisor):
    return shell(
        "Server", "/server", config,
        Div(
            form(config, spec),
            Div(preview(config, spec), run_panel(supervisor), log_panel(supervisor), cls="stack"),
            cls="split",
        ),
    )
