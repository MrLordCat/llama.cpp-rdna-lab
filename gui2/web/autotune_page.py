"""Autotune: measuring a server configuration, and searching for a better one.

bench2 loads the model once per run and measures every level and session asked
of it against that one server, so the two questions this page asks are of
different kinds. *What to measure* is free to grow -- another level costs the
prompt it sends, not another model load. *What to try* is not: batch, ubatch,
KV type and speculation are fixed for a whole run, so a second value on any of
them is a second run, and the page queues them rather than pretending
otherwise.

The server under test is the run's own settings, and it is edited in place on
this page: the model, the build, the devices, the layer split and the RPC
workers sit in the first panel instead of behind a detour through the Server
page. This page then chooses the workload and the settings to try.
"""

from __future__ import annotations

from dataclasses import fields
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
    Table,
    Tbody,
    Td,
    Th,
    Thead,
    Tr,
)

from gui2.config import AppConfig
from gui2.core import machine
from gui2.core.bench import (
    BENCH_BY_NAME,
    BENCH_DEFAULTS,
    B_FAIR,
    B_LIMITS,
    B_OUTPUT,
    B_PROMPT,
    B_SWEEP,
    B_WORK,
    BenchSpec,
    Configuration,
    LEVELS,
    MULTI_NAMES,
    Plan,
    SESSIONS,
    Scenario,
    Weighed,
    bench_commands,
    config_count,
    configurations,
    fit,
    items,
    plan,
    run_names,
    scenarios,
    server_context,
    validate_bench,
    varied,
)
from gui2.core.devices import Scan, pool
from gui2.core.gguf import ModelFacts, context_text
from gui2.core.memory import gib
from gui2.core.params import BY_NAME, Param, bounds
from gui2.core.results import Result, Setup, for_model, read_index, setup_of, taken_names
from gui2.core.runspec import Problem, RunSpec, mask_api_key
from gui2.proc import Busy, Supervisor
from gui2.proc.hidden import console_python
from gui2.web import server_page
from gui2.web.controls import toggle
from gui2.web.layout import command_lines, problem_lines, shell

#: These fields need a submission marker of their own. The Server page's link
#: carries `_form` so that the *run* reads back exactly as it was sent, and
#: reusing it here would let that link switch off every default the link never
#: mentioned.
AUTOTUNE_MARKER = "_autotune"

SECTIONS: tuple[tuple[str, str], ...] = (
    (B_WORK, "One server is loaded, sized to the largest scenario ticked, and every "
             "scenario is measured against it. So a second level costs the prompt it "
             "sends and nothing more."),
    (B_SWEEP, "These four are fixed for a whole run, so a second value on any row is a "
              "second run with the model loaded again. One value everywhere is a "
              "measurement; more than one is a search, done one run at a time."),
    (B_PROMPT, "The tokens each level asks for have to be made of something. Synthetic "
               "text is the same every time, which is what makes two runs comparable."),
    (B_FAIR, "The ways a number can flatter itself, or wander: a cold first request, "
             "a sampler long enough to change the answer's length."),
    (B_LIMITS, "A run left alone must end by itself."),
    (B_OUTPUT, "Where the results land."),
)


# -- reading the form ------------------------------------------------------


def autotune_from_params(params, spec: RunSpec) -> BenchSpec:
    """The bench settings from the form, or a run seeded from the server.

    Without the marker this is a link rather than a submission -- from the
    Server page, or the address bar -- so it starts as the one configuration
    that link describes, at the level its context has room for.
    """
    if not params or AUTOTUNE_MARKER not in params:
        seeded = BENCH_DEFAULTS.seeded_from(spec)
        return seeded.with_values(_read(params)) if params else seeded
    values = _read(params)
    # an unchecked box submits nothing; only a real submission may read that
    # silence as "off" -- or, for an axis, as emptied (the Server page's rule)
    for param in BENCH_BY_NAME.values():
        if param.kind == "bool":
            values[param.name] = param.name in params
        elif param.kind == "multi":
            values.setdefault(param.name, "")
    return BENCH_DEFAULTS.with_values(values)


def _read(params) -> dict:
    """The form as a flat dict, with the tick lists rejoined into one axis each."""
    getlist = getattr(params, "getlist", None)
    return {key: ",".join(getlist(key)) if getlist and key in MULTI_NAMES else params[key]
            for key in params.keys()}


def state_query(params) -> str:
    """The whole page as a link: the run, the workload, no secrets.

    The tick rows arrive as several boxes under one name, like the device list.
    Keeping only the last would silently narrow the search on reload.
    """
    return server_page.state_query(params, multi=server_page.MULTI_PARAMS | MULTI_NAMES)


def page_link(spec: RunSpec, bench: BenchSpec, /, **overrides: str) -> str:
    """This page as a link, with a few lines rewritten.

    The mirror of `server_page.spec_link`: written the way the form writes it,
    marker and all, so reading it back gives what it says. The two arguments
    are positional-only because `spec` is also the name of an axis.
    """
    pairs: list[tuple[str, str]] = []
    for field in fields(bench):
        value = overrides.get(field.name, getattr(bench, field.name))
        if isinstance(value, bool):
            # an unticked box sends nothing, and the marker says the silence was meant
            if value:
                pairs.append((field.name, "on"))
        else:
            pairs.append((field.name, str(value)))
    pairs.append((AUTOTUNE_MARKER, "1"))
    return "/autotune?" + server_page.spec_link(spec) + "&" + urlencode(pairs)


# -- fields ----------------------------------------------------------------


KV_CHIP_HELP = {
    "f16": "exact, and twice the size of everything else",
    "q8_0": "half the size, no measurable loss",
    "f8_e4m3": "half the size, newer format",
    "q4_0": "a quarter of the size, and it can show",
}

SPEC_CHIP_HELP = {
    "none": "the model generates every token itself",
    "mtp": "the model's own draft head guesses ahead",
}

CONTEXT_SOURCE_HELP = {
    "synthetic": "generated text, the same every run — the comparable one",
    "repo-snapshot": "this repository's own source, up to about 53K tokens",
    "file": "a text file of your own",
}


def _scenario_title(item: Scenario) -> str:
    if item.turns > 1:
        return (f"{item.name} — {item.turns} turns of ~{item.prompt_tokens} tokens in, "
                f"{item.decode_tokens} out, in {context_text(item.ctx)} of context")
    return (f"{item.name} — ~{item.prompt_tokens} tokens in, {item.decode_tokens} out, "
            f"in {context_text(item.ctx)} of context")


def _chip_label(param: Param, value: str) -> str:
    if param.name == "levels" and value in LEVELS:
        return f"L{value} · {context_text(LEVELS[value].ctx)}"
    if param.name == "session_levels" and value in SESSIONS:
        return f"SL{value} · {context_text(SESSIONS[value].ctx)}"
    return value


def _chip_title(param: Param, value: str) -> str:
    if param.name == "levels" and value in LEVELS:
        return _scenario_title(LEVELS[value])
    if param.name == "session_levels" and value in SESSIONS:
        return _scenario_title(SESSIONS[value])
    return {"kv": KV_CHIP_HELP, "spec": SPEC_CHIP_HELP}.get(param.name, {}).get(value, "")


def _offered(param: Param, bench: BenchSpec, facts: ModelFacts | None) -> list[str]:
    """The row of values to tick, with whatever is already chosen kept in it.

    A scenario the model cannot reach is not offered: bench2 would start a
    server that llama-server refuses, and wait out the whole health timeout
    finding that out.
    """
    ceiling = facts.n_ctx_train if facts else 0
    table = {"levels": LEVELS, "session_levels": SESSIONS}.get(param.name)
    offered = [choice for choice in param.choices
               if not (ceiling and table and table[choice].ctx > ceiling)]
    chosen = items(getattr(bench, param.name))
    return offered + [value for value in chosen if value not in offered]


def _chips(param: Param, bench: BenchSpec, facts: ModelFacts | None) -> Div:
    chosen = set(items(getattr(bench, param.name)))
    return Div(*[
        toggle(param.name, _chip_label(param, value), value in chosen,
               value=value, title=_chip_title(param, value))
        for value in _offered(param, bench, facts)
    ], cls="chips")


def _slider(param: Param, bench: BenchSpec):
    low, high, step = bounds(param)
    value = getattr(bench, param.name)
    shown = f"{value:g}" if isinstance(value, float) else str(value)
    caption = _caption(param, float(value))
    return Div(
        Div(
            # the range is nameless: the number box is what the form submits
            Input(type="range", value=shown, min=low, max=high, step=step, cls="range",
                  aria_label=param.label, oninput="this.nextElementSibling.value=this.value"),
            Input(type="number", name=param.name, value=shown, min=low, max=high, step=step,
                  cls="numberbox", oninput="this.previousElementSibling.value=this.value"),
            cls="slider",
        ),
        Span(caption, cls="ceiling") if caption else None,
        cls="sliderbox",
    )


def _caption(param: Param, value: float) -> str:
    """The number in the unit a person judges it by, not the one the flag wants."""
    if param.group == B_LIMITS:
        return "off" if value <= 0 else duration(value)
    if param.name == "temperature":
        return "the same answer every time" if value <= 0 else ""
    return ""


def _control(param: Param, bench: BenchSpec, facts: ModelFacts | None):
    """Nothing here is typed that can be ticked or dragged instead."""
    value = getattr(bench, param.name)
    if param.kind == "multi":
        return _chips(param, bench, facts)
    if param.kind == "slider":
        return _slider(param, bench)
    if param.kind == "bool":
        return toggle(param.name, param.label, bool(value), title=param.help)
    if param.kind == "choice":
        return Select(*[Option(_choice_label(param, choice), value=choice, selected=choice == value)
                        for choice in param.choices], name=param.name)
    if param.kind in {"int", "float"}:
        return Input(type="number", name=param.name,
                     value=f"{value:g}" if param.kind == "float" else str(value),
                     min=param.minimum, max=param.maximum,
                     step="any" if param.kind == "float" else None)
    return Input(type="text", name=param.name, value=str(value))


def _choice_label(param: Param, choice: str) -> str:
    if param.name == "context_source" and choice in CONTEXT_SOURCE_HELP:
        return f"{choice} — {CONTEXT_SOURCE_HELP[choice]}"
    return choice


def _hint(param: Param, bench: BenchSpec) -> str:
    if param.name == "context_file" and bench.context_source != "file":
        return f"{param.help}. The source above is {bench.context_source}, so this is unused."
    if param.name == "run_name" and not bench.run_name:
        return f"{param.help} — and a search adds what tells its runs apart"
    return param.help


#: choices whose options carry their own explanation, and tick rows that would
#: wrap: a column of a grid is not wide enough for either, and an axis that
#: wraps stops reading as one row of values to choose between
WIDE = frozenset({"context_source", "context_file", "run_name",
                  "levels", "session_levels", "batch", "ubatch", "kv", "spec", "spec_n"})


def _field(param: Param, bench: BenchSpec, facts: ModelFacts | None = None):
    hint = _hint(param, bench)
    if param.kind == "bool":
        # the button carries the label, so a Label around it would say it twice
        return Div(_control(param, bench, facts),
                   Span(hint, cls="hint") if hint else None, cls="field switch")
    return Label(
        Span(param.label),
        _control(param, bench, facts),
        Span(hint, cls="hint") if hint else None,
        cls="field wide" if param.name in WIDE else "field",
        title=hint,
    )


# -- what the run comes to -------------------------------------------------


def duration(seconds: float) -> str:
    """A number of seconds as a person would judge it, not to the second."""
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    hours = minutes / 60
    if hours < 36:
        return f"{hours:.1f} hours"
    return f"{hours / 24:.1f} days"


def thousands(value: int) -> str:
    return f"{value / 1000:.0f}K" if value >= 10000 else f"{value:,}"


def _plan_lines(bench: BenchSpec, run: Plan) -> list[str]:
    chosen = scenarios(bench)
    lines: list[str] = []
    named = ", ".join(f"{'SL' if item.turns > 1 else 'L'}{item.key}" for item in chosen)
    ctx = server_context(bench)
    lines.append(f"{named} against one server of {context_text(ctx)} — bench2 sizes it "
                 f"from the largest of them")
    if run.configs > 1:
        varying = ", ".join(sorted(varied(bench)))
        lines.append(f"{run.configs} runs, one per combination of {varying}, each loading "
                     f"the model again")
    else:
        lines.append("One configuration — a measurement rather than a search, until a "
                     "second value is ticked on one of the rows above")
    repeats = f" × {bench.runs} repeats" if bench.runs > 1 else ""
    lines.append(f"{run.requests} request{'s' if run.requests != 1 else ''} in all{repeats}: "
                 f"about {thousands(run.prefilled)} tokens read and "
                 f"{thousands(run.decoded)} generated")
    lines.append(f"each load is given {duration(run.startup_s)} to answer before the run "
                 f"counts as failed")
    return lines


def plan_panel(bench: BenchSpec) -> Div:
    run = plan(bench)
    if not run.requests:
        return Div(H3("What this comes to"),
                   Div("Nothing ticked to measure.", cls="muted"), cls="panel")
    return Div(H3("What this comes to"),
               *[Div(line, cls="hint block") for line in _plan_lines(bench, run)],
               cls="panel")


def _weighed_label(item: Weighed, kinds: int, ubatches: int) -> str:
    parts = [context_text(item.ctx)]
    if kinds > 1:
        parts.append(f"with {item.kv}")
    if ubatches > 1:
        parts.append(f"at ubatch {item.ubatch}")
    return " ".join(parts)


def memory_panel(spec: RunSpec, bench: BenchSpec, scan: Scan, backend: str) -> Div | None:
    """Which of the configurations the cards have room for.

    Arithmetic on the model header, so it arrives before the first load rather
    than after several health timeouts.
    """
    facts = server_page.model_facts(spec)
    devices = server_page.run_devices(scan, spec, backend)
    budget, parts, measured = pool(devices)
    report = fit(spec, facts, bench, budget, devices=max(1, len(devices)))
    if report is None or not report.weighed:
        return None

    kinds = len(set(items(bench.kv)))
    ubatches = len(set(items(bench.ubatch)))
    where = f"{gib(budget)} {'free' if measured else 'installed'} ({' + '.join(parts)})"
    heaviest, over = report.heaviest, report.over
    assert heaviest is not None

    name = _weighed_label(heaviest, kinds, ubatches)
    if not over:
        verdict = Div(
            f"It fits: {name} wants {gib(heaviest.mib)} of {where}" if report.total == 1
            else f"All {report.total} fit: the heaviest, {name}, wants "
                 f"{gib(heaviest.mib)} of {where}",
            cls="problem ok")
        rest: list = []
    else:
        largest = report.largest_fitting
        lost = duration(report.over_count * max(0.0, bench.health_timeout))
        verdict = Div(
            f"⚠ It will not fit in {where}: {name} wants {gib(heaviest.mib)}."
            if report.total == 1 else
            f"⚠ {report.over_count} of {report.total} will not fit in {where}. The heaviest, "
            f"{name}, wants {gib(heaviest.mib)}.",
            cls="problem err")
        rest = [Div(f"A server too big to load is not skipped: bench2 waits out the health "
                    f"timeout, records the failure and goes on to the next run. As set "
                    f"here that is up to {lost} spent measuring nothing.", cls="problem muted")]
        if largest is not None:
            rest.append(Div(f"The most it has room for is "
                            f"{_weighed_label(largest, kinds, ubatches)}, at "
                            f"{gib(largest.mib)}.", cls="problem muted"))

    return Div(H3("Room for it"), verdict, *rest, cls="panel memory")


# -- what has already been measured ----------------------------------------

#: name, and whether the column holds a number that should line up with the one above
EARLIER_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("When", False), ("Backend", False), ("Build", False), ("Scenario", False),
    ("batch", True), ("ubatch", True), ("KV", False), ("spec", False),
    ("Prefill", True), ("Decode", True), ("Slope", True), ("", False),
)


def _number(value: float | None, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if value else "—"


def _reuse_link(result: Result, setup: Setup, spec: RunSpec, bench: BenchSpec) -> A | str:
    """Put one earlier row back on the rows below, leaving everything else alone.

    A row is one scenario of one run, so the level travels with the settings:
    the two together are what was measured. The other axis is cleared, because
    a level and a session are different workloads rather than two of a kind.
    """
    if not setup.known:
        return "—"
    session = result.kind == "session"
    return A("use", href=page_link(
        spec, bench,
        levels="" if session else result.level,
        session_levels=result.level if session else "",
        batch=str(setup.batch), ubatch=str(setup.ubatch),
        kv=setup.kv, spec=setup.spec, spec_n=str(setup.spec_n or 2)),
        title=f"measure {result.scenario} again with this run's batch, ubatch, "
              f"KV and speculation")


def _earlier_row(result: Result, setup: Setup, spec: RunSpec, bench: BenchSpec) -> Tr:
    return Tr(
        # the year is always this one, and the seconds only matter as a tie-break
        Td(result.time_text[5:16], cls="when", title=result.time_text),
        Td(result.backend),
        Td(result.commit[:7], title=f"commit {result.commit}, as bench2 recorded it"),
        Td(result.scenario, title=f"{result.turns} turns" if result.turns else
           context_text(result.ctx)),
        Td(str(setup.batch) if setup.batch else "—", cls="num"),
        Td(str(setup.ubatch) if setup.ubatch else "—", cls="num"),
        Td(setup.kv or "—"),
        Td(f"{setup.spec} ×{setup.spec_n}" if setup.spec_n and setup.spec != "none"
           else (setup.spec or "—")),
        Td(_number(result.prefill_tps, 0), cls="num"),
        Td(_number(result.decode_tps, 2), cls="num"),
        Td(_number(result.decode_slope, 3) if result.turns else "—", cls="num",
           title="how much decode speed drops per turn as the context grows"),
        Td(_reuse_link(result, setup, spec, bench)),
        title=result.run_name,
    )


def earlier_panel(results: list[Result], spec: RunSpec, bench: BenchSpec,
                  oob: bool = False) -> Div:
    """What bench2 has already measured of this model, and how to try it again.

    The numbers come from bench2's index; the settings behind each of them come
    from that run's own run.json, which is the only place they are written down.
    A row whose folder has been deleted keeps its numbers and loses its link.
    """
    if not results:
        return Div(id="earlier", hx_swap_oob="true" if oob else None)
    cache: dict[str, Setup] = {}
    rows = [_earlier_row(result, setup_of(result, cache), spec, bench)
            for result in results]
    return Div(
        Details(
            Summary(f"What {len(results)} earlier measurement"
                    f"{'s' if len(results) != 1 else ''} of this model found"),
            Span("Every row is something someone already tried. \"use\" puts one back on "
                 "the rows below -- its level and its four settings, nothing else -- so "
                 "the next attempt can differ from it by a single thing. A session's "
                 "slope is how much decode speed drops per turn.", cls="hint block"),
            Div(Table(
                Thead(Tr(*[Th(name, cls="num" if numeric else None)
                           for name, numeric in EARLIER_COLUMNS])),
                Tbody(*rows),
            ), cls="table-wrap earlier"),
            cls="panel",
        ),
        id="earlier",
        hx_swap_oob="true" if oob else None,
    )


def _busy_problems() -> list[Problem]:
    """A llama-server already up, found the way bench2 will find it.

    bench2's preflight has no policy to soften this: it refuses outright, so
    the run would end before it started.
    """
    pids = machine.running_servers()
    if not pids:
        return []
    return [Problem("error", f"llama-server is already running (pid {', '.join(pids)}). "
                             f"bench2 refuses to start while one is, because it would be "
                             f"sharing the GPUs with whatever that is measuring.")]


# -- the run being measured ------------------------------------------------


#: The run's own settings, edited in place here because they are what the
#: runs above are measured on: bench2 loads this exact server once per
#: configuration. A hidden twin of any of these would arrive twice in one
#: submission and whichever won would be whichever the browser felt like.
EDITABLE_SERVER: frozenset[str] = frozenset({
    "model", "build_dir", "devices", "rpc_endpoints", "split_mode", "tensor_split",
    "gpu_layers", "gpu_layers_all", "parallel", "flash_attn",
})


def model_field(config: AppConfig, spec: RunSpec, scan: Scan, backend: str,
                options: dict):
    """The Model select as a field of this page's form.

    Changing the model re-reads everything it decides — the level chips, the
    fit verdict, the layer count — so it posts the whole form back rather than
    the preview alone. The verdict under the select waits for the device scan
    exactly the way the Server page's does.
    """
    facts = server_page.model_facts(spec)
    hooks = {}
    if scan is not None and not scan.ready:
        hooks = {"id": "modelfield",
                 "hx_get": f"/server/modelfield?autotune=1&{server_page.spec_link(spec)}",
                 "hx_trigger": "load delay:700ms",
                 "hx_target": "#modelfield",
                 "hx_swap": "outerHTML"}
    verdict = server_page._model_verdict(spec, facts, scan, backend)
    return Label(
        Span("Model"),
        Select(*[Option(label, value=item, selected=item == spec.model)
                 for label, item in options["model"]],
               name="model",
               hx_post="/autotune/form", hx_trigger="change consume",
               hx_target="#autotuneform", hx_swap="outerHTML"),
        Span(verdict, cls="problem err" if verdict.startswith("⚠") else "hint")
        if verdict else None,
        Span(BY_NAME["model"].help, cls="hint") if BY_NAME["model"].help else None,
        cls="field",
        **hooks,
    )


def build_field(spec: RunSpec, config: AppConfig, options: dict):
    """The Build select, with the same whole-form refresh: the build decides
    the backend, which decides which devices exist to tick below."""
    chosen = server_page.build_of(config, spec)
    when = (f"built {chosen.built_on} — {chosen.built_text}" if chosen and chosen.usable
            else "newest first, so the top one is the last thing built")
    return Label(
        Span("Build"),
        Select(*[Option(label, value=item, selected=item == spec.build_dir)
                 for label, item in options["build_dir"]],
               name="build_dir",
               hx_post="/autotune/form", hx_trigger="change consume",
               hx_target="#autotuneform", hx_swap="outerHTML"),
        Span(f"supplies llama-server; capabilities are read from CMakeCache. {when}",
             cls="hint"),
        cls="field",
    )


def rpc_address_field(spec: RunSpec, options: dict, facts) -> Label:
    """The Worker addresses box as the Autotune page needs it.

    It is the same field the Server page has, plus one behaviour: a change
    re-reads the device list, so a worker pasted in here shows up as an RPC
    card without visiting the Server page first.
    """
    param = BY_NAME["rpc_endpoints"]
    hint = server_page._hint(param, spec, facts)
    return Label(
        Span(param.label),
        Input(type="text", name=param.name, value=spec.rpc_endpoints,
              placeholder="192.168.1.60:50052",
              hx_get="/server/devices",
              hx_include=".paramform",
              hx_trigger="change consume",
              hx_target="#devicefield",
              hx_swap="outerHTML"),
        Span(hint, cls="hint") if hint else None,
        cls="field",
        title=hint,
    )


def server_panel(config: AppConfig, spec: RunSpec, scan: Scan, backend: str) -> Details:
    """The server under test, edited in place rather than by a detour.

    These are the settings bench2 builds its server from: the model, the
    binary, the devices it is spread over and how. Everything else the Server
    page knows — threads, cache policy, metrics, mmproj — still reaches the
    benchmark through --server-extra, unchanged, and the link carries the spec
    so nothing is lost on the way there.
    """
    options = server_page._options(config, spec)
    facts = server_page.model_facts(spec)
    devices = server_page.run_devices(scan, spec, backend)
    return Details(
        Summary("The server being measured"),
        Span("The runs are measurements of this exact server: the model, the build and "
             "the split decide what is being timed. The device list and the workers are "
             "passed to bench2 rather than left to its hardware profile, which names "
             "cards of its own. The rest — threads, cache policy, metrics, mmproj — "
             "stays on the Server page and still reaches bench2 through the command.",
             cls="hint block"),
        Div(
            model_field(config, spec, scan, backend, options),
            build_field(spec, config, options),
            cls="grid",
        ),
        server_page.devices_field(spec, scan, backend),
        Div(
            rpc_address_field(spec, options, facts),
            server_page._field(BY_NAME["split_mode"], spec, options, facts),
            cls="grid",
        ),
        server_page.balancer_field(spec, scan, backend),
        server_page.split_line(spec, facts, devices),
        Div(
            server_page._field(BY_NAME["gpu_layers"], spec, options, facts),
            server_page._field(BY_NAME["parallel"], spec, options, facts),
            server_page._field(BY_NAME["flash_attn"], spec, options, facts),
            cls="grid",
        ),
        Div(server_page._field(BY_NAME["gpu_layers_all"], spec, options, facts),
            cls="switches"),
        Div(facts.summary, cls="hint block muted") if facts and facts.summary else None,
        A("Everything else on the Server page →",
          href=f"/server?{server_page.spec_link(spec)}", cls="button"),
        cls="panel",
        open=True,
    )


def spec_inputs(spec: RunSpec, exclude: frozenset[str] = frozenset()):
    """The whole run as hidden fields, so posting this form round-trips it.

    `exclude` names the fields this form edits visibly; a hidden twin would
    arrive twice in one submission.
    """
    inputs = []
    for field in fields(spec):
        if field.name in exclude:
            continue
        value = getattr(spec, field.name)
        if isinstance(value, bool):
            if not value:
                continue
            value = "1"
        inputs.append(Input(type="hidden", name=field.name, value=str(value)))
    return inputs


# -- panels ----------------------------------------------------------------


def problems_for(config: AppConfig, spec: RunSpec, bench: BenchSpec) -> list[Problem]:
    build = server_page.build_of(config, spec)
    problems = list(validate_bench(spec, bench, server_page.model_facts(spec),
                                   taken_names(config.bench_results)))
    problems += _busy_problems()
    if not spec.model:
        problems.insert(0, Problem("error", "No model selected on the Server page"))
    elif not Path(spec.model).is_file():
        problems.insert(0, Problem("error", f"Model file not found: {spec.model}"))
    if build is None:
        problems.insert(0, Problem("error", "No build selected on the Server page"))
    elif not build.usable:
        problems.insert(0, Problem("error", f"{build.name} has no llama-server binary"))
    if not config.bench_script.is_file():
        problems.insert(0, Problem("error", f"bench2 not found at {config.bench_script}"))
    return problems


#: bench2's own spelling for the backends the builds report
BACKEND_FLAG = {"rocm": "rocm", "hip": "rocm", "vulkan": "vk", "cpu": "cpu"}


def commands(config: AppConfig, spec: RunSpec, bench: BenchSpec,
             python: str = "python") -> list[tuple[str, list[str]]]:
    build = server_page.build_of(config, spec)
    binary = build.server_bin if build and build.server_bin else Path("llama-server")
    backend = BACKEND_FLAG.get(build.backend if build else "", "")
    return bench_commands(spec, bench, config.bench_script, binary,
                          backend=backend, python=python)


def preview(config: AppConfig, spec: RunSpec, bench: BenchSpec, scan: Scan, backend: str,
            oob: bool = False) -> Div:
    runs = commands(config, spec, bench)
    if runs:
        # one command and a whole search both fold: the command is identical for
        # every run except batch, ubatch, KV and speculation, which the Results
        # table names anyway -- the right column is for reading the size of the
        # run, not a dozen copies of the same line
        blocks = [Details(
            Summary("Show the command" if len(runs) == 1 else f"Show the {len(runs)} commands"),
            Div(*[Div(Span(name, cls="hint block"),
                      Pre(command_lines(mask_api_key(argv))))
                  for name, argv in runs]),
            cls="inline-details",
        )]
    else:
        blocks = []

    return Div(
        Div(
            H3("Command" if len(runs) <= 1 else f"{len(runs)} runs"),
            *problem_lines(problems_for(config, spec, bench)),
            *blocks,
            Span("bench2 starts and stops llama-server itself; nothing here needs a "
                 "server running first, and it refuses to run while one is. Every run "
                 "is the same command with the batch, ubatch, KV and speculation "
                 "varied; the Results panel names each one.",
                 cls="hint block"),
            cls="panel",
        ),
        plan_panel(bench),
        memory_panel(spec, bench, scan, backend),
        id="autotunepreview",
        hx_swap_oob="true" if oob else None,
    )


def _section(title: str, hint: str, bench: BenchSpec, facts: ModelFacts | None) -> Details:
    named = [param for param in BENCH_BY_NAME.values() if param.group == title]
    # switches read as a list of statements about the run; boxes in a grid do not
    switches = [param for param in named if param.kind == "bool"]
    rest = [param for param in named if param.kind != "bool"]
    return Details(
        Summary(title),
        Span(hint, cls="hint block"),
        Div(*[_field(param, bench, facts) for param in rest], cls="grid") if rest else None,
        Div(*[_field(param, bench, facts) for param in switches],
            cls="switches") if switches else None,
        cls="panel",
        open=True if title in {B_WORK, B_SWEEP} else None,
    )


def form(config: AppConfig, spec: RunSpec, bench: BenchSpec, results: list[Result],
         scan: Scan, backend: str) -> Form:
    facts = server_page.model_facts(spec)
    panels = [server_panel(config, spec, scan, backend),
              earlier_panel(results, spec, bench)]
    panels += [_section(title, hint, bench, facts) for title, hint in SECTIONS]
    count = config_count(bench)
    panels.append(Div(
        Button("Start" if count == 1 else f"Start {count} runs", type="button", cls="primary",
               hx_post="/autotune/start", hx_target="#runstate", hx_swap="outerHTML"),
        Span("Runs exactly the commands shown on the right, one after another. Each takes "
             "the GPUs for as long as it lasts; stopping one abandons the rest.",
             cls="hint"),
        cls="panel runbar",
    ))
    return Form(
        *panels,
        *spec_inputs(spec, exclude=EDITABLE_SERVER),
        Input(type="hidden", name=server_page.FORM_MARKER, value="1"),
        Input(type="hidden", name=AUTOTUNE_MARKER, value="1"),
        cls="paramform",
        id="autotuneform",
        novalidate=True,
        hx_post="/autotune/preview",
        hx_target="#autotunepreview",
        hx_swap="outerHTML",
        hx_trigger="change, keyup changed delay:400ms",
    )


def start(config: AppConfig, supervisor: Supervisor, spec: RunSpec, bench: BenchSpec,
          board: list[tuple[str, Configuration]]):
    """Validate, then hand the runs to the same GPU slot the server uses."""
    blocking = [problem.message for problem in problems_for(config, spec, bench)
                if problem.level == "error"]
    if blocking:
        return server_page.run_panel(supervisor, "; ".join(dict.fromkeys(blocking)), "error")

    # console_python(): the GUI may itself be running under pythonw.exe, which
    # would hand the benchmark a child that cannot be signalled
    runs = commands(config, spec, bench, python=console_python())
    try:
        supervisor.start_all("autotune", [(f"bench2 · {name}", argv) for name, argv in runs],
                             cwd=config.bench_script.parent.parent)
    except Busy as busy:
        return server_page.run_panel(supervisor, f"{busy.current.label} is still running", "error")
    return (server_page.run_panel(supervisor),
            server_page.log_panel(supervisor, oob=True),
            results_panel(board, {}, supervisor, oob=True))


def page(config: AppConfig, spec: RunSpec, bench: BenchSpec, supervisor: Supervisor,
         scan: Scan, backend: str, results: list[Result],
         board: list[tuple[str, Configuration]], started: str = ""):
    return shell(
        "Autotune", "/autotune", config,
        Div(
            form(config, spec, bench, results, scan, backend),
            Div(A("Autotune history →", href="/autotune/history", cls="button"),
                results_panel(board, board_results(config, board, started), supervisor),
                preview(config, spec, bench, scan, backend),
                server_page.run_panel(supervisor),
                server_page.log_panel(supervisor), cls="stack"),
            cls="split",
        ),
        nav={"/server": server_page.spec_link(spec)},
    )


def measured(config: AppConfig, spec: RunSpec) -> list[Result]:
    """Earlier bench2 measurements of the model this page is about."""
    return for_model(read_index(config.bench_results / "index.csv"), spec.model)


# -- the run as it happens -------------------------------------------------


def board_for(config: AppConfig, spec: RunSpec, bench: BenchSpec
              ) -> list[tuple[str, Configuration]]:
    """The queue handed to the supervisor: one row per run, in its order."""
    build = server_page.build_of(config, spec)
    backend = BACKEND_FLAG.get(build.backend if build else "", "")
    return list(zip(run_names(spec, bench, backend), configurations(bench)))


def board_results(config: AppConfig, board: list[tuple[str, Configuration]],
                  started: str = "") -> dict[str, Result]:
    """The latest index row bench2 has recorded for each run of the queue.

    `started` is when the queue began: a run folder name is reused by every
    search of the same parameters, so rows written before that belong to the
    previous search and would read as this one's results while bench2 is still
    measuring — the table must stay empty until bench2 itself records first.
    """
    wanted = {name for name, _config in board}
    found: dict[str, Result] = {}
    for result in read_index(config.bench_results / "index.csv"):
        if result.run_name in wanted and (not started or result.when >= started):
            # one row per finished scenario, newest last: the last one wins
            found[result.run_name] = result
    return found


#: one row per run: what told it apart, what it measured, and where the queue is
RESULTS_COLUMNS = ("Run", "batch", "ubatch", "KV", "spec", "Prefill", "Decode", "Status")


def results_panel(board: list[tuple[str, Configuration]], found: dict[str, Result],
                  supervisor: Supervisor, oob: bool = False) -> Div:
    """The queue as one table, filling in as bench2 records each run.

    bench2's index gains a row per finished scenario, so the table reads the
    index rather than scraping the log: the numbers are the ones bench2 wrote,
    not a guess from its output. A run that died before recording anything
    stays empty and says so.
    """
    if not board:
        return Div(id="results", hx_swap_oob="true" if oob else None)
    snapshot = supervisor.snapshot()
    alive = bool(snapshot and snapshot.alive)
    current = snapshot.label.removeprefix("bench2 · ") if alive else ""
    with_draft = any(config.spec != "none" for _name, config in board)
    columns = RESULTS_COLUMNS[:5] + (("draft",) if with_draft else ()) + RESULTS_COLUMNS[5:]

    rows: list = []
    reached = False
    for name, config in board:
        result = found.get(name)
        if result is not None:
            prefill, decode = _number(result.prefill_tps, 0), _number(result.decode_tps, 2)
            status = f"{result.scenario} ok" if result.ok else f"{result.scenario} failed"
            reached = reached or name == current
        elif alive and name == current:
            prefill, decode, status = "…", "…", "measuring"
            reached = True
        elif reached:
            prefill, decode, status = "—", "—", "queued"
        else:
            prefill, decode, status = "—", "—", "no result"
        cells = [Td(config.suffix, title=name, cls="runname"),
                 Td(str(config.batch), cls="num"),
                 Td(str(config.ubatch), cls="num"),
                 Td(config.kv),
                 Td(config.spec)]
        if with_draft:
            cells.append(Td(str(config.spec_n) if config.spec != "none" else "—", cls="num"))
        cells += [Td(prefill, cls="num"),
                  Td(decode, cls="num"),
                  Td(status, cls="status")]
        rows.append(Tr(*cells))

    return Div(
        H3(f"Results — {len(board)} runs"),
        Span("Fills in as bench2 records each finished scenario in its index. The "
             "numbers are that run's last recorded scenario; a finished run keeps "
             "them, a run that died before recording anything stays empty.",
             cls="hint block"),
        Div(Table(
            Thead(Tr(*[Th(name, cls="num"
                          if name in {"batch", "ubatch", "draft", "Prefill", "Decode"} else None)
                       for name in columns])),
            Tbody(*rows),
        ), cls="table-wrap results"),
        id="results",
        cls="panel",
        hx_get="/autotune/results" if alive else None,
        hx_trigger="every 2s" if alive else None,
        hx_swap="outerHTML" if alive else None,
        hx_swap_oob="true" if oob else None,
    )
