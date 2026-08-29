"""Autotune: measuring a server configuration, and searching for a better one.

bench2 loads the model once per run and measures every level and session asked
of it against that one server, so the two questions this page asks are of
different kinds. *What to measure* is free to grow -- another level costs the
prompt it sends, not another model load. *What to try* is not: batch, ubatch,
KV type and speculation are fixed for a whole run, so a second value on any of
them is a second run, and the page queues them rather than pretending
otherwise.

The run itself is not described twice. The model, the build, the devices, the
layer split and the RPC workers come from the Server page and travel here in
the query string; this page chooses the workload and the settings under test.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from urllib.parse import urlencode

from fasthtml.common import (
    A,
    Button,
    Dd,
    Details,
    Div,
    Dl,
    Dt,
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
from gui2.core.params import Param, bounds
from gui2.core.results import Result, for_model, read_index, taken_names
from gui2.core.runspec import Problem, RunSpec, mask_api_key, parse_rpc_endpoints
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


def page_link(spec: RunSpec, bench: BenchSpec, **overrides: str) -> str:
    """This page as a link, with a few lines rewritten.

    The mirror of `server_page.spec_link`: written the way the form writes it,
    marker and all, so reading it back gives what it says.
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
    return server_page.spec_link(spec) + "&" + urlencode(pairs)


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
                  "levels", "session_levels", "batch", "ubatch", "kv", "spec"})


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
    ("When", False), ("Run", False), ("Backend", False), ("Build", False),
    ("Scenario", False), ("Prefill", True), ("Decode", True), ("Slope", True),
)


def _number(value: float | None, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if value else "—"


def _earlier_row(result: Result) -> Tr:
    return Tr(
        Td(result.time_text, cls="when"),
        Td(result.run_name),
        Td(result.backend),
        Td(result.commit, title="the commit bench2 recorded for that build"),
        Td(result.scenario, title=f"{result.turns} turns" if result.turns else
           context_text(result.ctx)),
        Td(_number(result.prefill_tps, 0), cls="num"),
        Td(_number(result.decode_tps, 2), cls="num"),
        Td(_number(result.decode_slope, 3) if result.turns else "—", cls="num",
           title="how much decode speed drops per turn as the context grows"),
    )


def earlier_panel(results: list[Result], oob: bool = False) -> Div:
    """What bench2 has already measured of this model.

    Not a list of settings to reuse -- bench2 records the numbers, not the
    configuration that produced them, which lives in that run's own run.json.
    It is here to answer "has this been measured already" before it is measured
    again.
    """
    if not results:
        return Div(id="earlier", hx_swap_oob="true" if oob else None)
    return Div(
        Details(
            Summary(f"What {len(results)} earlier measurement"
                    f"{'s' if len(results) != 1 else ''} of this model found"),
            Span("From bench2's own index. A session's slope is how much decode speed "
                 "drops per turn as the conversation grows.", cls="hint block"),
            Div(Table(
                Thead(Tr(*[Th(name, cls="num" if numeric else None)
                           for name, numeric in EARLIER_COLUMNS])),
                Tbody(*[_earlier_row(result) for result in results]),
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


def inherited(config: AppConfig, spec: RunSpec) -> Details:
    """The server settings this page does not choose, and where to change them.

    Context is deliberately absent: bench2 sizes the server from the levels it
    is given, so the Server page's context is not what any of this measures.
    """
    build = server_page.build_of(config, spec)
    facts = server_page.model_facts(spec)
    workers = parse_rpc_endpoints(spec.rpc_endpoints)
    rows = [
        ("Model", Path(spec.model).name if spec.model else "— none selected —"),
        # when it was linked, because a benchmark of a stale binary measures the wrong thing
        ("Build", f"{build.name} ({build.backend}) · built {build.built_text}" if build
         else "— none selected —"),
        ("Devices", spec.devices or "all of them"),
        ("Workers", ", ".join(workers) if workers else "none — this machine only"),
        ("Layer split", f"-sm {spec.split_mode or 'default'}"
                        + (f" -ts {spec.tensor_split}" if spec.tensor_split else "")),
        ("GPU layers", "all of them" if spec.gpu_layers_all else str(spec.gpu_layers)),
        ("Parallel slots", str(spec.parallel)),
        ("Flash attention", "auto — sent as on" if spec.flash_attn == "auto"
         else spec.flash_attn),
    ]
    return Details(
        Summary("The server being measured"),
        Span("Everything here belongs to the server, so it is changed in one place and "
             "read in both. The device list and the workers are passed to bench2 rather "
             "than left to its hardware profile, which names cards of its own.",
             cls="hint block"),
        Div(Dl(*[item for name, value in rows for item in (Dt(name), Dd(str(value)))]),
            cls="detail"),
        Div(facts.summary, cls="hint block muted") if facts and facts.summary else None,
        A("Change these on the Server page →", href=f"/server?{server_page.spec_link(spec)}",
          cls="button"),
        cls="panel",
        open=True,
    )


def spec_inputs(spec: RunSpec):
    """The whole run as hidden fields, so posting this form round-trips it."""
    inputs = []
    for field in fields(spec):
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
    varying = varied(bench)
    blocks: list = []
    for (name, argv), config_row in zip(runs, configurations(bench)):
        blocks.append(Div(Span(f"{name} — {config_row.describe(varying)}", cls="hint block")
                          if len(runs) > 1 else None,
                          Pre(command_lines(mask_api_key(argv)))))

    return Div(
        Div(
            H3("Command" if len(runs) == 1 else f"{len(runs)} commands, in this order"),
            *problem_lines(problems_for(config, spec, bench)),
            *blocks,
            Span("bench2 starts and stops llama-server itself; nothing here needs a "
                 "server running first, and it refuses to run while one is.",
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


def form(config: AppConfig, spec: RunSpec, bench: BenchSpec, results: list[Result]) -> Form:
    facts = server_page.model_facts(spec)
    panels = [inherited(config, spec), earlier_panel(results)]
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
        *spec_inputs(spec),
        Input(type="hidden", name=server_page.FORM_MARKER, value="1"),
        Input(type="hidden", name=AUTOTUNE_MARKER, value="1"),
        cls="paramform",
        novalidate=True,
        hx_post="/autotune/preview",
        hx_target="#autotunepreview",
        hx_swap="outerHTML",
        hx_trigger="change, keyup changed delay:400ms",
    )


def start(config: AppConfig, supervisor: Supervisor, spec: RunSpec, bench: BenchSpec):
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
    return server_page.run_panel(supervisor), server_page.log_panel(supervisor, oob=True)


def page(config: AppConfig, spec: RunSpec, bench: BenchSpec, supervisor: Supervisor,
         scan: Scan, backend: str, results: list[Result]):
    return shell(
        "Autotune", "/autotune", config,
        Div(
            form(config, spec, bench, results),
            Div(preview(config, spec, bench, scan, backend),
                server_page.run_panel(supervisor),
                server_page.log_panel(supervisor), cls="stack"),
            cls="split",
        ),
        nav={"/server": server_page.spec_link(spec)},
    )


def measured(config: AppConfig, spec: RunSpec) -> list[Result]:
    """Earlier bench2 measurements of the model this page is about."""
    return for_model(read_index(config.bench_results / "index.csv"), spec.model)
