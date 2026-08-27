"""Autotune: trying server configurations and finding out which is fastest.

There is no separate "just measure it" mode, because a sweep with one value on
every line is exactly that. Arriving from the Server page fills all five lines
with what that page chose, so the first thing this page offers is a measurement
of the run being described; it becomes a search the moment a second value is
typed anywhere.

The run itself is not described twice. The model, the build, the devices and
the layer split come from the Server page and travel here in the query string;
this page decides what is asked of the server, how many times, and which
configurations to try. The command is built by `gui2.core.bench.to_bench_argv`,
the same function the Server page previews.

What the page adds is arithmetic nobody does by hand: how many requests a
choice of prompt set, repeats and sweep axes works out to, and how long the
run's own timeouts would let that take.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

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
    B_WHAT,
    BenchSpec,
    Fit,
    Plan,
    TASK_IDS,
    Weighed,
    fit,
    plan,
    sweep_values,
    to_bench_argv,
    validate_bench,
)
from gui2.core.devices import Scan, pool
from gui2.core.gguf import context_text
from gui2.core.memory import gib
from gui2.core.params import Param
from gui2.core.runspec import Problem, RunSpec, mask_api_key
from gui2.proc import Busy, Supervisor
from gui2.proc.hidden import console_python
from gui2.web import server_page
from gui2.web.layout import command_lines, problem_lines, shell

#: These fields need a submission marker of their own. The Server page's link
#: carries `_form` so that the *run* reads back exactly as it was sent, and
#: reusing it here would let that link switch off every default the link never
#: mentioned.
AUTOTUNE_MARKER = "_autotune"

SECTIONS: tuple[tuple[str, str], ...] = (
    (B_SWEEP, "One value on a line measures that setting; two or more search it. Every "
              "extra value multiplies the run rather than adding to it — three contexts "
              "and two batch sizes are six server loads, not five — which is why there "
              "is a cap."),
    (B_WHAT, "Each configuration is judged by asking it the same prompts the same number "
             "of times. These decide how much work that is, and everything below is "
             "about keeping the comparison honest."),
    (B_PROMPT, "A short question measures almost nothing: real work arrives with a file, "
               "a diff or a stack trace in front of it. This is what puts one there."),
    (B_FAIR, "The ways a number can flatter itself — a warm cache, a second server "
             "sharing the GPUs, reasoning counted as output."),
    (B_LIMITS, "A sweep left alone must end by itself. These are the seconds after "
               "which it stops waiting."),
    (B_OUTPUT, "Where the result lands and how much of the server's own account of "
               "itself is kept with it."),
)


# -- reading the form ------------------------------------------------------


def autotune_from_params(params, spec: RunSpec) -> BenchSpec:
    """The autotune settings from the form, or a sweep seeded from the run.

    Without the marker this is a link rather than a submission — from the
    Server page, or the address bar — so the sweep starts as one configuration:
    the one that link describes.
    """
    if not params or AUTOTUNE_MARKER not in params:
        seeded = BENCH_DEFAULTS.seeded_from(spec)
        return seeded.with_values({key: params[key] for key in params.keys()}) \
            if params else seeded
    values = {key: params[key] for key in params.keys()}
    # an unchecked box submits nothing; only a real submission may read that
    # silence as "off" (the same rule the Server page follows)
    for param in BENCH_BY_NAME.values():
        if param.kind == "bool":
            values[param.name] = param.name in params
    return BENCH_DEFAULTS.with_values(values)


def state_query(params) -> str:
    """The whole page as a link: the run, the sweep, no secrets."""
    return server_page.state_query(params)


# -- fields ----------------------------------------------------------------


def _control(param: Param, bench: BenchSpec):
    """These flags are plainer than llama-server's: no sliders, no devices, no
    value whose limits another file decides. A separate small renderer costs
    less than teaching the Server page's one about a second dataclass."""
    value = getattr(bench, param.name)
    if param.kind == "bool":
        return Input(type="checkbox", name=param.name, checked=bool(value))
    if param.kind == "choice":
        return Select(*[Option(_choice_label(param, choice), value=choice, selected=choice == value)
                        for choice in param.choices], name=param.name)
    if param.kind in {"int", "float"}:
        return Input(type="number", name=param.name, value=f"{value:g}"
                     if param.kind == "float" else str(value),
                     min=param.minimum, max=param.maximum,
                     step="any" if param.kind == "float" else None)
    return Input(type="text", name=param.name, value=str(value))


TASK_SET_HELP = {
    "quick": "two short prompts — for checking that a build runs at all",
    "full": "the two quick prompts and two longer ones",
    "v2": "five prompts written like real agent work: review, write, debug, plan, analyse",
    "v2-mini": "one v2 prompt, writing a function",
    "v2-review": "one v2 prompt, reviewing code",
}

POLICY_HELP = {
    "fail": "stop before starting — the safe answer, and the reason a stale server "
            "is found now rather than in the numbers",
    "warn": "say so and measure anyway",
    "ignore": "say nothing",
}

CONTEXT_MODE_HELP = {
    "off": "just the prompt, a few hundred tokens",
    "repo-snapshot": "the prompt behind a slab of this repository's own source",
}


def _choice_label(param: Param, choice: str) -> str:
    help_text = {"tasks": TASK_SET_HELP, "background_server_policy": POLICY_HELP,
                 "real_context_mode": CONTEXT_MODE_HELP}.get(param.name, {}).get(choice)
    if help_text:
        return f"{choice} — {help_text}"
    return choice


def _hint(param: Param, bench: BenchSpec) -> str:
    if param.name == "task_ids":
        available = TASK_IDS.get(bench.tasks)
        if available:
            return f"{param.help}. In {bench.tasks}: {', '.join(available)}"
    if param.name == "label" and not bench.label:
        return f"{param.help} — it will be called rocm-baseline-<date>"
    return param.help


#: choices whose options carry their own explanation, and text whose hint
#: lists what may go in it: a column of a grid is not wide enough for either
WIDE = frozenset({"tasks", "task_ids", "real_context_mode", "background_server_policy",
                  "sweep_spec", "sweep_kv"})


def _field(param: Param, bench: BenchSpec):
    hint = _hint(param, bench)
    return Label(
        Span(param.label),
        _control(param, bench),
        Span(hint, cls="hint") if hint else None,
        cls="field inline" if param.kind == "bool"
        else ("field wide" if param.name in WIDE else "field"),
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


def _plan_lines(bench: BenchSpec, run: Plan) -> list[str]:
    lines: list[str] = []
    axes = " × ".join(str(len(values)) for values in sweep_values(bench).values())
    if run.configs > 1:
        lines.append(f"{run.configs} server configurations ({axes}), each loaded from scratch")
    else:
        lines.append("One configuration — a measurement rather than a search, until a "
                     "second value is added to one of the lines above")
    prompts = f"{run.tasks} prompt{'s' if run.tasks != 1 else ''}"
    each = " against each of them" if run.configs > 1 else ""
    repeats = f" × {run.runs} repeats" if run.runs > 1 else ""
    lines.append(f"{prompts}{each}{repeats} — {run.requests} requests in all")
    if run.primed:
        lines.append(f"{run.primed} of them first get an unmeasured pass to fill the "
                     f"n-gram cache, counted above")
    if run.per_request_s:
        lines.append(f"each answer is abandoned after {duration(run.per_request_s)}, "
                     f"and the server is given {duration(run.startup_s)} to load")
    lines.append(f"so the run cannot outlast {duration(run.worst_case_s)}, "
                 f"however badly it goes")
    return lines


def plan_panel(spec: RunSpec, bench: BenchSpec) -> Div:
    run = plan(spec, bench)
    if not run.tasks:
        return Div(H3("What this comes to"),
                   Div("No prompts selected, so there is nothing to measure.", cls="muted"),
                   cls="panel")
    return Div(
        H3("What this comes to"),
        *[Div(line, cls="hint block") for line in _plan_lines(bench, run)],
        Div(_under_test(bench), cls="hint block muted"),
        cls="panel",
    )


def _under_test(bench: BenchSpec) -> str:
    """The settings the numbers will belong to.

    The sweep sets these itself for every configuration it runs, so whatever
    the Server page chose for them is not what will be measured.
    """
    swept = sweep_values(bench)
    contexts = ", ".join(context_text(int(value)) if value.isdigit() else value
                         for value in swept["sweep_ctx"]) or "none"
    return (f"Contexts under test: {contexts}. The sweep sets the context, batch, ubatch, "
            f"KV type and speculation itself, so the Server page's choice of those five "
            f"is replaced by the lines above.")


def _weighed_label(item: Weighed, kinds: int, ubatches: int) -> str:
    """Name a configuration by whatever tells it apart from the others."""
    parts = [context_text(item.ctx)]
    if kinds > 1:
        parts.append(f"with {item.kv}")
    if ubatches > 1:
        parts.append(f"at ubatch {item.ubatch}")
    return " ".join(parts)


def memory_panel(spec: RunSpec, bench: BenchSpec, scan: Scan, backend: str) -> Div | None:
    """Which of the swept configurations the cards have room for.

    Arithmetic on the model header, so it arrives before the sweep rather than
    after a quarter of an hour of the server failing to load.
    """
    facts = server_page.model_facts(spec)
    devices = server_page.run_devices(scan, spec, backend)
    budget, parts, measured = pool(devices)
    report = fit(spec, facts, bench, budget, devices=max(1, len(devices)))
    if report is None or not report.weighed:
        return None

    swept = sweep_values(bench)
    kinds, ubatches = len(set(swept["sweep_kv"])), len(set(swept["sweep_ubatch"]))
    where = f"{gib(budget)} {'free' if measured else 'installed'} ({' + '.join(parts)})"
    heaviest, over = report.heaviest, report.over
    assert heaviest is not None

    if not over:
        verdict = Div(f"All {report.total} fit: the heaviest, "
                      f"{_weighed_label(heaviest, kinds, ubatches)}, wants "
                      f"{gib(heaviest.mib)} of {where}", cls="problem ok")
        rest: list = []
    else:
        largest = report.largest_fitting
        lost = duration(report.over_count * max(0.0, bench.startup_timeout))
        verdict = Div(f"⚠ {report.over_count} of {report.total} will not fit in {where}. "
                      f"The heaviest, {_weighed_label(heaviest, kinds, ubatches)}, wants "
                      f"{gib(heaviest.mib)}.", cls="problem err")
        rest = [Div(f"A configuration too big to load is not skipped: the server fails, "
                    f"the script writes CONFIG FAILED and moves to the next one. At the "
                    f"startup timeout set here that is up to {lost} spent measuring nothing.",
                    cls="problem muted")]
        if largest is not None:
            rest.append(Div(f"The most it has room for is "
                            f"{_weighed_label(largest, kinds, ubatches)}, at "
                            f"{gib(largest.mib)}.", cls="problem muted"))

    return Div(H3("Room for it"), verdict, *rest, cls="panel memory")


def _busy_problems(bench: BenchSpec) -> list[Problem]:
    """A llama-server already up, found the way the script will find it."""
    pids = machine.running_servers()
    if not pids:
        return []
    who = f"llama-server is already running (pid {', '.join(pids)})"
    if bench.background_server_policy == "fail":
        return [Problem("error", f"{who}. This run is set to stop rather than share the "
                                 f"GPUs with it, so it will not start.")]
    return [Problem("warn", f"{who}, and it will be using the same GPUs. Whatever is "
                            f"measured here includes its load.")]


# -- the run being swept ---------------------------------------------------


def inherited(config: AppConfig, spec: RunSpec) -> Details:
    """The server settings this page does not choose, and where to change them.

    Context, batch, ubatch, KV type and speculation are deliberately absent:
    the sweep replaces all five per configuration, so showing the Server page's
    values here would name settings no measurement uses.
    """
    build = server_page.build_of(config, spec)
    facts = server_page.model_facts(spec)
    rows = [
        ("Model", Path(spec.model).name if spec.model else "— none selected —"),
        ("Build", f"{build.name} ({build.backend})" if build else "— none selected —"),
        ("Devices", spec.devices or "all of them"),
        ("GPU layers", "all of them" if spec.gpu_layers_all else str(spec.gpu_layers)),
        ("Parallel slots", str(spec.parallel)),
        # the script has no 'auto': it passes on or off, and its own default is on
        ("Flash attention", "auto — sent as on" if spec.flash_attn == "auto"
         else spec.flash_attn),
    ]
    return Details(
        Summary("The server being tuned"),
        Span("Everything here belongs to the server, not to the sweep, so it is changed "
             "in one place and read in both. What the sweep varies is below.",
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
    """The whole run as hidden fields, so posting this form round-trips it.

    A false checkbox is left out rather than sent as "0": that is what a form
    does, and `_form` is what tells the reader the silence was deliberate.
    """
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


def preview(config: AppConfig, spec: RunSpec, bench: BenchSpec, scan: Scan, backend: str,
            oob: bool = False) -> Div:
    build = server_page.build_of(config, spec)
    binary = build.server_bin if build and build.server_bin else Path("llama-server")
    argv = to_bench_argv(spec, bench, config.bench_script, binary)

    problems = list(validate_bench(spec, bench)) + _busy_problems(bench)
    if not spec.model:
        problems.insert(0, Problem("error", "No model selected on the Server page"))
    elif not Path(spec.model).is_file():
        problems.insert(0, Problem("error", f"Model file not found: {spec.model}"))
    if build is None:
        problems.insert(0, Problem("error", "No build selected on the Server page"))
    elif not build.usable:
        problems.insert(0, Problem("error", f"{build.name} has no llama-server binary"))

    return Div(
        Div(
            H3("Command"),
            *problem_lines(problems),
            Pre(command_lines(mask_api_key(argv))),
            Span("The script starts and stops llama-server itself, once per configuration; "
                 "nothing here needs a server running first.", cls="hint block"),
            cls="panel",
        ),
        plan_panel(spec, bench),
        memory_panel(spec, bench, scan, backend),
        id="autotunepreview",
        hx_swap_oob="true" if oob else None,
    )


def _section(title: str, hint: str, bench: BenchSpec) -> Details:
    names = [param for param in BENCH_BY_NAME.values() if param.group == title]
    return Details(
        Summary(title),
        Span(hint, cls="hint block"),
        Div(*[_field(param, bench) for param in names], cls="grid"),
        cls="panel",
        open=True if title in {B_SWEEP, B_WHAT} else None,
    )


def form(config: AppConfig, spec: RunSpec, bench: BenchSpec) -> Form:
    panels = [inherited(config, spec)]
    panels += [_section(title, hint, bench) for title, hint in SECTIONS]
    panels.append(Div(
        Button("Start autotune", type="button", cls="primary",
               hx_post="/autotune/start", hx_target="#runstate", hx_swap="outerHTML"),
        Span("Runs exactly the command shown on the right. It loads the model once per "
             "configuration, so it takes the GPUs for as long as it lasts.", cls="hint"),
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
    """Validate, then hand the sweep to the same GPU slot the server uses."""
    build = server_page.build_of(config, spec)
    blocking = [problem.message for problem
                in validate_bench(spec, bench) + _busy_problems(bench)
                if problem.level == "error"]
    if not spec.model or not Path(spec.model).is_file():
        blocking.append("Select a model on the Server page")
    if build is None or not build.usable:
        blocking.append("Select a build with a llama-server binary on the Server page")
    if blocking:
        return server_page.run_panel(supervisor, "; ".join(dict.fromkeys(blocking)), "error")

    assert build is not None and build.server_bin is not None
    label = f"autotune · {bench.tasks} · {Path(spec.model).name}"
    # console_python(): the GUI may itself be running under pythonw.exe, which
    # would hand the sweep a child that cannot be signalled
    argv = to_bench_argv(spec, bench, config.bench_script, build.server_bin,
                         python=console_python())
    try:
        supervisor.start("autotune", label, argv, cwd=config.data_root)
    except Busy as busy:
        return server_page.run_panel(supervisor, f"{busy.current.label} is still running", "error")
    return server_page.run_panel(supervisor), server_page.log_panel(supervisor, oob=True)


def page(config: AppConfig, spec: RunSpec, bench: BenchSpec, supervisor: Supervisor,
         scan: Scan, backend: str):
    return shell(
        "Autotune", "/autotune", config,
        Div(
            form(config, spec, bench),
            Div(preview(config, spec, bench, scan, backend),
                server_page.run_panel(supervisor),
                server_page.log_panel(supervisor), cls="stack"),
            cls="split",
        ),
    )
