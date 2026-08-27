"""Bench and autotune: measuring the run the Server page describes.

The run itself is not described twice. Everything about the model, the build,
the context and the devices comes from the Server page and travels here in the
query string; this page only decides what is asked of that server and how many
times. The command is built by `gui2.core.bench.to_bench_argv`, the same
function the Server page previews.

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
    Plan,
    TASK_IDS,
    plan,
    sweep_values,
    to_bench_argv,
    validate_bench,
)
from gui2.core.gguf import context_text
from gui2.core.params import Param
from gui2.core.runspec import Problem, RunSpec, mask_api_key
from gui2.proc import Busy, Supervisor
from gui2.proc.hidden import console_python
from gui2.web import server_page
from gui2.web.layout import command_lines, problem_lines, shell

#: The bench fields need a submission marker of their own. The Server page's
#: link carries `_form` so that the *run* reads back exactly as it was sent,
#: and reusing it here would let that link switch off every bench default the
#: link never mentioned.
BENCH_MARKER = "_bench"

SECTIONS: tuple[tuple[str, str], ...] = (
    (B_WHAT, "A benchmark is a fixed set of prompts asked a fixed number of times. "
             "These four decide how much work that is, and everything else on the page "
             "is about keeping the answer honest."),
    (B_PROMPT, "A short question measures almost nothing: real work arrives with a file, "
               "a diff or a stack trace in front of it. This is what puts one there."),
    (B_FAIR, "The ways a number can flatter itself — a warm cache, a second server "
             "sharing the GPUs, reasoning counted as output."),
    (B_LIMITS, "A benchmark left alone must end by itself. These are the seconds after "
               "which it stops waiting."),
    (B_OUTPUT, "Where the result lands and how much of the server's own account of "
               "itself is kept with it."),
    (B_SWEEP, "Instead of measuring one configuration, work through a list of them and "
              "report which was fastest. Every value added multiplies the run rather "
              "than adding to it, which is why there is a cap."),
)


# -- reading the form ------------------------------------------------------


def bench_from_params(params) -> BenchSpec:
    """The bench settings from the form, or the defaults from a bare link."""
    if not params:
        return BENCH_DEFAULTS
    values = {key: params[key] for key in params.keys()}
    if BENCH_MARKER in params:
        # an unchecked box submits nothing; only a real submission may read
        # that silence as "off" (the same rule the Server page follows)
        for param in BENCH_BY_NAME.values():
            if param.kind == "bool":
                values[param.name] = param.name in params
    return BENCH_DEFAULTS.with_values(values)


def state_query(params) -> str:
    """The whole page as a link: the run, the bench settings, no secrets."""
    return server_page.state_query(params)


# -- fields ----------------------------------------------------------------


def _control(param: Param, bench: BenchSpec):
    """The bench flags are plainer than llama-server's: no sliders, no devices,
    no value whose limits another file decides. A separate small renderer costs
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
WIDE = frozenset({"tasks", "task_ids", "real_context_mode", "background_server_policy"})


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
    if run.configs > 1:
        axes = " × ".join(str(len(values)) for values in sweep_values(bench).values())
        lines.append(f"{run.configs} server configurations ({axes}), each loaded from scratch")
    prompts = f"{run.tasks} prompt{'s' if run.tasks != 1 else ''}"
    each = " against each of them" if run.configs > 1 else ""
    repeats = f" × {run.runs} repeats" if run.runs > 1 else ""
    priming = " plus one unmeasured priming pass" if run.prime else ""
    lines.append(f"{prompts}{each}{repeats}{priming} — {run.requests} requests in all")
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
        Div(_under_test(spec, bench), cls="hint block muted"),
        cls="panel",
    )


def _under_test(spec: RunSpec, bench: BenchSpec) -> str:
    """The settings the numbers will belong to.

    A sweep overrides the ones the Server page set, so naming those would
    describe a run that is not the one about to happen.
    """
    if not bench.autotune:
        return (f"Context under test: {context_text(spec.ctx_size)}, "
                f"batch {spec.batch_size}/{spec.ubatch_size}")
    swept = sweep_values(bench)
    contexts = ", ".join(context_text(int(value)) if value.isdigit() else value
                         for value in swept["sweep_ctx"]) or "none"
    return (f"Contexts under test: {contexts} — the sweep replaces the context and "
            f"batch sizes chosen on the Server page")


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


# -- the run the bench measures --------------------------------------------


def inherited(config: AppConfig, spec: RunSpec) -> Details:
    """The server settings this page did not choose, and where to change them."""
    build = server_page.build_of(config, spec)
    facts = server_page.model_facts(spec)
    rows = [
        ("Model", Path(spec.model).name if spec.model else "— none selected —"),
        ("Build", f"{build.name} ({build.backend})" if build else "— none selected —"),
        ("Context", context_text(spec.ctx_size)),
        ("Batch / ubatch", f"{spec.batch_size} / {spec.ubatch_size}"),
        ("KV cache", f"{spec.cache_type_k} / {spec.cache_type_v}"),
        ("Devices", spec.devices or "all of them"),
        ("Speculation", spec.spec_type),
    ]
    return Details(
        Summary("The run being measured"),
        Span("Everything here belongs to the server, not to the benchmark, so it is "
             "changed in one place and read in both.", cls="hint block"),
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


def preview(config: AppConfig, spec: RunSpec, bench: BenchSpec, oob: bool = False) -> Div:
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
            Span("The benchmark script starts and stops llama-server itself; nothing here "
                 "needs a server running first.", cls="hint block"),
            cls="panel",
        ),
        plan_panel(spec, bench),
        id="benchpreview",
        hx_swap_oob="true" if oob else None,
    )


def _section(title: str, hint: str, bench: BenchSpec) -> Details:
    names = [param for param in BENCH_BY_NAME.values() if param.group == title]
    return Details(
        Summary(title),
        Span(hint, cls="hint block"),
        Div(*[_field(param, bench) for param in names], cls="grid"),
        cls="panel",
        open=True if title in {B_WHAT, B_PROMPT} else None,
    )


def form(config: AppConfig, spec: RunSpec, bench: BenchSpec) -> Form:
    panels = [inherited(config, spec)]
    panels += [_section(title, hint, bench) for title, hint in SECTIONS]
    panels.append(Div(
        Button("Start benchmark", type="button", cls="primary",
               hx_post="/bench/start", hx_target="#runstate", hx_swap="outerHTML"),
        Span("Runs exactly the command shown on the right. It loads the model, so it "
             "takes the GPUs for as long as it lasts.", cls="hint"),
        cls="panel runbar",
    ))
    return Form(
        *panels,
        *spec_inputs(spec),
        Input(type="hidden", name=server_page.FORM_MARKER, value="1"),
        Input(type="hidden", name=BENCH_MARKER, value="1"),
        cls="paramform",
        novalidate=True,
        hx_post="/bench/preview",
        hx_target="#benchpreview",
        hx_swap="outerHTML",
        hx_trigger="change, keyup changed delay:400ms",
    )


def start(config: AppConfig, supervisor: Supervisor, spec: RunSpec, bench: BenchSpec):
    """Validate, then hand the benchmark to the same GPU slot the server uses."""
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
    label = f"benchmark · {bench.tasks} · {Path(spec.model).name}"
    # console_python(): the GUI may itself be running under pythonw.exe, which
    # would hand the benchmark a child that cannot be signalled
    argv = to_bench_argv(spec, bench, config.bench_script, build.server_bin,
                         python=console_python())
    try:
        supervisor.start("bench", label, argv, cwd=config.data_root)
    except Busy as busy:
        return server_page.run_panel(supervisor, f"{busy.current.label} is still running", "error")
    return server_page.run_panel(supervisor), server_page.log_panel(supervisor, oob=True)


def page(config: AppConfig, spec: RunSpec, bench: BenchSpec, supervisor: Supervisor):
    return shell(
        "Bench", "/bench", config,
        Div(
            form(config, spec, bench),
            Div(preview(config, spec, bench),
                server_page.run_panel(supervisor),
                server_page.log_panel(supervisor), cls="stack"),
            cls="split",
        ),
    )
