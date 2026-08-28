"""Autotune: trying server configurations and finding out which is fastest.

There is no separate "just measure it" mode, because a sweep with one value on
every axis is exactly that. Arriving from the Server page ticks one value on
each of the five, so the first thing this page offers is a measurement of the
run being described; it becomes a search the moment a second value is ticked
anywhere.

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
from urllib.parse import quote_plus, urlencode

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
    B_WHAT,
    BenchSpec,
    Fit,
    MULTI_NAMES,
    Plan,
    TASK_IDS,
    Weighed,
    fit,
    items,
    plan,
    sweep_values,
    to_bench_argv,
    validate_bench,
)
from gui2.core.devices import Scan, pool
from gui2.core.gguf import ModelFacts, context_text
from gui2.core.history import Run, past_sweeps, winning_config
from gui2.core.memory import gib
from gui2.core.params import Param, bounds
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
    (B_SWEEP, "One value ticked on a row measures that setting; two or more search it. "
              "Every extra value multiplies the run rather than adding to it — three "
              "contexts and two batch sizes are six server loads, not five — which is why "
              "there is a cap."),
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
    """The whole page as a link: the run, the sweep, no secrets.

    The five axes arrive as several boxes under one name, like the device
    list. Keeping only the last would silently narrow the sweep on reload.
    """
    return server_page.state_query(params, multi=server_page.MULTI_PARAMS | MULTI_NAMES)


def page_link(spec: RunSpec, bench: BenchSpec, **overrides: str) -> str:
    """This page as a link, with a few sweep lines rewritten.

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
    "ngram-mod": "repeats already in the prompt are guessed from it",
    "ngram-mtp": "both at once",
}


def _chip_label(param: Param, value: str) -> str:
    if param.name == "sweep_ctx" and value.isdigit():
        return context_text(int(value))
    return value


def _offered(param: Param, bench: BenchSpec, facts: ModelFacts | None) -> list[str]:
    """The row of values to tick, with whatever is already chosen kept in it.

    A context the model cannot reach is not offered, because the sweep would
    spend a whole startup timeout discovering that llama-server refuses it.
    """
    ceiling = facts.n_ctx_train if param.name == "sweep_ctx" and facts else 0
    offered = [choice for choice in param.choices
               if not (ceiling and choice.isdigit() and int(choice) > ceiling)]
    chosen = items(getattr(bench, param.name))
    return offered + [value for value in chosen if value not in offered]


def _chips(param: Param, bench: BenchSpec, facts: ModelFacts | None) -> Div:
    # the ticked look comes from the box's own state in CSS, not from a class
    # decided here: only the preview is swapped back, so a class would go stale
    chosen = set(items(getattr(bench, param.name)))
    helps = {"sweep_kv": KV_CHIP_HELP, "sweep_spec": SPEC_CHIP_HELP}.get(param.name, {})
    return Div(*[
        Label(
            Input(type="checkbox", name=param.name, value=value, checked=value in chosen),
            Span(_chip_label(param, value)),
            cls="chip",
            title=helps.get(value, ""),
        ) for value in _offered(param, bench, facts)
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
    if param.name == "max_tokens":
        return f"≈ {value * 0.75:.0f} words"
    if param.name == "real_context_chars":
        return "fill whatever context the configuration has" if value <= 0 \
            else f"≈ {value / 4 / 1024:.1f}K tokens in front of every prompt"
    return ""


def _control(param: Param, bench: BenchSpec, facts: ModelFacts | None):
    """Nothing here is typed that can be ticked or dragged instead: the five
    axes are closed lists, and every number worth setting has two ends."""
    value = getattr(bench, param.name)
    if param.kind == "multi":
        return _chips(param, bench, facts)
    if param.kind == "slider":
        return _slider(param, bench)
    if param.kind == "bool":
        return Input(type="checkbox", name=param.name, checked=bool(value), cls="switch")
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


#: choices whose options carry their own explanation, and tick rows that would
#: wrap: a column of a grid is not wide enough for either, and an axis that
#: wraps stops reading as one row of values to choose between
WIDE = frozenset({"tasks", "task_ids", "real_context_mode", "background_server_policy",
                  "sweep_ctx", "sweep_batch", "sweep_ubatch", "sweep_spec", "sweep_kv"})


def _field(param: Param, bench: BenchSpec, facts: ModelFacts | None = None):
    hint = _hint(param, bench)
    return Label(
        Span(param.label),
        _control(param, bench, facts),
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
                     "second value is ticked on one of the rows above")
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
            f"is replaced by what is ticked above.")


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
        lost = duration(report.over_count * max(0.0, bench.startup_timeout))
        verdict = Div(
            f"⚠ It will not fit in {where}: {name} wants {gib(heaviest.mib)}."
            if report.total == 1 else
            f"⚠ {report.over_count} of {report.total} will not fit in {where}. The heaviest, "
            f"{name}, wants {gib(heaviest.mib)}.",
            cls="problem err")
        rest = [Div(f"A configuration too big to load is not skipped: the server fails to "
                    f"start, the script writes CONFIG FAILED and carries on. At the startup "
                    f"timeout set here that is up to {lost} spent measuring nothing.",
                    cls="problem muted")]
        if largest is not None:
            rest.append(Div(f"The most it has room for is "
                            f"{_weighed_label(largest, kinds, ubatches)}, at "
                            f"{gib(largest.mib)}.", cls="problem muted"))

    return Div(H3("Room for it"), verdict, *rest, cls="panel memory")


#: name, and whether the column holds a number that should line up with the one above
EARLIER_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("When", False), ("Build", False), ("Context", False), ("Batch / ubatch", False),
    ("KV", False), ("Spec", False), ("t/s", True), ("Decode", True), ("", False),
)


def _number(value: float | None) -> str:
    return f"{value:.1f}" if value else "-"


def _earlier_row(spec: RunSpec, bench: BenchSpec, run: Run) -> Tr:
    chosen = winning_config(run)
    mine = bool(spec.build_dir) and run.build_name == Path(spec.build_dir).name
    return Tr(
        Td(run.time_text, cls="when"),
        Td(run.build_name, cls="mine" if mine else None,
           title="the build now selected" if mine else run.backend),
        Td(context_text(int(chosen["sweep_ctx"]))),
        Td(f"{chosen['sweep_batch']} / {chosen['sweep_ubatch']}"),
        Td(chosen["sweep_kv"]),
        Td(chosen["sweep_spec"]),
        Td(_number(run.aggregate_tps), cls="num"),
        Td(_number(run.decode_eval_tps), cls="num"),
        Td(A("use", href=f"/autotune?{page_link(spec, bench, **chosen)}", cls="button small")),
        # what the numbers were measured over, which is what makes them comparable
        title=f"{run.tasks} prompts, context from {run.real_context_mode or 'off'}, "
              f"{run.backend}",
    )


def earlier_panel(spec: RunSpec, bench: BenchSpec, runs: list[Run], oob: bool = False) -> Div:
    """What earlier sweeps of this model settled on, and a way to reuse it.

    A sweep writes one row for itself: batch, ubatch and KV type in it are the
    literal word "sweep", and the configuration it chose survives only in
    `best_config`. Reading that back is what turns a second sweep into a
    narrower one instead of the same three hours again.

    Redrawn with every edit, because each "use" link carries the rest of the
    page with it: a stale one would quietly undo whatever was changed since the
    page was opened.
    """
    earlier = past_sweeps(runs, spec.model)
    if not earlier:
        return Div(id="earlier", hx_swap_oob="true" if oob else None)
    return Div(
        Details(
            Summary(f"What {len(earlier)} earlier sweep{'s' if len(earlier) != 1 else ''} of "
                    f"this model chose"),
            Span("A sweep records only its winner, so this is what each one decided rather "
                 "than everything it tried. Using one turns the next sweep into a check of "
                 "it, or a search around it once a second value is added.", cls="hint block"),
            Div(Table(
                Thead(Tr(*[Th(name, cls="num" if numeric else None)
                           for name, numeric in EARLIER_COLUMNS])),
                Tbody(*[_earlier_row(spec, bench, run) for run in earlier]),
            ), cls="table-wrap earlier"),
            A("All of them in History →",
              href=f"/history?q={quote_plus(Path(spec.model).name)}&mode=autotune",
              cls="button small"),
            cls="panel",
        ),
        id="earlier",
        hx_swap_oob="true" if oob else None,
    )


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
        # when it was linked, because a sweep of a stale binary measures the wrong thing
        ("Build", f"{build.name} ({build.backend}) · built {build.built_text}" if build
         else "— none selected —"),
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


def _section(title: str, hint: str, bench: BenchSpec, facts: ModelFacts | None) -> Details:
    names = [param for param in BENCH_BY_NAME.values() if param.group == title]
    # switches read as a list of statements about the run; boxes in a grid do not
    switches = [param for param in names if param.kind == "bool"]
    rest = [param for param in names if param.kind != "bool"]
    return Details(
        Summary(title),
        Span(hint, cls="hint block"),
        Div(*[_field(param, bench, facts) for param in rest], cls="grid") if rest else None,
        Div(*[_field(param, bench, facts) for param in switches],
            cls="switches") if switches else None,
        cls="panel",
        open=True if title in {B_SWEEP, B_WHAT} else None,
    )


def form(config: AppConfig, spec: RunSpec, bench: BenchSpec, runs: list[Run]) -> Form:
    facts = server_page.model_facts(spec)
    panels = [inherited(config, spec), earlier_panel(spec, bench, runs)]
    panels += [_section(title, hint, bench, facts) for title, hint in SECTIONS]
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
         scan: Scan, backend: str, runs: list[Run]):
    return shell(
        "Autotune", "/autotune", config,
        Div(
            form(config, spec, bench, runs),
            Div(preview(config, spec, bench, scan, backend),
                server_page.run_panel(supervisor),
                server_page.log_panel(supervisor), cls="stack"),
            cls="split",
        ),
    )
