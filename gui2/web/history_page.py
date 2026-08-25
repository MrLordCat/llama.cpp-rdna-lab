"""History & Analytics: groups -> configs -> runs.

The drill-down exists because a single TPS axis over the whole CSV compares
runs that are not comparable. A group pins backend, model, context and
speculative mode; a config is one batch/ubatch/KV combination inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

from fasthtml.common import (
    A,
    Dd,
    Div,
    Dl,
    Dt,
    Form,
    H2,
    I,
    Input,
    Label,
    Option,
    Select,
    Span,
    Table,
    Tbody,
    Td,
    Th,
    Thead,
    Tr,
)

from gui2.config import AppConfig
from gui2.core.history import (
    GroupStat,
    LaneStat,
    Run,
    RunFilter,
    apply_filter,
    facets,
    group_stats,
    lane_stats,
    sort_runs,
    summarize,
)
from gui2.web.chart import backend_color, config_chart, progress_chart
from gui2.web.layout import number, shell

LIMITS = (50, 100, 250, 1000)
MAX_GROUPS = 60

RUN_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("time", "Time", False),
    ("backend", "Backend", False),
    ("mode", "Mode", False),
    ("", "Config", False),
    ("tps", "TPS", True),
    ("prompt", "Prompt t/s", True),
    ("decode", "Decode t/s", True),
    ("", "Err", True),
    ("label", "Label", False),
)


@dataclass(frozen=True, slots=True)
class ViewState:
    run_filter: RunFilter
    sort: str
    descending: bool
    limit: int

    def as_params(self, **overrides) -> dict[str, str]:
        params = {
            "family": self.run_filter.family,
            "lane": self.run_filter.lane,
            "backend": self.run_filter.backend,
            "model": self.run_filter.model,
            "mode": self.run_filter.mode,
            "spec": self.run_filter.spec,
            "q": self.run_filter.query,
            "min_tps": "" if self.run_filter.min_tps is None else f"{self.run_filter.min_tps:g}",
            "hide_errors": "on" if self.run_filter.hide_errors else "",
            "best_only": "on" if self.run_filter.best_only else "",
            "sort": self.sort,
            "desc": "1" if self.descending else "0",
            "limit": str(self.limit),
        }
        params.update({key: str(value) for key, value in overrides.items()})
        return {key: value for key, value in params.items() if value}

    def query(self, **overrides) -> str:
        return urlencode(self.as_params(**overrides))


def read_state(params) -> ViewState:
    def text(key: str) -> str:
        return str(params.get(key, "") or "").strip()

    try:
        min_tps = float(text("min_tps")) if text("min_tps") else None
    except ValueError:
        min_tps = None

    try:
        limit = int(text("limit") or 100)
    except ValueError:
        limit = 100

    return ViewState(
        run_filter=RunFilter(
            backend=text("backend"),
            model=text("model"),
            mode=text("mode"),
            spec=text("spec"),
            query=text("q"),
            min_tps=min_tps,
            hide_errors=bool(text("hide_errors")),
            best_only=bool(text("best_only")),
            family=text("family"),
            lane=text("lane"),
        ),
        sort=text("sort") or "time",
        descending=text("desc") != "0",
        limit=limit if limit in LIMITS else 100,
    )


def family_label(family: str) -> str:
    return family.replace("|", " · ") or "-"


def _drill(state: ViewState, **overrides) -> dict[str, str]:
    return {"hx_get": "/history/rows?" + state.query(**overrides), "hx_target": "#results"}


def _select(name: str, label: str, options: tuple[str, ...], selected: str):
    return Label(
        label,
        Select(
            Option("all", value="", selected=not selected),
            *[Option(item, value=item, selected=item == selected) for item in options],
            name=name,
        ),
    )


def _filters(state: ViewState, available) -> Form:
    return Form(
        _select("backend", "Backend", available.backends, state.run_filter.backend),
        _select("model", "Model", available.models, state.run_filter.model),
        _select("mode", "Mode", available.modes, state.run_filter.mode),
        _select("spec", "Speculative", available.specs, state.run_filter.spec),
        Label(
            "Search label / lane / args",
            Input(type="search", name="q", value=state.run_filter.query, placeholder="e.g. vulkan130k"),
        ),
        Label(
            "Min TPS",
            Input(type="number", name="min_tps", step="0.1", min="0",
                  value="" if state.run_filter.min_tps is None else f"{state.run_filter.min_tps:g}"),
        ),
        Label(
            "Rows",
            Select(*[Option(str(item), value=str(item), selected=item == state.limit) for item in LIMITS],
                   name="limit"),
        ),
        Label(
            Input(type="checkbox", name="hide_errors", checked=state.run_filter.hide_errors),
            "hide failed runs", cls="inline",
        ),
        Label(
            Input(type="checkbox", name="best_only", checked=state.run_filter.best_only),
            "lane bests only", cls="inline",
        ),
        A("reset", href="/history", cls="muted"),
        cls="filters panel",
        hx_get="/history/rows",
        hx_target="#results",
        hx_trigger="change, search, keyup changed delay:350ms from:input[type=search]",
        # drill-down and sort state are re-rendered with the results, so they survive a filter change
        hx_include="#view-state input",
    )


def _summary(runs: list[Run]) -> Div:
    stats = summarize(runs)
    best_label = "-"
    if stats.best_run is not None:
        best_label = f"{stats.best_run.backend} · {stats.best_run.label or stats.best_run.run_id}"
    cells = (
        ("Runs", str(stats.count), False),
        ("Failed", str(stats.with_errors), False),
        ("Best TPS", number(stats.best_tps), False),
        ("Median TPS", number(stats.median_tps), False),
        ("Best run", best_label, True),
        ("Range", f"{stats.first_time[:10]} → {stats.last_time[:10]}", True),
    )
    return Div(
        *[Div(Span(key, cls="k"), Span(value, cls="v small" if small else "v"), cls="stat")
          for key, value, small in cells],
        cls="stats",
    )


def _breadcrumb(state: ViewState, runs: list[Run]) -> Div:
    parts: list = [A("All groups", **_drill(state, family="", lane=""), cls="crumb")]
    family = state.run_filter.family
    if family:
        parts += [Span(" › ", cls="muted"),
                  A(family_label(family), **_drill(state, lane=""), cls="crumb")]
    if state.run_filter.lane:
        config = runs[0].config if runs else state.run_filter.lane
        parts += [Span(" › ", cls="muted"), Span(config)]
    return Div(*parts, cls="crumbs")


def _groups_view(state: ViewState, runs: list[Run]) -> Div:
    groups = group_stats(runs)[:MAX_GROUPS]
    header = ("Backend", "Model", "Ctx", "Spec", "Configs", "Runs", "Err", "Best TPS", "Median", "Last")
    rows = [
        Tr(
            Td(group.backend, style=f"color:{backend_color(group.backend)}"),
            Td(group.model),
            Td("-" if group.ctx is None else f"{group.ctx // 1024}K", cls="num"),
            Td(group.spec),
            Td(str(group.lanes), cls="num"),
            Td(str(group.runs), cls="num"),
            Td(str(group.failed), cls="num err" if group.failed else "num muted"),
            Td(number(group.best_tps), cls="num best"),
            Td(number(group.median_tps), cls="num muted"),
            Td(group.last_time[:10], cls="muted"),
            **_drill(state, family=group.key, lane=""),
        )
        for group in groups
    ]
    return Div(
        Div(f"{len(groups)} comparable groups — open one to compare its configs", cls="muted"),
        Div(Table(Thead(Tr(*[Th(name, cls="num" if name in {"Ctx", "Configs", "Runs", "Err", "Best TPS", "Median"} else None)
                             for name in header])),
                  Tbody(*rows)), cls="table-wrap"),
        cls="panel",
    )


def _lanes_view(state: ViewState, runs: list[Run]) -> Div:
    lanes: list[LaneStat] = lane_stats(runs)
    header = ("Config", "Backend", "Runs", "Err", "Best TPS", "Median", "Last")
    rows = [
        Tr(
            Td(lane.label, cls="label"),
            Td(lane.backend, style=f"color:{backend_color(lane.backend)}"),
            Td(str(lane.runs), cls="num"),
            Td(str(lane.failed), cls="num err" if lane.failed else "num muted"),
            Td(number(lane.best_tps), cls="num best"),
            Td(number(lane.median_tps), cls="num muted"),
            Td(lane.last_time[:10], cls="muted"),
            **_drill(state, lane=lane.key),
        )
        for lane in lanes
    ]
    return Div(
        Div(
            config_chart(lanes, link_for=lambda lane: "/history/rows?" + state.query(lane=lane.key)),
            cls="panel",
        ),
        Div(
            Div(f"{len(lanes)} configs in this group — open one to see its run history", cls="muted"),
            Div(Table(Thead(Tr(*[Th(name, cls="num" if name in {"Runs", "Err", "Best TPS", "Median"} else None)
                                 for name in header])),
                      Tbody(*rows)), cls="table-wrap"),
            cls="panel",
        ),
    )


def _header_cell(state: ViewState, sort_key: str, label: str, numeric: bool):
    if not sort_key:
        return Th(label, cls="num" if numeric else None)
    active = state.sort == sort_key
    descending = not state.descending if active else True
    arrow = (" ▾" if state.descending else " ▴") if active else ""
    return Th(
        A(label + arrow, **_drill(state, sort=sort_key, desc="1" if descending else "0"),
          cls="sorted" if active else None),
        cls="num" if numeric else None,
    )


def _run_row(run: Run) -> Tr:
    return Tr(
        Td(run.time_text),
        Td(run.backend, style=f"color:{backend_color(run.backend)}"),
        Td(run.mode),
        Td(run.config, cls="muted"),
        Td(number(run.aggregate_tps), cls="num best" if run.is_group_best else "num"),
        Td(number(run.prompt_eval_tps), cls="num"),
        Td(number(run.decode_eval_tps), cls="num"),
        Td(str(run.errors), cls="num err" if run.errors else "num muted"),
        Td(run.label or run.run_id, cls="label", title=run.label),
        hx_get=f"/history/run/{run.index}",
        hx_target="#detail",
        hx_swap="innerHTML",
    )


def _runs_view(state: ViewState, runs: list[Run]) -> Div:
    ordered = sort_runs(runs, state.sort, state.descending)
    shown = ordered[: state.limit]
    return Div(
        Div(progress_chart(sort_runs(runs, "time", descending=False)), cls="panel"),
        Div(
            Div(f"Showing {len(shown)} of {len(ordered)} runs — click a row for details", cls="muted"),
            Div(Table(Thead(Tr(*[_header_cell(state, key, label, numeric)
                                 for key, label, numeric in RUN_COLUMNS])),
                      Tbody(*[_run_row(run) for run in shown])), cls="table-wrap"),
            cls="panel",
        ),
    )


def _legend(runs: list[Run]) -> Div:
    backends = sorted({run.backend for run in runs})
    return Div(
        *[Span(I(style=f"background:{backend_color(name)}"), name) for name in backends],
        Span(I(style="background:#f0d98a"), "best"),
        cls="legend",
    )


def results(state: ViewState, runs: list[Run]) -> Div:
    selected = apply_filter(runs, state.run_filter)
    if not state.run_filter.family:
        body = _groups_view(state, selected)
    elif not state.run_filter.lane:
        body = _lanes_view(state, selected)
    else:
        body = _runs_view(state, selected)

    return Div(
        Div(
            Input(type="hidden", name="family", value=state.run_filter.family),
            Input(type="hidden", name="lane", value=state.run_filter.lane),
            Input(type="hidden", name="sort", value=state.sort),
            Input(type="hidden", name="desc", value="1" if state.descending else "0"),
            id="view-state", style="display:none",
        ),
        Div(_breadcrumb(state, selected), cls="panel"),
        Div(_summary(selected), _legend(selected), cls="panel"),
        body,
        id="results",
    )


def detail(run: Run) -> Div:
    rows = (
        ("Run ID", run.run_id or "-"),
        ("Label", run.label or "-"),
        ("Timestamp", run.time_text),
        ("Build", f"{run.build_name or '-'} ({run.backend})"),
        ("Model", run.model_path or "-"),
        ("Group", family_label(run.family)),
        ("Config", run.config),
        ("Context", "-" if run.ctx is None else str(run.ctx)),
        ("Speculative", run.spec_mode),
        ("Extra args", run.extra_args or "-"),
        ("GPU layers / parallel", f"{run.gpu_layers} / {run.parallel}"),
        ("Flash attention", run.flash_attn),
        ("Tasks", f"{run.tasks} {run.task_ids}".strip()),
        ("Max tokens", "-" if run.max_tokens is None else str(run.max_tokens)),
        ("Real context", f"{run.real_context_mode} {run.real_context_chars or ''}".strip()),
        ("Aggregate TPS", number(run.aggregate_tps)),
        ("Prompt eval", f"{number(run.prompt_eval_tps)} t/s · {number(run.prompt_eval_ms)} ms"),
        ("Decode eval", f"{number(run.decode_eval_tps)} t/s · {number(run.decode_eval_ms)} ms"),
        ("Errors", str(run.errors)),
        ("Metric scope", run.metric_scope or "-"),
        ("Best config", run.best_config or "-"),
        ("Artifacts", ", ".join(run.artifacts) or "-"),
    )
    return Div(
        H2(run.label or run.run_id or "run"),
        Dl(*[item for key, value in rows for item in (Dt(key), Dd(value))]),
        cls="panel detail",
    )


def page(state: ViewState, runs: list[Run], config: AppConfig):
    return shell(
        "History & Analytics", "/history", config,
        _filters(state, facets(runs)),
        Div(id="detail"),
        results(state, runs),
    )


__all__ = ["GroupStat", "ViewState", "detail", "page", "read_state", "results"]
