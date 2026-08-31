"""The Autotune history: every bench2 measurement, filterable and expandable.

bench2's index holds one row per scenario; a run is a whole folder of them.
This page shows one row per run — the best scenario's numbers — and expands a
row into the scenarios it is made of, because "best decode 61 t/s" hides the
three other configurations the same search measured.

Everything here reads bench2's own index.csv and run.json files; nothing
starts a server or asks a driver.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, urlencode

from fasthtml.common import (
    A,
    Div,
    Form,
    H3,
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
from gui2.core.results import Placement, Result, build_of_run, placement_of, read_index
from gui2.web.layout import shell

#: (query key, header, numeric) — numeric columns sort and right-align
COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("time", "When", False),
    ("run", "Run", False),
    ("backend", "Backend", False),
    ("build", "Build", False),
    ("model", "Model", False),
    ("placement", "GPU order / split", False),
    ("lane", "Lane", False),
    ("spec", "Spec", False),
    ("prefill", "Prefill t/s", True),
    ("decode", "Decode t/s", True),
    ("total", "Total t/s", True),
    ("scenarios", "Scenarios", True),
)
COLSPAN = len(COLUMNS)

SORT_KEYS = {
    "time": lambda row: row[0].when,
    "run": lambda row: row[0].run_name,
    "backend": lambda row: row[0].backend,
    "build": lambda row: row[1],
    "model": lambda row: row[0].model,
    "placement": lambda row: row[2].sort_key,
    "lane": lambda row: row[0].scenario,
    "spec": lambda row: row[0].spec_mode,
    "prefill": lambda row: row[0].prefill_tps,
    "decode": lambda row: row[0].decode_tps,
    "total": lambda row: row[0].aggregate_tps,
    "scenarios": lambda row: len(row[3]),
}


def _text(params, key: str) -> str:
    return str(params.get(key, "") or "").strip()


def grouped(rows: list[Result], lane: str) -> dict[str, list[Result]]:
    """One run name -> its scenarios, after the lane filter has its say.

    The lane filter picks which scenario of a run the row stands for: a search
    that measured L1-L4 still appears when L2 is asked, as its L2 numbers.
    """
    groups: dict[str, list[Result]] = {}
    for row in rows:
        if lane and row.scenario != lane:
            continue
        groups.setdefault(row.run_name, []).append(row)
    return groups


def representative(run: list[Result]) -> Result:
    """The scenario the row's numbers come from: the fastest decode recorded."""
    return max(run, key=lambda row: (row.decode_tps, row.aggregate_tps))


def sorted_runs(groups: dict[str, list[Result]], build_cache: dict[str, str],
                placement_cache: dict[str, Placement], sort: str,
                descending: bool) -> list[tuple[Result, str, Placement, list[Result]]]:
    """(representative, build, placement, scenarios) in the requested order.

    A non-numeric column sorts as text; a numeric one as its number. Reversed
    only flips the order, never the meaning.
    """
    rows = [
        (representative(run), build_of_run(representative(run), build_cache),
         placement_of(representative(run), placement_cache), run)
        for run in groups.values()
    ]
    key = SORT_KEYS.get(sort, SORT_KEYS["time"])
    return sorted(rows, key=key, reverse=descending)


def facets(rows: list[Result], cache: dict[str, str]) -> dict[str, list[str]]:
    """The filter values that have at least one row behind them, in order."""
    lanes: list[str] = []
    for row in rows:
        if row.scenario not in lanes:
            lanes.append(row.scenario)
    return {
        "lane": lanes,
        "backend": sorted({row.backend for row in rows if row.backend}),
        "build": sorted({build_of_run(row, cache) for row in rows if row.path}),
    }


def _filter(form: str, params, key: str, value: str, sort: str, desc: bool) -> str:
    """A select box whose change submits the filters and keeps the sort."""
    chosen = _text(params, key)
    options = [Option("all", value="", selected=chosen == "")]
    options += [Option(value, value=value, selected=value == chosen) for value in value]
    return Select(*options, name=key, value=chosen,
                  hx_get=f"/autotune/history?{form}&sort={sort}&desc={1 if desc else 0}",
                  hx_trigger="change", hx_target="#history", hx_swap="outerHTML",
                  hx_push_url="true")


def filters_row(params, rows: list[Result], cache: dict[str, str],
                sort: str, desc: bool) -> Div:
    """The four selectors, one per comparison family: lane, backend, spec, build."""
    choices = facets(rows, cache)
    base = urlencode({key: _text(params, key)
                      for key in ("lane", "backend", "mtp", "build") if _text(params, key)})
    selects = [
        ("lane", "Lane", choices["lane"]),
        ("backend", "Backend", choices["backend"]),
        ("mtp", "Speculation", ["mtp", "none"]),
        ("build", "Build", choices["build"]),
    ]
    return Div(
        *[Div(Span(title), _filter(base, params, key, values, sort, desc), cls="field")
          for key, title, values in selects],
        cls="toolbar",
    )


def sort_link(params, form: str, key: str, name: str, numeric: bool,
              sort: str, desc: bool) -> A:
    """A column header that re-sorts: click again to flip the direction."""
    active = sort == key
    arrow = " ↓" if active and desc else (" ↑" if active else "")
    return A(name + arrow, href=f"/autotune/history?{form}&sort={key}"
                                f"&desc={0 if active and desc else 1}",
             cls=f"sorted num" if active else ("num" if numeric else None))


def table(config: AppConfig, params) -> Div:
    """The run rows, with the filters above and an empty stub under each."""
    rows = read_index(config.bench_results / "index.csv")
    cache: dict[str, str] = {}
    placement_cache: dict[str, Placement] = {}
    lane = _text(params, "lane")
    backend = _text(params, "backend")
    mtp = _text(params, "mtp")
    build = _text(params, "build")
    sort = _text(params, "sort") or "time"
    desc = params.get("desc", "1") != "0"

    rows = [row for row in rows
            if (not backend or row.backend == backend)
            and (not mtp or row.spec_mode == mtp)]
    groups = grouped(rows, lane)
    if build:
        groups = {name: run for name, run in groups.items()
                  if build_of_run(representative(run), cache) == build}
    chosen = sorted_runs(groups, cache, placement_cache, sort, desc)
    limit = 100
    chosen = chosen[:limit]

    form = urlencode({key: value for key, value in
                      (("lane", lane), ("backend", backend), ("mtp", mtp),
                       ("build", build)) if value})
    if not rows:
        body = Div(Span("No measurements yet — run an Autotune search and its results "
                        "land here from bench2's own index.", cls="hint block"),
                   cls="panel")
    elif not chosen:
        body = Div(Span("Nothing matches these filters.", cls="hint block"), cls="panel")
    else:
        header = [Th(A("▸", title="expand"))] + [
            Th(sort_link(params, form, key, name, numeric, sort, desc),
               cls="num" if numeric else None)
            for key, name, numeric in COLUMNS
        ]
        run_rows = []
        for representative_row, run_build, placement, scenarios in chosen:
            when = representative_row.time_text
            draft = "none"
            if representative_row.spec_mode == "mtp":
                draft = (f"mtp n{representative_row.draft_n}"
                         if representative_row.draft_n else "mtp")
            run_rows += [
                Tr(Td(A("▸", hx_get=f"/autotune/history/run?run_name={quote(representative_row.run_name)}",
                       hx_target=f"#detail-{representative_row.run_name}",
                       hx_swap="outerHTML")),
                   Td(when),
                   Td(Path(representative_row.run_name).name, title=representative_row.run_name,
                      cls="label"),
                   Td(representative_row.backend),
                   Td(run_build or "—"),
                   Td(Path(representative_row.model).name, title=representative_row.model,
                      cls="label"),
                         Td(placement.text, title="GPU order · tensor split proportions",
                             cls="label"),
                   Td(representative_row.scenario, cls="num"),
                   Td(draft),
                   Td(f"{representative_row.prefill_tps:.0f}", cls="num"),
                   Td(f"{representative_row.decode_tps:.1f}", cls="num"),
                   Td(f"{representative_row.aggregate_tps:.1f}", cls="num"),
                   Td(str(len(scenarios)), cls="num"),
                   cls="run-row"),
                Tr(id=f"detail-{representative_row.run_name}"),
            ]
        body = Div(
            Div(Table(Thead(Tr(*header)), Tbody(*run_rows)),
                cls="table-wrap history"),
            cls="panel",
        )

    return Div(
        H3("Autotune runs"),
        Span("One row per run: the numbers are its fastest decode, and ▸ lists every "
             "scenario that run measured. Filtering keeps the run's other scenarios "
             "visible when expanded.", cls="hint block"),
        filters_row(params, rows, cache, sort, desc),
        body,
        id="history",
    )


def run_detail(config: AppConfig, run_name: str) -> Tr:
    """Every scenario of one run, as the row the main table expands into."""
    scenarios = [row for row in read_index(config.bench_results / "index.csv")
                 if row.run_name == run_name]
    scenarios.sort(key=lambda row: (row.when, row.level))
    scenario_rows = []
    for row in scenarios:
        row_spec = row.spec_mode
        if row_spec == "mtp":
            row_spec += f" · draft {row.draft_n}" if row.draft_n else " · draft ?"
        scenario_rows.append(Tr(
            Td(row.time_text),
            Td(row.scenario, cls="num"),
            Td(f"{row.prefill_tps:.0f}", cls="num"),
            Td(f"{row.decode_tps:.1f}", cls="num"),
            Td(f"{row.aggregate_tps:.1f}", cls="num"),
            Td(str(row.ctx), cls="num"),
            Td(row.status or "ok"),
            Td(row_spec),
        ))
    return Tr(
        Td(Div(
            H3(f"{Path(run_name).name} — every scenario"),
            Div(Table(
                Thead(Tr(*[Th(name, cls="num" if name != "When" and name != "Status" else None)
                           for name in ("When", "Lane", "Prefill t/s", "Decode t/s",
                                        "Total t/s", "Context", "Status", "Spec")])),
                Tbody(*scenario_rows),
            ), cls="table-wrap"),
            A("hide", hx_get=f"/autotune/history/hide?run_name={run_name}",
              hx_target=f"#detail-{run_name}", hx_swap="outerHTML", cls="hint"),
            cls="run-detail",
        ), colspan=COLSPAN + 1),
        id=f"detail-{run_name}",
    )


def hidden_detail(run_name: str) -> Tr:
    """The collapsed stub the expand link fills in."""
    return Tr(id=f"detail-{run_name}")


def page(config: AppConfig, params) -> str:
    return shell(
        "Autotune history", "/autotune", config,
        table(config, params),
        nav={"/autotune": ""},
    )
