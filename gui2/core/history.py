"""Benchmark history: read BENCH_RUNS.csv, filter, sort, summarise.

The CSV is the canonical generated history produced by
`scripts/agent_workload_bench.py`. Autotune rows store literal "sweep" in the
numeric batch/ubatch columns, so every numeric field is parsed defensively.
"""

from __future__ import annotations

import csv
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# lane_key parts that describe the tuned config rather than the comparison family
_CONFIG_BATCH = re.compile(r"^b[^|]*/ub")

SORT_KEYS: dict[str, str] = {
    "time": "timestamp",
    "backend": "backend",
    "model": "model",
    "mode": "mode",
    "ctx": "ctx",
    "tps": "aggregate_tps",
    "prompt": "prompt_eval_tps",
    "decode": "decode_eval_tps",
    "label": "label",
}


def _text(row: dict[str, str], key: str) -> str:
    return str(row.get(key, "") or "").strip()


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(_text(row, key))
    except ValueError:
        return None


def _integer(row: dict[str, str], key: str) -> int | None:
    value = _number(row, key)
    return None if value is None else int(value)


def _flag(row: dict[str, str], key: str) -> bool:
    return _text(row, key).lower() in {"1", "on", "true", "yes"}


def split_lane(lane_key: str) -> tuple[str, str]:
    """Split a lane key into (comparison family, tuned config).

    Only runs of the same family may be compared: it pins backend, model,
    context, speculative mode, task set and prompt source. Batch/ubatch and KV
    types are what a sweep varies inside that family.
    """
    family: list[str] = []
    config: list[str] = []
    for part in (piece for piece in lane_key.split("|") if piece):
        if part.startswith("kv"):
            config.append("kv " + part[2:])
        elif _CONFIG_BATCH.match(part):
            config.append(part)
        else:
            family.append(part)
    return "|".join(family), " · ".join(config)


@dataclass(frozen=True, slots=True)
class Run:
    """One benchmark or autotune run."""

    index: int
    timestamp: datetime | None
    run_id: str
    label: str
    build_name: str
    backend: str
    mode: str
    model: str
    model_path: str
    tasks: str
    task_ids: str
    ctx: int | None
    batch: str
    ubatch: str
    kv_k: str
    kv_v: str
    spec_mode: str
    extra_args: str
    gpu_layers: int | None
    parallel: int | None
    flash_attn: str
    max_tokens: int | None
    real_context_mode: str
    real_context_chars: int | None
    aggregate_tps: float | None
    prompt_eval_tps: float | None
    decode_eval_tps: float | None
    prompt_eval_ms: float | None
    decode_eval_ms: float | None
    errors: int
    metric_scope: str
    lane_key: str
    family: str
    config: str
    best_config: str
    is_group_best: bool
    is_mtp_model: bool
    artifacts: tuple[str, ...]
    raw: dict[str, str] = field(repr=False, default_factory=dict)

    @property
    def time_text(self) -> str:
        return self.timestamp.strftime(TIMESTAMP_FORMAT) if self.timestamp else "-"

    @property
    def scored(self) -> bool:
        return self.aggregate_tps is not None and self.errors == 0


def _to_run(index: int, row: dict[str, str]) -> Run:
    try:
        stamp: datetime | None = datetime.strptime(_text(row, "timestamp"), TIMESTAMP_FORMAT)
    except ValueError:
        stamp = None

    model_path = _text(row, "model")
    lane_key = _text(row, "lane_key")
    family, config = split_lane(lane_key)
    if not family:
        family = "|".join(
            (_text(row, "build_backend") or "unknown", Path(model_path).name, f"ctx{_text(row, 'ctx')}")
        )
    if not config:
        config = f"b{_text(row, 'batch')}/ub{_text(row, 'ubatch')} · kv {_text(row, 'kv_k')}/{_text(row, 'kv_v')}"

    artifacts = tuple(
        name
        for name in (
            _text(row, "jsonl_file"),
            _text(row, "csv_file"),
            _text(row, "summary_file"),
            _text(row, "server_log_file"),
        )
        if name
    )

    return Run(
        index=index,
        timestamp=stamp,
        run_id=_text(row, "run_id"),
        label=_text(row, "label"),
        build_name=_text(row, "build_name"),
        backend=_text(row, "build_backend") or "unknown",
        mode=_text(row, "mode"),
        model=Path(model_path).name or "-",
        model_path=model_path,
        tasks=_text(row, "tasks"),
        task_ids=_text(row, "task_ids"),
        ctx=_integer(row, "ctx"),
        batch=_text(row, "batch") or "-",
        ubatch=_text(row, "ubatch") or "-",
        kv_k=_text(row, "kv_k") or "-",
        kv_v=_text(row, "kv_v") or "-",
        spec_mode=_text(row, "spec_mode") or "none",
        extra_args=_text(row, "extra_args"),
        gpu_layers=_integer(row, "gpu_layers"),
        parallel=_integer(row, "parallel"),
        flash_attn=_text(row, "flash_attn") or "-",
        max_tokens=_integer(row, "max_tokens"),
        real_context_mode=_text(row, "real_context_mode"),
        real_context_chars=_integer(row, "real_context_chars"),
        aggregate_tps=_number(row, "aggregate_tps"),
        prompt_eval_tps=_number(row, "prompt_eval_tps"),
        decode_eval_tps=_number(row, "decode_eval_tps"),
        prompt_eval_ms=_number(row, "prompt_eval_ms"),
        decode_eval_ms=_number(row, "decode_eval_ms"),
        errors=_integer(row, "errors") or 0,
        metric_scope=_text(row, "metric_scope"),
        lane_key=lane_key,
        family=family,
        config=config,
        best_config=_text(row, "best_config"),
        is_group_best=_flag(row, "is_group_best"),
        is_mtp_model=_flag(row, "is_mtp_model"),
        artifacts=artifacts,
        raw=dict(row),
    )


def load_runs(csv_path: Path) -> list[Run]:
    """Newest-first list of runs; a missing history file yields an empty list."""
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    runs = [_to_run(index, row) for index, row in enumerate(rows)]
    runs.sort(key=lambda run: (run.timestamp or datetime.min, run.index), reverse=True)
    return runs


@dataclass(frozen=True, slots=True)
class RunFilter:
    backend: str = ""
    model: str = ""
    mode: str = ""
    spec: str = ""
    query: str = ""
    min_tps: float | None = None
    hide_errors: bool = False
    best_only: bool = False
    family: str = ""
    lane: str = ""

    def matches(self, run: Run) -> bool:
        if self.family and run.family != self.family:
            return False
        if self.lane and run.lane_key != self.lane:
            return False
        if self.backend and run.backend != self.backend:
            return False
        if self.model and run.model != self.model:
            return False
        if self.mode and run.mode != self.mode:
            return False
        if self.spec and run.spec_mode != self.spec:
            return False
        if self.hide_errors and run.errors:
            return False
        if self.best_only and not run.is_group_best:
            return False
        if self.min_tps is not None and (run.aggregate_tps or 0.0) < self.min_tps:
            return False
        if self.query:
            needle = self.query.lower()
            haystack = " ".join((run.label, run.lane_key, run.extra_args, run.build_name, run.model)).lower()
            if needle not in haystack:
                return False
        return True


def apply_filter(runs: list[Run], run_filter: RunFilter) -> list[Run]:
    return [run for run in runs if run_filter.matches(run)]


def sort_runs(runs: list[Run], sort: str = "time", descending: bool = True) -> list[Run]:
    """Sort by one column; runs without a value for it stay at the end."""
    attribute = SORT_KEYS.get(sort, "timestamp")

    def value_of(run: Run):
        value = getattr(run, attribute)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str):
            return value.lower()
        return None if value is None else float(value)

    present = [run for run in runs if value_of(run) is not None]
    missing = [run for run in runs if value_of(run) is None]
    present.sort(key=lambda run: (value_of(run), run.index), reverse=descending)
    return present + missing


@dataclass(frozen=True, slots=True)
class Facets:
    backends: tuple[str, ...]
    models: tuple[str, ...]
    modes: tuple[str, ...]
    specs: tuple[str, ...]


def facets(runs: list[Run]) -> Facets:
    def distinct(attribute: str) -> tuple[str, ...]:
        return tuple(sorted({getattr(run, attribute) for run in runs if getattr(run, attribute)}))

    return Facets(
        backends=distinct("backend"),
        models=distinct("model"),
        modes=distinct("mode"),
        specs=distinct("spec_mode"),
    )


@dataclass(frozen=True, slots=True)
class Summary:
    count: int
    with_errors: int
    best_tps: float | None
    median_tps: float | None
    best_run: Run | None
    first_time: str
    last_time: str


def summarize(runs: list[Run]) -> Summary:
    scored = [run for run in runs if run.aggregate_tps is not None]
    values = [run.aggregate_tps or 0.0 for run in scored]
    best = max(scored, key=lambda run: run.aggregate_tps or 0.0, default=None)
    stamps = sorted(run.time_text for run in runs if run.timestamp)
    return Summary(
        count=len(runs),
        with_errors=sum(1 for run in runs if run.errors),
        best_tps=max(values) if values else None,
        median_tps=median(values) if values else None,
        best_run=best,
        first_time=stamps[0] if stamps else "-",
        last_time=stamps[-1] if stamps else "-",
    )


@dataclass(frozen=True, slots=True)
class LaneStat:
    """One tuned config (lane) inside a comparison family."""

    key: str
    label: str
    backend: str
    runs: int
    failed: int
    best_tps: float
    median_tps: float
    best_run: Run
    last_time: str


@dataclass(frozen=True, slots=True)
class GroupStat:
    """A set of runs that may legitimately be compared with each other."""

    key: str
    backend: str
    model: str
    ctx: int | None
    spec: str
    lanes: int
    runs: int
    failed: int
    best_tps: float
    median_tps: float
    best_config: str
    best_run: Run
    last_time: str


#: how a sweep records what it chose, e.g.
#: `ctx=12288 b=8192 ub=1024 kv=f8_e4m3 spec=none extra=base extra_args=<none>`
_BEST_CONFIG = re.compile(r"(\w+)=(\S+)")

#: the abbreviations that row uses, against the sweep axis each one names
BEST_CONFIG_AXES: tuple[tuple[str, str], ...] = (
    ("ctx", "sweep_ctx"), ("b", "sweep_batch"), ("ub", "sweep_ubatch"),
    ("kv", "sweep_kv"), ("spec", "sweep_spec"),
)


def winning_config(run: Run) -> dict[str, str]:
    """The configuration a sweep settled on, keyed by sweep axis.

    An autotune run writes one row for the whole sweep: batch, ubatch and the
    KV types are the literal word "sweep", and what it actually chose survives
    only in `best_config`. Reading it back is the only way to answer "what did
    the last sweep of this decide" without opening the summary CSV.
    """
    found = dict(_BEST_CONFIG.findall(run.best_config or ""))
    chosen = {axis: found[key] for key, axis in BEST_CONFIG_AXES
              if found.get(key, "<none>") != "<none>"}
    return chosen if len(chosen) == len(BEST_CONFIG_AXES) else {}


def past_sweeps(runs: list[Run], model_path: str, limit: int = 4) -> list[Run]:
    """Finished sweeps of the same model, newest first, that chose something.

    Matched on the file name rather than the path: the same model moved to
    another directory is still the same model, and the answer it gave still
    applies.
    """
    wanted = Path(model_path).name.lower()
    if not wanted:
        return []
    found = [run for run in runs
             if run.mode == "autotune" and not run.errors
             and Path(run.model_path).name.lower() == wanted
             and winning_config(run)]
    found.sort(key=lambda run: run.timestamp or datetime.min, reverse=True)
    return found[:limit]


def _bucket(runs: list[Run], attribute: str) -> dict[str, list[Run]]:
    buckets: dict[str, list[Run]] = {}
    for run in runs:
        buckets.setdefault(getattr(run, attribute), []).append(run)
    return buckets


def _last_time(runs: list[Run]) -> str:
    stamps = sorted(run.time_text for run in runs if run.timestamp)
    return stamps[-1] if stamps else "-"


def lane_stats(runs: list[Run]) -> list[LaneStat]:
    """Per-config summary, best first. Failed runs never define a best."""
    stats: list[LaneStat] = []
    for key, bucket in _bucket(runs, "lane_key").items():
        scored = [run for run in bucket if run.scored]
        if not scored:
            continue
        values = [run.aggregate_tps or 0.0 for run in scored]
        best = max(scored, key=lambda run: run.aggregate_tps or 0.0)
        stats.append(
            LaneStat(
                key=key,
                label=bucket[0].config,
                backend=bucket[0].backend,
                runs=len(bucket),
                failed=sum(1 for run in bucket if run.errors),
                best_tps=max(values),
                median_tps=median(values),
                best_run=best,
                last_time=_last_time(bucket),
            )
        )
    stats.sort(key=lambda item: item.best_tps, reverse=True)
    return stats


def group_stats(runs: list[Run]) -> list[GroupStat]:
    """Per-family summary, best first."""
    stats: list[GroupStat] = []
    for key, bucket in _bucket(runs, "family").items():
        scored = [run for run in bucket if run.scored]
        if not scored:
            continue
        values = [run.aggregate_tps or 0.0 for run in scored]
        best = max(scored, key=lambda run: run.aggregate_tps or 0.0)
        head = bucket[0]
        stats.append(
            GroupStat(
                key=key,
                backend=head.backend,
                model=head.model,
                ctx=head.ctx,
                spec=head.spec_mode,
                lanes=len({run.lane_key for run in bucket}),
                runs=len(bucket),
                failed=sum(1 for run in bucket if run.errors),
                best_tps=max(values),
                median_tps=median(values),
                best_config=best.config,
                best_run=best,
                last_time=_last_time(bucket),
            )
        )
    stats.sort(key=lambda item: item.best_tps, reverse=True)
    return stats


class HistoryStore:
    """Reloads the history CSV only when the file changes on disk."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._stamp: tuple[float, int] | None = None
        self._runs: list[Run] = []

    def _current_stamp(self) -> tuple[float, int] | None:
        try:
            info = self.csv_path.stat()
        except OSError:
            return None
        return (info.st_mtime, info.st_size)

    def runs(self) -> list[Run]:
        stamp = self._current_stamp()
        with self._lock:
            if stamp != self._stamp:
                self._runs = load_runs(self.csv_path)
                self._stamp = stamp
            return self._runs
