"""Benchmark history: read BENCH_RUNS.csv, filter, sort, summarise.

The CSV is the canonical generated history produced by
`scripts/agent_workload_bench.py`. Autotune rows store literal "sweep" in the
numeric batch/ubatch columns, so every numeric field is parsed defensively.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

from gui2.core.results import Result, build_of_run, read_index

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


# -- autotune -> canonical history -------------------------------------------
# BENCH_RUNS.csv is what History & Analytics reads. bench2, which the Autotune
# page measures with, keeps its own index instead. The exporter below carries
# finished autotune runs into the canonical CSV in the same schema, so the
# history page shows everything the lab measured, not only what
# agent_workload_bench.py recorded.

#: the canonical column order, matching scripts/agent_workload_bench.py
HISTORY_FIELDS: tuple[str, ...] = (
    "timestamp", "run_id", "build_id", "build_name", "build_backend",
    "mode", "label", "model", "is_mtp_model", "tasks", "task_ids", "runs",
    "ctx", "batch", "ubatch", "kv_k", "kv_v", "spec_mode", "extra_preset",
    "extra_args", "no_reuse", "gpu_layers", "parallel", "flash_attn",
    "max_tokens", "real_context_mode", "real_context_chars",
    "real_context_safe_fill", "no_v2_prime_pass", "temperature", "top_p",
    "aggregate_tps", "mean_task_tps", "prompt_eval_tps", "decode_eval_tps",
    "prompt_eval_ms", "decode_eval_ms", "errors", "metric_scope", "lane_key",
    "best_config", "jsonl_file", "csv_file", "summary_file", "server_log_file",
    "is_group_best",
)

AUTOTUNE_MODE = "autotune"

#: bench2 run names name everything: <base>-b<batch>-u<ubatch>-<kv>-<spec>[-n<draft>]
_RUN_SUFFIX = re.compile(r"-b(\d+)-u(\d+)-([a-zA-Z0-9_]+)-(none|mtp)(?:-n(\d+))?$")

_SYNC_LOCK = threading.Lock()


def _autotune_run_id(run_name: str, when: str, model: str) -> str:
    """Stable per launch; a rerun of the same configuration gets a new id."""
    seed = f"{when}|{run_name}|autotune|{model}"
    digest = hashlib.sha1(seed.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"run-{digest}"


def _config_from_name(run_name: str) -> dict[str, str]:
    match = _RUN_SUFFIX.search(run_name)
    if not match:
        return {}
    return {"batch": match.group(1), "ubatch": match.group(2), "kv": match.group(3),
            "spec": match.group(4), "spec_n": match.group(5) or ""}


def _autotune_lane(backend: str, model_path: str, ctx: int, batch: str, ubatch: str,
                   kv_k: str, kv_v: str, spec: str, tasks: str, task_ids: str,
                   context_source: str, devices: str, tensor_split: str) -> str:
    """The same lane shape as the legacy script, so both feeds stay comparable."""
    model = Path(model_path).name or "-"
    kv = f"{kv_k or '-'}/{kv_v or '-'}"
    task_part = f"{tasks}:{task_ids}" if task_ids else (tasks or "-")
    return (f"{backend or '-'}|{model}|ctx{ctx or '-'}|b{batch or '-'}/ub{ubatch or '-'}"
            f"|kv{kv}|spec={spec or '-'}|reuse|tasks={task_part}"
            f"|max=-|ctxsrc={context_source or '-'}"
            f"|dev={devices or 'auto'}|ts={tensor_split or 'auto'}")


def _read_run_meta(items: list[Result]) -> dict:
    for item in items:
        if not item.path:
            continue
        try:
            return json.loads((Path(item.path) / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def _autotune_row(run_name: str, rows: list[Result], meta: dict,
                  build: str) -> dict[str, str] | None:
    """One BENCH_RUNS row for a finished run, named by its fastest decode."""
    finished = [row for row in rows if row.ok]
    if not finished:
        return None
    best = max(finished, key=lambda row: (row.decode_tps, row.aggregate_tps))
    opts = meta.get("server") or {}
    from_name = _config_from_name(run_name)
    spec = str(opts.get("spec") or best.spec_mode or "none")
    batch = opts.get("batch_size") or from_name.get("batch", "")
    ubatch = opts.get("ubatch_size") or from_name.get("ubatch", "")
    kv_k = opts.get("kv_k") or from_name.get("kv", "")
    kv_v = opts.get("kv_v") or from_name.get("kv", "")
    draft = best.mtp_draft_n or from_name.get("spec_n", "")
    tasks = str(meta.get("type") or best.kind)
    levels, sessions = meta.get("levels") or ([], [])
    task_ids = ",".join(str(item) for item in list(levels) + list(sessions))
    model_path = str(meta.get("model") or best.model)
    when = best.time_text

    def text(key: str, default: str = "") -> str:
        value = opts.get(key)
        return default if value in (None, "") else str(value)

    def number(value) -> str:
        return f"{value:.4g}" if value else "0"

    devices = text("dev")
    tensor_split = text("ts")
    extra_parts: list[str] = []
    if devices:
        extra_parts += ["--dev", devices]
    if tensor_split:
        extra_parts += ["--ts", tensor_split]
    if draft and spec == "mtp":
        extra_parts += ["--spec-n", str(draft)]

    return {
        "timestamp": when,
        "run_id": _autotune_run_id(run_name, when, best.model),
        "build_id": best.commit or "",
        "build_name": build,
        "build_backend": best.backend,
        "mode": AUTOTUNE_MODE,
        "label": run_name,
        "model": model_path,
        "is_mtp_model": "1" if spec == "mtp" else "",
        "tasks": tasks,
        "task_ids": task_ids,
        "runs": text("runs", "1"),
        "ctx": str(best.ctx),
        "batch": str(batch),
        "ubatch": str(ubatch),
        "kv_k": str(kv_k),
        "kv_v": str(kv_v),
        "spec_mode": spec,
        "extra_preset": "",
        "extra_args": " ".join(extra_parts),
        "no_reuse": "",
        "gpu_layers": text("gpu_layers"),
        "parallel": text("parallel"),
        "flash_attn": "on" if opts.get("flash_attn") is True else ("off" if opts.get("flash_attn") is False else ""),
        "max_tokens": "",
        "real_context_mode": text("context_source"),
        "real_context_chars": "",
        "real_context_safe_fill": "",
        "no_v2_prime_pass": "",
        "temperature": text("temperature"),
        "top_p": text("top_p"),
        "aggregate_tps": number(best.aggregate_tps),
        "mean_task_tps": "",
        "prompt_eval_tps": number(best.prefill_tps),
        "decode_eval_tps": number(best.decode_tps),
        "prompt_eval_ms": "",
        "decode_eval_ms": "",
        "errors": str(sum(1 for row in rows if not row.ok)),
        "metric_scope": AUTOTUNE_MODE,
        "lane_key": _autotune_lane(best.backend, model_path, best.ctx, str(batch), str(ubatch),
                                   str(kv_k), str(kv_v), spec, tasks, task_ids,
                                   text("context_source"), devices, tensor_split),
        "best_config": "",
        "jsonl_file": f"{run_name}.jsonl" if best.path else "",
        "csv_file": "",
        "summary_file": best.path,
        "server_log_file": "",
        "is_group_best": "",
    }


def _existing_run_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row.get("run_id", "") for row in csv.DictReader(handle) if row.get("run_id")}


def sync_autotune_runs(runs_csv: Path, index_csv: Path) -> int:
    """Carry finished autotune runs into the canonical history CSV.

    bench2 records each finished scenario in its own index; History & Analytics
    reads BENCH_RUNS.csv. Every finished run is appended as one row, named by
    its fastest decode exactly as the Autotune history page names it; a run
    already present (stable run_id per launch) is skipped. Returns the number
    of rows added. Safe to call while bench2 rewrites the index: a torn read
    exports nothing rather than a half row.
    """
    if not index_csv.is_file():
        return 0
    with _SYNC_LOCK:
        try:
            rows = read_index(index_csv)
            existing = _existing_run_ids(runs_csv)
        except (OSError, ValueError, csv.Error):
            return 0
        groups: dict[str, list[Result]] = {}
        for row in rows:
            groups.setdefault(row.run_name, []).append(row)
        added: list[dict[str, str]] = []
        cache: dict[str, str] = {}
        for run_name, items in groups.items():
            meta = _read_run_meta(items)
            build = build_of_run(items[0], cache) if items else ""
            row = _autotune_row(run_name, items, meta, build)
            if row and row["run_id"] not in existing:
                existing.add(row["run_id"])
                added.append(row)
        if not added:
            return 0
        runs_csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not runs_csv.is_file()
        with runs_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS,
                                    extrasaction="ignore")
            if new_file:
                writer.writeheader()
            for row in added:
                writer.writerow(row)
        return len(added)
