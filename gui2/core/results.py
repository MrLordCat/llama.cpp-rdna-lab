"""What bench2 has already measured, read from its own index.

bench2 writes one row per measurement into `build_logs/bench/index.csv` and one
folder per run beside it. Reading both back is what lets the Autotune page say
"this has been measured" instead of measuring it again, and what lets it warn
before a name is written over.

Nothing here starts anything or asks a driver: it is a CSV and a directory
listing.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Result:
    """One measurement: a level or a whole session, of one run."""

    run_name: str = ""
    kind: str = ""
    level: str = ""
    when: str = ""
    backend: str = ""
    model: str = ""
    commit: str = ""
    ctx: int = 0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    aggregate_tps: float = 0.0
    decode_slope: float = 0.0
    turns: int = 0
    status: str = ""
    #: how far ahead the draft head guessed; empty means no speculation
    mtp_draft_n: str = ""
    #: where bench2 wrote the run folder, for reaching its run.json
    path: str = ""

    @property
    def scenario(self) -> str:
        """How bench2's own tables name it: L2 is a level, SL2 a session."""
        return f"{'SL' if self.kind == 'session' else 'L'}{self.level}"

    @property
    def spec_mode(self) -> str:
        """The mode bench2 ran under, spelled as the sweep axes spell it."""
        return "mtp" if self.mtp_draft_n else "none"

    @property
    def time_text(self) -> str:
        """The timestamp without its timezone, which is always this machine's."""
        return self.when[:19].replace("T", " ")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _number(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except ValueError:
        return 0.0


def read_index(path: Path) -> list[Result]:
    """Every measurement in the index, newest last, or nothing if there is none."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []
    return [
        Result(
            run_name=row.get("run_name", ""),
            kind=row.get("type", ""),
            level=row.get("level", ""),
            when=row.get("timestamp", ""),
            backend=row.get("backend", ""),
            model=row.get("model", ""),
            commit=row.get("commit", ""),
            ctx=int(_number(row, "ctx")),
            prefill_tps=_number(row, "prefill_tps"),
            decode_tps=_number(row, "decode_tps"),
            aggregate_tps=_number(row, "aggregate_tps"),
            decode_slope=_number(row, "decode_slope"),
            turns=int(_number(row, "session_turns")),
            status=row.get("status", ""),
            mtp_draft_n=row.get("mtp_draft_n", ""),
            path=row.get("path", ""),
        )
        for row in rows
    ]


def for_model(results: list[Result], model: str, limit: int = 8) -> list[Result]:
    """The most recent measurements of one model file, newest first.

    bench2 records the file name rather than the path, so a model moved between
    checkouts still matches what was measured of it.
    """
    if not model:
        return []
    name = Path(model).name
    mine = [item for item in results if item.model == name and item.ok]
    return sorted(mine, key=lambda item: item.when, reverse=True)[:limit]


def taken_names(results_dir: Path) -> frozenset[str]:
    """Run folders that already exist, and so would be written over."""
    try:
        return frozenset(entry.name for entry in results_dir.iterdir() if entry.is_dir())
    except OSError:
        return frozenset()


def server_opts(path: str) -> dict:
    """The `server` block of a run's run.json: what that run was told to be."""
    try:
        data = json.loads((Path(path) / "run.json").read_text(encoding="utf-8"))
        opts = data.get("server")
    except (OSError, ValueError, AttributeError):
        return {}
    return opts if isinstance(opts, dict) else {}


@dataclass(frozen=True, slots=True)
class Setup:
    """The axes a finished run was measured with, spelled as Autotune spells them."""

    batch: int = 0
    ubatch: int = 0
    kv: str = ""
    spec: str = ""
    spec_n: int = 0

    @property
    def known(self) -> bool:
        """False when the run folder is gone, or older than what bench2 records."""
        return bool(self.batch and self.ubatch and self.kv and self.spec)


def _whole(value: object) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def setup_of(result: Result, cache: dict[str, Setup]) -> Setup:
    """What a past run was set to, from the run.json bench2 wrote beside it.

    The index records what a run measured and never what it was measured with;
    only the run folder knows that. Without it a table of earlier results can
    be read but not reused.
    """
    if result.path not in cache:
        opts = server_opts(result.path) if result.path else {}
        cache[result.path] = Setup(
            batch=_whole(opts.get("batch")),
            ubatch=_whole(opts.get("ubatch")),
            kv=str(opts.get("kv_k") or ""),
            spec=str(opts.get("spec") or ""),
            spec_n=_whole(opts.get("spec_n")),
        )
    return cache[result.path]


def build_of_run(result: Result, cache: dict[str, str]) -> str:
    """The build directory the run's server binary came from, if it said so.

    The index does not name the binary; each run's run.json does. The cache
    keeps one page of history from re-reading the same folders per row.
    """
    if not result.path:
        return ""
    if result.path in cache:
        return cache[result.path]
    server_bin = server_opts(result.path).get("server_bin")
    build = Path(str(server_bin)).parent.parent.name if server_bin else ""
    cache[result.path] = build
    return build
