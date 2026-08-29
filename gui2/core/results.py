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

    @property
    def scenario(self) -> str:
        """How bench2's own tables name it: L2 is a level, SL2 a session."""
        return f"{'SL' if self.kind == 'session' else 'L'}{self.level}"

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
