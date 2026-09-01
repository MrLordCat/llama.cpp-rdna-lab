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
import re
from dataclasses import dataclass
from pathlib import Path

#: bench2 names a run after the configuration it measures, so the speculation
#: token is in the name either at the end (-mtp, -mtp-n2) or followed by the
#: workload and run tokens (-mtp-n2-l0-l4-r3). The index column is declared but
#: never filled, and run.json keeps the mode but not the lookahead -- the name
#: is the one place the lookahead is written down.
_NAME_MTP = re.compile(r"-mtp(?:-n\d+)?(?:-|$)")
_NAME_DRAFT_N = re.compile(r"-mtp-n(\d+)")


@dataclass(frozen=True, slots=True)
class Result:
    """One measurement: a level or a whole session, of one run."""

    run_name: str = ""
    #: all configurations queued by one press of Autotune Start
    series_id: str = ""
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
    #: real draft counts recorded by bench2, 'accepted/draft_n' (e.g. '149/212')
    mtp_accepted: str = ""
    #: effective decode t/s under speculation (== decode_tps when MTP is on)
    eff_decode_tps: float = 0.0
    #: explicit GPU order and tensor proportions recorded by bench2
    devices: str = ""
    tensor_split: str = ""
    #: where bench2 wrote the run folder, for reaching its run.json
    path: str = ""

    @property
    def mtp_acc_pct(self) -> float:
        """Acceptance percentage from 'accepted/draft_n'; 0.0 when unknown."""
        if not self.mtp_accepted or not self.spec_mode == "mtp":
            return 0.0
        try:
            accepted, total = self.mtp_accepted.split("/", 1)
            return 100.0 * float(accepted) / float(total)
        except (ValueError, ZeroDivisionError):
            return 0.0

    @property
    def scenario(self) -> str:
        """How bench2's own tables name it: L2 is a level, SL2 a session."""
        return f"{'SL' if self.kind == 'session' else 'L'}{self.level}"

    @property
    def spec_mode(self) -> str:
        """The mode bench2 ran under, spelled as the sweep axes spell it.

        bench2 declares `mtp_draft_n` in the index but does not write it, so an
        empty cell is not proof of none: the run name carries the mode too, and
        is checked before concluding.
        """
        return "mtp" if (self.mtp_draft_n or _NAME_MTP.search(self.run_name)) else "none"

    @property
    def draft_n(self) -> str:
        """How far ahead the draft head guessed; empty means run.json did not say.

        The lookahead lives only in the run name (run.json has the mode but not
        the number), so it is read from there when the index cell is empty.
        """
        if self.mtp_draft_n:
            return self.mtp_draft_n
        match = _NAME_DRAFT_N.search(self.run_name)
        return match.group(1) if match else ""

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
            series_id=row.get("series_id", ""),
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
            mtp_accepted=row.get("mtp_accepted", ""),
            eff_decode_tps=_number(row, "eff_decode_tps"),
            devices=row.get("devices", ""),
            tensor_split=row.get("tensor_split", ""),
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
class Placement:
    """The order and relative share assigned to the GPUs of one run."""

    devices: str = ""
    tensor_split: str = ""

    @property
    def known(self) -> bool:
        return bool(self.devices or self.tensor_split)

    @property
    def text(self) -> str:
        if not self.known:
            return "—"
        devices = "all (detected order)" if self.devices in {"", "auto"} else self.devices
        split = "automatic" if self.tensor_split in {"", "auto"} else self.tensor_split
        return f"{devices} · {split}"

    @property
    def sort_key(self) -> str:
        return f"{self.devices}|{self.tensor_split}"


def placement_of(result: Result, cache: dict[str, Placement]) -> Placement:
    """Read placement from the index, falling back to an older run.json.

    New bench2 indexes keep these fields themselves, so history survives a
    cleaned run folder. Runs made before those columns existed still have the
    effective values in the server block beside their measurements.
    """
    key = result.path or f"index:{result.run_name}"
    if key not in cache:
        opts = server_opts(result.path) if result.path else {}
        has_placement = "dev" in opts or "ts" in opts
        devices = result.devices or (str(opts.get("dev") or "auto") if has_placement else "")
        tensor_split = result.tensor_split or (
            str(opts.get("ts") or "auto") if has_placement else "")
        cache[key] = Placement(devices, tensor_split)
    return cache[key]


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
