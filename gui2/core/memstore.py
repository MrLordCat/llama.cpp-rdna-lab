"""What earlier runs actually cost, kept so the next one does not have to guess.

The estimate in `gui2.core.memory` is arithmetic on a model header. It is
close, but it cannot know what a driver will do: fragmentation, the buffers no
allocator claims, the other program on the card. A run that has already
happened knows all of that, and the only reason to throw it away is that
nobody wrote it down.

So it is written down, keyed by the command that produced it. Two levels of
answer come back out:

* the same command again -- the measurement, verbatim. Nothing is estimated.
* the same command at a different context -- the measured weights and compute
  buffers reused as they are, with only the KV cache rescaled. The KV cache is
  the one term that is exactly linear in the context length, which is what
  makes this honest rather than a fudge.

Nothing else is inferred. A different model, a different device list or a
different batch size is a different run, and gets the arithmetic.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from gui2.core.measured import DeviceUse, Measurement

#: how many runs are worth remembering. Old entries fall off the end; the
#: value of a measurement is mostly that it is recent and from this machine.
LIMIT = 200

SCHEMA = 1

#: the flag whose value changes the size of the KV cache and nothing else, so
#: a record can be reused across it by rescaling
_CONTEXT_FLAGS = frozenset({"-c", "--ctx-size"})

#: flags that decide who may talk to the server, not what it allocates.
#: Dropping them means changing a port does not throw away a measurement --
#: and, for the API key, means the key is never written to this file at all.
_IRRELEVANT_PAIRS = frozenset({
    "--api-key", "--api-key-file", "--host", "--port", "--threads-http",
    "--alias", "--model-alias", "--path", "--log-file",
})
_IRRELEVANT_ALONE = frozenset({"--metrics", "--no-webui", "--verbose", "-v"})


def _family_and_context(argv: Sequence[str]) -> tuple[tuple[str, ...], int]:
    """The command with its context removed, and the context that was removed.

    argv[0] goes too: rebuilding into another directory does not change what a
    run costs, and a person who moved their build should not lose the answer.
    """
    rest = list(argv[1:])
    context = 0
    trimmed: list[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        if token in _CONTEXT_FLAGS and index + 1 < len(rest):
            try:
                context = int(rest[index + 1])
            except ValueError:
                context = 0
            index += 2
            continue
        if token in _IRRELEVANT_PAIRS and index + 1 < len(rest):
            index += 2
            continue
        if token in _IRRELEVANT_ALONE:
            index += 1
            continue
        trimmed.append(token)
        index += 1
    return tuple(trimmed), context


@dataclass(frozen=True, slots=True)
class Record:
    """One finished run's memory, and enough of its command to recognise it."""

    family: tuple[str, ...]
    context: int
    devices: tuple[DeviceUse, ...]
    complete: bool = False
    at: float = 0.0

    @property
    def measurement(self) -> Measurement:
        return Measurement(self.devices, self.complete)

    @property
    def age_text(self) -> str:
        seconds = max(0.0, time.time() - self.at)
        if seconds < 3600:
            return f"{seconds / 60:.0f} minutes ago"
        if seconds < 86400:
            return f"{seconds / 3600:.0f} hours ago"
        return f"{seconds / 86400:.0f} days ago"


def rescaled(record: Record, context: int) -> Measurement:
    """The record's measurement, with only the KV cache moved to a new context.

    Weights do not change with the context and neither, to any degree worth
    reporting, do the compute buffers -- those follow the ubatch, which is not
    part of what is being changed here. The KV cache is exactly linear, so it
    is the only term that moves.
    """
    if context <= 0 or record.context <= 0 or context == record.context:
        return record.measurement
    factor = context / record.context
    return Measurement(
        tuple(replace(device, kv_mib=device.kv_mib * factor,
                      state_mib=device.state_mib)  # recurrent state is per sequence, not per token
              for device in record.devices),
        record.complete,
    )


class MemoryStore:
    """A short, plain JSON file of finished runs. Never blocks a request."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: list[Record] | None = None
        # written by a job's own thread as it finishes, read by request threads
        self._lock = threading.Lock()

    # -- reading -----------------------------------------------------------

    def records(self) -> list[Record]:
        with self._lock:
            if self._records is None:
                self._records = list(_load(self.path))
            return list(self._records)

    def exact(self, argv: Sequence[str]) -> Record | None:
        """The same command, run before. Its answer needs no adjusting."""
        family, context = _family_and_context(argv)
        return next((record for record in self.records()
                     if record.family == family and record.context == context), None)

    def nearest(self, argv: Sequence[str]) -> Record | None:
        """The same command at some other context, closest one first.

        Closest, because the rescaling is linear and exact for the KV cache but
        says nothing about whatever else drifts with a very different run.
        """
        family, context = _family_and_context(argv)
        candidates = [record for record in self.records()
                      if record.family == family and record.context > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda record: abs(record.context - context))

    def recall(self, argv: Sequence[str]) -> tuple[Measurement, Record | None, bool]:
        """What is known about this command: the figures, the run, and whether
        the figures are that run's own rather than rescaled from it."""
        exact = self.exact(argv)
        if exact is not None:
            return exact.measurement, exact, True
        near = self.nearest(argv)
        if near is None:
            return Measurement(), None, False
        _family, context = _family_and_context(argv)
        return rescaled(near, context), near, False

    # -- writing -----------------------------------------------------------

    def remember(self, argv: Sequence[str], measurement: Measurement) -> Record | None:
        """Keep a finished run. An incomplete one replaces nothing complete.

        A run killed during load reports its weights and no compute buffers.
        That is a worse answer than the one already stored, so it is not
        allowed to overwrite it.
        """
        if not measurement.vram:
            return None
        family, context = _family_and_context(argv)
        record = Record(family=family, context=context, devices=measurement.devices,
                        complete=measurement.complete, at=time.time())

        kept = self.records()
        previous = next((item for item in kept
                         if item.family == family and item.context == context), None)
        if previous is not None:
            if previous.complete and not record.complete:
                return previous
            kept.remove(previous)
        kept.insert(0, record)
        del kept[LIMIT:]
        with self._lock:
            self._records = kept
        _save(self.path, kept)
        return record


# -- the file itself -------------------------------------------------------


def _device_json(device: DeviceUse) -> dict:
    return {
        "name": device.name,
        "description": device.description,
        "model": round(device.model_mib, 2),
        "kv": round(device.kv_mib, 2),
        "state": round(device.state_mib, 2),
        "output": round(device.output_mib, 2),
        "compute": round(device.compute_mib, 2),
        "total": device.total_mib,
        "overhead": device.overhead_mib,
    }


def _device_from(data: dict) -> DeviceUse:
    return DeviceUse(
        name=str(data.get("name", "")),
        description=str(data.get("description", "")),
        model_mib=float(data.get("model", 0.0)),
        kv_mib=float(data.get("kv", 0.0)),
        state_mib=float(data.get("state", 0.0)),
        output_mib=float(data.get("output", 0.0)),
        compute_mib=float(data.get("compute", 0.0)),
        total_mib=data.get("total"),
        overhead_mib=data.get("overhead"),
    )


def _load(path: Path) -> Iterable[Record]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return []
    records: list[Record] = []
    for item in data.get("runs", []):
        try:
            records.append(Record(
                family=tuple(str(token) for token in item["family"]),
                context=int(item.get("context", 0)),
                devices=tuple(_device_from(entry) for entry in item.get("devices", [])),
                complete=bool(item.get("complete")),
                at=float(item.get("at", 0.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return records


def _save(path: Path, records: list[Record]) -> None:
    payload = {
        "schema": SCHEMA,
        "runs": [
            {
                "family": list(record.family),
                "context": record.context,
                "complete": record.complete,
                "at": round(record.at, 3),
                "devices": [_device_json(device) for device in record.devices],
            }
            for record in records
        ],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # written whole and moved into place: a half-written file would be
        # read back as no history at all
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


__all__ = ["LIMIT", "MemoryStore", "Record", "rescaled"]
