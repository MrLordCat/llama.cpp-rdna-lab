"""What a run actually took, read back out of its own log.

`memory.py` predicts. This reads. llama-server announces every buffer it
allocates as it starts, and prints a reconciled table on the way out, so a run
that has happened once never has to be guessed at again.

Two views come out of a log, and they are deliberately kept apart:

* **allocations** -- the `... buffer size = N MiB` lines. They appear as the
  model loads, so they are available long before the run ends, and they name
  every buffer separately. A speculative draft context or a second draft model
  adds its own lines; those are real memory and are summed in.
* **the breakdown** -- `common_memory_breakdown_print`, emitted at shutdown.
  It alone knows the card's total, and it alone measures the gap between what
  ggml asked the driver for and what the driver actually took.

The trap, visible in every log: ::

    load_tensors:   CPU_Mapped model buffer size =   682.03 MiB

`CPU_Mapped` is the model file mapped into RAM, and `Vulkan_Host` is pinned
host memory. Neither is video memory. Counting them lands the total several
gigabytes high, so only names ending in a device index -- `Vulkan0`, `ROCm1`,
`RPC0[...]` -- are treated as VRAM.

Nothing here runs anything. It parses text a run already produced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Literal

#: the buffer kinds llama.cpp reports, in the order a person would read them
Kind = Literal["model", "KV", "RS", "output", "compute"]

# load_tensors:      Vulkan0 model buffer size =  4615.84 MiB
# llama_kv_cache: RPC0[192.168.1.60:50052] KV buffer size =  2304.00 MiB
# The (?<![~\w]) keeps out ~llama_context, the destructor's complaint about a
# compute buffer that came back a different size; it reports no allocation.
_ALLOCATION = re.compile(
    r"(?<![~\w])(?:load_tensors|llama_kv_cache|llama_memory_recurrent|llama_context):\s+"
    r"(?P<name>\S+)\s+(?P<kind>model|KV|RS|output|compute) buffer size\s*=\s*(?P<mib>[\d.]+) MiB"
)

# |   - Vulkan1 (RX 9070 XT)      | 16304 = 8721 + (6341 = 5035 + 1284 + 20) + 1241 |
_BREAKDOWN = re.compile(
    r"\|\s+- (?P<name>\S+) \((?P<description>[^)]*)\)\s+\|\s+"
    r"(?P<total>\d+) =\s*(?P<free>-?\d+) \+ \(\s*(?P<self>\d+) =\s*(?P<model>-?\d+) \+"
    r"\s*(?P<context>-?\d+) \+\s*(?P<compute>-?\d+)\) \+\s*(?P<unaccounted>-?\d+)"
)

# | memory breakdown [MiB]        | total   free   self ...  -- the table's
# first line, and the only reliable mark of where a new one begins
_BREAKDOWN_HEADER = re.compile(r"memory breakdown \[MiB\]")

#: RPC devices carry their endpoint in the name at allocation time and in the
#: description at breakdown time; one spelling has to win, and RPC0 is the one
#: the command line uses
_RPC_ADDRESS = re.compile(r"^(?P<name>RPC\d+)\[(?P<address>[^\]]*)\]$")

#: a device that holds VRAM ends in its index: Vulkan0, ROCm1, CUDA0, RPC0.
#: CPU, CPU_Mapped, Host, Vulkan_Host and Vulkan_Host_Direct do not.
_ACCELERATOR = re.compile(r"^[A-Za-z]+\d+$")


def split_name(name: str) -> tuple[str, str]:
    """`RPC0[192.168.1.60:50052]` is one device with two names in it."""
    match = _RPC_ADDRESS.match(name)
    if match:
        return match["name"], match["address"]
    return name, ""


def is_accelerator(name: str) -> bool:
    """Whether a buffer on this device costs video memory."""
    return bool(_ACCELERATOR.match(split_name(name)[0]))


@dataclass(frozen=True, slots=True)
class DeviceUse:
    """Every buffer one device held during one run."""

    name: str
    description: str = ""
    model_mib: float = 0.0
    kv_mib: float = 0.0
    state_mib: float = 0.0
    output_mib: float = 0.0
    compute_mib: float = 0.0
    total_mib: int | None = None
    #: what the driver took beyond ggml's own accounting: fragmentation, the
    #: buffers no allocator claims, and anything else sharing the card
    overhead_mib: float | None = None

    @property
    def is_vram(self) -> bool:
        return is_accelerator(self.name)

    @property
    def used_mib(self) -> float:
        """The sum ggml asked for. What a next run of the same shape needs."""
        return self.model_mib + self.kv_mib + self.state_mib + self.output_mib + self.compute_mib

    @property
    def parts(self) -> tuple[tuple[str, float], ...]:
        """The non-zero pieces, largest first, ready to be shown."""
        named = (
            ("weights", self.model_mib),
            ("KV cache", self.kv_mib),
            ("recurrent state", self.state_mib),
            ("compute", self.compute_mib),
            ("output", self.output_mib),
        )
        return tuple(sorted((p for p in named if p[1] > 0), key=lambda p: -p[1]))


@dataclass(frozen=True, slots=True)
class Measurement:
    """One run's memory, as the run itself reported it."""

    devices: tuple[DeviceUse, ...] = ()
    #: set once the shutdown table has been seen; until then compute buffers
    #: and the driver's own overhead are still missing
    complete: bool = False

    def __bool__(self) -> bool:
        return bool(self.devices)

    @property
    def vram(self) -> tuple[DeviceUse, ...]:
        return tuple(device for device in self.devices if device.is_vram)

    @property
    def host(self) -> tuple[DeviceUse, ...]:
        return tuple(device for device in self.devices if not device.is_vram)

    @property
    def vram_mib(self) -> float:
        return sum(device.used_mib for device in self.vram)

    def find(self, name: str) -> DeviceUse | None:
        wanted = split_name(name)[0]
        for device in self.devices:
            if device.name == wanted:
                return device
        return None


@dataclass
class _Accumulator:
    """A device under construction, before it is frozen into a `DeviceUse`."""

    name: str
    description: str = ""
    sums: dict[str, float] = field(default_factory=dict)
    total_mib: int | None = None
    overhead_mib: float | None = None
    breakdown_compute: float | None = None

    def add(self, kind: str, mib: float) -> None:
        self.sums[kind] = self.sums.get(kind, 0.0) + mib


_FIELDS: dict[str, str] = {
    "model": "model_mib",
    "KV": "kv_mib",
    "RS": "state_mib",
    "output": "output_mib",
    "compute": "compute_mib",
}


class Reader:
    """Follows a log as it is written.

    A run's allocations are announced in its first hundred lines and its
    reconciliation in its last, and hours of generation can scroll between
    them -- past the end of any bounded buffer. So the reading happens once,
    as each line arrives, and only the totals are kept.
    """

    __slots__ = ("_devices", "_complete")

    def __init__(self) -> None:
        self._devices: dict[str, _Accumulator] = {}
        self._complete = False

    def _slot(self, raw: str) -> _Accumulator:
        name, address = split_name(raw)
        entry = self._devices.get(name)
        if entry is None:
            entry = self._devices[name] = _Accumulator(name, address)
        elif address and not entry.description:
            entry.description = address
        return entry

    def feed(self, line: str) -> None:
        allocation = _ALLOCATION.search(line)
        if allocation:
            self._slot(allocation["name"]).add(allocation["kind"], float(allocation["mib"]))
            return

        if _BREAKDOWN_HEADER.search(line):
            # a second table means the process was restarted into the same
            # log; the newest one describes the run being asked about
            for entry in self._devices.values():
                entry.total_mib = entry.overhead_mib = entry.breakdown_compute = None
            self._complete = True
            return

        row = _BREAKDOWN.search(line)
        if row:
            entry = self._slot(row["name"])
            if not entry.description:
                entry.description = row["description"]
            entry.total_mib = int(row["total"])
            entry.overhead_mib = float(row["unaccounted"])
            entry.breakdown_compute = float(row["compute"])

    def result(self) -> Measurement:
        return Measurement(
            tuple(_freeze(entry) for entry in self._devices.values()), self._complete
        )


def parse(lines: Iterable[str]) -> Measurement:
    """Read a launch log, whole or partial, into a `Measurement`.

    Safe to call on a log still being written: what has been printed so far is
    what comes back, and `complete` says whether the shutdown table arrived.
    """
    reader = Reader()
    for line in lines:
        reader.feed(line)
    return reader.result()


def _freeze(entry: _Accumulator) -> DeviceUse:
    values = {_FIELDS[kind]: mib for kind, mib in entry.sums.items() if kind in _FIELDS}
    device = DeviceUse(
        name=entry.name,
        description=entry.description,
        total_mib=entry.total_mib,
        overhead_mib=entry.overhead_mib,
        **values,
    )
    # some builds never print a compute buffer line and only own up to it in
    # the shutdown table; take that figure rather than report none
    if not device.compute_mib and entry.breakdown_compute:
        device = replace(device, compute_mib=entry.breakdown_compute)
    return device


def parse_text(text: str) -> Measurement:
    return parse(text.splitlines())


def notes(measurement: Measurement) -> Iterator[str]:
    """Anything about the measurement a person should be told, in words."""
    if not measurement:
        return
    if not measurement.complete:
        yield "Measured as the model loaded; compute buffers are not in this figure yet."
    for device in measurement.vram:
        overhead = device.overhead_mib
        if overhead is not None and overhead > 512:
            yield (
                f"{device.name} held {overhead / 1024:.1f} GiB beyond the buffers "
                "llama.cpp accounts for — driver overhead, or another program on the card."
            )
    host = sum(device.used_mib for device in measurement.host)
    if host:
        yield f"A further {host / 1024:.1f} GiB sat in system RAM, not on a card."
