"""Which devices llama-server can use, discovered without touching a GPU.

Starting a backend to enumerate devices is exactly what must not happen while
the GPUs are busy -- it can drop the driver. So the list is assembled from
evidence that already exists:

* the device lines llama-server printed in earlier runs. These are
  authoritative: real llama.cpp names, real order, real free VRAM.
* the display adapters in the Windows registry, which is a plain registry read
  and never calls into the driver.
* the RPC endpoints the user configured, whose names are positional by
  definition (RPC0, RPC1, ... in --rpc order).

No llama.cpp binary is executed here, ever.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

from gui2.core.memory import gib

if TYPE_CHECKING:  # only for the annotation: this module must not need a probe
    from gui2.core.rpc import Fleet

LOG_GLOB = "*.server.log"
#: newest logs only: a topology from months ago is not worth the read
LOG_LIMIT = 25

# llama_prepare_model_devices: using device Vulkan1 (AMD Radeon RX 9070 XT) (unknown id) - 15416 MiB free
_USING_DEVICE = re.compile(
    r"using device (?P<name>\S+) \((?P<description>[^)]*)\)(?: \([^)]*\))? - (?P<free>\d+) MiB free"
)
# common_memory_breakdown_print: |   - Vulkan1 (RX 9070 XT)  | 16304 = 4202 + (...
_MEMORY_ROW = re.compile(r"\|\s+- (?P<name>\S+) \((?P<description>[^)]*)\)\s+\|\s+(?P<total>\d+) =")

_ADAPTER_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"

Status = Literal["idle", "scanning", "ready"]


@dataclass(frozen=True, slots=True)
class Device:
    """One entry of llama-server's `-dev` list."""

    name: str
    description: str = ""
    backend: str = ""
    free_mib: int | None = None
    total_mib: int | None = None
    source: str = ""
    confirmed: bool = False

    @property
    def memory_text(self) -> str:
        if self.free_mib is not None:
            return f"{self.free_mib / 1024:.1f} GiB free"
        if self.total_mib is not None:
            return f"{self.total_mib / 1024:.1f} GiB"
        return ""

    @property
    def label(self) -> str:
        parts = [self.description, self.memory_text]
        detail = " · ".join(part for part in parts if part)
        return f"{self.name} — {detail}" if detail else self.name


@dataclass(frozen=True, slots=True)
class Scan:
    """Result of one discovery pass."""

    status: Status = "idle"
    devices: tuple[Device, ...] = ()
    adapters: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    scanned_at: float | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def for_backend(self, backend: str) -> tuple[Device, ...]:
        """Devices a build with this backend could actually address."""
        if not backend or backend == "cpu":
            return tuple(device for device in self.devices if device.backend == "rpc")
        return tuple(device for device in self.devices
                     if device.backend in {backend, "rpc"})

    @property
    def local(self) -> tuple[Device, ...]:
        """The accelerators in this machine.

        No RPC: a worker on another machine is a choice made per run, and a
        model's size is a fact about this one.
        """
        return tuple(device for device in self.devices
                     if device.backend not in {"rpc", "cpu"})


def pool(devices: Iterable[Device]) -> tuple[float, list[str], bool]:
    """Memory these devices offer, per device and in total.

    Free memory where a real run reported it, total capacity otherwise; the
    flag says which, because "16 GiB installed" and "15.4 GiB free" answer
    different questions and only one of them is about today.
    """
    total = 0.0
    parts: list[str] = []
    seen = False
    measured = True
    for device in devices:
        seen = True
        value = device.free_mib if device.free_mib is not None else device.total_mib
        if value is None:
            measured = False
            continue
        measured = measured and device.free_mib is not None
        total += value
        parts.append(f"{device.name} {gib(value)}")
    return total, parts, seen and measured


def _backend_of(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("vulkan"):
        return "vulkan"
    if lowered.startswith(("rocm", "cuda", "hip")):
        return "rocm"
    if lowered.startswith("rpc"):
        return "rpc"
    return "cpu"


def newest_logs(roots: Iterable[Path], limit: int = LOG_LIMIT) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root and root.is_dir():
            found.extend(root.glob(LOG_GLOB))
    found.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    return found[:limit]


def devices_from_log(text: str) -> list[Device]:
    """Devices named in one llama-server log, in the order it listed them."""
    found: dict[str, Device] = {}
    for match in _USING_DEVICE.finditer(text):
        name = match["name"]
        found[name] = Device(
            name=name,
            description=match["description"],
            backend=_backend_of(name),
            free_mib=int(match["free"]),
            confirmed=True,
        )
    for match in _MEMORY_ROW.finditer(text):
        name = match["name"]
        total = int(match["total"])
        if name in found:
            found[name] = replace(found[name], total_mib=total)
        else:
            found[name] = Device(name=name, description=match["description"],
                                 backend=_backend_of(name), total_mib=total, confirmed=True)
    return list(found.values())


def _from_logs(roots: Iterable[Path]) -> tuple[dict[str, Device], str]:
    """Merge the newest logs; the newest mention of a device wins."""
    merged: dict[str, Device] = {}
    newest = ""
    for path in newest_logs(roots):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for device in devices_from_log(text):
            if device.name in merged:
                continue
            stamp = time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
            merged[device.name] = replace(device, source=f"{path.name} ({stamp})")
            newest = newest or path.name
    return merged, newest


def display_adapters() -> list[tuple[str, int | None]]:
    """GPU names from the Windows registry: no driver call, no subprocess."""
    try:
        import winreg
    except ImportError:
        return []

    adapters: list[tuple[str, int | None]] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _ADAPTER_CLASS) as root:
            index = 0
            while True:
                try:
                    subkey = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                if not subkey.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, subkey) as node:
                        description = winreg.QueryValueEx(node, "DriverDesc")[0]
                        try:
                            memory = int(winreg.QueryValueEx(node, "HardwareInformation.qwMemorySize")[0])
                        except (OSError, ValueError, TypeError):
                            memory = None
                except OSError:
                    continue
                adapters.append((str(description), memory))
    except OSError:
        return []
    return adapters


def reachable(endpoint: str, timeout: float = 0.4) -> bool:
    """TCP reachability only: the RPC worker is never asked to do any work."""
    host, _, port = endpoint.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def rpc_entries(endpoints: Iterable[str], fleet: "Fleet | None" = None,
                known: dict[str, Device] | None = None) -> list[Device]:
    """Remote devices, numbered the way llama.cpp numbers them.

    The names are positional and run straight through the workers: if the
    first address offers two GPUs it takes RPC0 and RPC1, and the second
    address starts at RPC2. Until a worker has been asked how many devices it
    has, one name per address is the only honest guess -- and it is wrong for
    exactly the machine that has two cards, which is why the check exists.
    """
    known = known or {}
    answered = {worker.endpoint: worker for worker in fleet.workers} if fleet else {}
    entries: list[Device] = []
    for endpoint in endpoints:
        worker = answered.get(endpoint)
        if worker is not None and worker.devices:
            for remote in worker.devices:
                entries.append(Device(
                    name=f"RPC{len(entries)}",
                    description=f"{endpoint} · device {remote.index}",
                    backend="rpc",
                    free_mib=int(remote.free_mib),
                    total_mib=int(remote.total_mib),
                    source="reported by the worker itself",
                    confirmed=True,
                ))
            continue

        name = f"RPC{len(entries)}"
        alive = worker.reachable if worker is not None else reachable(endpoint)
        previous = known.get(name)
        entries.append(Device(
            name=name,
            description=endpoint if alive else f"{endpoint} (not answering)",
            backend="rpc",
            free_mib=(previous.free_mib if previous
                      and previous.description.startswith(endpoint) else None),
            source="--rpc order",
            confirmed=alive,
        ))
    return entries


def scan(log_roots: Iterable[Path], endpoints: Iterable[str] = (),
         backend_hint: str = "", fleet: "Fleet | None" = None) -> Scan:
    """One discovery pass. Safe to call at any time, including under load."""
    notes: list[str] = []
    known, newest = _from_logs(log_roots)
    if newest:
        notes.append(f"device names confirmed by earlier runs (newest: {newest})")

    adapters = display_adapters()
    local = [device for device in known.values() if device.backend in {"vulkan", "rocm"}]

    if not local and adapters and backend_hint in {"vulkan", "rocm"}:
        # No run to learn from: assume llama.cpp enumerates the adapters in the
        # order Windows lists them. Flagged as unconfirmed, because it is.
        prefix = "Vulkan" if backend_hint == "vulkan" else "ROCm"
        for index, (description, memory) in enumerate(adapters):
            name = f"{prefix}{index}"
            known[name] = Device(
                name=name,
                description=description,
                backend=backend_hint,
                total_mib=int(memory / 1024 / 1024) if memory else None,
                source="display adapter list",
            )
        notes.append("no earlier run to learn from: device order assumed from the adapter list")
    elif not local and not adapters:
        notes.append("no local GPU found in earlier runs or in the adapter list")

    ordered = sorted(known.values(), key=lambda device: (device.backend != "rpc", device.name))

    rpc = rpc_entries(endpoints, fleet, known)
    if endpoints and not any(device.confirmed for device in rpc):
        notes.append("RPC workers are configured but none answered a TCP connect")
    if fleet is None and endpoints:
        notes.append("one RPC name per worker until one is checked; a worker with two GPUs "
                     "takes two of the names")

    devices = [device for device in ordered if device.backend != "rpc"] + rpc
    return Scan(
        status="ready",
        devices=tuple(devices),
        adapters=tuple(name for name, _ in adapters),
        notes=tuple(notes),
        scanned_at=time.time(),
    )


class DeviceService:
    """Runs the scan off the request thread and caches the result."""

    def __init__(self, log_roots: Iterable[Path]) -> None:
        self._log_roots = tuple(log_roots)
        self._lock = threading.Lock()
        self._scan = Scan(status="idle")
        self._thread: threading.Thread | None = None
        self._signature: tuple[str, ...] = ()
        #: what the last worker check found, if anyone has asked
        self._fleet: "Fleet | None" = None

    def state(self) -> Scan:
        with self._lock:
            return self._scan

    def remember(self, fleet: "Fleet") -> None:
        """Keep what a worker check learned, and redo the list with it.

        The probe is the only thing that knows a worker has two GPUs rather
        than one, and that changes every RPC name after it.
        """
        with self._lock:
            self._fleet = fleet
            self._signature = ()

    def start(self, endpoints: Iterable[str] = (), backend_hint: str = "") -> None:
        """Kick off a scan unless an identical one already ran or is running."""
        signature = (backend_hint, *endpoints)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._scan.ready and signature == self._signature:
                return
            self._signature = signature
            self._scan = replace(self._scan, status="scanning")
            fleet = self._fleet
            self._thread = threading.Thread(
                target=self._run, args=(tuple(endpoints), backend_hint, fleet),
                name="gui2-devices", daemon=True,
            )
            self._thread.start()

    def refresh(self, endpoints: Iterable[str] = (), backend_hint: str = "") -> None:
        with self._lock:
            self._signature = ()
        self.start(endpoints, backend_hint)

    def _run(self, endpoints: tuple[str, ...], backend_hint: str,
             fleet: "Fleet | None" = None) -> None:
        result = scan(self._log_roots, endpoints, backend_hint, fleet)
        with self._lock:
            self._scan = result
