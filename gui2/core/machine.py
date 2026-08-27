"""What this machine can offer, answered without starting anything.

The Server section asks questions an ordinary person has no way to answer:
how many threads, which host address, which port. Every one of them has a
right answer the operating system already knows -- how many cores are real
rather than hyper-threaded, whether a port is already taken, which address
another machine on the same desk would have to dial. Reading them costs
microseconds and touches no GPU, no driver and no binary.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

#: A physical core runs one llama.cpp worker at full speed; its hyper-thread
#: sibling shares the same arithmetic units and mostly adds contention. This
#: is why the recommended thread count is the physical count, not the one
#: Task Manager shows.
@dataclass(frozen=True, slots=True)
class Cores:
    """How many workers this CPU can really keep busy."""

    logical: int = 1
    physical: int = 0

    @property
    def usable(self) -> int:
        """The thread count worth defaulting to."""
        return self.physical or max(1, self.logical // 2 if self.logical > 2 else self.logical)

    @property
    def text(self) -> str:
        if self.physical and self.physical != self.logical:
            return f"{self.physical} cores, {self.logical} hardware threads"
        return f"{self.logical} hardware threads"


def _physical_windows() -> int:
    """Physical cores from kernel32, which is the only place Windows says so."""
    import ctypes
    from ctypes import wintypes

    relation_processor_core = 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.GetLogicalProcessorInformationEx
    # spelled out because the default int conversion would truncate the
    # buffer pointer on 64-bit and hand the kernel a bad address
    query.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.BOOL

    length = wintypes.DWORD(0)
    query(relation_processor_core, None, ctypes.byref(length))  # asks for the size
    if not length.value:
        return 0

    buffer = (ctypes.c_byte * length.value)()
    if not query(relation_processor_core, ctypes.cast(buffer, ctypes.c_void_p),
                 ctypes.byref(length)):
        return 0

    # a packed sequence of variable-length records; only the size field is
    # needed to walk it, and every record here is one physical core
    offset, cores = 0, 0
    while offset + 8 <= length.value:
        size = int.from_bytes(bytes(buffer[offset + 4:offset + 8]), sys.byteorder)
        if size <= 0:
            break
        cores += 1
        offset += size
    return cores


def _physical_linux() -> int:
    """Physical cores from /proc/cpuinfo: distinct (package, core) pairs."""
    seen: set[tuple[str, str]] = set()
    package = core = ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "physical id":
                    package = value
                elif key == "core id":
                    core = value
                elif not line.strip() and core:
                    seen.add((package, core))
                    package = core = ""
    except OSError:
        return 0
    if core:
        seen.add((package, core))
    return len(seen)


@lru_cache(maxsize=1)
def cores() -> Cores:
    """This CPU's core count. Cached: it cannot change while we run."""
    logical = os.cpu_count() or 1
    physical = 0
    try:
        if os.name == "nt":
            physical = _physical_windows()
        elif sys.platform.startswith("linux"):
            physical = _physical_linux()
    except Exception:  # noqa: BLE001 - a core count is never worth an exception
        physical = 0
    return Cores(logical=logical, physical=min(physical, logical) if physical else 0)


def auto_threads_http(parallel: int) -> int:
    """What llama-server picks for itself when --threads-http is not given.

    Mirrors server-http.cpp: ``max(n_parallel + 4, hardware_concurrency - 1)``.
    Saying the number out loud is the whole point -- it is almost always
    better than anything a person would guess, and knowing that is what makes
    "leave it alone" an informed choice rather than a shrug.
    """
    return max(max(1, parallel) + 4, (os.cpu_count() or 1) - 1)


#: Long enough to cross a loopback interface, short enough that a typo in the
#: port box does not stall the preview.
PROBE_TIMEOUT = 0.12


def port_taken(port: int, host: str = "127.0.0.1", timeout: float = PROBE_TIMEOUT) -> bool:
    """Whether something already answers on this port.

    A connect, not a bind: on Windows a bind can succeed against a port
    another process holds with SO_REUSEADDR, which would report a busy port
    as free -- the one mistake this check exists to prevent. The connection
    is closed immediately and nothing is sent on it.
    """
    if not 1 <= int(port) <= 65535:
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError, OverflowError):
        return False


def free_port(near: int = 8080, tries: int = 40) -> int | None:
    """The first free port at or after `near`, so the advice is actionable."""
    port = max(1, int(near))
    for candidate in range(port, min(65536, port + max(1, tries))):
        if not port_taken(candidate):
            return candidate
    return None


@lru_cache(maxsize=1)
def lan_address() -> str:
    """This machine's address on the local network, or "" if it has none.

    Found by asking the routing table which interface would be used to reach
    the outside world. The socket is UDP and never connected in the wire
    sense: no packet leaves the machine.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.settimeout(0.2)
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1: reserved, never routed
        address = probe.getsockname()[0]
    except OSError:
        return ""
    finally:
        probe.close()
    return "" if address.startswith("127.") else str(address)


def hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""


#: A process list is cheap but not free, and a preview redraws on every
#: keystroke. Two seconds is far shorter than anyone can start a server in.
SERVERS_TTL = 2.0
_servers_cache: tuple[float, tuple[str, ...]] = (0.0, ())


def _server_pids() -> tuple[str, ...]:
    """PIDs of every llama-server on this machine, from the process list.

    The same question `agent_workload_bench.find_background_llama_servers`
    asks, and for the same reason: a benchmark shares its GPUs with anything
    already loaded, so a second server means the numbers are not measuring
    what they claim to. Reading the process table starts no GPU work and
    touches no driver.
    """
    import subprocess

    if os.name == "nt":
        command = ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"]
        # a console window would flash in the user's face on every preview
        options = {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    else:
        command, options = ["pgrep", "-x", "llama-server"], {}
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=5, **options)
    except (OSError, subprocess.SubprocessError):
        return ()
    if os.name != "nt":
        return tuple(line.strip() for line in done.stdout.splitlines() if line.strip().isdigit())
    if done.returncode != 0:
        return ()
    pids = []
    for raw in done.stdout.splitlines():
        # CSV without a header: "llama-server.exe","1234","Console","1","900 K"
        columns = [cell.strip().strip('"') for cell in raw.strip().split('","')]
        if len(columns) >= 2 and columns[0].lower() == "llama-server.exe":
            pids.append(columns[1])
    return tuple(pids)


def running_servers(ttl: float = SERVERS_TTL) -> tuple[str, ...]:
    """`_server_pids`, at most once every `ttl` seconds."""
    global _servers_cache
    now = time.monotonic()
    when, pids = _servers_cache
    if now - when > ttl:
        pids = _server_pids()
        _servers_cache = (now, pids)
    return pids


__all__ = [
    "Cores",
    "auto_threads_http",
    "cores",
    "free_port",
    "hostname",
    "lan_address",
    "port_taken",
    "running_servers",
]
