"""Standing up a worker on the other machine, and checking that it answered.

The RPC backend is the one part of llama.cpp that needs a second computer set
up by hand, and no GUI can do that for you: the command has to run over there.
So this does the next best thing. It writes the command out, correct for the
machine it is meant for, and afterwards says whether whatever is at that
address is really a worker, which protocol it speaks, and how much memory it
is offering -- the three things that otherwise only come out as a failed model
load ninety seconds into a launch.

Nothing here starts anything. `worker_command` returns text for a person to
paste. `probe` opens one socket, says hello, asks two questions and hangs up.

The wire format is llama.cpp's own, from ggml/src/ggml-rpc:

    request   | cmd (1 byte) | size (8 bytes) | payload |
    response  | size (8 bytes) | payload |

HELLO is pinned to command 14 by a static_assert in rpc_types.h precisely so
that a stranger can ask a worker who it is before agreeing on anything else.
Every other command number is only valid for a matching major version, so
nothing else is sent until the version has been read back.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass, field

#: the protocol this GUI knows how to read. Only the major number decides
#: whether the remaining commands mean what we think they mean.
KNOWN_PROTOCOL = (5, 0, 0)

#: rpc-server's own defaults, so the generated command matches the docs
DEFAULT_PORT = 50052

CMD_GET_DEVICE_MEMORY = 11
CMD_HELLO = 14           # pinned by static_assert; safe to send to any version
CMD_DEVICE_COUNT = 15

#: rpc_msg_hello_req is RPC_CONN_CAPS_SIZE bytes of capability flags. All zero
#: is "plain TCP, nothing to negotiate", which is what a bystander should say.
_CAPS_SIZE = 24
_HELLO_RSP = 4 + _CAPS_SIZE

#: one guess at a worker must not hold the page up
PROBE_TIMEOUT = 1.5

#: a worker with more devices than this is far likelier to be a wrong port
_MAX_DEVICES = 32


@dataclass(frozen=True, slots=True)
class RemoteDevice:
    """One device a worker is offering, as the worker measures it."""

    index: int
    free_bytes: int
    total_bytes: int

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def free_mib(self) -> float:
        return self.free_bytes / (1024 * 1024)


@dataclass(frozen=True, slots=True)
class Worker:
    """What one address turned out to be."""

    endpoint: str
    reachable: bool = False
    protocol: tuple[int, int, int] | None = None
    devices: tuple[RemoteDevice, ...] = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.reachable and self.protocol is not None and not self.error

    @property
    def compatible(self) -> bool:
        """Whether llama-server would accept this worker at all.

        Mirrors negotiate_hello: a different major number is refused outright.
        """
        return bool(self.protocol) and self.protocol[0] == KNOWN_PROTOCOL[0]

    @property
    def version_text(self) -> str:
        return ".".join(str(part) for part in self.protocol) if self.protocol else "unknown"

    @property
    def total_bytes(self) -> int:
        return sum(device.total_bytes for device in self.devices)

    @property
    def free_bytes(self) -> int:
        return sum(device.free_bytes for device in self.devices)


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    """The command to run on the other machine, and why each flag is there."""

    port: int = DEFAULT_PORT
    #: empty means "everything the machine has", which is rpc-server's default
    devices: tuple[str, ...] = ()
    open_to_network: bool = True
    #: on by default: the first run copies the model over the network, the
    #: next runs read it from the worker's own disk (%LOCALAPPDATA%\llama.cpp\rpc)
    cache: bool = True
    threads: int = 0
    #: the other machine's address as this machine reaches it; fills the boxes
    host: str = ""

    @property
    def address(self) -> str:
        """`host:port` — the one string the form's Worker addresses box wants."""
        return f"{self.host}:{self.port}" if self.host else ""

    def command(self, binary: str = "rpc-server") -> list[str]:
        argv = [binary]
        # 127.0.0.1 would only be reachable from the worker machine itself,
        # which is the one machine that does not need it
        argv += ["-H", "0.0.0.0" if self.open_to_network else "127.0.0.1"]
        argv += ["-p", str(self.port)]
        if self.devices:
            argv += ["-d", ",".join(self.devices)]
        if self.threads:
            argv += ["-t", str(self.threads)]
        if self.cache:
            argv += ["-c"]
        return argv

    def text(self, binary: str = "rpc-server") -> str:
        return " ".join(self.command(binary))


def worker_command(plan: WorkerPlan, binary: str = "rpc-server") -> str:
    return plan.text(binary)


def worker_bat(plan: WorkerPlan) -> str:
    """The one file to run on the other machine: opens the port, starts the worker.

    A .bat rather than prose, because the GUI cannot reach that machine and
    this is the whole setup a person has to do there: run it as Administrator,
    and nothing else. The firewall rule is added only on worker machines, so
    it is harmless to re-run — the rule is replaced, not duplicated.
    """
    argv = ["\"%~dp0rpc-server.exe\""]
    argv += ["-H", "0.0.0.0" if plan.open_to_network else "127.0.0.1", "-p", str(plan.port)]
    if plan.devices:
        argv += ["-d", ",".join(plan.devices)]
    if plan.threads:
        argv += ["-t", str(plan.threads)]
    if plan.cache:
        argv += ["-c"]
    lines = [
        "@echo off",
        f"REM Generated by GUI 2.0 - rpc-server worker on port {plan.port}",
        "net session >nul 2>&1",
        "if errorlevel 1 (echo Run this file as Administrator. & pause & exit /b 1)",
        "if not exist \"%~dp0rpc-server.exe\" (echo rpc-server.exe is missing - put this file next to it. & pause & exit /b 1)",
        f"netsh advfirewall firewall delete rule name=\"GUI 2.0 rpc-server {plan.port}\" >nul 2>&1",
        f"netsh advfirewall firewall add rule name=\"GUI 2.0 rpc-server {plan.port}\" dir=in action=allow protocol=TCP localport={plan.port} >nul",
        f"set \"PORT={plan.port}\"",
        "set \"ADDR=\"",
        "powershell -NoProfile -Command \"Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -First 1 -ExpandProperty IPAddress | Set-Content -NoNewline -Encoding ascii '%~dp0rpc-addr.txt'\" >nul 2>&1",
        "set /p \"ADDR=\" < \"%~dp0rpc-addr.txt\" >nul 2>&1",
        "del \"%~dp0rpc-addr.txt\" >nul 2>&1",
        "if not defined ADDR echo Could not find this machine's IPv4 address - run ipconfig and type it into the GUI manually.",
        "echo.",
        "echo Your address for the GUI is:",
        "echo   %ADDR%:%PORT%",
        "if defined ADDR echo(%ADDR%:%PORT%|clip",
        "if defined ADDR echo It is already on your clipboard: go to the GUI, open More than one machine and press Ctrl+V into the address box.",
        "echo.",
        "echo The port is open. This window must stay open while the GUI uses this machine -",
        "echo press Ctrl+C to stop the worker.",
        "echo.",
        *(["echo Cache is ON: the first model load copies the weights here,",
           "echo later loads read them from this machine\\'s disk - much faster."]
          if plan.cache else []),
        " ".join(argv),
        "pause",
    ]
    return "\r\n".join(lines) + "\r\n"


def _exchange(sock: socket.socket, cmd: int, payload: bytes, expect: int) -> bytes:
    """One request and its reply, or a ValueError describing what went wrong."""
    sock.sendall(bytes([cmd]) + struct.pack("<Q", len(payload)) + payload)
    header = _recv_exact(sock, 8)
    size = struct.unpack("<Q", header)[0]
    if size != expect:
        raise ValueError(f"expected {expect} bytes back, got {size}")
    return _recv_exact(sock, expect) if expect else b""


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ValueError("the worker closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def probe(endpoint: str, timeout: float = PROBE_TIMEOUT) -> Worker:
    """Ask one address who it is. One connection, three questions, goodbye.

    A worker serves one client at a time, so a probe sent while the worker is
    computing for someone else waits in the accept queue and times out here
    rather than interrupting anything. That is why this is never automatic:
    it runs when a person asks it to.
    """
    host, _, port = endpoint.rpartition(":")
    try:
        address = (host, int(port))
    except ValueError:
        return Worker(endpoint, error="not a host:port address")

    try:
        with socket.create_connection(address, timeout=timeout) as sock:
            sock.settimeout(timeout)
            reply = _exchange(sock, CMD_HELLO, bytes(_CAPS_SIZE), _HELLO_RSP)
            protocol = (reply[0], reply[1], reply[2])
            if protocol[0] != KNOWN_PROTOCOL[0]:
                # the command numbers below are only meaningful within a major
                # version; asking anything else would be guessing
                return Worker(endpoint, reachable=True, protocol=protocol)
            return Worker(endpoint, reachable=True, protocol=protocol,
                          devices=_devices(sock))
    except (TimeoutError, socket.timeout) as exc:  # noqa: UP041 - socket.timeout on 3.9
        return Worker(endpoint, error=f"no answer within {timeout:g}s ({exc or 'timed out'})")
    except ConnectionRefusedError:
        return Worker(endpoint, error="nothing is listening there")
    except (OSError, ValueError, struct.error, IndexError) as exc:
        return Worker(endpoint, reachable=True, error=f"answered, but not like a worker: {exc}")


def _devices(sock: socket.socket) -> tuple[RemoteDevice, ...]:
    count = struct.unpack("<I", _exchange(sock, CMD_DEVICE_COUNT, b"", 4))[0]
    found: list[RemoteDevice] = []
    for index in range(min(count, _MAX_DEVICES)):
        free, total = struct.unpack(
            "<QQ", _exchange(sock, CMD_GET_DEVICE_MEMORY, struct.pack("<I", index), 16))
        found.append(RemoteDevice(index=index, free_bytes=free, total_bytes=total))
    return tuple(found)


@dataclass(frozen=True, slots=True)
class Fleet:
    """Every configured endpoint, in the order --rpc lists them.

    That order is not decoration: llama.cpp names the first worker's devices
    RPC0, the second's RPC1 and so on, and -dev and -ts both refer to those
    names. Getting the order wrong sends the model to the wrong machine.
    """

    workers: tuple[Worker, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """RPC device names in --rpc order, one per device each worker offers."""
        names: list[str] = []
        for worker in self.workers:
            names += [f"RPC{len(names) + offset}" for offset in range(max(1, len(worker.devices)))]
        return tuple(names)

    def naming(self) -> list[tuple[str, Worker, RemoteDevice | None]]:
        """Each RPC name paired with the worker and device it stands for."""
        pairs: list[tuple[str, Worker, RemoteDevice | None]] = []
        for worker in self.workers:
            devices: tuple[RemoteDevice | None, ...] = worker.devices or (None,)
            for device in devices:
                pairs.append((f"RPC{len(pairs)}", worker, device))
        return pairs


def probe_all(endpoints: list[str], timeout: float = PROBE_TIMEOUT) -> Fleet:
    return Fleet(tuple(probe(endpoint, timeout) for endpoint in endpoints))


@dataclass(frozen=True, slots=True)
class Guide:
    """The two-machine recipe, as steps rather than as prose."""

    steps: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())


def guide(plan: WorkerPlan, this_machine: str = "") -> Guide:
    """What to do on the other machine, in the order it has to be done."""
    return Guide(
        steps=(
            f"Run the generated rpc-worker-{plan.port}.bat on the other machine as "
            f"Administrator. It opens the firewall, starts the worker, prints the "
            "machine's own address and copies it to the clipboard.",
            "Back here: press Ctrl+V into the address box (or press the button that "
            "puts it there), then press Check — it confirms whether the worker "
            "answered and how much memory it is offering.",
            "Tick the RPC0 row in the Devices list below, then start the server — or "
            "run the same search on the Autotune page.",
        ),
        warnings=(
            "The RPC backend has no authentication and no encryption: anyone who can "
            "reach that port can run code on that machine. Keep it on a network you "
            "trust, never on the open internet.",
            "Every layer placed on a remote device sends its activations over the "
            "network on each token. A gigabit link is the usual reason a two-machine "
            "setup is slower than one.",
        ),
    )


__all__ = [
    "DEFAULT_PORT",
    "Fleet",
    "Guide",
    "KNOWN_PROTOCOL",
    "RemoteDevice",
    "Worker",
    "WorkerPlan",
    "guide",
    "probe",
    "probe_all",
    "worker_bat",
    "worker_command",
]
