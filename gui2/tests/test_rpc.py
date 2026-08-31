"""The RPC worker check, against a worker made of nothing but a socket.

No rpc-server is started here and no GPU is touched: the fake below speaks
the same eleven bytes of handshake, which is the whole of what the check
depends on. If llama.cpp ever changes that handshake, these tests keep
passing and the real thing stops working -- so the numbers below are quoted
from ggml/src/ggml-rpc/rpc_types.h and are the thing to re-read when a probe
starts saying "answered, but not like a worker".
"""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from gui2.core.rpc import (
    DEFAULT_PORT,
    Fleet,
    KNOWN_PROTOCOL,
    RemoteDevice,
    Worker,
    WorkerPlan,
    guide,
    probe,
    worker_bat,
)

GIB = 1024 ** 3


class FakeWorker(threading.Thread):
    """Answers HELLO, DEVICE_COUNT and GET_DEVICE_MEMORY, then hangs up."""

    def __init__(self, version=KNOWN_PROTOCOL, devices=((8 * GIB, 16 * GIB),), mute=False):
        super().__init__(daemon=True)
        self.version = version
        self.devices = devices
        self.mute = mute            # accepts the connection and says nothing
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.endpoint = "127.0.0.1:%d" % self.listener.getsockname()[1]

    def run(self) -> None:
        try:
            client, _ = self.listener.accept()
        except OSError:
            return
        with client:
            if self.mute:
                client.recv(64)
                return
            while True:
                head = client.recv(9)
                if len(head) < 9:
                    return
                cmd = head[0]
                size = struct.unpack("<Q", head[1:])[0]
                payload = client.recv(size) if size else b""
                reply = self._reply(cmd, payload)
                if reply is None:
                    return
                client.sendall(struct.pack("<Q", len(reply)) + reply)

    def _reply(self, cmd: int, payload: bytes) -> bytes | None:
        if cmd == 14:                                    # HELLO
            major, minor, patch = self.version
            return bytes([major, minor, patch, 0]) + bytes(24)
        if cmd == 15:                                    # DEVICE_COUNT
            return struct.pack("<I", len(self.devices))
        if cmd == 11:                                    # GET_DEVICE_MEMORY
            index = struct.unpack("<I", payload)[0]
            free, total = self.devices[index]
            return struct.pack("<QQ", free, total)
        return None

    def close(self) -> None:
        self.listener.close()


@pytest.fixture
def worker():
    fake = FakeWorker()
    fake.start()
    yield fake
    fake.close()


def test_a_worker_says_its_version_and_what_it_is_offering(worker):
    found = probe(worker.endpoint, timeout=2.0)

    assert found.ok and found.compatible
    assert found.version_text == "5.0.0"
    assert found.devices == (RemoteDevice(index=0, free_bytes=8 * GIB, total_bytes=16 * GIB),)
    assert found.devices[0].total_mib == 16384


def test_nothing_listening_is_said_plainly():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()                    # the port is now certainly free

    found = probe(f"127.0.0.1:{port}", timeout=1.0)
    assert not found.ok and not found.reachable
    assert "nothing is listening" in found.error or "no answer" in found.error


def test_a_different_major_version_is_reported_and_nothing_else_is_asked():
    fake = FakeWorker(version=(4, 9, 0))
    fake.start()
    try:
        found = probe(fake.endpoint, timeout=2.0)
    finally:
        fake.close()

    assert found.reachable and found.version_text == "4.9.0"
    # llama-server refuses a major mismatch outright, and so must the report
    assert not found.compatible
    # the command numbers only mean something inside a major version
    assert found.devices == ()


def test_something_that_is_not_a_worker_is_not_mistaken_for_one():
    fake = FakeWorker(mute=True)
    fake.start()
    try:
        found = probe(fake.endpoint, timeout=0.4)
    finally:
        fake.close()

    assert not found.ok
    assert found.error


def test_the_worker_command_binds_where_the_other_machine_can_reach_it():
    plan = WorkerPlan(devices=("Vulkan0",))
    assert plan.text() == f"rpc-server -H 0.0.0.0 -p {DEFAULT_PORT} -d Vulkan0"

    # 127.0.0.1 would only be reachable from the worker itself
    assert "-H 127.0.0.1" in WorkerPlan(open_to_network=False).text()
    assert WorkerPlan(port=50055, cache=True, threads=4).text().endswith("-t 4 -c")
    assert "-d" not in WorkerPlan().command(), "no -d means every device it has"


def test_rpc_names_follow_the_order_the_workers_are_listed_in():
    fleet = Fleet((
        Worker("a:1", reachable=True, protocol=KNOWN_PROTOCOL, devices=(
            RemoteDevice(0, GIB, GIB), RemoteDevice(1, GIB, GIB))),
        Worker("b:2", reachable=True, protocol=KNOWN_PROTOCOL, devices=(
            RemoteDevice(0, GIB, GIB),)),
    ))
    assert fleet.names == ("RPC0", "RPC1", "RPC2")

    naming = fleet.naming()
    assert [name for name, _worker, _device in naming] == ["RPC0", "RPC1", "RPC2"]
    assert naming[2][1].endpoint == "b:2", "swapping the two would send layers to the wrong machine"


def test_an_unreachable_worker_still_claims_its_place_in_the_numbering():
    # it has to: the names are positional, so a worker that is down does not
    # renumber the ones after it, it only fails to answer
    fleet = Fleet((Worker("a:1", error="nothing is listening"),
                   Worker("b:2", reachable=True, protocol=KNOWN_PROTOCOL,
                          devices=(RemoteDevice(0, GIB, GIB),))))
    assert fleet.names == ("RPC0", "RPC1")


def test_the_guide_says_the_dangerous_part_out_loud():
    said = " ".join(guide(WorkerPlan()).warnings).lower()
    assert "no authentication" in said
    assert "network" in said


def test_a_worker_bat_opens_the_port_and_starts_the_worker():
    bat = worker_bat(WorkerPlan(port=50052, devices=("Vulkan0",), cache=True))
    assert "@echo off" in bat and "\r\n" in bat
    assert f"localport=50052" in bat, "the firewall rule is part of the file"
    assert "rpc-server.exe\" -H 0.0.0.0 -p 50052 -d Vulkan0 -c" in bat
    assert "Run this file as Administrator" in bat or "Administrator" in bat
    # a closed worker still leaves the firewall alone: 127.0.0.1 only binds
    assert "-H 127.0.0.1" in worker_bat(WorkerPlan(open_to_network=False))


def test_a_plan_knows_the_address_the_local_box_wants():
    assert WorkerPlan(host="192.168.1.60", port=50052).address == "192.168.1.60:50052"
    assert WorkerPlan().address == ""


def test_a_worker_with_two_gpus_takes_two_of_the_names():
    """The count is positional, so guessing one per address renumbers the rest."""
    from gui2.core.devices import rpc_entries
    from gui2.core.rpc import Fleet, RemoteDevice, Worker

    endpoints = ["a:1", "b:2"]

    # before any check: one name per address, and it is wrong
    guessed = rpc_entries(endpoints)
    assert [device.name for device in guessed] == ["RPC0", "RPC1"]

    fleet = Fleet((
        Worker("a:1", reachable=True, protocol=KNOWN_PROTOCOL, devices=(
            RemoteDevice(0, 15 * GIB, 16 * GIB), RemoteDevice(1, 7 * GIB, 8 * GIB))),
        Worker("b:2", reachable=True, protocol=KNOWN_PROTOCOL, devices=(
            RemoteDevice(0, 23 * GIB, 24 * GIB),)),
    ))
    known = rpc_entries(endpoints, fleet)
    assert [device.name for device in known] == ["RPC0", "RPC1", "RPC2"]
    assert known[2].description.startswith("b:2")
    assert known[0].free_mib == 15 * 1024 and known[0].total_mib == 16 * 1024
    assert all(device.confirmed for device in known)
