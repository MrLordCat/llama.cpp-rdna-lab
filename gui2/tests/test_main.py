"""The launcher's port helpers: a second instance must join, not fight.

A click on the desktop shortcut while the GUI already runs used to end in a
uvicorn bind traceback inside a terminal that closed instantly, which reads as
"the GUI did not start". `python -m gui2` now notices the running instance and
only opens the browser, and it opens that browser only once the port answers.
"""

from __future__ import annotations

import socket

from gui2.__main__ import open_when_ready, port_in_use, probe_host, wait_for_port


def _listen() -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    return srv


def _free_port() -> int:
    srv = _listen()
    port = srv.getsockname()[1]
    srv.close()
    return port


def test_probe_host_maps_wildcards_to_loopback():
    assert probe_host("0.0.0.0") == "127.0.0.1"
    assert probe_host("") == "127.0.0.1"
    assert probe_host("::") == "127.0.0.1"
    assert probe_host("127.0.0.1") == "127.0.0.1"
    assert probe_host("192.168.1.5") == "192.168.1.5"


def test_port_in_use_is_false_for_a_free_port():
    assert not port_in_use("127.0.0.1", _free_port())


def test_port_in_use_sees_a_listening_socket():
    with _listen() as srv:
        assert port_in_use("127.0.0.1", srv.getsockname()[1])


def test_wait_for_port_returns_once_something_listens():
    with _listen() as srv:
        assert wait_for_port("127.0.0.1", srv.getsockname()[1], timeout=2.0)


def test_wait_for_port_gives_up_on_a_dead_port():
    assert not wait_for_port("127.0.0.1", _free_port(), timeout=0.4, interval=0.1)


def test_open_when_ready_opens_the_url(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("gui2.__main__.webbrowser.open", opened.append)
    with _listen() as srv:
        thread = open_when_ready("http://127.0.0.1:1/history", "127.0.0.1",
                                 srv.getsockname()[1], timeout=2.0)
        thread.join(timeout=5.0)
    assert opened == ["http://127.0.0.1:1/history"]


def test_open_when_ready_stays_silent_when_nothing_listens(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("gui2.__main__.webbrowser.open", opened.append)
    thread = open_when_ready("http://127.0.0.1:1/history", "127.0.0.1",
                             _free_port(), timeout=0.4)
    thread.join(timeout=5.0)
    assert opened == []
