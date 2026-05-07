"""Lightweight local HTTP API for GUI automation/e2e testing."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot


class _UiExecutor(QObject):
    """Bridge object to execute queued calls in Qt main thread."""

    trigger = pyqtSignal()

    def __init__(self, api_server: "GuiApiServer"):
        super().__init__()
        self.api_server = api_server
        self.trigger.connect(self.run_pending, Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def run_pending(self) -> None:
        self.api_server._run_pending_ui_call()


class GuiApiServer:
    """HTTP bridge that drives GUI actions and returns structured results."""

    def __init__(self, main_window, host: str = "127.0.0.1", port: int = 8765):
        self.main_window = main_window
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._executor = _UiExecutor(self)
        self._ui_lock = threading.Lock()
        self._ui_pending = None
        self._ui_done: threading.Event | None = None
        self._ui_out: dict[str, Any] = {}

    def start(self) -> None:
        if self._server is not None:
            return

        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args) -> None:
                return

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                body = self.rfile.read(length)
                if not body:
                    return {}
                try:
                    return json.loads(body.decode("utf-8"))
                except Exception:
                    return {}

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                try:
                    path = urlparse(self.path).path
                    if path == "/api/ping":
                        self._send(HTTPStatus.OK, {"ok": True, "service": "llama-gui-api"})
                        return
                    if path == "/api/state":
                        data = api.call_ui(api._state_snapshot)
                        self._send(HTTPStatus.OK, {"ok": True, "state": data})
                        return
                    self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not-found"})
                except Exception as exc:
                    self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": repr(exc)})

            def do_POST(self) -> None:
                try:
                    path = urlparse(self.path).path
                    payload = self._read_json()

                    if path == "/api/autotune":
                        self._send(HTTPStatus.OK, api.run_autotune(payload))
                        return

                    if path == "/api/apply-preset":
                        self._send(HTTPStatus.OK, api.apply_preset(payload))
                        return

                    if path == "/api/scenario/autotune-apply":
                        self._send(HTTPStatus.OK, api.scenario_autotune_apply(payload))
                        return

                    self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not-found"})
                except Exception as exc:
                    self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": repr(exc)})

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    def call_ui(self, fn, timeout_sec: float = 10.0):
        done = threading.Event()
        with self._ui_lock:
            self._ui_pending = fn
            self._ui_done = done
            self._ui_out = {}
        self._executor.trigger.emit()

        if not done.wait(timeout=timeout_sec):
            raise TimeoutError(f"ui call timeout after {timeout_sec}s")
        with self._ui_lock:
            out = dict(self._ui_out)
        if "error" in out:
            raise RuntimeError(out["error"])
        return out.get("result")

    def _run_pending_ui_call(self) -> None:
        with self._ui_lock:
            fn = self._ui_pending
            done = self._ui_done
        if fn is None or done is None:
            return
        try:
            result = fn()
            with self._ui_lock:
                self._ui_out = {"result": result}
        except Exception as exc:
            with self._ui_lock:
                self._ui_out = {"error": repr(exc)}
        finally:
            done.set()
            with self._ui_lock:
                self._ui_pending = None
                self._ui_done = None

    def _state_snapshot(self) -> dict[str, Any]:
        server_tab = self.main_window.server_tab
        return {
            "model_path": server_tab.server_model_path.text().strip(),
            "context": server_tab.server_context_spinbox.value(),
            "batch": server_tab.server_batch_spinbox.value(),
            "ubatch": server_tab.server_ubatch_spinbox.value(),
            "parallel": server_tab.server_parallel_spinbox.value(),
            "kv": server_tab.server_kv_type_combo.currentText(),
            "status": server_tab.server_status_label.text(),
            "build_status": self.main_window.build_tab.build_status_label.text(),
        }

    def run_autotune(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_path = str(payload.get("model_path") or "").strip() or None
        wait = bool(payload.get("wait", True))
        timeout_sec = float(payload.get("timeout_sec", 5400.0))
        sweep_mode = str(payload.get("sweep_mode") or "full").strip().lower()

        done = threading.Event()
        result: dict[str, Any] = {}

        def start_on_ui() -> dict[str, Any]:
            def on_done(success: bool, info: dict[str, Any]) -> None:
                result.update(info)
                result["success"] = bool(success)
                done.set()

            started = self.main_window.build_tab.run_large_context_autotune(
                model_path=model_path,
                silent=True,
                completion_callback=on_done,
                sweep_mode=sweep_mode,
            )
            return {"started": bool(started)}

        start_info = self.call_ui(start_on_ui)
        if not start_info.get("started"):
            return {"ok": False, "error": "autotune-not-started", "result": result}

        if not wait:
            return {"ok": True, "started": True, "waiting": False}

        if not done.wait(timeout=timeout_sec):
            return {"ok": True, "started": True, "waiting": True, "status": "running"}

        return {"ok": bool(result.get("success", False)), "started": True, "result": result}

    def apply_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_path = str(payload.get("model_path") or "").strip() or None

        def apply_on_ui() -> dict[str, Any]:
            server_tab = self.main_window.server_tab
            if model_path:
                server_tab.server_model_path.setText(model_path)
            result = server_tab.apply_model_file_preset() or {}
            return result

        result = self.call_ui(apply_on_ui)
        return {"ok": bool(result.get("matched", False)), "result": result}

    def scenario_autotune_apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        auto = self.run_autotune(payload)
        if not auto.get("ok"):
            return {"ok": False, "stage": "autotune", "autotune": auto}

        model_path = str(payload.get("model_path") or "").strip() or None
        preset = self.apply_preset({"model_path": model_path})
        if not preset.get("ok"):
            return {"ok": False, "stage": "apply-preset", "autotune": auto, "preset": preset}

        state = self.call_ui(self._state_snapshot)
        return {
            "ok": True,
            "autotune": auto,
            "preset": preset,
            "state": state,
        }
