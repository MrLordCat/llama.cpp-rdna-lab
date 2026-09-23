"""What a page remembers about itself: a form, as the query that reproduces it."""

from __future__ import annotations

import json

from gui2.config import AppConfig
from gui2.core.form_state import SCHEMA, FormStateStore

def test_nothing_remembered_yet_reads_as_empty(tmp_path):
    store = FormStateStore(tmp_path / "server-state.json")
    assert store.query() == ""

def test_a_form_comes_back_after_a_restart(tmp_path):
    path = tmp_path / "server-state.json"
    FormStateStore(path).remember("model=m.gguf&ctx_size=32768&devices=ROCm1&devices=ROCm0")

    # a second store, as after the GUI is restarted, reads the same file
    assert FormStateStore(path).query() == "model=m.gguf&ctx_size=32768&devices=ROCm1&devices=ROCm0"

def test_an_empty_form_is_not_written_down(tmp_path):
    path = tmp_path / "server-state.json"
    FormStateStore(path).remember("model=m.gguf")
    FormStateStore(path).remember("")

    assert FormStateStore(path).query() == "model=m.gguf"

def test_a_broken_file_means_defaults_rather_than_a_crash(tmp_path):
    path = tmp_path / "server-state.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert FormStateStore(path).query() == ""

    path.write_text(json.dumps({"schema": SCHEMA + 1, "query": "model=m.gguf"}), encoding="utf-8")
    assert FormStateStore(path).query() == "", "an older or newer schema is not guessed at"

    path.write_text(json.dumps(["model=m.gguf"]), encoding="utf-8")
    assert FormStateStore(path).query() == ""

def test_the_file_is_written_whole_not_in_place(tmp_path):
    """A reader must never catch half a form, so it is renamed into place."""
    path = tmp_path / "server-state.json"
    FormStateStore(path).remember("model=m.gguf")

    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA
    assert not list(tmp_path.glob("*.tmp")), "the temporary file is gone after the write"

def test_the_state_lives_beside_the_other_gui_state(tmp_path):
    """The path comes from the config, so Windows and Linux write it the same."""
    config = AppConfig(data_root=tmp_path)
    assert config.server_state_json == tmp_path / "build_logs" / "gui2" / "server-state.json"
    assert config.autotune_state_json.parent == config.server_state_json.parent
