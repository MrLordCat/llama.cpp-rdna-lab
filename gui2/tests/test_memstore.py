"""Remembering what a run cost, and what may honestly be inferred from it."""

from __future__ import annotations

import json
import time

from gui2.core.measured import parse_text
from gui2.core.memstore import MemoryStore, Record, rescaled

#: one real launch, trimmed to the four lines that carry the numbers
LAUNCH = (
    "load_tensors:      Vulkan0 model buffer size =  4615.84 MiB\n"
    "llama_kv_cache:    Vulkan0 KV buffer size =  1280.00 MiB\n"
    "common_memory_breakdown_print: | memory breakdown [MiB] | total free self model context compute unaccounted |\n"
    "common_memory_breakdown_print: |   - Vulkan0 (RX 9070 XT) | 16304 = 8817 +"
    " (6264 =  4615 +    1629 +      19) +        1221 |\n"
)

ARGV = ["D:/build/bin/llama-server.exe", "-m", "m.gguf", "-c", "32768", "-ngl", "999"]


def store_at(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.json")


def test_a_finished_run_comes_back_verbatim(tmp_path):
    store = store_at(tmp_path)
    store.remember(ARGV, parse_text(LAUNCH))

    measurement, record, exact = store.recall(ARGV)
    assert exact and record is not None and record.context == 32768
    assert measurement.vram[0].kv_mib == 1280.00
    assert measurement.vram[0].used_mib == 4615.84 + 1280.00 + 19


def test_it_survives_the_gui_being_restarted(tmp_path):
    store_at(tmp_path).remember(ARGV, parse_text(LAUNCH))

    # a second store, as after a restart, reads the same file
    measurement, record, exact = store_at(tmp_path).recall(ARGV)
    assert exact and record is not None
    assert measurement.vram[0].model_mib == 4615.84


def test_only_the_kv_cache_moves_with_the_context(tmp_path):
    store = store_at(tmp_path)
    store.remember(ARGV, parse_text(LAUNCH))

    doubled = list(ARGV)
    doubled[doubled.index("-c") + 1] = "65536"
    measurement, record, exact = store.recall(doubled)

    assert not exact and record is not None and record.context == 32768
    device = measurement.vram[0]
    assert device.kv_mib == 2560.00, "the KV cache is exactly proportional to the context"
    assert device.model_mib == 4615.84, "weights do not grow with the context"
    assert device.compute_mib == 19, "nor do the compute buffers, which follow the ubatch"


def test_a_different_run_is_not_answered_from_the_wrong_measurement(tmp_path):
    store = store_at(tmp_path)
    store.remember(ARGV, parse_text(LAUNCH))

    other = [*ARGV[:-2], "-ngl", "20"]           # fewer layers offloaded
    measurement, record, _exact = store.recall(other)
    assert record is None and not measurement.vram


def test_rebuilding_elsewhere_or_moving_the_port_keeps_the_answer(tmp_path):
    store = store_at(tmp_path)
    store.remember([*ARGV, "--port", "8080", "--api-key", "hunter2"], parse_text(LAUNCH))

    moved = ["E:/other-build/bin/llama-server.exe", *ARGV[1:], "--port", "9090"]
    _measurement, record, exact = store.recall(moved)
    assert exact and record is not None

    # and the key was never written down in the first place
    assert "hunter2" not in (tmp_path / "memory.json").read_text(encoding="utf-8")


def test_a_run_killed_during_load_does_not_overwrite_a_complete_one(tmp_path):
    store = store_at(tmp_path)
    store.remember(ARGV, parse_text(LAUNCH))
    partial = parse_text("load_tensors:      Vulkan0 model buffer size =  4615.84 MiB\n")
    assert not partial.complete

    store.remember(ARGV, partial)
    measurement, _record, _exact = store.recall(ARGV)
    assert measurement.vram[0].kv_mib == 1280.00, "the better answer stays"


def test_the_newest_measurement_of_the_same_command_wins(tmp_path):
    store = store_at(tmp_path)
    store.remember(ARGV, parse_text(LAUNCH))
    store.remember(ARGV, parse_text(LAUNCH.replace("1280.00", "1290.00")))

    measurement, _record, _exact = store.recall(ARGV)
    assert measurement.vram[0].kv_mib == 1290.00
    assert len(store.records()) == 1, "one entry per command, not one per launch"


def test_the_nearest_context_is_the_one_rescaled(tmp_path):
    store = store_at(tmp_path)
    for context, kv in ((16384, "640.00"), (131072, "5120.00")):
        argv = list(ARGV)
        argv[argv.index("-c") + 1] = str(context)
        store.remember(argv, parse_text(LAUNCH.replace("1280.00", kv)))

    wanted = list(ARGV)
    wanted[wanted.index("-c") + 1] = "98304"
    _measurement, record, exact = store.recall(wanted)
    assert not exact and record is not None and record.context == 131072


def test_a_damaged_file_is_no_history_rather_than_a_crash(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert MemoryStore(path).records() == []

    path.write_text(json.dumps({"schema": 99, "runs": []}), encoding="utf-8")
    assert MemoryStore(path).records() == []


def test_ages_are_said_in_units_a_person_uses():
    now = time.time()
    assert Record((), 0, (), at=now - 120).age_text == "2 minutes ago"
    assert Record((), 0, (), at=now - 7200).age_text == "2 hours ago"
    assert Record((), 0, (), at=now - 3 * 86400).age_text == "3 days ago"


def test_rescaling_to_the_same_context_changes_nothing():
    measurement = parse_text(LAUNCH)
    record = Record(family=("-m", "m.gguf"), context=32768, devices=measurement.devices,
                    complete=True, at=time.time())
    assert rescaled(record, 32768) == record.measurement
    assert rescaled(record, 0) == record.measurement
