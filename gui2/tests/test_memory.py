"""The VRAM estimate, pinned to sizes real runs printed.

The reference is a Qwen3.8-27B run at 163 840 tokens, whose log reports

    llama_kv_cache: size = 8960.00 MiB (163840 cells, 16 layers, 1/1 seqs)
    llama_memory_recurrent: size = 149.63 MiB (65 layers, 1 seqs)

with the last twelve KV layers held in f16 and the rest in f8_e4m3. The whole
point of these tests is that a formula change has to keep matching that log.
"""

from __future__ import annotations

from pathlib import Path

from gui2.core.gguf import read_facts
from gui2.core.memory import (
    CONTEXT_PAD,
    MIB,
    capacity,
    context_for_budget,
    estimate,
    kv_bytes,
    layer_split,
    state_bytes,
    weight_bytes,
)
from gui2.core.runspec import DEFAULTS
from gui2.tests.fixtures import QWEN35_27B, write_gguf

REFERENCE_CTX = 163840


def qwen35(tmp_path: Path, **kwargs) -> Path:
    return write_gguf(tmp_path / "qwen35.gguf", architecture="qwen35", layers=65,
                      context=262144, embedding=5120, hparams=QWEN35_27B, **kwargs)


def test_a_hybrid_model_gives_a_kv_cache_to_every_fourth_layer(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path))

    # 65 blocks, one of them a NextN head; 64 main layers, every 4th keeps KV
    assert layer_split(facts) == (16, 48)


def test_the_kv_cache_is_the_size_the_run_reported(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path))

    # the logged run held 12 of its 16 layers in f16 and 4 in f8_e4m3
    mixed = (12 * kv_bytes(facts, REFERENCE_CTX, "f16", "f16")
             + 4 * kv_bytes(facts, REFERENCE_CTX, "f8_e4m3", "f8_e4m3")) / 16
    assert round(mixed / MIB) == 8960
    assert round(kv_bytes(facts, REFERENCE_CTX) / MIB) == 10240


def test_the_recurrent_state_is_the_size_the_run_reported(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path))

    assert round(state_bytes(facts, 1) / MIB, 2) == 149.62
    # it is per sequence and, unlike the KV cache, indifferent to the context
    assert state_bytes(facts, 4) == 4 * state_bytes(facts, 1)


def test_a_cheaper_cache_type_is_cheaper_in_proportion(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path))
    full = kv_bytes(facts, 32768, "f16", "f16")

    assert kv_bytes(facts, 32768, "q8_0", "q8_0") / full == 34 / 64
    assert kv_bytes(facts, 32768, "q4_0", "q4_0") / full == 18 / 64
    # mixing the two is allowed, and lands in between
    assert full > kv_bytes(facts, 32768, "f16", "q8_0") > kv_bytes(facts, 32768, "q8_0", "q8_0")


def test_a_plain_model_caches_every_layer(tmp_path: Path):
    facts = read_facts(write_gguf(tmp_path / "plain.gguf", layers=32, embedding=4096,
                                  hparams={"attention.head_count": 32,
                                           "attention.head_count_kv": 8,
                                           "attention.key_length": 128,
                                           "attention.value_length": 128}))

    assert layer_split(facts) == (32, 0)
    assert state_bytes(facts) == 0
    # 32 layers x 4096 tokens x 8 heads x 128 wide x 2 bytes, K and V
    assert kv_bytes(facts, 4096) / MIB == 32 * 4096 * 8 * 128 * 2 * 2 / MIB


def test_a_partly_offloaded_model_only_pays_for_what_is_on_the_gpu(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path, pad_to=66 * 1024 * 1024))

    assert weight_bytes(facts, True, 0) == facts.file_bytes
    # -ngl counts the output layer too, so 33 of 66 is half the file
    assert weight_bytes(facts, False, 33) == facts.file_bytes / 2
    assert weight_bytes(facts, False, 999) == facts.file_bytes


def test_the_context_that_fits_is_the_inverse_of_the_cache_size(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path))
    room = kv_bytes(facts, 32768) / MIB

    assert context_for_budget(facts, room) == 32768
    assert context_for_budget(facts, room, "q8_0", "q8_0") > 32768
    assert context_for_budget(facts, 0) == 0


def test_the_estimate_adds_up_and_says_what_a_hybrid_model_costs(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path, pad_to=4 * 1024 * 1024))
    spec = DEFAULTS.with_values({"ctx_size": 32768, "parallel": 2})

    report = estimate(spec, facts, devices=2)

    assert report.complete
    assert report.total_mib == sum(term.mib for term in report.terms)
    assert [term.label for term in report.terms] == [
        "Weights", "KV cache", "Recurrent state", "Compute buffers"]
    assert any("every 4th layer" in note for note in report.notes)


def test_without_a_model_the_estimate_says_so_instead_of_guessing():
    report = estimate(DEFAULTS, None)

    assert not report.complete
    assert report.total_mib == 0
    assert report.notes


def test_capacity_spends_what_the_weights_leave_and_no_more(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path, pad_to=4 * MIB))
    room = 16304.0

    cap = capacity(facts, room)

    assert cap.known and cap.loads
    assert cap.spare_mib == room - cap.weights_mib - cap.fixed_mib
    # every token has a price, and the context is what the spare room buys
    assert cap.context * cap.per_token_mib <= cap.spare_mib
    assert (cap.context + CONTEXT_PAD) * cap.per_token_mib > cap.spare_mib
    assert cap.context % CONTEXT_PAD == 0, "llama.cpp pads the context to 256 tokens"


def test_capacity_never_promises_more_context_than_the_model_was_trained_for(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path, pad_to=4 * MIB))

    cap = capacity(facts, 200_000.0)

    assert cap.whole and cap.context == facts.n_ctx_train
    # and the room the model cannot use is reported as still free
    assert round(cap.leftover_mib) == round(cap.spare_mib - cap.context * cap.per_token_mib)


def test_capacity_reports_a_model_that_cannot_load_rather_than_a_context_of_zero(tmp_path: Path):
    facts = read_facts(qwen35(tmp_path, pad_to=64 * MIB))

    cap = capacity(facts, 32.0)

    assert cap.known and not cap.loads and not cap.fits
    assert cap.spare_mib < 0


def test_capacity_of_an_unreadable_header_claims_nothing():
    cap = capacity(None, 16304.0)

    assert not cap.known and not cap.loads and cap.context == 0
