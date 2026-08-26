"""GGUF header reading and the slider range it feeds."""

from __future__ import annotations

from pathlib import Path

from gui2.core.gguf import context_text, read_facts
from gui2.core.params import BY_NAME, context_bounds
from gui2.tests.fixtures import QWEN35_27B, write_gguf


def test_facts_come_from_the_header(tmp_path: Path):
    facts = read_facts(write_gguf(tmp_path / "qwen.gguf"))

    assert facts.architecture == "qwen3"
    assert facts.n_layers == 48
    assert facts.n_ctx_train == 40960
    assert facts.known
    assert "40K" in facts.summary


def test_the_tokenizer_list_gives_the_vocabulary_and_ends_the_read(tmp_path: Path):
    facts = read_facts(write_gguf(tmp_path / "qwen.gguf", vocab=1024, hparams=QWEN35_27B))

    assert facts.n_vocab == 1024
    # every architecture key is written before the token list and must survive it
    assert facts.n_head_kv == 4
    assert facts.full_attention_interval == 4
    assert facts.ssm_inner == 6144


def test_the_attention_shape_falls_back_to_what_can_be_derived(tmp_path: Path):
    facts = read_facts(write_gguf(tmp_path / "plain.gguf", embedding=4096,
                                  hparams={"attention.head_count": 32}))

    # no key_length and no head_count_kv: one head's width is n_embd / n_head,
    # and without grouped attention every head keeps its own K and V
    assert facts.head_dim_k == 128
    assert facts.n_embd_k_gqa == 4096


def test_a_file_that_is_not_a_gguf_reports_instead_of_raising(tmp_path: Path):
    plain = tmp_path / "notes.gguf"
    plain.write_bytes(b"not a model")
    facts = read_facts(plain)

    assert not facts.known
    assert facts.error
    assert not read_facts(tmp_path / "missing.gguf").known


def test_rereading_a_rewritten_file_sees_the_new_header(tmp_path: Path):
    path = write_gguf(tmp_path / "m.gguf", context=32768)
    assert read_facts(path).n_ctx_train == 32768

    # a re-quantized file keeps its name; the cache must not answer for it
    write_gguf(path, architecture="qwen3moe", context=131072, layers=64)
    facts = read_facts(path)
    assert facts.n_ctx_train == 131072
    assert facts.n_layers == 64


def test_context_slider_stops_at_what_the_model_was_trained_for():
    assert context_bounds(40960)[1] == 40960
    assert context_bounds(32768)[1] == 32768
    # an unknown model falls back to the schema's own ceiling
    assert context_bounds(None)[1] == BY_NAME["ctx_size"].maximum


def test_context_slider_can_reach_both_ends_of_its_range():
    for trained in (2048, 4096, 8192, 32768, 40960, 131072, 262144, 1010000):
        low, high, step = context_bounds(trained)
        assert low <= high <= trained
        assert (high - low) % step == 0, "the model's limit must land on the step ladder"
        assert low >= step
        # a short model must not be given a range that starts above what it has
        assert low <= trained


def test_context_text_reads_as_a_size():
    assert context_text(131072) == "128K"
    assert context_text(40960) == "40K"
    assert context_text(1500) == "1500"
