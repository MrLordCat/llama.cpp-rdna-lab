"""The Models page: what a file will cost, before anything is started.

Every number here comes from a header and a device list, so the tests can
build both and assert on the exact sentence a person would read. The card is
constructed rather than discovered: a test that depends on what an old log
happened to say is a test that fails on someone else's machine.
"""

from __future__ import annotations

from pathlib import Path

from fasthtml.common import to_xml

from gui2.config import AppConfig
from gui2.core.devices import Device, Scan
from gui2.tests.fixtures import QWEN35_27B, write_gguf
from gui2.web import models_page
from gui2.web.models_page import View

#: the reference card of this lab, as its logs report it
CARD_MIB = 16304


def card(free_mib: int = CARD_MIB) -> Scan:
    return Scan(status="ready", devices=(
        Device(name="Vulkan0", description="RX 9070 XT", backend="vulkan",
               free_mib=free_mib, total_mib=CARD_MIB, source="log", confirmed=True),
    ))


def library(tmp_path: Path, *, weights_mib: int = 4) -> AppConfig:
    """A models directory holding one hybrid model and one projector."""
    folder = tmp_path / "models"
    folder.mkdir(exist_ok=True)
    write_gguf(folder / "qwen35-27b.gguf", architecture="qwen35", layers=65,
               context=262144, embedding=5120, hparams=QWEN35_27B,
               pad_to=weights_mib * 1024 * 1024)
    write_gguf(folder / "mmproj-model-f16.gguf", layers=24, context=None, embedding=1024)
    return AppConfig(data_root=tmp_path)


def render(config: AppConfig, scan: Scan, kv: str = "f16") -> str:
    return to_xml(models_page.results(config, scan, View(kv=kv)))


def test_a_model_the_card_can_hold_whole_says_so_and_what_is_left(tmp_path: Path):
    html = render(library(tmp_path), card(20000))

    # 16 attention layers x 1024 wide, K and V in f16: 16 GiB of cache at 256K,
    # which a 20 GiB card can hold along with the weights
    assert "all 256K" in html
    assert "to spare" in html


def test_a_card_with_room_for_only_part_of_the_context_says_how_much(tmp_path: Path):
    html = render(library(tmp_path), card())

    # the honest number, rounded down: promising 256K on a card that cannot
    # hold it is exactly the failure this page exists to prevent
    assert "up to 252K of its 256K" in html
    assert "all 256K" not in html


def test_a_cheaper_cache_type_turns_the_same_card_into_enough(tmp_path: Path):
    config = library(tmp_path)

    assert "up to 252K" in render(config, card(), "f16")
    assert "all 256K" in render(config, card(), "q4_0")


def test_a_model_too_big_to_load_says_by_how_much(tmp_path: Path):
    html = render(library(tmp_path, weights_mib=300), card(200))

    assert "too big to load" in html
    assert "up to" not in html


def test_the_price_of_context_is_stated_per_thousand_tokens(tmp_path: Path):
    config = library(tmp_path)

    # 16 layers x 2 x 1024 elements x 2 bytes x 1024 tokens
    assert "64 MiB per 1K" in render(config, card())
    assert "18 MiB per 1K" in render(config, card(), "q4_0")


def test_a_projector_is_not_offered_as_something_to_launch(tmp_path: Path):
    html = render(library(tmp_path), card())

    assert "loaded with a model, not on its own" in html
    assert "mmproj-model-f16.gguf" in html
    assert html.count("Set up") == 1, "only the model itself can be set up"
    # its vision tower has layers, but none of them hold a context
    assert html.count("per 1K") == 1


def test_a_conversion_that_kept_its_nextn_block_is_marked(tmp_path: Path):
    config = library(tmp_path)
    write_gguf(config.models / "qwen35-27b-outq6-mtp.gguf", architecture="qwen35",
               layers=65, context=262144, embedding=5120, hparams=QWEN35_27B)

    assert ">MTP<" in render(config, card())


def test_the_setup_link_hands_the_file_to_the_server_page(tmp_path: Path):
    html = render(library(tmp_path), card())

    assert 'href="/server?model=' in html
    assert "qwen35-27b.gguf" in html


def test_without_a_measured_card_the_page_says_so_rather_than_guessing(tmp_path: Path):
    html = render(library(tmp_path), Scan())

    assert "No card has been measured yet" in html
    assert "no card measured yet" in html
    # the header facts do not depend on a card, so they are still shown
    assert "64 MiB per 1K" in html
    assert "trained for 256K" in html


def test_a_split_model_is_one_row_priced_by_all_of_its_parts(tmp_path: Path):
    from gui2.tests.test_gguf import split_model

    config = library(tmp_path)
    split_model(config.models, parts=3, part_mib=1024)

    html = render(config, card(20000))

    assert "3 parts" in html
    assert "3.0 GiB" in html, "the size is the whole model, not its first part"
    assert "-00001-of-00003.gguf" in html
    assert "-00002-of-00003.gguf" not in html, "only the part llama.cpp accepts is listed"


def test_a_split_model_missing_a_part_is_not_priced_as_if_it_were_there(tmp_path: Path):
    from gui2.tests.test_gguf import split_model

    config = library(tmp_path)
    split_model(config.models, parts=3, part_mib=1024)
    (config.models / "big-IQ1_S-00003-of-00003.gguf").unlink()

    html = render(config, card(20000))

    assert "1 of its 3 parts is not here" in html
    assert "2.0 GiB" in html


def test_the_page_offers_the_cache_types_and_keeps_the_chosen_one(tmp_path: Path):
    config = library(tmp_path)

    html = to_xml(models_page.page(config, card(), {"kv": "q8_0"}))

    assert 'value="q8_0" selected' in html
    assert "half the size of f16" in html
    # an unknown value is not a reason to render nothing
    assert models_page.read_state({"kv": "nonsense"}).kv == "f16"
