"""Models: what is on disk, and what each one would cost to run.

A directory listing answers "is it there". The question worth a page is the
next one: will this file load on the cards this machine has, and how much
context is left once its weights are down. Both answers come from the GGUF
header and the device list already gathered from old logs -- no GPU is
touched, no binary is run, nothing is started.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

from fasthtml.common import (A, Div, Form, Label, Option, Select, Span, Table, Tbody, Td, Th,
                             Thead, Tr)

from gui2.config import AppConfig
from gui2.core.devices import Scan, pool
from gui2.core.gguf import ModelFacts, context_text, read_facts
from gui2.core.inventory import ModelFile, discover_models
from gui2.core.memory import Capacity, capacity, gib
from gui2.web.layout import shell

#: the KV cache types worth comparing; the rest are variations on these
CACHE_TYPES = ("f16", "q8_0", "q4_0")
CACHE_HELP = {
    "f16": "full precision, which is what llama-server uses unless told otherwise",
    "q8_0": "half the size of f16, at a quality difference nobody has managed to measure",
    "q4_0": "a quarter of the size, and the first thing to try when nothing else fits",
}

K = 1024


@dataclass(frozen=True, slots=True)
class View:
    """What the page is showing. Everything else is read off the disk."""

    kv: str = "f16"

    def query(self) -> str:
        return urlencode({"kv": self.kv})


def read_state(params) -> View:
    kv = (params.get("kv") or "f16").strip()
    return View(kv=kv if kv in CACHE_TYPES else "f16")


def _rounded(tokens: int) -> str:
    """Tokens as a person would say them, never rounded up.

    Rounding down matters here: a number in this column is a promise that the
    run will start, and 31 999 tokens presented as 32K is a broken one.
    """
    if tokens >= 2 * K:
        return context_text(tokens // K * K)
    return str(tokens)


def _tags(model: ModelFile, facts: ModelFacts) -> list:
    tags = []
    if model.is_mmproj:
        tags.append(Span("mmproj", cls="tag muted",
                         title="a vision projector — loaded alongside a model, "
                               "not instead of one"))
    if model.is_mtp or facts.nextn_layers:
        tags.append(Span("MTP", cls="tag best",
                         title="carries a NextN block, so it can draft its own tokens"))
    if model.is_split:
        tags.append(Span(f"{model.declared_parts} parts", cls="tag muted",
                         title="a split model — llama.cpp is given this part and "
                               "finds the rest itself"))
    if facts.full_attention_interval > 1:
        tags.append(Span("hybrid", cls="tag muted",
                         title=f"only every {facts.full_attention_interval}th layer keeps a "
                               f"KV cache, so context is unusually cheap here"))
    elif facts.sliding_window:
        tags.append(Span("windowed", cls="tag muted",
                         title=f"some layers only keep the last {facts.sliding_window} "
                               f"tokens, so the real cache is smaller than the estimate"))
    return tags


def _describe(facts: ModelFacts) -> str:
    if facts.error:
        return "header could not be read"
    return facts.summary or "no architecture in the header"


def _price(cap: Capacity, model: ModelFile) -> str:
    """What a thousand tokens of context costs on this model.

    A projector holds no context, so quoting it one would be inventing a
    number out of the vision tower's layer count.
    """
    if model.is_mmproj or not cap.known or cap.per_token_mib <= 0:
        return ""
    return f"{gib(cap.per_token_mib * K)} per 1K"


def _verdict(cap: Capacity, model: ModelFile) -> tuple[str, str]:
    """The sentence in the space column, and the class that colours it."""
    if model.is_mmproj:
        return "loaded with a model, not on its own", "muted"
    if model.missing_parts:
        missing = model.missing_parts
        return (f"⚠ {missing} of its {model.declared_parts} parts "
                f"{'is' if missing == 1 else 'are'} not here", "err")
    if not cap.known:
        return "", "muted"
    if not cap.room_mib:
        return "no card measured yet", "muted"
    if not cap.loads:
        return f"⚠ {gib(-cap.spare_mib)} too big to load", "err"
    if not cap.fits:
        return "⚠ loads, but with no room left for context", "err"
    if cap.whole:
        return f"all {context_text(cap.trained)}, {gib(cap.leftover_mib)} to spare", "good"
    return f"up to {_rounded(cap.context)} of its {context_text(cap.trained)}", "warn"


def _row(model: ModelFile, room: float, view: View) -> Tr:
    facts = read_facts(model.path)
    cap = capacity(facts, room, view.kv, view.kv)
    verdict, tone = _verdict(cap, model)
    setup = A("Set up →", href="/server?" + urlencode({"model": str(model.path)}), cls="crumb")
    return Tr(
        Td(Span(model.name, title=str(model.path)), *_tags(model, facts), cls="modelname"),
        Td(model.size_text, cls="num"),
        Td(_describe(facts), cls="muted"),
        Td(verdict, cls=tone),
        Td(_price(cap, model), cls="num muted"),
        Td("" if model.is_mmproj else setup, cls="label"),
    )


def _room_line(scan: Scan, view: View) -> Div:
    """What the numbers to the right were measured against."""
    total, parts, measured = pool(scan.local)
    if not parts:
        return Div("No card has been measured yet, so only what the headers say is shown. "
                   "Start a server once and this page can size every model against it.",
                   cls="hint block")
    basis = "free right now" if measured else "installed capacity"
    return Div(
        f"Sized against {' + '.join(parts)} — {gib(total)} {basis}, with every layer "
        f"offloaded, one conversation at a time and a KV cache in {view.kv}.",
        cls="hint block wide",
    )


def cache_hint(view: View, oob: bool = False):
    """The explanation beside the picker, which has to follow the picker."""
    return Span(CACHE_HELP[view.kv], cls="hint", id="kvhint",
                hx_swap_oob="true" if oob else None)


def results(config: AppConfig, scan: Scan, view: View) -> Div:
    models = discover_models(config.models)
    room, _parts, _measured = pool(scan.local)
    if models:
        body = Table(
            Thead(Tr(
                Th("Model"), Th("Size", cls="num"), Th("What it is"),
                Th("Context it has room for"), Th("Price of context", cls="num"), Th(""),
            )),
            Tbody(*[_row(model, room, view) for model in models]),
        )
    else:
        body = Div(f"No .gguf files in {config.models}", cls="muted")
    launchable = sum(1 for model in models if not model.is_mmproj)
    return Div(
        Div(f"{launchable} models in {config.models}", cls="muted"),
        _room_line(scan, view),
        Div(body, cls="table-wrap models"),
        id="modelrows",
    )


def picker(view: View) -> Form:
    return Form(
        Label(
            Span("KV cache type"),
            Select(
                *[Option(name, value=name, selected=name == view.kv, title=CACHE_HELP[name])
                  for name in CACHE_TYPES],
                name="kv",
            ),
            cls="field picker",
        ),
        cache_hint(view),
        hx_get="/models/rows", hx_target="#modelrows", hx_swap="outerHTML",
        hx_trigger="change", cls="row toolbar",
    )


def page(config: AppConfig, scan: Scan, params=None):
    view = read_state(params or {})
    return shell(
        "Models", "/models", config,
        Div(picker(view), results(config, scan, view), cls="panel"),
    )
