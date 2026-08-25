"""Models: read-only inventory of the configured models directory."""

from __future__ import annotations

from fasthtml.common import Div, Span, Table, Tbody, Td, Th, Thead, Tr

from gui2.config import AppConfig
from gui2.core.inventory import discover_models
from gui2.web.layout import shell


def page(config: AppConfig):
    models = discover_models(config.models)
    rows = [
        Tr(
            Td(model.name),
            Td(model.size_text, cls="num"),
            Td(Span("MTP", cls="tag best") if model.is_mtp else ""),
            Td(Span("mmproj", cls="tag muted") if model.is_mmproj else ""),
            Td(str(model.path), cls="muted label", title=str(model.path)),
        )
        for model in models
    ]
    body = Table(
        Thead(Tr(Th("Model"), Th("Size", cls="num"), Th(""), Th(""), Th("Path"))),
        Tbody(*rows),
    ) if rows else Div(f"No .gguf files in {config.models}", cls="muted")

    return shell(
        "Models", "/models", config,
        Div(
            Div(f"{len(models)} models in {config.models}", cls="muted"),
            Div(body, cls="table-wrap"),
            cls="panel",
        ),
    )
