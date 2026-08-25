"""Page shell and formatting helpers shared by every page."""

from __future__ import annotations

from fasthtml.common import H1, A, Header, Main, Nav, Span, Title

from gui2.config import AppConfig

NAV: tuple[tuple[str, str], ...] = (
    ("/history", "History & Analytics"),
    ("/server", "Server"),
    ("/models", "Models"),
)


def number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def shell(title: str, active: str, config: AppConfig, *content):
    return (
        Title(f"GUI 2.0 — {title}"),
        Header(
            H1("llama.cpp RDNA lab — GUI 2.0"),
            Nav(*[A(label, href=href, cls="active" if href == active else None) for href, label in NAV]),
            Span(str(config.data_root), cls="path"),
            cls="top",
        ),
        Main(*content),
    )
