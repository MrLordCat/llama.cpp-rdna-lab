"""Controls the pages share, so that the same idea looks the same everywhere."""

from __future__ import annotations

from fasthtml.common import Input, Label, Span


def toggle(name: str, text: str, on: bool, value: str = "", title: str = "", **attrs) -> Label:
    """A checkbox as a button: the word is the control, not a note beside it.

    The pressed look comes from `:checked` in CSS rather than from a class
    decided here, so it survives a click the server has not answered yet.
    """
    box: dict = {"type": "checkbox", "name": name}
    if value:
        box["value"] = value
    box["checked"] = on
    return Label(Input(**box), Span(text), cls="chip", title=title, **attrs)
