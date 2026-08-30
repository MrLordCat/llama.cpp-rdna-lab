"""Page shell and formatting helpers shared by every page."""

from __future__ import annotations

from pathlib import Path

from fasthtml.common import H1, A, Div, Header, Main, Nav, Span, Title

from gui2.config import AppConfig
from gui2.core.runspec import Problem

NAV: tuple[tuple[str, str], ...] = (
    ("/history", "History & Analytics"),
    ("/server", "Server"),
    ("/autotune", "Autotune"),
    ("/models", "Models"),
)

#: how a problem is shown: the prefix carries the weight, the class the colour
PROBLEM_STYLE: dict[str, tuple[str, str]] = {
    "error": ("⚠ ", "problem err"),
    "warn": ("⚠ ", "problem warn"),
    "note": ("note: ", "problem muted"),
}


def number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def problem_lines(problems: list[Problem]) -> list:
    return [Div(PROBLEM_STYLE[problem.level][0] + problem.message,
                cls=PROBLEM_STYLE[problem.level][1]) for problem in problems]


def command_lines(argv: list[str]) -> str:
    """A command line broken so each flag and its value sit on one line.

    Anything before the first flag stays on the first line with the program:
    an interpreter and the script it is given are one thought, not two.
    """
    lines: list[str] = []
    current = Path(argv[0]).name
    for token in argv[1:]:
        if token.startswith("-"):
            lines.append(current)
            current = "  " + token
        else:
            # an empty value is an argument, and one that means something:
            # left invisible the flag above it would read as having none
            current += f" {token}" if token else " ''"
    lines.append(current)
    return "\n".join(lines)


BALANCER_JS = """
function splitBars(box) {
    return Array.from(box.querySelectorAll(".splitbar"));
}
function splitShow(box) {
    // the percent label is the visible truth; the hidden box is what submits
    splitBars(box).forEach((bar) => {
        const slider = bar.querySelector("input[type=range]");
        const label = bar.querySelector(".share");
        if (label) label.textContent = Math.round((Number(slider.value) || 0) * 10) + "%";
    });
}
function splitWrite(box) {
    const field = box.querySelector("input[name=tensor_split]");
    if (field) field.value = splitBars(box)
        .map((bar) => bar.querySelector("input[type=range]").value).join(",");
}
function splitAuto(btn) {
    // back to llama.cpp deciding: equal bars, and an empty -ts list
    const box = btn.closest(".splitbalance");
    if (!box) return;
    const bars = splitBars(box);
    const each = Math.floor(10 / bars.length);
    bars.forEach((bar, i) => {
        bar.querySelector("input[type=range]").value =
            i === bars.length - 1 ? 10 - each * (bars.length - 1) : each;
    });
    const field = box.querySelector("input[name=tensor_split]");
    if (field) field.value = "";
    splitShow(box);
}
document.addEventListener("input", (event) => {
    const slider = event.target;
    if (!(slider instanceof HTMLInputElement) || slider.type !== "range") return;
    const box = slider.closest(".splitbalance");
    if (!box) return;
    const bars = splitBars(box);
    const i = bars.indexOf(slider.closest(".splitbar"));
    if (i < 0) return;
    const value = Math.max(0, Math.min(10, Number(slider.value) || 0));
    slider.value = String(value);
    let others = 0;
    bars.forEach((bar, j) => { if (j !== i) others += Number(bar.querySelector("input[type=range]").value) || 0; });
    const rest = 10 - value;
    bars.forEach((bar, j) => {
        if (j === i) return;
        const other = bar.querySelector("input[type=range]");
        const next = others > 0 ? Math.round((Number(other.value) || 0) * rest / others)
                                : Math.round(rest / (bars.length - 1));
        other.value = String(Math.max(0, Math.min(10, next)));
    });
    splitShow(box);
    splitWrite(box);
});
"""


def shell(title: str, active: str, config: AppConfig, *content,
          nav: dict[str, str] | None = None):
    """`nav` keeps the page the link comes from: Server's "Autotune" opens the
    sweep of what is on screen, Autotune's "Server" opens the same run, rather
    than two empty forms."""
    return (
        Title(f"GUI 2.0 — {title}"),
        Header(
            H1("llama.cpp RDNA lab — GUI 2.0"),
            Nav(*[A(label,
                    href=f"{href}?{nav[href]}" if nav and href in nav else href,
                    cls="active" if href == active else None)
                  for href, label in NAV]),
            Span(str(config.data_root), cls="path"),
            cls="top",
        ),
        Main(*content),
    )
