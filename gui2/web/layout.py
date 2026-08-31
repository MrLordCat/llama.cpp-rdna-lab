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
    // the share label is the visible truth; the hidden box is what submits.
    // Values are relative weights (100,100,100 is a perfect third each), so
    // the label reads weight over the sum rather than the raw number.
    const bars = splitBars(box);
    const total = bars.reduce((sum, bar) =>
        sum + (Number(bar.querySelector("input[type=range]").value) || 0), 0);
    bars.forEach((bar) => {
        const slider = bar.querySelector("input[type=range]");
        const label = bar.querySelector(".share");
        if (label) {
            const v = Number(slider.value) || 0;
            label.textContent = total > 0 ? Math.round(v / total * 100) + "%" : "0%";
        }
    });
}
function splitWrite(box) {
    const field = box.querySelector("input[name=tensor_split]");
    if (field) field.value = splitBars(box)
        .map((bar) => bar.querySelector("input[type=range]").value).join(",");
}
function splitEqual(btn) {
    // one share per card, however many there are: 100,100,100 divides evenly
    const box = btn.closest(".splitbalance");
    if (!box) return;
    splitBars(box).forEach((bar) => { bar.querySelector("input[type=range]").value = "100"; });
    const field = box.querySelector("input[name=tensor_split]");
    if (field) field.value = splitBars(box)
        .map((bar) => bar.querySelector("input[type=range]").value).join(",");
    splitShow(box);
}
function splitAuto(btn) {
    // back to llama.cpp deciding: equal-looking bars, and an empty -ts list
    const box = btn.closest(".splitbalance");
    if (!box) return;
    splitBars(box).forEach((bar) => { bar.querySelector("input[type=range]").value = "100"; });
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
    const value = Math.max(0, Math.min(100, Number(slider.value) || 0));
    slider.value = String(value);
    let others = 0;
    bars.forEach((bar, j) => { if (j !== i) others += Number(bar.querySelector("input[type=range]").value) || 0; });
    const rest = 100 - value;
    bars.forEach((bar, j) => {
        if (j === i) return;
        const other = bar.querySelector("input[type=range]");
        const next = others > 0 ? Math.round((Number(other.value) || 0) * rest / others)
                                : Math.round(rest / (bars.length - 1));
        other.value = String(Math.max(0, Math.min(100, next)));
    });
    splitShow(box);
    splitWrite(box);
});
"""

#: dragging a card to the top makes it first in -dev (and its split bar follows)
DEVICE_ORDER_JS = """
function deviceOrderSync(devlist) {
    // the split bars pair with the device rows by name; drag one and the other
    // follows, so -ts never points at the wrong card
    const names = Array.from(devlist.querySelectorAll(".devrow"))
        .map((row) => row.querySelector(".devname")?.textContent?.trim() || "");
    document.querySelectorAll(".splitbalance").forEach((box) => {
        const place = box.querySelector(".splitbars");
        if (!place) return;
        const byName = new Map(Array.from(box.querySelectorAll(".splitbar")).map((bar) => {
            const input = bar.querySelector("input[name^=split_]");
            return [input ? input.name.slice("split_".length) : "", bar];
        }));
        names.forEach((name) => { const bar = byName.get(name); if (bar) place.appendChild(bar); });
        splitWrite(box);
        splitShow(box);
    });
    // the "-dev a,b,c" line under the list follows the rows, not the alphabet
    const hint = devlist.nextElementSibling;
    const chosen = Array.from(devlist.querySelectorAll("input[name=devices]:checked"))
        .map((input) => input.value);
    if (hint && hint.classList.contains("hint") && chosen.length) {
        hint.textContent = "-dev " + chosen.join(",");
    }
}
function deviceDrag(root) {
    let dragged = null;
    root.addEventListener("dragstart", (event) => {
        const row = event.target.closest(".devrow");
        if (!row) return;
        dragged = row;
        row.classList.add("grabbing");
        event.dataTransfer.effectAllowed = "move";
    });
    root.addEventListener("dragover", (event) => {
        const row = event.target.closest(".devrow");
        if (!row || row === dragged) return;
        event.preventDefault();
        const rect = row.getBoundingClientRect();
        const after = event.clientY > rect.top + rect.height / 2;
        root.insertBefore(dragged, after ? row.nextSibling : row);
    });
    root.addEventListener("drop", (event) => { event.preventDefault(); });
    root.addEventListener("dragend", () => {
        if (!dragged) return;
        dragged.classList.remove("grabbing");
        dragged = null;
        deviceOrderSync(root);
    });
    root.addEventListener("change", (event) => {
        if (event.target.name === "devices") deviceOrderSync(root);
    });
}
const __devlist = document.getElementById("devicefield")?.querySelector(".devlist");
if (__devlist) deviceDrag(__devlist);
"""


def _nav_link(href: str, label: str, active: str, nav: dict[str, str] | None):
    """One tab. Server and Autotune share a form, so their links re-read the
    address bar when clicked: a worker typed after the page loaded has already
    updated the URL but not the baked href, and a click would silently drop it.
    """
    attrs: dict = {
        "href": f"{href}?{nav[href]}" if nav and href in nav else href,
        "cls": "active" if href == active else None,
    }
    if href in ("/server", "/autotune"):
        attrs["data-path"] = href
        attrs["onclick"] = "this.href=this.dataset.path+location.search"
    return A(label, **attrs)


def shell(title: str, active: str, config: AppConfig, *content,
          nav: dict[str, str] | None = None):
    """`nav` keeps the page the link comes from: Server's "Autotune" opens the
    sweep of what is on screen, Autotune's "Server" opens the same run, rather
    than two empty forms."""
    return (
        Title(f"GUI 2.0 — {title}"),
        Header(
            H1("llama.cpp RDNA lab — GUI 2.0"),
            Nav(*[_nav_link(href, label, active, nav) for href, label in NAV]),
            Span(str(config.data_root), cls="path"),
            cls="top",
        ),
        Main(*content),
    )
