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
// the one document listener registers once; an OOB swap re-runs this script
if (!window.__gui2BalancerInputWired) {
window.__gui2BalancerInputWired = true;
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
}
"""

#: dragging a card to the top makes it first in -dev (and its split bar follows)
DEVICE_ORDER_JS = """
(() => {
const displayKeyPrefix = "gui2.display-device.";
function deviceDisplaySync(root) {
    const backend = root.dataset.backend || "";
    let saved = "";
    try { saved = localStorage.getItem(displayKeyPrefix + backend) || ""; } catch (_) {}
    if (!saved) {
        saved = root.querySelector(".displaymark.active")?.dataset.device || "";
    }
    root.querySelectorAll(".displaymark").forEach((mark) => {
        const active = mark.dataset.device === saved;
        mark.classList.toggle("active", active);
        mark.setAttribute("aria-pressed", String(active));
        mark.title = active
            ? mark.dataset.device + " is the display-attached GPU"
            : "Mark " + mark.dataset.device + " as the display-attached GPU";
    });
}
function deviceDisplaySet(root, mark) {
    const backend = root.dataset.backend || "";
    try { localStorage.setItem(displayKeyPrefix + backend, mark.dataset.device || ""); } catch (_) {}
    deviceDisplaySync(root);
}
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
function deviceOrderSubmit(devlist) {
    // An empty device list means "all" but carries no order. The first drag
    // makes that implicit choice explicit, otherwise a form submission could
    // not preserve the order the user just made.
    let chosen = Array.from(devlist.querySelectorAll("input[name=devices]:checked"));
    if (!chosen.length) {
        devlist.querySelectorAll("input[name=devices]").forEach((input) => { input.checked = true; });
        chosen = Array.from(devlist.querySelectorAll("input[name=devices]:checked"));
    }
    deviceOrderSync(devlist);
    // Both Server and Autotune forms listen for change. Submitting the current
    // DOM order updates their preview and URL, so the order survives navigation.
    if (chosen[0]) chosen[0].dispatchEvent(new Event("change", { bubbles: true }));
}
function deviceDrag(root) {
    if (root.dataset.dragReady === "true") return;
    root.dataset.dragReady = "true";
    let dragged = null;
    let moved = false;
    root.addEventListener("click", (event) => {
        const display = event.target.closest(".displaymark");
        if (display) {
            event.preventDefault();
            event.stopPropagation();
            deviceDisplaySet(root, display);
            return;
        }
        if (event.target.closest(".draghandle")) {
            // A handle lives inside a label; do not let a click on it toggle
            // the checkbox. It changes order only.
            event.preventDefault();
            event.stopPropagation();
        }
    });
    root.addEventListener("keydown", (event) => {
        const display = event.target.closest(".displaymark");
        if (display && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            event.stopPropagation();
            deviceDisplaySet(root, display);
        }
    });
    root.addEventListener("dragstart", (event) => {
        const handle = event.target.closest(".draghandle");
        const row = handle?.closest(".devrow");
        if (!row) {
            event.preventDefault();
            return;
        }
        dragged = row;
        moved = false;
        row.classList.add("grabbing");
        if (event.dataTransfer) {
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/plain", row.querySelector(".devname")?.textContent || "device");
        }
    });
    root.addEventListener("dragover", (event) => {
        const row = event.target.closest(".devrow");
        if (!dragged || !row || row === dragged) return;
        event.preventDefault();
        const rect = row.getBoundingClientRect();
        const after = event.clientY > rect.top + rect.height / 2;
        const before = after ? row.nextSibling : row;
        if (before !== dragged && dragged.nextSibling !== before) {
            root.insertBefore(dragged, before);
            moved = true;
        }
    });
    root.addEventListener("drop", (event) => {
        if (dragged) event.preventDefault();
    });
    root.addEventListener("dragend", () => {
        if (!dragged) return;
        dragged.classList.remove("grabbing");
        dragged = null;
        if (moved) deviceOrderSubmit(root);
        moved = false;
    });
    root.addEventListener("change", (event) => {
        if (event.target.name === "devices") deviceOrderSync(root);
    });
    deviceDisplaySync(root);
}
const __devlist = document.getElementById("devicefield")?.querySelector(".devlist");
if (__devlist) deviceDrag(__devlist);
})();
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
