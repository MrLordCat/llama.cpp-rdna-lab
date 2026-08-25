"""Server-rendered SVG charts. No JS charting library, no build step.

Only runs from one comparison family share an axis: putting different models,
contexts or KV types on one TPS scale compares things that are not comparable.
"""

from __future__ import annotations

from collections.abc import Callable

from fasthtml.common import A
from fasthtml.svg import Circle, G, Line, Polyline, Rect, Svg, Text, Title

from gui2.core.history import LaneStat, Run

BACKEND_COLORS = {
    "rocm": "#e0785b",
    "vulkan": "#6fb8d9",
    "cpu": "#9ecf8a",
}
FALLBACK_COLOR = "#8a94a3"
BEST_COLOR = "#f0d98a"

GRID = "#2a3038"
AXIS = "#3a424d"
MUTED = "#8a94a3"


def backend_color(backend: str) -> str:
    return BACKEND_COLORS.get(backend.lower(), FALLBACK_COLOR)


def _empty(message: str, width: int, height: int = 90):
    return Svg(
        Text(message, x=width // 2, y=height // 2, fill=MUTED, font_size=13, text_anchor="middle"),
        width=width, height=height, viewBox=f"0 0 {width} {height}", cls="chart",
    )


def config_chart(
    lanes: list[LaneStat],
    width: int = 900,
    label_width: int = 250,
    row_height: int = 26,
    link_for: Callable[[LaneStat], str] | None = None,
):
    """Best TPS per tuned config inside one family; the tick marks the median."""
    if not lanes:
        return _empty("No scored runs in this group", width)

    height = row_height * len(lanes) + 26
    bar_left = label_width
    bar_max = width - label_width - 96
    top_tps = max(lane.best_tps for lane in lanes) or 1.0

    rows = []
    for index, lane in enumerate(lanes):
        y = index * row_height + 8
        length = max(2.0, lane.best_tps / top_tps * bar_max)
        median_x = bar_left + max(1.0, lane.median_tps / top_tps * bar_max)
        color = backend_color(lane.backend)
        row = G(
            Rect(width=width, height=row_height, x=0, y=y - 4, fill="transparent"),
            Text(lane.label[:44] or "-", x=0, y=y + 12, fill="#d8dee6", font_size=11.5),
            Rect(width=length, height=15, x=bar_left, y=y + 1, fill=color, rx=2, opacity=0.75),
            Line(x1=median_x, y1=y - 1, x2=median_x, y2=y + 17, stroke="#14181d", stroke_width=2),
            Text(f"{lane.best_tps:.2f}", x=bar_left + length + 8, y=y + 13,
                 fill=BEST_COLOR, font_size=11.5, font_weight="600"),
            Text(f"med {lane.median_tps:.2f} · {lane.runs} runs"
                 + (f" · {lane.failed} failed" if lane.failed else ""),
                 x=bar_left + length + 52, y=y + 13, fill=MUTED, font_size=10.5),
        )
        row = row(Title(f"{lane.label}\nbest {lane.best_tps:.2f} · median {lane.median_tps:.2f} tok/s\n"
                        f"{lane.runs} runs, last {lane.last_time}"))
        rows.append(
            A(row, hx_get=link_for(lane), hx_target="#results", style="cursor:pointer")
            if link_for else row
        )

    return Svg(
        G(*rows),
        Line(x1=bar_left, y1=4, x2=bar_left, y2=height - 18, stroke=AXIS, stroke_width=1),
        width=width, height=height, viewBox=f"0 0 {width} {height}", cls="chart",
    )


def progress_chart(runs: list[Run], width: int = 900, height: int = 240):
    """Run-by-run TPS for a single config; equal spacing, newest on the right."""
    points = [run for run in runs if run.aggregate_tps is not None]
    if not points:
        return _empty("No scored runs in this lane", width)

    points.sort(key=lambda run: (run.timestamp is not None, run.timestamp, run.index))
    pad_left, pad_right, pad_top, pad_bottom = 58, 14, 14, 30
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    top_tps = (max(run.aggregate_tps or 0.0 for run in points) * 1.12) or 1.0
    step = plot_w / (len(points) - 1) if len(points) > 1 else 0.0

    def to_x(index: int) -> float:
        return pad_left + (index * step if len(points) > 1 else plot_w / 2)

    def to_y(tps: float) -> float:
        return pad_top + plot_h - (tps / top_tps) * plot_h

    grid = []
    for line_index in range(5):
        value = top_tps * line_index / 4
        y = to_y(value)
        grid.append(Line(x1=pad_left, y1=y, x2=width - pad_right, y2=y, stroke=GRID, stroke_width=1))
        grid.append(Text(f"{value:.1f}", x=pad_left - 8, y=y + 4, fill=MUTED, font_size=11, text_anchor="end"))

    scored = [run.aggregate_tps or 0.0 for run in points if run.scored]
    reference = []
    if scored:
        best = max(scored)
        reference = [
            Line(x1=pad_left, y1=to_y(best), x2=width - pad_right, y2=to_y(best),
                 stroke=BEST_COLOR, stroke_width=1, stroke_dasharray="4 3", opacity=0.7),
            Text(f"best {best:.2f}", x=width - pad_right, y=to_y(best) - 5,
                 fill=BEST_COLOR, font_size=10.5, text_anchor="end"),
        ]

    trace = " ".join(f"{to_x(i):.1f},{to_y(run.aggregate_tps or 0.0):.1f}" for i, run in enumerate(points))
    marks = []
    for index, run in enumerate(points):
        mark = Circle(
            r=4 if run.errors else 3.4,
            cx=to_x(index),
            cy=to_y(run.aggregate_tps or 0.0),
            fill="#e0785b" if run.errors else backend_color(run.backend),
            stroke=BEST_COLOR if run.is_group_best else "none",
            stroke_width=1.5,
        )
        marks.append(
            mark(Title(f"{run.time_text} · {run.aggregate_tps:.2f} tok/s"
                       + (f" · {run.errors} errors" if run.errors else "")
                       + f"\n{run.label or run.run_id}"))
        )

    labels = [
        Text(points[0].time_text[:10], x=pad_left, y=height - 10, fill=MUTED, font_size=11),
        Text(points[-1].time_text[:10], x=width - pad_right, y=height - 10,
             fill=MUTED, font_size=11, text_anchor="end"),
    ]

    return Svg(
        G(*grid),
        *reference,
        Polyline(points=trace, fill="none", stroke=MUTED, stroke_width=1, opacity=0.5),
        Line(x1=pad_left, y1=pad_top, x2=pad_left, y2=pad_top + plot_h, stroke=AXIS, stroke_width=1),
        G(*marks),
        G(*labels),
        width=width, height=height, viewBox=f"0 0 {width} {height}", cls="chart",
    )
