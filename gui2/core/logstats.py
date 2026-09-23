"""What the log says a request cost, kept as the last few turns.

llama-server prints one timing pair per finished request:

    prompt eval time = 3217.81 ms /  55 tokens (  58.51 ms per token,   17.09 tokens per second)
       eval time =   950.79 ms /  43 tokens (  22.11 ms per token,   45.23 tokens per second)

The pair arrives as the request finishes, possibly hours after the one before
it and far past the end of the bounded log buffer, so it is read once, as each
line arrives, and only the last few requests are kept. The Log panel shows
their average next to its heading, so the speed a run is doing is visible in
the same line the log scrolls in.
"""

from __future__ import annotations

import re
import threading
from collections import deque

#: how many finished requests the average is taken over
RECENT_TURNS = 3

# "prompt eval time" starts at column zero; the decode line is indented, which
# is what keeps the two apart. The numbers are right-justified by the server
# (%8.2f / %5d), so the spaces before them vary and are matched as whitespace.
# Speeds are computed from ms and tokens rather than parsed from the
# parenthetical, because the parenthetical is a rounding of the same two numbers.
_PROMPT = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
_DECODE = re.compile(r"^\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")

def _tps(milliseconds: str, tokens: str) -> float | None:
    count = int(tokens)
    span = float(milliseconds) / 1000.0
    if count <= 0 or span <= 0:
        return None  # a zero span is a rounding artifact, not infinite speed
    return count / span

class TurnTracker:
    """The prompt and decode speeds of the last few requests, as the log says."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: deque[tuple[float | None, float | None]] = deque(maxlen=RECENT_TURNS)

    def feed(self, line: str) -> None:
        """One line of the child's output; most lines are not timing lines."""
        prompt = _PROMPT.search(line)
        if prompt:
            with self._lock:
                self._turns.append((_tps(prompt.group(1), prompt.group(2)), None))
            return
        decode = _DECODE.match(line)
        if not decode:
            return
        with self._lock:
            speed = _tps(decode.group(1), decode.group(2))
            if self._turns and self._turns[-1][1] is None:
                # the indented line that follows a prompt line is its decode
                last = self._turns[-1]
                self._turns[-1] = (last[0], speed)
            else:
                self._turns.append((None, speed))

    def summary(self) -> str:
        """`last 3: prompt 17.1 t/s · decode 45.2 t/s`; "" until one finishes."""
        with self._lock:
            turns = list(self._turns)
        prompts = [prompt for prompt, _decode in turns if prompt is not None]
        decodes = [decode for _prompt, decode in turns if decode is not None]
        parts = []
        if prompts:
            parts.append(f"prompt {sum(prompts) / len(prompts):.1f} t/s")
        if decodes:
            parts.append(f"decode {sum(decodes) / len(decodes):.1f} t/s")
        if not parts:
            return ""
        return f"last {len(turns)}: " + " · ".join(parts)

__all__ = ["TurnTracker", "RECENT_TURNS"]
