#!/usr/bin/env python3
"""Entry point for GUI 2.0 — the local web UI in gui2/.

The whole GUI lives in gui2/ and starts with `python -m gui2`; this file only
keeps the old launch habit working.
"""

from gui2.__main__ import main

if __name__ == "__main__":
    main()
