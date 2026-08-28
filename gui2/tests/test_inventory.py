"""Builds read off disk: which one is newest, and how old each one is.

The ordering is the point of these tests. Eight `build-*` directories accrete
over a few months of trying backends, and the question asked of the list is
almost never "which is first alphabetically" but "which one did I just build".
"""

from __future__ import annotations

import os
import time

import pytest

from gui2.core.inventory import Build, discover_builds, read_build

HOUR = 3600.0


def make_build(root, name: str, *, age_hours: float | None = 0.0,
               cache: str = "GGML_VULKAN:BOOL=ON") -> None:
    """A build directory as `read_build` expects one; `age_hours=None` links nothing."""
    folder = root / name
    (folder / "bin").mkdir(parents=True)
    (folder / "CMakeCache.txt").write_text(cache, encoding="utf-8")
    if age_hours is None:
        return
    binary = folder / "bin" / "llama-server.exe"
    binary.write_text("", encoding="utf-8")
    stamp = time.time() - age_hours * HOUR
    os.utime(binary, (stamp, stamp))


def test_the_newest_build_is_offered_first(tmp_path):
    """Name order would have put the oldest of these on top."""
    make_build(tmp_path, "build-alpha", age_hours=200)
    make_build(tmp_path, "build-zeta", age_hours=2)
    make_build(tmp_path, "build-middle", age_hours=50)

    assert [build.name for build in discover_builds(tmp_path)] == [
        "build-zeta", "build-middle", "build-alpha"]


def test_a_build_with_no_binary_sinks_below_the_ones_that_have_one(tmp_path):
    """Configured but never compiled is not a candidate, whatever its date."""
    make_build(tmp_path, "build-configured", age_hours=None)
    make_build(tmp_path, "build-old", age_hours=900)

    builds = discover_builds(tmp_path)
    assert [build.name for build in builds] == ["build-old", "build-configured"]
    assert builds[-1].usable is False
    assert builds[-1].built_text == "never built"


@pytest.mark.parametrize("age_hours, expected", [
    (0.25, "15 min ago"),
    (5, "5 h ago"),
    (47, "47 h ago"),
    (24 * 8, "8 days ago"),
])
def test_the_age_is_said_in_the_unit_that_still_means_something(age_hours, expected):
    """Minutes while a build is warm, days once "312 h ago" has stopped landing."""
    build = Build(path=None, name="b", backend="vulkan", server_bin=None,
                  supports_rpc=None, built_at=time.time() - age_hours * HOUR)
    assert build.built_text == expected


def test_the_date_comes_from_the_binary_not_the_cmake_cache(tmp_path):
    """CMakeCache.txt survives every rebuild that reuses it, so it dates the wrong event."""
    make_build(tmp_path, "build-rebuilt", age_hours=1)
    cache = tmp_path / "build-rebuilt" / "CMakeCache.txt"
    old = time.time() - 900 * HOUR
    os.utime(cache, (old, old))

    assert read_build(tmp_path / "build-rebuilt").built_text == "1 h ago"
