"""AppConfig and the child-process environment it brings."""

from __future__ import annotations

import os
import sys

from gui2.config import AppConfig


def test_runtime_env_empty_without_paths():
    config = AppConfig()
    assert config.runtime_env("rocm") == {}


def test_runtime_env_rocm_linux_sets_library_path():
    config = AppConfig(library_paths=("/opt/rocm/lib",))
    env = config.runtime_env("rocm")
    if sys.platform == "win32":
        assert env == {}
    else:
        assert env["LD_LIBRARY_PATH"] == "/opt/rocm/lib"


def test_runtime_env_keeps_existing_library_path():
    config = AppConfig(library_paths=("/opt/rocm/lib",))
    old = os.environ.get("LD_LIBRARY_PATH")
    try:
        os.environ["LD_LIBRARY_PATH"] = "/existing"
        env = config.runtime_env("rocm")
        if sys.platform != "win32":
            assert env["LD_LIBRARY_PATH"] == "/opt/rocm/lib:/existing"
    finally:
        if old is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = old


def test_runtime_env_only_for_gpu_backends():
    config = AppConfig(library_paths=("/opt/rocm/lib",))
    assert config.runtime_env("cpu") == {}
    assert config.runtime_env("vulkan") == {}

def test_runtime_env_needs_a_build_to_prepend_a_path():
    """Without a build there is nothing to pair a runtime with."""
    config = AppConfig(library_paths=("/opt/rocm/lib",))
    assert config.runtime_env("rocm", None) == {}

def test_runtime_env_windows_prepends_the_compiler_of_the_build(tmp_path):
    """A ROCm build must load the SDK that compiled it, not whichever is first."""
    if sys.platform != "win32":
        return
    sdk = tmp_path / "ROCm" / "7.2" / "bin"
    sdk.mkdir(parents=True)
    build = tmp_path / "build-rocm72"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_CXX_COMPILER:STRING={sdk / 'clang++.exe'}\n", encoding="utf-8")

    env = AppConfig().runtime_env("rocm", build)
    assert env["PATH"].split(os.pathsep)[0] == str(sdk)
    # a CPU build has no library of its own to find
    assert AppConfig().runtime_env("cpu", build) == {}

def test_runtime_env_windows_path_keeps_the_rest_of_path(tmp_path, monkeypatch):
    if sys.platform != "win32":
        return
    sdk = tmp_path / "ROCm" / "7.2" / "bin"
    sdk.mkdir(parents=True)
    build = tmp_path / "build-rocm72"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_CXX_COMPILER:STRING={sdk / 'clang++.exe'}\n", encoding="utf-8")
    monkeypatch.setenv("PATH", os.pathsep.join([r"C:\Windows", str(sdk)]))

    parts = AppConfig().runtime_env("rocm", build)["PATH"].split(os.pathsep)
    assert parts[0] == str(sdk)
    assert parts.count(str(sdk)) == 1
    assert r"C:\Windows" in parts
