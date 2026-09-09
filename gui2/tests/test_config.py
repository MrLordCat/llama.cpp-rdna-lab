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
