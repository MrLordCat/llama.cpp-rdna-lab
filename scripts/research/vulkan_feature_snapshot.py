#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


FEATURE_TESTS = ("coopmat", "coopmat2", "integer_dot", "bfloat16")
VULKANINFO_FILTER = re.compile(
    r"GPU id|deviceName|driverName|driverInfo|apiVersion|driverVersion|"
    r"VK_KHR_cooperative_matrix|VK_NV_cooperative_matrix2|"
    r"CooperativeMatrix|cooperativeMatrix|subgroupSize|subgroupSizeControl|"
    r"shaderBFloat16|shaderFloat8",
    re.IGNORECASE,
)


def run_command(args: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout + completed.stderr


def print_glslc_section(repo_root: Path) -> None:
    glslc = shutil.which("glslc")
    print("## glslc")
    print()
    if glslc is None:
        print("- glslc: not found")
        print()
        return

    print(f"- path: {glslc}")
    code, version = run_command([glslc, "--version"])
    version_line = next((line.strip() for line in version.splitlines() if line.strip()), "unknown")
    print(f"- version: {version_line}")
    print()

    print("| feature_test | status |")
    print("|---|---|")
    for name in FEATURE_TESTS:
        shader = repo_root / "ggml/src/ggml-vulkan/vulkan-shaders/feature-tests" / f"{name}.comp"
        if not shader.exists():
            print(f"| {name} | missing shader |")
            continue
        code, output = run_command([
            glslc,
            "-o",
            "-",
            "-fshader-stage=compute",
            "--target-env=vulkan1.3",
            str(shader),
        ])
        status = "OK" if code == 0 else "FAIL"
        detail = ""
        if code != 0:
            detail = " - " + re.sub(r"\s+", " ", output.strip())[:160]
        print(f"| {name} | {status}{detail} |")
    print()


def filtered_lines(text: str, context: int = 1) -> list[str]:
    lines = text.splitlines()
    selected: set[int] = set()
    for idx, line in enumerate(lines):
        if VULKANINFO_FILTER.search(line):
            for pos in range(max(0, idx - context), min(len(lines), idx + context + 1)):
                selected.add(pos)
    return [lines[idx] for idx in sorted(selected)]


def print_vulkaninfo_section() -> None:
    vulkaninfo = shutil.which("vulkaninfo")
    print("## vulkaninfo")
    print()
    if vulkaninfo is None:
        print("- vulkaninfo: not found")
        print()
        return

    print(f"- path: {vulkaninfo}")
    code, summary = run_command([vulkaninfo, "--summary"])
    if code != 0:
        print(f"- summary_status: FAIL ({code})")
    else:
        print("- summary_status: OK")
    print()

    code, full = run_command([vulkaninfo], timeout=90)
    if code != 0:
        print(f"- full_status: FAIL ({code})")
        print()
        return

    print("### Capability Signals")
    print()
    print(f"- VK_KHR_cooperative_matrix: {'yes' if 'VK_KHR_cooperative_matrix' in full else 'no'}")
    print(f"- VK_NV_cooperative_matrix2: {'yes' if 'VK_NV_cooperative_matrix2' in full else 'no'}")
    print(f"- subgroupSizeControl: {'yes' if 'subgroupSizeControl' in full else 'no'}")
    print()

    print("### Filtered Output")
    print()
    print("```text")
    for line in filtered_lines(summary + "\n" + full, context=1)[:220]:
        print(line)
    print("```")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Vulkan GLSLC and device feature snapshot for research gates")
    parser.add_argument("--repo-root", default=".", help="repository root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    print("# Vulkan Feature Snapshot")
    print()
    print_glslc_section(repo_root)
    print_vulkaninfo_section()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())