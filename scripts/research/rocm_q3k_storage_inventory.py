#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pattern:
    file: str
    regex: str
    label: str
    reason: str


PATTERNS = [
    Pattern(
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        r"ggml_backend_cuda_buffer_type_get_alloc_size|ggml_backend_cuda_buffer_init_tensor|ggml_backend_cuda_buffer_(set|get)_tensor|ggml_backend_cuda_buffer_(set|get)_tensor_2d|ggml_backend_cuda_buffer_cpy_tensor",
        "plain CUDA buffer API",
        "must translate host Q3_K byte offsets/sizes to padded device byte offsets/sizes",
    ),
    Pattern(
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        r"ggml_backend_cuda_split_buffer_(init_tensor|set_tensor|get_tensor)|ggml_backend_cuda_split_buffer_type_get_alloc_size",
        "split CUDA buffer API",
        "split tensors allocate/copy per device and currently use raw host row bytes",
    ),
    Pattern(
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        r"ggml_cuda_cpy_tensor_2d|ggml_backend_cuda_(set|get)_tensor_2d_async|ggml_backend_cuda_cpy_tensor_async|ggml_cuda_trace_tensor_buffer_offset",
        "async/view/copy helpers",
        "prompt staging, views, traces, and device-device copies must not mix 110-byte and 112-byte offsets",
    ),
    Pattern(
        "ggml/src/ggml-cuda/convert.cu",
        r"dequantize_block_q3_K|block_q3_K",
        "Q3_K dequant/getrows path",
        "cuBLAS staging and dequant correctness require the same device stride as storage",
    ),
    Pattern(
        "ggml/src/ggml-cuda/vecdotq.cuh",
        r"vec_dot_q3_K_q8_1|block_q3_K|get_int_b2\(bq3_K",
        "Q3_K MMVQ vecdot path",
        "decode-heavy H39 path reads Q3_K blocks directly",
    ),
    Pattern(
        "ggml/src/ggml-cuda/mmq.cuh",
        r"load_tiles_q3_K|vec_dot_q3_K_q8_1_dp4a|block_q3_K|GGML_TYPE_Q3_K",
        "Q3_K MMQ path",
        "large prompt/precompute and correctness tests use Q3_K MMQ/direct routes",
    ),
    Pattern(
        "ggml/src/ggml-cuda/mmvq.cu",
        r"GGML_TYPE_Q3_K|vec_dot_q3_K_q8_1",
        "Q3_K MMVQ dispatch/policy",
        "decode route selection must select padded-aware kernels only when storage is padded",
    ),
    Pattern(
        "ggml/src/ggml-cuda/common.cuh",
        r"ggml_cuda_type_traits<GGML_TYPE_Q3_K>|GGML_TYPE_Q3_K",
        "Q3_K type traits",
        "traits connect quant type to vecdot/dequant functions and route selectors",
    ),
]


def find_matches(repo_root: Path, pattern: Pattern) -> list[int]:
    path = repo_root / pattern.file
    if not path.exists():
        return []
    rx = re.compile(pattern.regex)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [idx for idx, line in enumerate(lines, start=1) if rx.search(line)]


def print_pattern_table(repo_root: Path) -> None:
    print("## Static Touchpoints")
    print()
    print("| Area | File | Matches | First lines | Why it matters |")
    print("|---|---|---:|---|---|")
    for pattern in PATTERNS:
        matches = find_matches(repo_root, pattern)
        first = ", ".join(str(v) for v in matches[:6])
        if len(matches) > 6:
            first += ", ..."
        print(f"| {pattern.label} | `{pattern.file}` | {len(matches)} | {first or '-'} | {pattern.reason} |")
    print()


def print_vulkan_anchor() -> None:
    print("## Vulkan Anchor")
    print()
    print("- `ggml_vk_device_type_size(Q3_K)` returns `ggml_type_size(Q3_K) + 2`, so Q3_K device blocks are `112` bytes.")
    print("- Vulkan set/get paths call padded read/write helpers when host size differs from device size.")
    print("- Vulkan view offsets are translated from host type-size units into device type-size units.")
    print("- Vulkan Q3_K MMVQ shaders can use `block_q3_K_packed32` because the storage contract already provides the padding.")
    print()


def print_cut_plan() -> None:
    print("## Prototype Cut Plan")
    print()
    print("| Step | Scope | Gate | Decision rule |")
    print("|---|---|---|---|")
    print("| P0 | helpers and docs only | build/py_compile | no runtime behavior change |")
    print("| P1 | non-split CUDA buffer Q3_K padded set/get + reverse get | dedicated roundtrip or `test-backend-ops` smoke | continue only if host<->device bytes roundtrip exactly |")
    print("| P2 | padded-aware Q3_K dequant + MMVQ + MMQ accessors | `test-backend-ops test -b ROCm0 -o MUL_MAT -p type_a=q3_K` | continue only if Q3_K correctness passes for `n=1` and prompt-shaped cases |")
    print("| P3 | split/async/cpy/view offsets and cuBLAS staging | `test-backend-ops` broader Q3_K coverage plus small real-server sanity | continue only if no fallback/corruption and traces show intended route |")
    print("| P4 | real H39 server A/B | E196 lane, q4/q4, no reuse, thinking on | keep only with clean wall/decode gain and normal text |")
    print()


def print_rejections() -> None:
    print("## Immediate Rejections")
    print()
    print("- Decode-only MMVQ padded kernels without padded dequant/MMQ/copy support: model prefill and correctness smoke can still touch Q3_K through other routes.")
    print("- Local `vecdotq.cuh` packed32 loads over raw `110`-byte storage: every other block has a `2 mod 4` start and this does not reproduce Vulkan's storage contract.")
    print("- Duplicate persistent padded Q3_K copy: memory cost is a second Q3_K model copy, not the `1.818%` replacement padding delta.")
    print("- Runtime per-node repack: adds large immutable-weight traffic on the hot path before every matvec family.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory ROCm Q3_K padded-storage touchpoints")
    parser.add_argument("--repo-root", default=".", help="repository root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    print("# ROCm Q3_K Padded Storage Inventory")
    print()
    print(f"- repo_root: `{repo_root}`")
    print("- host_q3_block_bytes: `110`")
    print("- target_device_q3_block_bytes: `112`")
    print()
    print_vulkan_anchor()
    print_pattern_table(repo_root)
    print_cut_plan()
    print_rejections()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
