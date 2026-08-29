#!/usr/bin/env python3
"""Split ggml-rpc.cpp into a layered RPC file stack.

Pure text split guided by line ranges from the current symbol map; each range
is sanity-checked against an expected marker before extraction. Never modifies
the source file.

Run: python scripts/research/rpc_split.py
"""
import re
import sys
from pathlib import Path

SRC = Path("ggml/src/ggml-rpc/ggml-rpc.cpp")

# (target, start_line_1based, end_line_inclusive, expected_marker at start)
SEG = [
    ("common", 37, 51, "static double rpc_wall_ms()"),
    ("types", 53, 277, "struct rpc_tensor {"),
    ("client", 278, 289, "static ggml_guid_t ggml_backend_rpc_guid"),
    ("common", 296, 326, "static uint64_t graph_structure_hash"),
    ("client", 327, 352, "struct graph_cache {"),
    ("client", 353, 372, "struct ggml_backend_rpc_context {"),
    ("client", 374, 383, "static uint64_t fnv_hash"),
    ("common", 385, 416, "static bool send_msg"),
    ("common", 417, 446, "static bool parse_endpoint"),
    ("common", 448, 532, "struct rpc_send_task {"),
    ("common", 534, 646, "static bool send_rpc_cmd_direct"),
    ("common", 647, 697, "static bool negotiate_hello"),
    ("client", 699, 726, "static void ggml_backend_rpc_buffer_free_buffer"),
    ("common", 728, 745, "static bool is_causal_mask_name"),
    ("client", 746, 753, "static bool is_rpc_activation_name"),
    ("client", 754, 814, "static rpc_tensor serialize_tensor"),
    ("client", 815, 833, "static enum ggml_status ggml_backend_rpc_buffer_init_tensor"),
    ("client", 835, 869, "static bool pack_causal_mask"),
    ("server", 870, 895, "static bool unpack_causal_mask"),
    ("client", 896, 1007, "static bool rpc_send_tensor_data"),
    ("client", 1008, 1097, "static void ggml_backend_rpc_buffer_set_tensor"),
    ("client", 1099, 1116, "static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer"),
    ("client", 1117, 1154, "static size_t get_alignment"),
    ("client", 1155, 1252, "static size_t ggml_backend_rpc_buffer_type_get_alloc_size"),
    ("client", 1254, 1300, "static bool rpc_async_copy_submit"),
    ("client", 1302, 1346, "static void ggml_backend_rpc_free"),
    ("client", 1354, 1395, "static void add_tensor"),
    ("client", 1396, 1466, "static enum ggml_status ggml_backend_rpc_graph_compute"),
    ("client", 1467, 1482, "static bool ggml_backend_rpc_cpy_tensor_async"),
    ("client", 1483, 1602, "static void ggml_backend_rpc_get_async"),
    ("client", 1603, 1646, "ggml_backend_t ggml_backend_rpc_init"),
    ("server", 1647, 1707, "class rpc_server {"),
    ("server", 1708, 2557, "void rpc_server::hello"),
    ("server", 2558, 2913, "static void rpc_serve_client"),
    ("server", 2914, 3237, "void ggml_backend_rpc_start_server"),
]

OUT = {
    "types": "rpc_types.h",
    "client": "rpc_client.cpp",
    "server": "rpc_server.cpp",
    "common": "rpc_common.cpp",
}


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = text.split("\n")
    total = len(lines)
    out: dict[str, list[str]] = {k: [] for k in OUT}

    # validate each range against its expected marker
    prev_seg: tuple[str, int, int, str] | None = None
    for tgt, a, b, marker in SEG:
        la = lines[a - 1]
        assert la.startswith(marker.split("(")[0]) or marker in la, (
            f"marker mismatch for {tgt}@{a}: got {la!r} expected {marker!r}")
        if prev_seg and prev_seg[2] > b:
            raise ValueError(f"overlapping ranges: {prev_seg} vs {(tgt, a, b, marker)}")
        prev_seg = (tgt, a, b, marker)

    # assign every source line to a target file (gaps -> preceding block)
    order: list[tuple[int, str]] = []
    seg_idx = 0
    last_tgt = SEG[0][0]
    for ln in range(1, total + 1):
        while seg_idx < len(SEG) and ln > SEG[seg_idx][2]:
            seg_idx += 1
        if seg_idx < len(SEG) and SEG[seg_idx][1] <= ln <= SEG[seg_idx][2]:
            last_tgt = SEG[seg_idx][0]
        order.append((ln, last_tgt))

    per_file: dict[str, int] = {k: 0 for k in OUT}
    for _ln, tgt in order:
        per_file[tgt] += 1
    print(f"source lines: {total}, assigned: {sum(per_file.values())}")

    for tgt, fname in OUT.items():
        p = Path("ggml/src/ggml-rpc") / fname
        content_lines = [lines[ln - 1] for ln, t in order if t == tgt]
        p.write_text("\n".join(content_lines) + "\n", encoding="utf-8")
        print(f"wrote {fname:16} {p} : {len(content_lines):4d} lines")


if __name__ == "__main__":
    sys.exit(main())
