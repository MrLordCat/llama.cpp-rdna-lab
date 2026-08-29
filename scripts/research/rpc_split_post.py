#!/usr/bin/env python3
"""Post-process the rpc_split.py output:
1. prepend #pragma once + includes to rpc_types.h
2. prepend rpc_internal.h include to the three .cpp files
3. drop 'static' from shared helpers in rpc_common.cpp (extern linkage)
4. update CMakeLists.txt to the new file stack
"""
import re
from pathlib import Path

DIR = Path("ggml/src/ggml-rpc")

TYPES_PROLOGUE = """#pragma once
// RPC wire protocol types (structs are packed and sent verbatim).
#include "ggml.h"
#include <array>
#include <cstdint>
#include <cstring>

"""

CPP_PROLOGUE = "#include \"rpc_internal.h\"\n\n"

UNSTATIC = [
    "static double rpc_wall_ms(",
    "static uint64_t graph_structure_hash(",
    "static bool send_msg(",
    "static bool recv_msg(",
    "static bool parse_endpoint(",
    "static bool rpc_send_submit(",
    "static bool send_rpc_cmd_direct(",
    "static bool send_rpc_cmd(",
    "static bool send_rpc_cmd_async(",
    "static bool negotiate_hello(",
    "static std::shared_ptr<socket_t> get_socket(",
    "static bool is_causal_mask_name(",
]

NEW_SOURCES = ["rpc_common.cpp", "rpc_client.cpp", "rpc_server.cpp", "transport.cpp"]


def prepend(p: Path, text: str) -> None:
    cur = p.read_text(encoding="utf-8")
    p.write_text(text + cur, encoding="utf-8")


def main() -> None:
    prepend(DIR / "rpc_types.h", TYPES_PROLOGUE)
    for name in ("rpc_client.cpp", "rpc_server.cpp"):
        prepend(DIR / name, CPP_PROLOGUE)

    common = DIR / "rpc_common.cpp"
    txt = common.read_text(encoding="utf-8")
    for sig in UNSTATIC:
        assert sig in txt, f"missing signature {sig!r}"
        txt = txt.replace(sig, sig[len("static "):], 1)
    common.write_text(CPP_PROLOGUE + txt, encoding="utf-8")

    cmake = DIR / "CMakeLists.txt"
    txt = cmake.read_text(encoding="utf-8")
    old = "                         ggml-rpc.cpp\n"
    new = ("                         rpc_common.cpp\n"
           "                         rpc_client.cpp\n"
           "                         rpc_server.cpp\n")
    if old in txt:
        txt = txt.replace(old, new, 1)
    cmake.write_text(txt, encoding="utf-8")
    print("post-process done")


if __name__ == "__main__":
    main()
