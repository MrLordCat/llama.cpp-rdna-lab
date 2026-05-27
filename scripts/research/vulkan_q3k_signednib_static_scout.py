#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


FOCUS_OPS = (
    "OpControlBarrier",
    "OpLoad",
    "OpStore",
    "OpFConvert",
    "OpFMul",
    "OpFAdd",
    "OpIAdd",
    "OpISub",
    "OpIMul",
    "OpUDiv",
    "OpUMod",
    "OpShiftRightLogical",
    "OpShiftRightArithmetic",
    "OpShiftLeftLogical",
    "OpBitwiseAnd",
    "OpBitwiseOr",
    "OpBitwiseXor",
    "OpCooperativeMatrixLoadKHR",
    "OpCooperativeMatrixMulAddKHR",
    "OpCooperativeMatrixStoreKHR",
)

OP_RE = re.compile(r"\b(Op[A-Za-z0-9_]+)\b")

TYPE_Q3_BLOCK = """#define QUANT_K_Q3_K 256

struct block_q3_K
{
    uint8_t hmask[QUANT_K_Q3_K/8];
    uint8_t qs[QUANT_K_Q3_K/4];
    uint8_t scales[12];
    float16_t d;
    uint16_t _pad;
};

struct block_q3_K_packed16
{
    uint16_t hmask[QUANT_K_Q3_K/8/2];
    uint16_t qs[QUANT_K_Q3_K/4/2];
    uint16_t scales[12/2];
    float16_t d;
    uint16_t _pad;
};

struct block_q3_K_packed32
{
    uint32_t hmask[QUANT_K_Q3_K/8/4];
    uint32_t qs[QUANT_K_Q3_K/4/4];
    uint32_t scales[12/4];
    float16_t d;
    uint16_t _pad;
};

#if defined(DATA_A_Q3_K)
#define QUANT_K QUANT_K_Q3_K
#define QUANT_R 1
#define A_TYPE block_q3_K
#define A_TYPE_PACKED16 block_q3_K_packed16
#define A_TYPE_PACKED32 block_q3_K_packed32
#define DATA_A_QUANT_K
#endif
"""

TYPE_Q3_SIGNED_NIBBLE_BLOCK = """#define QUANT_K_Q3_K 256

struct block_q3_K
{
    uint8_t hmask[QUANT_K_Q3_K/8];
    uint8_t qs[QUANT_K_Q3_K/4];
    uint8_t scales[12];
    float16_t d;
    uint16_t _pad;
};

struct block_q3_K_packed16
{
    uint16_t hmask[QUANT_K_Q3_K/8/2];
    uint16_t qs[QUANT_K_Q3_K/4/2];
    uint16_t scales[12/2];
    float16_t d;
    uint16_t _pad;
};

struct block_q3_K_packed32
{
    uint32_t hmask[QUANT_K_Q3_K/8/4];
    uint32_t qs[QUANT_K_Q3_K/4/4];
    uint32_t scales[12/4];
    float16_t d;
    uint16_t _pad;
};

struct block_q3_K_signed_nibble
{
    uint8_t qsnib[QUANT_K_Q3_K/2];
    uint8_t scales[12];
    float16_t d;
    uint16_t _pad;
};

#if defined(DATA_A_Q3_K)
#define QUANT_K QUANT_K_Q3_K
#define QUANT_R 1
#if defined(GGML_VK_Q3K_SIGNED_NIBBLE_LAYOUT)
#define A_TYPE block_q3_K_signed_nibble
#else
#define A_TYPE block_q3_K
#define A_TYPE_PACKED16 block_q3_K_packed16
#define A_TYPE_PACKED32 block_q3_K_packed32
#endif
#define DATA_A_QUANT_K
#endif
"""

Q3_DEQUANT_BLOCK = """#if defined(DATA_A_Q3_K)
FLOAT_TYPEV2 dequant_q3_k_pair(const uint idx) {
            const uint ib = idx / 128;                   // 2 values per idx
            const uint iqs = idx % 128;                  // 0..127

            const uint n = iqs / 64;                     // 0,1
            const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
            const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
            const uint is = iqs / 8;                     // 0..15
            const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
            const uint qsshift = halfsplit * 2;          // 0,2,4,6

            const int8_t us = int8_t(((data_a[ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                                  | (((data_a[ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
            const float dl = float(data_a[ib].d) * float(us - 32);

            const vec2 qs = vec2(unpack8((uint(data_a_packed16[ib].qs[qsi / 2]) >> qsshift) & 0x0303).xy);
            const vec2 hm = vec2(unpack8(((uint(data_a_packed16[ib].hmask[hmi / 2]) >> (4 * n + halfsplit)) & 0x0101 ^ 0x0101) << 2).xy);

            return FLOAT_TYPEV2(dl * (qs.x - hm.x),
                                dl * (qs.y - hm.y));
}
#endif
"""

Q3_DEQUANT_SIGNED_NIBBLE_BLOCK = """#if defined(DATA_A_Q3_K)
FLOAT_TYPEV2 dequant_q3_k_pair(const uint idx) {
            const uint ib = idx / 128;                   // 2 values per idx
            const uint iqs = idx % 128;                  // 0..127
            const uint is = iqs / 8;                     // 0..15

            const int8_t us = int8_t(((data_a[ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                                  | (((data_a[ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
            const float dl = float(data_a[ib].d) * float(us - 32);

#if defined(GGML_VK_Q3K_SIGNED_NIBBLE_LAYOUT)
            const uint packed = uint(data_a[ib].qsnib[iqs]);
            const ivec2 q_raw = ivec2(int(packed & 0x0F), int((packed >> 4) & 0x0F));
            const ivec2 q = q_raw - ((q_raw & ivec2(8)) << 1);
            return FLOAT_TYPEV2(dl * float(q.x),
                                dl * float(q.y));
#else
            const uint n = iqs / 64;                     // 0,1
            const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
            const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
            const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
            const uint qsshift = halfsplit * 2;          // 0,2,4,6
            const vec2 qs = vec2(unpack8((uint(data_a_packed16[ib].qs[qsi / 2]) >> qsshift) & 0x0303).xy);
            const vec2 hm = vec2(unpack8(((uint(data_a_packed16[ib].hmask[hmi / 2]) >> (4 * n + halfsplit)) & 0x0101 ^ 0x0101) << 2).xy);
            return FLOAT_TYPEV2(dl * (qs.x - hm.x),
                                dl * (qs.y - hm.y));
#endif
}
#endif
"""

COMPILE_DEFINES = {
    "FLOAT16": "1",
    "ACC_TYPE": "float16_t",
    "ACC_TYPEV2": "f16vec2",
    "ACC_TYPE_MAX": "float16_t(65504.0)",
    "COOPMAT": "1",
    "FLOAT_TYPE": "float16_t",
    "FLOAT_TYPEV2": "f16vec2",
    "FLOAT_TYPEV4": "f16vec4",
    "FLOAT_TYPEV8": "f16mat2x4",
    "DATA_A_Q3_K": "1",
    "LOAD_VEC_A": "4",
    "LOAD_VEC_B": "8",
    "B_TYPE": "mat2x4",
    "D_TYPE": "float",
    "ALIGNED": "1",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_cache_tool(root: Path, key: str) -> str | None:
    cache = root / "build-vulkan/CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(key + ":"):
            _, value = line.split("=", 1)
            if value and Path(value).exists():
                return value
    return None


def find_tool(root: Path, name: str, cache_key: str | None = None) -> str:
    if cache_key:
        cached = parse_cache_tool(root, cache_key)
        if cached:
            return cached
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"ERROR: {name} not found")


def copy_shader_tree(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns("CMakeFiles", "CMakeCache.txt", "*.spv", "*.d")
    shutil.copytree(src, dst, ignore=ignore)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def patch_signed_nibble_layout(shader_dir: Path) -> None:
    replace_once(shader_dir / "types.glsl", TYPE_Q3_BLOCK, TYPE_Q3_SIGNED_NIBBLE_BLOCK)
    replace_once(shader_dir / "mul_mm_funcs.glsl", Q3_DEQUANT_BLOCK, Q3_DEQUANT_SIGNED_NIBBLE_BLOCK)


def compile_shader(glslc: str, shader_dir: Path, out_path: Path, extra_defines: dict[str, str] | None = None) -> None:
    defines = dict(COMPILE_DEFINES)
    if extra_defines:
        defines.update(extra_defines)
    cmd = [
        glslc,
        "-fshader-stage=compute",
        "--target-env=vulkan1.2",
        str(shader_dir / "mul_mm.comp"),
        "-o",
        str(out_path),
    ]
    for key, value in sorted(defines.items()):
        cmd.append(f"-D{key}={value}")

    completed = subprocess.run(
        cmd,
        cwd=shader_dir,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "glslc failed")


def disassemble(spirv_dis: str, path: Path) -> str:
    completed = subprocess.run(
        [spirv_dis, str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"spirv-dis failed for {path}")
    return completed.stdout


def opcode_counts(disassembly: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for line in disassembly.splitlines():
        match = OP_RE.search(line)
        if match:
            counts[match.group(1)] += 1
    return counts


def write_report(
    path: Path,
    glslc: str,
    spirv_dis: str,
    base_spv: Path,
    variant_spv: Path,
    base_dis: str,
    variant_dis: str,
    base_counts: collections.Counter[str],
    variant_counts: collections.Counter[str],
) -> None:
    bitwise_ops = ("OpShiftRightLogical", "OpShiftRightArithmetic", "OpShiftLeftLogical", "OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor")
    base_bitwise = sum(base_counts.get(op, 0) for op in bitwise_ops)
    variant_bitwise = sum(variant_counts.get(op, 0) for op in bitwise_ops)
    base_ops = sum(base_counts.values())
    variant_ops = sum(variant_counts.values())

    lines: list[str] = []
    lines.append("# S001 Vulkan Q3_K Signed-Nibble Static Scout")
    lines.append("")
    lines.append("Status: compile/static scout only; no runtime wiring and no TPS claim.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- glslc: `{glslc}`")
    lines.append(f"- spirv-dis: `{spirv_dis}`")
    lines.append(f"- base SPIR-V: `{base_spv}`")
    lines.append(f"- candidate SPIR-V: `{variant_spv}`")
    lines.append("- shader route: `matmul_q3_k_f32_aligned_f16acc_cm1` defines, compiled from a temporary shader copy")
    lines.append("")
    lines.append("## Size And Opcode Summary")
    lines.append("")
    lines.append("| shader | bytes | disasm lines | opcodes | bitwise/shift ops |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(f"| base | {base_spv.stat().st_size} | {len(base_dis.splitlines())} | {base_ops} | {base_bitwise} |")
    lines.append(f"| signed-nibble | {variant_spv.stat().st_size} | {len(variant_dis.splitlines())} | {variant_ops} | {variant_bitwise} |")
    lines.append("")
    lines.append("## Focus Ops")
    lines.append("")
    lines.append("| op | base | signed-nibble | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for op in FOCUS_OPS:
        base_value = base_counts.get(op, 0)
        variant_value = variant_counts.get(op, 0)
        if base_value or variant_value:
            lines.append(f"| {op} | {base_value} | {variant_value} | {variant_value - base_value:+d} |")
    lines.append("")
    lines.append("## Gate Read")
    lines.append("")
    if variant_bitwise < base_bitwise and variant_ops <= base_ops:
        lines.append("- PASS-static: candidate reduces bitwise/shift work without increasing total opcode count in this compile-only fingerprint.")
    elif variant_bitwise < base_bitwise:
        lines.append("- MIXED-static: candidate reduces bitwise/shift work, but total opcode count increases. Require pipeline resource stats before runtime wiring.")
    else:
        lines.append("- FAIL-static: candidate does not reduce bitwise/shift work enough to justify runtime wiring.")
    lines.append("- This scout does not prove memory conversion cost or correctness; a runtime prototype would still need a default-off layout converter and narrow tensor gate.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile/static scout for a Vulkan Q3_K signed-nibble prompt-layout shader")
    parser.add_argument("--output-dir", default="build_logs/agent-workload", help="artifact directory")
    parser.add_argument("--keep-workdir", action="store_true", help="keep temporary shader copies for inspection")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    shader_src = root / "ggml/src/ggml-vulkan/vulkan-shaders"
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    glslc = find_tool(root, "glslc", "Vulkan_GLSLC_EXECUTABLE")
    spirv_dis = find_tool(root, "spirv-dis")

    work_parent = Path(tempfile.mkdtemp(prefix="s001-q3k-signednib-", dir=output_dir))
    try:
        base_dir = work_parent / "base"
        variant_dir = work_parent / "signed-nibble"
        copy_shader_tree(shader_src, base_dir)
        copy_shader_tree(shader_src, variant_dir)
        patch_signed_nibble_layout(variant_dir)

        base_spv = output_dir / "s001-matmul_q3_k_f32_aligned_f16acc_cm1-base.spv"
        variant_spv = output_dir / "s001-matmul_q3_k_f32_aligned_f16acc_cm1-signednib.spv"
        report = output_dir / "s001-vulkan-q3k-signednib-static.md"

        compile_shader(glslc, base_dir, base_spv)
        compile_shader(glslc, variant_dir, variant_spv, {"GGML_VK_Q3K_SIGNED_NIBBLE_LAYOUT": "1"})

        base_dis = disassemble(spirv_dis, base_spv)
        variant_dis = disassemble(spirv_dis, variant_spv)
        write_report(
            report,
            glslc,
            spirv_dis,
            base_spv,
            variant_spv,
            base_dis,
            variant_dis,
            opcode_counts(base_dis),
            opcode_counts(variant_dis),
        )
        print(f"Wrote {report}")
        print(f"Wrote {base_spv}")
        print(f"Wrote {variant_spv}")
        if args.keep_workdir:
            print(f"Kept workdir {work_parent}")
        return 0
    finally:
        if not args.keep_workdir:
            shutil.rmtree(work_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())