#!/usr/bin/env python3
"""Transform generated D096 P2/P3 cm1 S-stages to native E4M3 WMMA.

The input is deliberately generated from canonical flash_attn_cm1.comp. P2
reads Q8 from a descriptor; P3 redirects the Q cooperative load to an
in-workgroup uint8 staging array and bitcasts only its encoded stores to fp8.
This tool is fail-closed: bindings/names, cooperative shapes, pointer
provenance, and the untouched PV-stage are audited before and after.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from d096_fp8_fa_spirv_audit import AuditError, audit_disassembly


LOAD_RE = re.compile(
    r"^(\s*)(%\S+)\s*=\s*OpCooperativeMatrixLoadKHR\s+(%\S+)\s+(%\S+)\s+(%\S+)\s+(%\S+)(.*)$"
)
ACCESS_RE = re.compile(
    r"^(\s*)(%\S+)\s*=\s*OpAccessChain\s+(%\S+)\s+(%\S+)\s+(.*)$"
)
VARIABLE_RE = re.compile(r"^(\s*)(%\S+)\s*=\s*OpVariable\s+(%\S+)\s+StorageBuffer\s*$")
WORKGROUP_VARIABLE_RE = re.compile(r"^(\s*)(%\S+)\s*=\s*OpVariable\s+(%\S+)\s+Workgroup\s*$")
DECORATE_BINDING_RE = re.compile(r"^\s*OpDecorate\s+(%\S+)\s+Binding\s+(\d+)\s*$")
NAME_RE = re.compile(r'^\s*OpName\s+(%\S+)\s+"([^"]+)"\s*$')
TYPE_POINTER_RE = re.compile(r"^\s*(%\S+)\s*=\s*OpTypePointer\s+(\S+)\s+(%\S+)\s*$")
TYPE_ARRAY_RE = re.compile(r"^\s*(%\S+)\s*=\s*OpTypeArray\s+(%\S+)\s+(%\S+)\s*$")
TYPE_INT_RE = re.compile(r"^\s*(%\S+)\s*=\s*OpTypeInt\s+(\d+)\s+(\d+)\s*$")
STORE_RE = re.compile(r"^(\s*)OpStore\s+(%\S+)\s+(%\S+)(.*)$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def find_tool(explicit: str | None, name: str) -> str:
    if explicit:
        return explicit
    if found := shutil.which(name):
        return found
    if sdk := __import__("os").environ.get("VULKAN_SDK"):
        candidate = Path(sdk) / "Bin" / f"{name}.exe"
        if candidate.is_file():
            return str(candidate)
    raise AuditError(f"{name} not found")


def disassemble(path: Path, spirv_dis: str) -> str:
    if path.suffix.lower() != ".spv":
        return path.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="d096-p2-dis-") as td:
        out = Path(td) / "base.spvasm"
        subprocess.run([spirv_dis, str(path), "-o", str(out)], check=True)
        return out.read_text(encoding="utf-8")


def transform(text: str, source: str, fused_q: bool = False, v_f16: bool = False, pv_f8: bool = False) -> str:
    if pv_f8:
        base_profile, final_profile = "p5-base", "fp8-p5"
    elif v_f16:
        base_profile, final_profile = "p4-base", "fp8-p4"
    else:
        base_profile = "p3-base" if fused_q else "p2-base"
        final_profile = "fp8-p3" if fused_q else "fp8-p2"
    report = audit_disassembly(text, source, base_profile)
    lines = text.splitlines()

    bindings: dict[str, int] = {}
    names: dict[str, str] = {}
    for line in lines:
        if match := DECORATE_BINDING_RE.match(line):
            bindings[match.group(1)] = int(match.group(2))
        if match := NAME_RE.match(line):
            names[match.group(1)] = match.group(2)

    stage_load_indices = [
        i for i in range(report.s_stage.first_load_line - 1, report.s_stage.mul_add_line - 1)
        if LOAD_RE.match(lines[i])
    ]
    require(len(stage_load_indices) == 2, f"expected two P2 S loads, found {len(stage_load_indices)}")

    load_info = []
    for index in stage_load_indices:
        match = LOAD_RE.match(lines[index])
        assert match
        _, result_id, old_type, pointer_id, layout_id, stride_id, tail = match.groups()
        access_matches = [ACCESS_RE.match(line) for line in lines]
        access = next((item for item in access_matches if item and item.group(2) == pointer_id), None)
        require(access is not None, f"cannot trace load pointer {pointer_id}")
        base_variable = access.group(4)
        load_info.append((index, result_id, old_type, pointer_id, layout_id, stride_id, tail,
                          access, base_variable, bindings.get(base_variable)))

    by_binding = {item[-1]: item for item in load_info if item[-1] is not None}
    expected_bindings = {1} if fused_q else {1, 7}
    require(
        set(by_binding) == expected_bindings,
        f"S loads must originate from bindings {sorted(expected_bindings)}, got {sorted(by_binding)}",
    )
    require(by_binding[1][4].endswith("int_0"), "K load is no longer row-major")
    if fused_q:
        q_items = [item for item in load_info if names.get(item[-2]) == "Qf_p3_dummy"]
        require(len(q_items) == 1, f"expected one Qf_p3_dummy S load, found {len(q_items)}")
        q_item = q_items[0]
        require(q_item[4].endswith("int_1"), "fused Q8 load is no longer column-major")
        q8_variables = [result_id for result_id, name in names.items() if name == "Q8f"]
        require(len(q8_variables) == 1, f"expected one named Q8f variable, found {len(q8_variables)}")
        q8_variable = q8_variables[0]
    else:
        q_item = by_binding[7]
        q8_variable = ""
        require(q_item[4].endswith("int_1"), "Q8 load is no longer column-major")

    variable_types: dict[str, tuple[int, re.Match[str]]] = {}
    for i, line in enumerate(lines):
        if match := VARIABLE_RE.match(line):
            variable_types[match.group(2)] = (i, match)
    for binding in ((1,) if fused_q else (1, 7)):
        variable = by_binding[binding][-2]
        require(variable in variable_types, f"descriptor variable {variable} declaration not found")

    q8_length_id = ""
    q8_array_type = ""
    if fused_q:
        workgroup_variables = {
            match.group(2): match.group(3)
            for line in lines
            if (match := WORKGROUP_VARIABLE_RE.match(line))
        }
        pointer_types = {
            match.group(1): (match.group(2), match.group(3))
            for line in lines
            if (match := TYPE_POINTER_RE.match(line))
        }
        array_types = {
            match.group(1): (match.group(2), match.group(3))
            for line in lines
            if (match := TYPE_ARRAY_RE.match(line))
        }
        int_types = {
            match.group(1): (int(match.group(2)), int(match.group(3)))
            for line in lines
            if (match := TYPE_INT_RE.match(line))
        }
        require(q8_variable in workgroup_variables, "Q8f is not a Workgroup variable")
        q8_pointer_type = workgroup_variables[q8_variable]
        require(q8_pointer_type in pointer_types, "Q8f pointer type not found")
        q8_storage, q8_array_type = pointer_types[q8_pointer_type]
        require(q8_storage == "Workgroup", "Q8f pointer storage class changed")
        require(q8_array_type in array_types, "Q8f array type not found")
        q8_component_type, q8_length_id = array_types[q8_array_type]
        require(int_types.get(q8_component_type) == (8, 0), "Q8f is no longer an unsigned 8-bit array")

    # SPIR-V logical layout: capabilities/extensions, annotations, then types.
    capability_anchor = next(i for i, line in enumerate(lines) if "OpCapability Float16" in line)
    lines[capability_anchor + 1:capability_anchor + 1] = [
        "               OpCapability Float8EXT",
        "               OpCapability Float8CooperativeMatrixEXT",
    ]
    extension_anchor = max(i for i, line in enumerate(lines) if "OpExtension " in line)
    lines[extension_anchor + 1:extension_anchor + 1] = [
        '               OpExtension "SPV_EXT_float8"',
    ]

    annotation_anchor = next(i for i, line in enumerate(lines) if "OpDecorate" in line)
    lines[annotation_anchor:annotation_anchor] = [
        "               OpDecorate %d096_arr_fp8 ArrayStride 1",
        "               OpDecorate %d096_fp8_block Block",
        "               OpMemberDecorate %d096_fp8_block 0 NonWritable",
        "               OpMemberDecorate %d096_fp8_block 0 Offset 0",
    ]

    # Q8f's Workgroup variable is declared before glslc's cooperative types,
    # so the fp8 component and its fixed array/pointer types must precede that
    # variable. Descriptor/cooperative helper types can remain at the first
    # cooperative-type anchor.
    float16_anchor = next(i for i, line in enumerate(lines) if re.search(r"OpTypeFloat\s+16\s*$", line))
    lines[float16_anchor + 1:float16_anchor + 1] = [
        " %d096_fp8 = OpTypeFloat 8 Float8E4M3EXT",
    ]
    if fused_q:
        q8_array_anchor = next(
            i for i, line in enumerate(lines)
            if (match := TYPE_ARRAY_RE.match(line)) and match.group(1) == q8_array_type
        )
        lines[q8_array_anchor:q8_array_anchor] = [
            # P3 is deliberately shape-gated to Br=16, HSK=256 in runtime.
            " %d096_q8_len = OpConstant %uint 4096",
            " %d096_arr_wg_fp8 = OpTypeArray %d096_fp8 %d096_q8_len",
            " %d096_ptr_wg_arr_fp8 = OpTypePointer Workgroup %d096_arr_wg_fp8",
            " %d096_ptr_wg_fp8 = OpTypePointer Workgroup %d096_fp8",
            # P5: Psh8 workgroup array (Bc*Br = 1024 encoded P bytes).
            " %d096_pv8_len = OpConstant %uint 1024",
            " %d096_arr_wg_pv_fp8 = OpTypeArray %d096_fp8 %d096_pv8_len",
            " %d096_ptr_wg_arr_pv_fp8 = OpTypePointer Workgroup %d096_arr_wg_pv_fp8",
        ]

    type_anchor = next(i for i, line in enumerate(lines) if "OpTypeCooperativeMatrixKHR" in line)
    injected_types = [
        " %d096_arr_fp8 = OpTypeRuntimeArray %d096_fp8",
        " %d096_fp8_block = OpTypeStruct %d096_arr_fp8",
        " %d096_ptr_fp8_block = OpTypePointer StorageBuffer %d096_fp8_block",
        " %d096_ptr_fp8 = OpTypePointer StorageBuffer %d096_fp8",
        " %d096_cm_f8_a = OpTypeCooperativeMatrixKHR %d096_fp8 %uint_3 %uint_16 %uint_16 %uint_0",
        " %d096_cm_f8_b = OpTypeCooperativeMatrixKHR %d096_fp8 %uint_3 %uint_16 %uint_16 %uint_1",
        " %d096_ptr_fn_cm_f8_a = OpTypePointer Function %d096_cm_f8_a",
        " %d096_ptr_fn_cm_f8_b = OpTypePointer Function %d096_cm_f8_b",
    ]
    lines[type_anchor:type_anchor] = injected_types

    # Re-find all anchors after insertions, then rewrite only the two proven
    # descriptor variables/access chains/loads and their local matrix holders.
    def replace_single(pattern: re.Pattern[str], predicate, replacement) -> None:
        matches = [(i, pattern.match(line)) for i, line in enumerate(lines)]
        matches = [(i, m) for i, m in matches if m and predicate(m)]
        require(len(matches) == 1, f"expected one rewrite target, found {len(matches)}")
        i, match = matches[0]
        lines[i] = replacement(match)

    def bypass_holder(old_result: str, cm_type: str, label: str) -> None:
        store_re = re.compile(rf"^(\s*)OpStore\s+(%\S+)\s+{re.escape(old_result)}\s*$")
        stores = [(i, store_re.match(line)) for i, line in enumerate(lines)]
        stores = [(i, match) for i, match in stores if match]
        require(len(stores) == 1, f"cannot identify matrix holder for {label}")
        store_index = stores[0][0]
        holder = stores[0][1].group(2)
        holder_load_re = re.compile(rf"^(\s*)(%\S+)\s*=\s*OpLoad\s+%\S+\s+{re.escape(holder)}\s*$")
        holder_loads = [(i, holder_load_re.match(line)) for i, line in enumerate(lines)]
        holder_loads = [(i, match) for i, match in holder_loads if match and i > store_index]
        require(holder_loads, f"matrix holder load not found for {label}")
        i, match = holder_loads[0]
        lines[store_index] = f"; D096: bypass canonical {holder} f16 holder ({label})"
        lines[i] = f"{match.group(1)}{match.group(2)} = OpCopyObject {cm_type} {old_result}"

    for binding, cm_type in ((1, "%d096_cm_f8_a"),) if fused_q else ((1, "%d096_cm_f8_a"), (7, "%d096_cm_f8_b")):
        item = by_binding[binding]
        old_result, pointer_id, base_variable = item[1], item[3], item[-2]
        replace_single(VARIABLE_RE, lambda m, v=base_variable: m.group(2) == v,
                       lambda m: f"{m.group(1)}{m.group(2)} = OpVariable %d096_ptr_fp8_block StorageBuffer")
        replace_single(ACCESS_RE, lambda m, p=pointer_id: m.group(2) == p,
                       lambda m: f"{m.group(1)}{m.group(2)} = OpAccessChain %d096_ptr_fp8 {m.group(4)} {m.group(5)}")
        replace_single(LOAD_RE, lambda m, r=old_result: m.group(2) == r,
                       lambda m, c=cm_type: f"{m.group(1)}{m.group(2)} = OpCooperativeMatrixLoadKHR {c} {m.group(4)} {m.group(5)} {m.group(6)}{m.group(7)}")
        bypass_holder(old_result, cm_type, f"binding {binding}")

    if fused_q:
        old_result, pointer_id = q_item[1], q_item[3]
        replace_single(
            WORKGROUP_VARIABLE_RE,
            lambda match: match.group(2) == q8_variable,
            lambda match: f"{match.group(1)}{match.group(2)} = OpVariable %d096_ptr_wg_arr_fp8 Workgroup",
        )

        # Rewrite the one encoder store into Q8f. The bitcast preserves the
        # exact E4M3 byte produced by canonical GLSL while making the Workgroup
        # array's logical component type float8 for the cooperative load.
        q8_accesses = [
            (i, match)
            for i, line in enumerate(lines)
            if (match := ACCESS_RE.match(line)) and match.group(4) == q8_variable
        ]
        q8_store_sites = []
        for access_index, access in q8_accesses:
            pointer = access.group(2)
            stores = [
                (i, match)
                for i, line in enumerate(lines)
                if (match := STORE_RE.match(line)) and match.group(2) == pointer
            ]
            if stores:
                require(len(stores) == 1, f"Q8f pointer {pointer} has multiple stores")
                q8_store_sites.append((access_index, access, stores[0]))
        require(len(q8_store_sites) == 1, f"expected one Q8f encoder store, found {len(q8_store_sites)}")
        access_index, access, (store_index, store) = q8_store_sites[0]
        lines[access_index] = (
            f"{access.group(1)}{access.group(2)} = OpAccessChain %d096_ptr_wg_fp8 "
            f"{access.group(4)} {access.group(5)}"
        )
        bitcast_id = "%d096_q8_store_value"
        lines[store_index:store_index + 1] = [
            f"{store.group(1)}{bitcast_id} = OpBitcast %d096_fp8 {store.group(3)}",
            f"{store.group(1)}OpStore {store.group(2)} {bitcast_id}{store.group(4)}",
        ]

        replace_single(
            ACCESS_RE,
            lambda match: match.group(2) == pointer_id,
            lambda match: (
                f"{match.group(1)}{match.group(2)} = OpAccessChain %d096_ptr_wg_fp8 "
                f"{q8_variable} {match.group(5)}"
            ),
        )
        replace_single(
            LOAD_RE,
            lambda match: match.group(2) == old_result,
            lambda match: (
                f"{match.group(1)}{match.group(2)} = OpCooperativeMatrixLoadKHR %d096_cm_f8_b "
                f"{match.group(4)} {match.group(5)} {match.group(6)}{match.group(7)}"
            ),
        )
        bypass_holder(old_result, "%d096_cm_f8_b", "fused Q8f")

    if pv_f8:
        # P5: retype the two PV cooperative loads (P and V) to fp8.
        # P loads from the shared f16vec4 Psh placeholder; the transform
        # redirects it to the parallel uint8 Psh8 array (dense [col*Br+row]
        # layout). V loads directly from the unused f16 placeholder descriptor
        # data_v, which is retyped to a raw fp8 block; the f16 kvsh fallback
        # load in the SHMEM_STAGING!=0 branch is intentionally left untouched.
        psh_variable = next(result_id for result_id, name in names.items() if name == "Psh")
        psh8_variable = next(result_id for result_id, name in names.items() if name == "Psh8")
        # glslc names used descriptors as empty strings; identify the V
        # placeholder through its struct member name instead.
        member_owner = next(
            match.group(1) for line in lines
            if (match := re.match(r'^\s*OpMemberName\s+(%\S+)\s+0\s+"data_v"\s*$', line))
        )
        v_pointer_type = next(
            match.group(1) for line in lines
            if (match := re.match(rf'^\s*(%\S+)\s*=\s*OpTypePointer\s+StorageBuffer\s+{re.escape(member_owner)}\s*$', line))
        )
        data_v_variable = next(
            match.group(2) for line in lines
            if (match := re.match(rf'^(\s*)(%\S+)\s*=\s*OpVariable\s+{re.escape(v_pointer_type)}\s+StorageBuffer\s*$', line))
        )
        vv4_owner = next(
            match.group(1) for line in lines
            if (match := re.match(r'^\s*OpMemberName\s+(%\S+)\s+0\s+"data_vv4"\s*$', line))
        )
        vv4_pointer_type = next(
            match.group(1) for line in lines
            if (match := re.match(rf'^\s*(%\S+)\s*=\s*OpTypePointer\s+StorageBuffer\s+{re.escape(vv4_owner)}\s*$', line))
        )
        data_vv4_variable = next(
            match.group(2) for line in lines
            if (match := re.match(rf'^(\s*)(%\S+)\s*=\s*OpVariable\s+{re.escape(vv4_pointer_type)}\s+StorageBuffer\s*$', line))
        )

        # Earlier edits insert prologue lines that shift line numbers, so find
        # the two PV cooperative loads (Psh, data_vv4 bases) structurally
        # instead of reusing the pre-edit audit line numbers.
        pv_loads = []
        for index, line in enumerate(lines):
            load_match = LOAD_RE.match(line)
            if not load_match:
                continue
            load_pointer = load_match.group(4)
            access = next(
                (item for item in (ACCESS_RE.match(candidate) for candidate in lines)
                 if item and item.group(2) == load_pointer),
                None,
            )
            require(access is not None, f"cannot trace PV load pointer {load_pointer}")
            if access.group(4) in {psh_variable, data_v_variable}:
                pv_loads.append(index)
        require(len(pv_loads) == 2, f"expected two P5 PV loads, found {len(pv_loads)}")

        pv_items = []
        for index in pv_loads:
            match = LOAD_RE.match(lines[index])
            assert match
            _, result_id, old_type, pointer_id, layout_id, stride_id, tail = match.groups()
            access = next(
                (item for item in (ACCESS_RE.match(line) for line in lines)
                 if item and item.group(2) == pointer_id),
                None,
            )
            require(access is not None, f"cannot trace PV load pointer {pointer_id}")
            pv_items.append((index, result_id, old_type, pointer_id, layout_id, stride_id, tail, access))

        p_item = next(item for item in pv_items if item[7].group(4) == psh_variable)
        v_item = next(item for item in pv_items if item[7].group(4) == data_v_variable)

        # --- P load: Psh (f16vec4 Workgroup) -> Psh8 (fp8 Workgroup) ---
        p_load_line, p_result, _, p_pointer, p_layout, p_stride, p_tail, p_access = p_item
        require(p_layout.endswith("int_1"), "PV P load is no longer column-major")
        require(p_stride == "%psh_stride", "PV P load no longer uses the psh_stride placeholder")
        p_offset_id = p_access.group(5)
        p_offset_mul = next(
            (i, match)
            for i, line in enumerate(lines)
            if (match := re.match(rf"^(\s*){re.escape(p_offset_id)}\s*=\s*OpIMul\s+%uint\s+(%\S+)\s+%psh_stride\s*$", line))
        )
        require(p_offset_mul is not None, "cannot locate the PV P load offset multiply")
        p_mul_line, p_mul = p_offset_mul
        # Dense Psh8 layout: every KV row occupies Br bytes, so the access
        # chain offset (bc_chunk*MatBc*psh_stride) becomes bc_chunk*MatBc*Br.
        lines[p_mul_line] = f"{p_mul.group(1)}{p_mul.group(0)[:p_mul.end() - len('psh_stride')]}uint_16"
        lines[p_load_line] = (
            f"{p_item[7].group(1)}{p_result} = OpCooperativeMatrixLoadKHR %d096_cm_f8_a "
            f"{p_pointer} {p_layout} %uint_16{p_tail}"
        )
        replace_single(
            WORKGROUP_VARIABLE_RE,
            lambda match: match.group(2) == psh8_variable,
            lambda match: f"{match.group(1)}{match.group(2)} = OpVariable %d096_ptr_wg_arr_pv_fp8 Workgroup",
        )
        # Convert the Psh8 byte stores to bitcast fp8 stores (like the Q8f
        # encoder) so the fp8 array's logical component type stays float8.
        psh8_accesses = [
            (i, match)
            for i, line in enumerate(lines)
            if (match := ACCESS_RE.match(line)) and match.group(4) == psh8_variable
        ]
        require(psh8_accesses, "no Psh8 access chains found")
        # Rewrite the access chains 1:1 first (no insertions, indices stable).
        for access_index, access in psh8_accesses:
            lines[access_index] = (
                f"{access.group(1)}{access.group(2)} = OpAccessChain %d096_ptr_wg_fp8 "
                f"{access.group(4)} {access.group(5)}"
            )
        psh8_stores = []
        for access_index, access in psh8_accesses:
            pointer = access.group(2)
            stores = [
                (i, match)
                for i, line in enumerate(lines)
                if (match := STORE_RE.match(line)) and match.group(2) == pointer
            ]
            require(len(stores) == 1, f"Psh8 pointer {pointer} has {len(stores)} stores")
            psh8_stores.append(stores[0])
        require(psh8_stores, "no Psh8 encoder stores found")
        # Insert the bitcast+store pairs bottom-up so earlier indices hold.
        for n, (store_index, store) in reversed(list(enumerate(psh8_stores))):
            bitcast_id = f"%d096_pv8_bc{n}"
            lines[store_index:store_index + 1] = [
                f"{store.group(1)}{bitcast_id} = OpBitcast %d096_fp8 {store.group(3)}",
                f"{store.group(1)}OpStore {store.group(2)} {bitcast_id}{store.group(4)}",
            ]
        replace_single(
            ACCESS_RE,
            lambda match, p=p_pointer: match.group(2) == p,
            lambda match: (
                f"{match.group(1)}{match.group(2)} = OpAccessChain %d096_ptr_wg_fp8 "
                f"{psh8_variable} {p_offset_id}"
            ),
        )

        # --- V load: data_v (f16 element placeholder) -> raw fp8 block ---
        v_load_line, v_result, _, v_pointer, v_layout, v_stride, v_tail, v_access = v_item
        require(v_layout.endswith("int_0"), "PV V load is no longer row-major")
        # data_v is used only by this cooperative load; retype it to a raw fp8
        # block. Offsets and stride are element counts, which match fp8 bytes.
        replace_single(
            VARIABLE_RE,
            lambda match, v=data_v_variable: match.group(2) == v,
            lambda match: f"{match.group(1)}{match.group(2)} = OpVariable %d096_ptr_fp8_block StorageBuffer",
        )
        replace_single(
            ACCESS_RE,
            lambda match, p=v_pointer: match.group(2) == p,
            lambda match: (
                f"{match.group(1)}{match.group(2)} = OpAccessChain %d096_ptr_fp8 "
                f"{data_v_variable} {match.group(5)}"
            ),
        )
        # The P-block edits inserted lines, so relocate the V load by its
        # (now unique) pointer instead of the stale recorded index.
        v_load_index = next(
            i for i, line in enumerate(lines)
            if (match := LOAD_RE.match(line)) and match.group(4) == v_pointer
        )
        lines[v_load_index] = (
            f"{v_item[7].group(1)}{v_result} = OpCooperativeMatrixLoadKHR %d096_cm_f8_b "
            f"{v_pointer} {v_layout} {v_stride}{v_tail}"
        )
        # Retype the f16 Function holders KMat/QMat to fp8 so the retyped P/V
        # loads flow through them; the merge-block holder loads (canonical,
        # defined on all paths for Function variables) then feed the fp8
        # MulAdd with no dominance issues. The dead f16 kvsh fallback store is
        # removed; the S-stage holder loads are already bypassed above.
        holder_re = re.compile(r'^(\s*)(%\S+)\s*=\s*OpVariable\s+(%\S+)\s+Function\s*$')
        holders = {m.group(2): m for i, m in enumerate(holder_re.match(l) for l in lines) if m}
        require("KMat" in names.values() and "QMat" in names.values(), "KMat/QMat names missing")
        k_mat = next(rid for rid, name in names.items() if name == "KMat")
        q_mat = next(rid for rid, name in names.items() if name == "QMat")
        require(k_mat in holders and q_mat in holders, "KMat/QMat Function holders missing")
        for holder_id, holder_type in ((k_mat, "%d096_ptr_fn_cm_f8_a"), (q_mat, "%d096_ptr_fn_cm_f8_b")):
            holder_line = next(i for i, m in enumerate(holder_re.match(l) for l in lines) if m and m.group(2) == holder_id)
            m = holder_re.match(lines[holder_line])
            lines[holder_line] = f"{m.group(1)}{m.group(2)} = OpVariable {holder_type} Function"
        # The f16 kvsh fallback store into the now-fp8 QMat holder is dead
        # (SHMEM_STAGING == 0); drop it and retype the merge-block loads.
        kvsh_variable = next(rid for rid, name in names.items() if name == "kvsh")
        kvsh_result = next(
            (m.group(2) for i, line in enumerate(lines)
             if (m := LOAD_RE.match(line)) and next(
                 (a for a in (ACCESS_RE.match(c) for c in lines)
                  if a and a.group(2) == m.group(4)),
                 None,
             ) and next(
                 a for a in (ACCESS_RE.match(c) for c in lines)
                 if a and a.group(2) == m.group(4)
             ).group(4) == kvsh_variable),
            None,
        )
        require(kvsh_result is not None, "cannot locate the kvsh fallback load")
        kvsh_store = next(
            (i, m) for i, line in enumerate(lines)
            if (m := STORE_RE.match(line)) and m.group(3) == kvsh_result
        )
        kvsh_store_index, kvsh_store_match = kvsh_store
        lines[kvsh_store_index] = f"; D096: remove dead f16 kvsh store into fp8 QMat holder ({kvsh_result})"
        holder_load_re = re.compile(r'^(\s*)(%\S+)\s*=\s*OpLoad\s+(%\S+)\s+(' + re.escape(k_mat) + '|' + re.escape(q_mat) + r')\s*$')
        holder_loads = [(i, m) for i, line in enumerate(lines) if (m := holder_load_re.match(line))]
        require(len(holder_loads) == 2, f"expected two holder loads, found {len(holder_loads)}")
        for i, m in holder_loads:
            target = "%d096_cm_f8_a" if m.group(4) == k_mat else "%d096_cm_f8_b"
            lines[i] = f"{m.group(1)}{m.group(2)} = OpLoad {target} {m.group(4)}"

    transformed = "\n".join(lines) + "\n"
    audit_disassembly(transformed, source + " (transformed)", final_profile)
    return transformed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="output .spv")
    parser.add_argument("--spvasm-output", type=Path)
    parser.add_argument("--spirv-dis")
    parser.add_argument("--spirv-as")
    parser.add_argument("--spirv-val")
    parser.add_argument("--fused-q", action="store_true", help="redirect Q load to shared Q8f (P3)")
    parser.add_argument("--v-f16", action="store_true", help="V comes from a dense f16 preconvert buffer (P4)")
    parser.add_argument("--pv-f8", action="store_true", help="PV stage runs fp8: P quantized to shared E4M3, V raw fp8, f32 acc (P5)")
    args = parser.parse_args()

    try:
        spirv_dis = find_tool(args.spirv_dis, "spirv-dis")
        spirv_as = find_tool(args.spirv_as, "spirv-as")
        spirv_val = find_tool(args.spirv_val, "spirv-val")
        transformed = transform(disassemble(args.input, spirv_dis), str(args.input), args.fused_q, args.v_f16, args.pv_f8)
        asm_out = args.spvasm_output
        with tempfile.TemporaryDirectory(prefix="d096-p2-transform-") as td:
            temporary = Path(td) / "p2.spvasm"
            assembly = asm_out or temporary
            assembly.parent.mkdir(parents=True, exist_ok=True)
            assembly.write_text(transformed, encoding="utf-8", newline="\n")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([spirv_as, "--target-env=spv1.5", str(assembly), "-o", str(args.output)], check=True)
            subprocess.run([spirv_val, "--target-env", "vulkan1.2", "--allow-localsizeid", str(args.output)], check=True)
    except (AuditError, OSError, subprocess.CalledProcessError) as exc:
        print(f"D096 SPIR-V transform failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())