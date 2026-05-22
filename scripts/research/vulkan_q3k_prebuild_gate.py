#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultRow:
    date: str
    exp_id: str
    short_name: str
    baseline: str
    candidate: str
    delta: str
    decision: str
    artifacts: str


@dataclass(frozen=True)
class CurrentState:
    q3_stride_padding: int | None
    q3_stride: int | None
    q3_load_vec: int | None
    corrected_loadvec4: bool
    rejected_probe_leftovers: list[str]


@dataclass(frozen=True)
class Prior:
    key: str
    label: str
    gain_pct: float
    decision: str
    evidence: str


PRIORS: tuple[Prior, ...] = (
    Prior("wn48", "E091 wn48 tile profile", 1.09, "needs-layout-validation", "pp7488 r3 972.31 vs E086 961.82, but E093 static scout marks WN48 invalid for BN=128"),
    Prior("loadvec4", "E086 corrected Q3_K LOAD_VEC_A=4", 4.25, "keep", "pp7488 r3 961.82 vs E082 922.62"),
    Prior("stride18", "E082 Q3_K coopmat stride18", 1.58, "keep", "pp7488 r3 922.62 vs 908.23"),
    Prior("loadvec8", "E087 corrected Q3_K LOAD_VEC_A=8", -1.50, "reject", "pp7488 r1 947.44 vs E086 961.82"),
    Prior("pairscale", "E088 pair-scale helper", -0.20, "reject", "pp7488 r1 959.89 vs E086 961.82"),
    Prior("scalecache", "E088 calibrated scale-reuse signal", -0.20, "low-ceiling", "removing half of per-pair scale decode did not improve pp7488"),
    Prior("dequantreuse", "Q3_K dequant reuse without pair-count reduction", 0.10, "low-ceiling", "scale/helper reuse is E088-calibrated non-positive unless the idea removes substantial dequant work"),
    Prior("bk64", "Q3_K BK=64 static scout", 0.0, "needs-resource-proof", "halves K-loop barriers but leaves full-K dequant/B traffic unchanged and raises Q3 LDS to 34816 B"),
    Prior("bk16", "Q3_K BK=16 static scout", 0.0, "low-ceiling", "doubles K-loop barriers; only plausible if pipeline resources improve materially"),
    Prior("bm256", "E098/E146 Q3_K BM256 large tile", -5.78, "reject", "E098 bm256 909.59 vs 983.21; E146 bm256 916.62 vs 972.84 with 31744 B LDS"),
    Prior("bn256", "E098/E143 Q3_K BN256 large tile", -3.67, "reject", "E098 bn256 947.12 vs 983.21; E143 BN256 variants regressed -32% with high LDS/register pressure"),
    Prior("bn192", "E143 Q3_K BN192 large-N route", -21.90, "reject", "bn192-wn96 760.78 vs 974.19 and nearby large-N variants were worse"),
    Prior("largetile", "E098/E143/E146 large-tile family", -5.0, "reject-without-new-topology", "plain BM/BN growth in current mul_mm.comp repeatedly loses to LDS/register/occupancy pressure"),
    Prior("packed32", "E090 packed32 pair helper", -1.04, "reject", "pp7488 r1 951.79 vs E086 961.82"),
    Prior("stride20", "E089 stride20 recheck", -5.21, "reject", "pp7488 r1 911.74 vs E086 961.82"),
    Prior("stride22", "E084 stride22", -3.06, "reject", "pp7488 r1 894.36 vs E082 922.62"),
    Prior("stride19", "E083 stride19", -31.30, "reject", "pp7488 r1 633.65 vs E082 922.62"),
    Prior("stride16", "E081 stride16", -9.33, "reject", "pp7488 r1 802.36 vs 884.96"),
    Prior("f16dequant", "E079 f16 dequant arithmetic", -1.46, "reject", "pp7488 r1 872.01 vs 884.96"),
    Prior("unsignedscale", "E080 unsigned scale arithmetic", -0.44, "reject", "pp7488 r1 881.07 vs 884.96"),
)


TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "wn48": ("wn48", "wn 48", "wn=48"),
    "loadvec4": ("load_vec_a=4", "load_vec_a 4", "loadvec4", "load vec 4", "load_vec_a == 4"),
    "loadvec8": ("load_vec_a=8", "load_vec_a 8", "loadvec8", "load vec 8", "load_vec_a == 8"),
    "stride18": ("stride18", "stride 18", "bk / 2 + 2"),
    "stride20": ("stride20", "stride 20", "bk / 2 + 4"),
    "stride22": ("stride22", "stride 22", "bk / 2 + 6"),
    "stride19": ("stride19", "stride 19", "bk / 2 + 3"),
    "stride16": ("stride16", "stride 16", "bk / 2"),
    "pairscale": ("pairscale", "pair-scale", "pair scale", "dequant_q3_k_pair2"),
    "scalecache": ("scale cache", "cache scale", "scale reuse", "precompute scale", "precomputed scale", "scale table", "dl cache"),
    "dequantreuse": ("dequant reuse", "reuse dequant", "block-level dequant", "block level dequant", "block reuse", "fused pair", "fused q3 decode"),
    "bk64": ("bk64", "bk 64", "bk=64", "bk = 64"),
    "bk16": ("bk16", "bk 16", "bk=16", "bk = 16"),
    "bm256": ("bm256", "bm 256", "bm=256", "bm = 256"),
    "bn256": ("bn256", "bn 256", "bn=256", "bn = 256"),
    "bn192": ("bn192", "bn 192", "bn=192", "bn = 192", "bn192-wn96", "bn 192 wn 96"),
    "largetile": ("large tile", "large-tile", "larger tile", "warptile", "tile growth", "larger-n", "larger-m"),
    "packed32": ("packed32", "packed 32", "data_a_packed32"),
    "f16dequant": ("f16 dequant", "float16 dequant", "half dequant"),
    "unsignedscale": ("unsigned scale", "uint scale", "uint8_t scale"),
}


def required_local_speedup(target_share: float, target_total_speedup: float) -> float | None:
    denom = (1.0 / target_total_speedup) - (1.0 - target_share)
    if denom <= 0.0:
        return None
    return target_share / denom


def total_speedup_from_local(target_share: float, local_speedup: float) -> float:
    return 1.0 / ((1.0 - target_share) + target_share / local_speedup)


def parse_results_log(path: Path) -> list[ResultRow]:
    if not path.exists():
        return []

    rows: list[ResultRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("| 20"):
            continue
        cols = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cols) < 8:
            continue
        rows.append(ResultRow(*cols[:8]))
    return rows


def first_match_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.S)
    return int(match.group(1)) if match else None


def detect_q3_load_vec(gen_text: str) -> int | None:
    lines = gen_text.splitlines()
    for idx, line in enumerate(lines):
        if 'q3_k' not in line:
            continue
        window = "\n".join(lines[idx : idx + 4])
        match = re.search(r'load_vec_quant\s*=\s*"(\d+)"', window)
        if match:
            return int(match.group(1))
    return None


def extract_q3_branch(funcs_text: str) -> str:
    marker = "#elif defined(DATA_A_Q3_K)"
    start = funcs_text.find(marker)
    if start < 0:
        return ""
    end = funcs_text.find("#elif defined(DATA_A_Q4_K)", start)
    if end < 0:
        return funcs_text[start:]
    return funcs_text[start:end]


def detect_current_state(root: Path) -> CurrentState:
    mul_mm = root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp"
    funcs = root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl"
    gen = root / "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"

    mul_text = mul_mm.read_text(encoding="utf-8") if mul_mm.exists() else ""
    funcs_text = funcs.read_text(encoding="utf-8") if funcs.exists() else ""
    gen_text = gen.read_text(encoding="utf-8") if gen.exists() else ""
    q3_branch = extract_q3_branch(funcs_text)

    padding = first_match_int(r"defined\(DATA_A_Q3_K\).*?#define\s+SHMEM_STRIDE\s+\(BK\s*/\s*2\s*\+\s*(\d+)\)", mul_text)
    stride = 16 + padding if padding is not None else None
    load_vec = detect_q3_load_vec(gen_text)

    corrected = all(
        needle in funcs_text
        for needle in (
            "#if LOAD_VEC_A == 4",
            "const uint pair_idx = idx * 2;",
            "dequant_q3_k_pair(pair_idx + 1)",
        )
    )

    leftovers: list[str] = []
    leftover_checks = {
        "E087 LOAD_VEC_A=8 branch": "#if LOAD_VEC_A == 8",
        "E088 pair-scale helper": "dequant_q3_k_pair2(",
        "E090 packed32 pair helper": "dequant_q3_k_pair2_packed32",
    }
    for label, needle in leftover_checks.items():
        if needle in q3_branch:
            leftovers.append(label)

    return CurrentState(padding, stride, load_vec, corrected, leftovers)


def get_git_diff(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "diff", "--", "ggml/src/ggml-vulkan", "scripts/research"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""
    return completed.stdout


def candidate_text_from_args(args: argparse.Namespace, root: Path) -> str:
    pieces: list[str] = []
    if args.candidate:
        pieces.append(args.candidate)
    if args.diff_file:
        diff_path = Path(args.diff_file)
        if not diff_path.is_absolute():
            diff_path = root / diff_path
        if diff_path.exists():
            pieces.append(diff_path.read_text(encoding="utf-8", errors="replace"))
    if args.use_git_diff:
        pieces.append(get_git_diff(root))
    return "\n".join(pieces).strip()


def matched_priors(candidate_text: str) -> list[Prior]:
    text = candidate_text.lower()
    matches: list[Prior] = []
    for prior in PRIORS:
        aliases = TOKEN_ALIASES.get(prior.key, (prior.key,))
        if any(alias in text for alias in aliases):
            matches.append(prior)
    return matches


def h31_rows(rows: list[ResultRow]) -> list[ResultRow]:
    out = []
    for row in rows:
        haystack = f"{row.exp_id} {row.short_name} {row.decision} {row.delta}".lower()
        if "h31" in haystack or ("vulkan" in haystack and "q3" in haystack):
            out.append(row)
    return out


def historical_analogs(rows: list[ResultRow], candidate_text: str) -> list[tuple[int, ResultRow]]:
    if not candidate_text:
        return [(1, row) for row in h31_rows(rows)[-16:]]

    text_tokens = set(re.findall(r"[a-z0-9_]+", candidate_text.lower()))
    scored: list[tuple[int, ResultRow]] = []
    for row in h31_rows(rows):
        row_text = f"{row.exp_id} {row.short_name} {row.candidate} {row.delta} {row.decision}".lower()
        row_tokens = set(re.findall(r"[a-z0-9_]+", row_text))
        score = len(text_tokens & row_tokens)
        for key, aliases in TOKEN_ALIASES.items():
            candidate_has_alias = any(alias in candidate_text.lower() for alias in aliases)
            row_has_alias = any(alias in row_text for alias in aliases)
            row_has_key = key in row_text or key.replace("loadvec", "load_vec") in row_text
            if candidate_has_alias and (row_has_alias or row_has_key):
                score += 3
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: (item[0], item[1].date, item[1].exp_id), reverse=True)
    return scored[:8]


def row_is_rejected_or_negative(row: ResultRow) -> bool:
    text = f"{row.delta} {row.decision}".lower()
    return (
        "reject" in text
        or "negative" in text
        or "regress" in text
        or bool(re.search(r"(^|[^0-9])-[0-9]+(?:\.[0-9]+)?\s*%", text))
    )


def print_current_state(state: CurrentState) -> None:
    print("## Current Static State")
    print()
    print(f"- q3_coopmat_stride: {state.q3_stride if state.q3_stride is not None else 'unknown'}")
    print(f"- q3_stride_padding_expr: BK/2+{state.q3_stride_padding if state.q3_stride_padding is not None else '?'}")
    print(f"- q3_load_vec_a: {state.q3_load_vec if state.q3_load_vec is not None else 'unknown'}")
    print(f"- corrected_loadvec4_present: {str(state.corrected_loadvec4).lower()}")
    if state.q3_stride is not None and state.q3_load_vec:
        bm = 64
        bk = 32
        pair_dequants = bm * bk // 2
        load_invocations = bm * bk // state.q3_load_vec
        padding_slots = bm * max(state.q3_stride - bk // 2, 0)
        print(f"- q3_pair_dequants_per_A_tile: {pair_dequants}")
        print(f"- A_load_invocations_per_tile: {load_invocations}")
        print(f"- A_pairs_per_invocation: {state.q3_load_vec / 2:.1f}")
        print(f"- A_padding_f16vec2_slots: {padding_slots}")
        q3_scale_groups = bm * (bk // 16)
        repeated_scale_decodes = max(pair_dequants - q3_scale_groups, 0)
        print(f"- theoretical_q3_scale_groups_per_A_tile: {q3_scale_groups}")
        print(f"- repeated_scale_decodes_if_no_reuse: {repeated_scale_decodes}")
        print("- scale_reuse_calibration: E088 removed roughly half of repeated scale decode and measured -0.20%, so scale-cache-only ideas are low priority")
    if state.rejected_probe_leftovers:
        print(f"- rejected_probe_leftovers: {', '.join(state.rejected_probe_leftovers)}")
    else:
        print("- rejected_probe_leftovers: none detected")
    print()


def print_target_math(args: argparse.Namespace) -> tuple[float, float | None]:
    target_total = args.goal_total_speedup
    if target_total is None:
        target_total = args.target_pp / args.baseline_pp

    req_local = required_local_speedup(args.target_share, target_total)
    print("## Target Math")
    print()
    print(f"- baseline_pp: {args.baseline_pp:.2f}")
    print(f"- target_pp: {args.target_pp:.2f}")
    print(f"- target_total_speedup: {target_total:.4f} ({(target_total - 1.0) * 100.0:+.2f}%)")
    print(f"- target_hotspot_share: {args.target_share:.4f}")
    if req_local is None:
        print("- required_local_speedup: impossible for this hotspot share")
    else:
        print(f"- required_local_speedup_if_only_hotspot_changes: {req_local:.4f} ({(req_local - 1.0) * 100.0:+.2f}%)")
    if args.local_gain_pct is not None:
        local_speedup = 1.0 + args.local_gain_pct / 100.0
        total = total_speedup_from_local(args.target_share, local_speedup)
        print(f"- user_estimated_local_gain: {args.local_gain_pct:+.2f}%")
        print(f"- projected_total_speedup_from_estimate: {total:.4f} ({(total - 1.0) * 100.0:+.2f}%)")
    print()
    return target_total, req_local


def print_dequant_reuse_sanity(state: CurrentState, req_local: float | None) -> None:
    if state.q3_stride is None or not state.q3_load_vec:
        return

    bm = 64
    bk = 32
    pair_dequants = bm * bk // 2
    load_invocations = bm * bk // state.q3_load_vec
    scale_groups = bm * (bk // 16)
    repeated_scale_decodes = max(pair_dequants - scale_groups, 0)
    repeated_share = repeated_scale_decodes / pair_dequants if pair_dequants else 0.0

    print("## Q3 Dequant Reuse Sanity")
    print()
    print(f"- current_pair_dequants_per_A_tile: {pair_dequants}")
    print(f"- current_A_load_invocations_per_tile: {load_invocations}")
    print(f"- repeated_scale_decode_share_if_no_reuse: {repeated_share:.3f}")
    print("- calibration: E088 reused roughly half of the repeated scale/dequant helper work and measured `-0.20%` pp7488 vs E086")
    if req_local is not None:
        req_gain = (req_local - 1.0) * 100.0
        print(f"- gate: reuse-only candidates need an explicit mechanism for about `{req_gain:.1f}%` local hotspot speedup; scale/helper reuse alone is below that bar")
    print("- prebuild implication: require an instruction/load-count model showing substantial pair-count, LDS traffic, or coopmat-work reduction before building another dequant-reuse probe")
    print()


def print_candidate_signal(
    candidate_text: str,
    priors: list[Prior],
    analogs: list[tuple[int, ResultRow]],
    args: argparse.Namespace,
    req_local: float | None,
) -> str:
    print("## Candidate Signal")
    print()
    if not candidate_text:
        print("- candidate_text: not provided; showing current-state gate only")
    else:
        compact = re.sub(r"\s+", " ", candidate_text).strip()
        print(f"- candidate_text: {compact[:220]}{'...' if len(compact) > 220 else ''}")

    if priors:
        print()
        print("| matched_prior | measured_gain_pct | decision | evidence |")
        print("|---|---:|---|---|")
        for prior in priors:
            print(f"| {prior.label} | {prior.gain_pct:+.2f} | {prior.decision} | {prior.evidence} |")
    else:
        print("- matched_prior: none")

    estimated_gain = max((p.gain_pct for p in priors), default=None)
    blocking_priors = [
        prior
        for prior in priors
        if prior.gain_pct <= 0.0 or "reject" in prior.decision
    ]
    blocking_analogs = [
        (score, row)
        for score, row in analogs
        if score >= args.min_blocking_analog_score and row_is_rejected_or_negative(row)
    ]
    decision = "needs-mechanism-estimate"
    reason = "no measured analogue matched; provide --local-gain-pct or add a cheap trace/stat gate"

    if blocking_priors:
        decision = "skip-build-unless-new-topology"
        reason = f"matched rejected prior: {blocking_priors[0].label}"
    elif blocking_analogs:
        score, row = blocking_analogs[0]
        decision = "skip-build-unless-new-topology"
        reason = f"matched rejected historical analogue {row.exp_id} with score {score}"
    elif estimated_gain is not None:
        if any("needs-layout-validation" in prior.decision for prior in priors):
            decision = "validate-layout-before-build"
            reason = "matched positive prior has invalid/static-suspect warptile layout"
        elif any("needs-resource-proof" in prior.decision for prior in priors):
            decision = "needs-resource-proof"
            reason = "static scout shows a tradeoff, not a clear work reduction; collect pipeline/resource proof before build"
        elif estimated_gain <= 0.0:
            decision = "skip-build"
            reason = "closest measured analogue is non-positive"
        elif estimated_gain < args.min_build_gain_pct:
            decision = "skip-or-doc-only"
            reason = f"estimated gain {estimated_gain:.2f}% is below min build gate {args.min_build_gain_pct:.2f}%"
        else:
            decision = "build-if-new-lane-or-confirming"
            reason = "positive measured analogue exists"

    if args.local_gain_pct is not None and not blocking_priors and not blocking_analogs:
        local_speedup = 1.0 + args.local_gain_pct / 100.0
        projected_total = total_speedup_from_local(args.target_share, local_speedup)
        projected_gain_pct = (projected_total - 1.0) * 100.0
        if projected_gain_pct < args.min_build_gain_pct:
            decision = "skip-build"
            reason = "projected Amdahl gain is below build gate"
        elif req_local is not None and local_speedup < req_local and args.require_target_closing:
            decision = "defer-for-higher-ceiling"
            reason = "candidate does not have enough local gain to close the selected target"
        else:
            decision = "build-candidate"
            reason = "projected Amdahl gain clears gate"

    print()
    print(f"- prebuild_decision: {decision}")
    print(f"- reason: {reason}")
    print()
    return decision


def print_analogs(analogs: list[tuple[int, ResultRow]]) -> None:
    print("## Historical Analogs")
    print()
    if not analogs:
        print("No matching H31/Vulkan Q3_K rows found.")
        print()
        return
    print("| score | id | short_name | delta | decision |")
    print("|---:|---|---|---|---|")
    for score, row in analogs:
        print(f"| {score} | {row.exp_id} | {row.short_name} | {row.delta} | {row.decision} |")
    print()


def print_guidance(args: argparse.Namespace, req_local: float | None) -> None:
    print("## Guidance")
    print()
    if req_local is not None:
        req_gain = (req_local - 1.0) * 100.0
        print(f"- To close the selected pp target via the Q3_K hotspot alone, look for mechanisms plausibly worth about `{req_gain:.1f}%` local speedup, not repeated `1-2%` micro-edits.")
    print("- Build only when the candidate clears the Amdahl gate, has no close rejected analogue, or targets a new high-share mechanism.")
    print("- For H31, prefer candidates that change algorithmic work, valid tile coverage, or memory layout at the active `matmul_q3_k_f32_f16acc_aligned_l` route.")
    print("- For BK-depth ideas, use `vulkan_warptile_static_scout.py` first: BK changes mostly trade K-loop barriers against LDS/resources without reducing full-K dequant/B traffic.")
    print("- Treat helper-only rewrites, wider packed loads, and neighboring stride tweaks as low-ceiling unless this gate shows a new reason.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-build gate for Vulkan Q3_K prefill candidates")
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument("--results-log", default="docs/research/RESULTS_LOG.md")
    parser.add_argument("--candidate", default="", help="freeform candidate description")
    parser.add_argument("--diff-file", default="", help="optional diff/prototype text file to analyze")
    parser.add_argument("--use-git-diff", action="store_true", help="include current git diff for shader/research files")
    parser.add_argument("--baseline-pp", type=float, default=961.82, help="current accepted Vulkan pp gate baseline")
    parser.add_argument("--target-pp", type=float, default=1097.66, help="target pp gate, usually ROCm control")
    parser.add_argument("--target-share", type=float, default=0.7184, help="hotspot wall share in [0,1]")
    parser.add_argument("--goal-total-speedup", type=float, default=None, help="override target total speedup")
    parser.add_argument("--local-gain-pct", type=float, default=None, help="candidate local hotspot gain estimate")
    parser.add_argument("--min-build-gain-pct", type=float, default=0.75, help="minimum projected/measured gain to justify a build")
    parser.add_argument("--min-blocking-analog-score", type=int, default=3, help="minimum historical analog score that can block a build when rejected/negative")
    parser.add_argument("--require-target-closing", action="store_true", help="require candidate to close selected target, not just improve")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    if args.target_share <= 0.0 or args.target_share >= 1.0:
        raise SystemExit("ERROR: --target-share must be in (0,1)")
    if args.baseline_pp <= 0.0 or args.target_pp <= 0.0:
        raise SystemExit("ERROR: --baseline-pp and --target-pp must be positive")

    results_path = Path(args.results_log)
    if not results_path.is_absolute():
        results_path = root / results_path

    rows = parse_results_log(results_path)
    state = detect_current_state(root)
    candidate_text = candidate_text_from_args(args, root)
    priors = matched_priors(candidate_text)
    analogs = historical_analogs(rows, candidate_text)

    print("# Vulkan Q3_K Prebuild Gate")
    print()
    print_current_state(state)
    _, req_local = print_target_math(args)
    print_dequant_reuse_sanity(state, req_local)
    print_candidate_signal(candidate_text, priors, analogs, args, req_local)
    print_analogs(analogs)
    print_guidance(args, req_local)

    if state.rejected_probe_leftovers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
