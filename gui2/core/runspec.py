"""RunSpec: the single description of a run.

`to_argv` is the only place a llama-server command line is built. Bench and
autotune reuse it through `gui2.core.bench`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Literal

from gui2.core import params
from gui2.core.gguf import context_text, read_facts
from gui2.core.params import SCHEMA, Param, aliases_of, flags_in, parse_extra

THINKING_OFF = '{"enable_thinking":false,"preserve_thinking":false}'
RPC_ENDPOINT_RE = re.compile(r"^[A-Za-z0-9_.\-]+:\d{1,5}$")

#: addresses that mean "every network card", and therefore "the whole network"
OPEN_HOSTS = frozenset({"0.0.0.0", "::", "*"})


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Every value a run needs. Defaults are the shipped defaults."""

    model: str = ""
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: int = 131072
    # 0 means "say nothing and let llama-server decide": it already picks the
    # physical core count for -t and sizes --threads-http from the slots, and
    # both answers beat a number a person would guess
    threads: int = 0
    threads_http: int = 0
    batch_size: int = 512
    ubatch_size: int = 128
    parallel: int = 1
    kv_unified: bool = False
    gpu_layers_all: bool = True
    gpu_layers: int = 64
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    conversation_cache: bool = False
    metrics: bool = True
    cache_ram: int = 8192
    ctx_checkpoints: int = 8
    checkpoint_every_n_tokens: int = 4096
    mmproj: str = ""
    mmproj_offload: bool = True
    spec_type: str = "none"
    spec_draft_n_max: int = 2
    ngram_n_min: int = 12
    ngram_n_match: int = 16
    ngram_n_max: int = 32
    embeddings: bool = False
    flash_attn: str = "auto"
    no_warmup: bool = False
    no_mmap: bool = False
    disable_thinking: bool = False
    fit: str = "on"
    rpc_endpoints: str = ""
    devices: str = ""
    split_mode: str = ""
    tensor_split: str = ""
    api_key: str = ""

    # not a llama-server flag: selects which build supplies the binary
    build_dir: str = ""
    extra_args: str = ""

    def with_values(self, values: dict[str, Any]) -> "RunSpec":
        """Copy with only known fields applied, coerced to the field type."""
        known = {field.name: field.type for field in fields(self)}
        updates: dict[str, Any] = {}
        for key, value in values.items():
            if key not in known:
                continue
            updates[key] = _coerce(value, getattr(self, key))
        return replace(self, **updates)


DEFAULTS = RunSpec()


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "on", "true", "yes"}
        return bool(value)
    if isinstance(current, int):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return current
    if isinstance(current, float):
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return current
    return str(value).strip()


def parse_rpc_endpoints(text: str) -> list[str]:
    """Valid, de-duplicated host:port endpoints from free-form text."""
    endpoints: list[str] = []
    for chunk in re.split(r"[,;\s]+", (text or "").strip()):
        if not chunk or not RPC_ENDPOINT_RE.match(chunk):
            continue
        if int(chunk.rsplit(":", 1)[1]) > 65535:
            continue
        if chunk not in endpoints:
            endpoints.append(chunk)
    return endpoints


#: llama_context pads both the total context and each slot's share to this
CONTEXT_PAD = 256


def _pad_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def slot_context(spec: RunSpec) -> tuple[int, int]:
    """The total context, and how much of it one conversation may use.

    Mirrors llama_context: unless a single cache is shared, the context is
    divided between the parallel slots and each share padded separately -- so
    asking for 128K with four slots gives four conversations of 32K, not one
    of 128K. The memory is the same either way; what changes is how long a
    single conversation may get before it is cut short, which is the part
    nobody discovers until it happens.
    """
    total = _pad_up(max(1, int(spec.ctx_size)), CONTEXT_PAD)
    slots = max(1, int(spec.parallel))
    if spec.kv_unified:
        return total, total
    per_slot = _pad_up(max(1, total // slots), CONTEXT_PAD)
    return per_slot * slots, per_slot


def rpc_device_names(count: int) -> list[str]:
    """RPC device ids in --rpc order; llama.cpp names them RPC0, RPC1, ..."""
    return [f"RPC{index}" for index in range(max(0, count))]


def _emit_spec_type(spec: RunSpec) -> list[str]:
    if spec.spec_type == "mtp":
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(spec.spec_draft_n_max)]
    if spec.spec_type == "ngram-mod":
        return [
            "--spec-type", "ngram-mod",
            "--spec-ngram-mod-n-min", str(spec.ngram_n_min),
            "--spec-ngram-mod-n-match", str(spec.ngram_n_match),
            "--spec-ngram-mod-n-max", str(spec.ngram_n_max),
        ]
    return []


#: -ngl 999 is llama.cpp's idiom for "as many layers as fit"
ALL_LAYERS = "999"

COMPOSITES = {
    "gpu_layers_all": lambda spec: [],
    "gpu_layers": lambda spec: ["-ngl", ALL_LAYERS if spec.gpu_layers_all else str(spec.gpu_layers)],
    "spec_type": _emit_spec_type,
    "spec_draft_n_max": lambda spec: [],
    "ngram_n_min": lambda spec: [],
    "ngram_n_match": lambda spec: [],
    "ngram_n_max": lambda spec: [],
    "disable_thinking": lambda spec: (
        ["--chat-template-kwargs", THINKING_OFF] if spec.disable_thinking else []
    ),
    "rpc_endpoints": lambda spec: (
        ["--rpc", ",".join(parse_rpc_endpoints(spec.rpc_endpoints))]
        if parse_rpc_endpoints(spec.rpc_endpoints) else []
    ),
}


def emit(param: Param, spec: RunSpec) -> list[str]:
    """Tokens one parameter contributes, before extra-argument overrides."""
    value = getattr(spec, param.name)
    if param.emit == "composite":
        return COMPOSITES[param.name](spec)
    if param.emit == "presence":
        return [param.flag] if value else []
    if param.emit == "absence":
        return [] if value else [param.flag]
    if value == "" or value is None:
        return []
    if param.skip_default and value == getattr(DEFAULTS, param.name):
        return []
    return [param.flag, str(value)]


def _suppressed(spec: RunSpec, param: Param) -> bool:
    """Checkpoint knobs belong to the conversation cache once it is enabled."""
    return spec.conversation_cache and param.name in {"ctx_checkpoints", "checkpoint_every_n_tokens"}


def to_argv(spec: RunSpec, binary: str | Path = "llama-server") -> list[str]:
    """The command line. Extra arguments win over any generated flag."""
    extra = parse_extra(spec.extra_args)
    overridden = flags_in(extra)

    argv = [str(binary)]
    for param in SCHEMA:
        if _suppressed(spec, param):
            continue
        tokens = emit(param, spec)
        if not tokens:
            continue
        if aliases_of(tokens[0]) & overridden:
            continue
        argv.extend(tokens)
    argv.extend(extra)
    return argv


API_KEY_MASK = "••••"
_API_KEY_RE = re.compile(r"(--api-key[=\s]+)(\S+)")


def mask_api_key(argv: list[str]) -> list[str]:
    """Copy of argv with API key values replaced, for previews and logs.

    Covers ``--api-key VALUE``, ``--api-key=VALUE`` and keys carried inside a
    packed argument such as ``--server-extra=-t 8 --api-key VALUE``.
    """
    masked: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            masked.append(API_KEY_MASK)
            hide_next = False
        elif token == "--api-key":
            masked.append(token)
            hide_next = True
        elif "--api-key" in token:
            masked.append(_API_KEY_RE.sub(rf"\g<1>{API_KEY_MASK}", token))
        else:
            masked.append(token)
    return masked


#: "error" refuses to start; "warn" starts anyway. The difference matters: a
#: server open to the network is a decision, not a mistake, and a GUI that
#: refuses to make it has taken the choice away from the person who owns it.
Level = Literal["error", "warn", "note"]


@dataclass(frozen=True, slots=True)
class Problem:
    level: Level
    message: str


def validate(spec: RunSpec, backend: str = "", supports_rpc: bool | None = None,
             available_devices: Iterable[str] | None = None) -> list[Problem]:
    """Blocking errors and behaviour notes for a spec.

    `backend` and `supports_rpc` come from the selected build's CMakeCache and
    `available_devices` from the device scan; no binary is ever probed. Model
    limits are read from the GGUF header, which is a plain file read.
    """
    problems: list[Problem] = []
    extra = parse_extra(spec.extra_args)
    extra_flags = flags_in(extra)

    facts = None
    if not spec.model:
        problems.append(Problem("error", "Select a model file"))
    elif not Path(spec.model).is_file():
        problems.append(Problem("error", f"Model file not found: {spec.model}"))
    else:
        facts = read_facts(spec.model)

    if not spec.build_dir:
        problems.append(Problem("error", "Select a build"))

    if spec.ubatch_size > spec.batch_size:
        problems.append(Problem("error", "Ubatch size must not exceed batch size"))

    if spec.host in OPEN_HOSTS and not spec.api_key:
        problems.append(Problem(
            "warn",
            f"--host {spec.host} accepts connections from the whole network and no API key "
            "is set: anyone who can reach this machine can use the model and read the "
            "conversations. Set an API key, or use 127.0.0.1.",
        ))

    if facts is not None and facts.n_ctx_train and spec.ctx_size > facts.n_ctx_train:
        problems.append(Problem(
            "note",
            f"{context_text(spec.ctx_size)} context exceeds the "
            f"{context_text(facts.n_ctx_train)} this model was trained for",
        ))

    if not spec.gpu_layers_all and spec.gpu_layers == 0:
        problems.append(Problem("note", "No layer is offloaded: the model will run on the CPU"))

    total_ctx, per_slot = slot_context(spec)
    if total_ctx != spec.ctx_size:
        problems.append(Problem(
            "note",
            f"llama.cpp will use {context_text(total_ctx)} of context: the request has to "
            f"divide into {spec.parallel} slot(s) of a multiple of {CONTEXT_PAD}",
        ))
    if per_slot < total_ctx:
        problems.append(Problem(
            "note",
            f"one conversation may use {context_text(per_slot)} of it — the rest belongs to "
            f"the other {spec.parallel - 1} slot(s); --kv-unified pools them instead",
        ))

    kv_types = {spec.cache_type_k, spec.cache_type_v}
    flash_on = spec.flash_attn == "on" or "-fa" in extra_flags or "--flash-attn" in extra_flags
    if kv_types & {"q8_0", "q4_0"} and spec.flash_attn == "off":
        problems.append(Problem("error", "A quantized KV cache needs flash attention"))
    if "f8_e4m3" in kv_types:
        if backend and backend not in {"vulkan", "rocm"}:
            problems.append(Problem("error", "f8_e4m3 KV requires a Vulkan or ROCm build"))
        if not flash_on:
            problems.append(Problem("error", "f8_e4m3 KV requires flash attention set to on"))

    if spec.spec_type == "mtp":
        if spec.parallel != 1:
            problems.append(Problem("error", "MTP requires --parallel 1"))
        if kv_types & {"q8_0", "f8_e4m3"}:
            hybrid_n = 12 if "f8_e4m3" in kv_types and spec.ctx_size >= 98304 else 8
            problems.append(Problem(
                "note",
                f"MTP auto policy keeps the last {hybrid_n} KV layers in f16 "
                "unless LLAMA_VK_MTP_KV_LAST_F16 overrides it",
            ))

    if spec.mmproj and not Path(spec.mmproj).is_file():
        problems.append(Problem("error", f"mmproj file not found: {spec.mmproj}"))

    endpoints = parse_rpc_endpoints(spec.rpc_endpoints)
    if spec.rpc_endpoints.strip() and not endpoints:
        problems.append(Problem("error", "RPC workers must be given as host:port"))
    if endpoints and supports_rpc is False:
        problems.append(Problem("error", "Selected build has GGML_RPC=OFF — --rpc will be rejected"))
    if endpoints:
        problems.append(Problem("note", f"Remote RPC workers: {', '.join(endpoints)}"))

    requested = [name for name in re.split(r"[,\s]+", spec.devices) if name]
    unavailable = {name for name in requested if name.upper().startswith("RPC")} - set(
        rpc_device_names(len(endpoints)))
    if unavailable:
        problems.append(Problem(
            "error",
            f"-dev references {', '.join(sorted(unavailable))} but no matching --rpc worker is configured",
        ))

    if available_devices is not None:
        known = set(available_devices)
        unknown = [name for name in requested if name not in known and name not in unavailable]
        if unknown:
            problems.append(Problem(
                "error",
                f"{', '.join(unknown)} is not among the devices found: {', '.join(sorted(known)) or 'none'}",
            ))

    if spec.tensor_split and spec.split_mode not in {"layer", "row"}:
        problems.append(Problem("note", "-ts is only used with -sm layer or -sm row"))

    overridden = [
        param.flag for param in SCHEMA
        if param.flag and aliases_of(param.flag) & extra_flags and emit(param, spec)
    ]
    if overridden:
        problems.append(Problem("note", f"Extra arguments override: {', '.join(sorted(set(overridden)))}"))

    return problems


def errors(problems: list[Problem]) -> list[Problem]:
    return [problem for problem in problems if problem.level == "error"]


__all__ = [
    "DEFAULTS",
    "Problem",
    "RunSpec",
    "errors",
    "mask_api_key",
    "params",
    "parse_rpc_endpoints",
    "rpc_device_names",
    "to_argv",
    "validate",
]
