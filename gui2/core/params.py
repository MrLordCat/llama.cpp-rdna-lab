"""Declarative llama-server parameter schema.

One `Param` per flag, in the order it is emitted on the command line. Forms and
argv are both generated from this list, so adding a flag is a one-line change
instead of an edit in every tab that composes a command.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Literal

Kind = Literal["int", "float", "text", "bool", "choice"]
Emit = Literal["value", "presence", "absence", "composite"]

KV_TYPES = ("f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f8_e4m3")
SPEC_TYPES = ("none", "mtp", "ngram-mod")
SPLIT_MODES = ("", "auto", "layer", "none", "row")

# flags llama-server accepts under more than one name; extra arguments must
# suppress a generated flag no matter which spelling the user typed
ALIASES: tuple[frozenset[str], ...] = (
    frozenset({"-m", "--model"}),
    frozenset({"-c", "--ctx-size"}),
    frozenset({"-t", "--threads"}),
    frozenset({"-b", "--batch-size"}),
    frozenset({"-ub", "--ubatch-size"}),
    frozenset({"-np", "--parallel"}),
    frozenset({"-ngl", "--gpu-layers", "--n-gpu-layers"}),
    frozenset({"-ctk", "--cache-type-k"}),
    frozenset({"-ctv", "--cache-type-v"}),
    frozenset({"-fa", "--flash-attn"}),
    frozenset({"-dev", "--device"}),
    frozenset({"-sm", "--split-mode"}),
    frozenset({"-ts", "--tensor-split"}),
    frozenset({"-mm", "--mmproj"}),
    frozenset({"-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"}),
    frozenset({"-cpent", "--checkpoint-every-n-tokens"}),
    frozenset({"-fit", "--fit"}),
)


@dataclass(frozen=True, slots=True)
class Param:
    """Presentation and argv metadata for one flag. Values live in RunSpec."""

    name: str
    label: str
    kind: Kind
    group: str
    flag: str = ""
    help: str = ""
    choices: tuple[str, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    emit: Emit = "value"
    skip_default: bool = False


G_MODEL = "Model & build"
G_CONTEXT = "Context & batching"
G_CACHE = "Cache & checkpoints"
G_DEVICE = "Devices"
G_SPEC = "Speculative"
G_SERVER = "Server"
G_ADVANCED = "Advanced"

SCHEMA: tuple[Param, ...] = (
    Param("model", "Model", "text", G_MODEL, "-m", help="path to the GGUF file"),
    Param("host", "Host", "text", G_SERVER, "--host"),
    Param("port", "Port", "int", G_SERVER, "--port", minimum=1, maximum=65535),
    Param("ctx_size", "Context", "int", G_CONTEXT, "-c", minimum=512, step=1024),
    Param("threads", "CPU threads", "int", G_CONTEXT, "-t", minimum=1, maximum=128),
    Param("threads_http", "HTTP threads", "int", G_SERVER, "--threads-http", minimum=1, maximum=64),
    Param("batch_size", "Batch", "int", G_CONTEXT, "--batch-size", minimum=1),
    Param("ubatch_size", "Ubatch", "int", G_CONTEXT, "--ubatch-size", minimum=1),
    Param("parallel", "Parallel sequences", "int", G_CONTEXT, "--parallel", minimum=1, maximum=64),
    Param("gpu_layers", "GPU layers", "int", G_DEVICE, "-ngl", minimum=-1),
    Param("cache_type_k", "KV cache K", "choice", G_CACHE, "--cache-type-k",
          choices=KV_TYPES, skip_default=True),
    Param("cache_type_v", "KV cache V", "choice", G_CACHE, "--cache-type-v",
          choices=KV_TYPES, skip_default=True),
    Param("conversation_cache", "Conversation cache", "bool", G_CACHE, "--conversation-cache",
          emit="presence",
          help="high-hit chat cache; supplies its own checkpoint policy"),
    Param("metrics", "Prometheus /metrics", "bool", G_SERVER, "--metrics", emit="presence"),
    Param("cache_ram", "Prompt cache RAM (MiB)", "int", G_CACHE, "--cache-ram", minimum=0),
    Param("ctx_checkpoints", "Context checkpoints", "int", G_CACHE, "--ctx-checkpoints", minimum=0),
    Param("checkpoint_every_n_tokens", "Checkpoint interval", "int", G_CACHE,
          "--checkpoint-every-n-tokens", minimum=0),
    Param("mmproj", "Vision mmproj", "text", G_MODEL, "--mmproj"),
    Param("mmproj_offload", "Offload mmproj to GPU", "bool", G_MODEL, "--no-mmproj-offload",
          emit="absence"),
    Param("spec_type", "Speculative mode", "choice", G_SPEC, "--spec-type",
          choices=SPEC_TYPES, emit="composite"),
    Param("spec_draft_n_max", "Draft tokens", "int", G_SPEC, minimum=1, maximum=16,
          emit="composite", help="--spec-draft-n-max, emitted with the MTP mode"),
    Param("ngram_n_min", "Ngram n-min", "int", G_SPEC, minimum=1, emit="composite"),
    Param("ngram_n_match", "Ngram n-match", "int", G_SPEC, minimum=1, emit="composite"),
    Param("ngram_n_max", "Ngram n-max", "int", G_SPEC, minimum=1, emit="composite"),
    Param("embeddings", "Embedding mode", "bool", G_SERVER, "--embeddings", emit="presence"),
    Param("flash_attn", "Flash attention", "choice", G_ADVANCED, "--flash-attn",
          choices=("auto", "on", "off"), skip_default=True),
    Param("no_warmup", "Skip warmup", "bool", G_ADVANCED, "--no-warmup", emit="presence"),
    Param("no_mmap", "Disable mmap", "bool", G_ADVANCED, "--no-mmap", emit="presence"),
    Param("disable_thinking", "Disable thinking", "bool", G_ADVANCED, "--chat-template-kwargs",
          emit="composite"),
    Param("fit", "Auto fit", "choice", G_ADVANCED, "-fit", choices=("on", "off"), skip_default=True),
    Param("rpc_endpoints", "RPC workers", "text", G_DEVICE, "--rpc", emit="composite",
          help="host:port list; must precede -dev so RPC0..RPCn resolve"),
    Param("devices", "Devices (-dev)", "text", G_DEVICE, "-dev"),
    Param("split_mode", "Split mode (-sm)", "choice", G_DEVICE, "-sm", choices=SPLIT_MODES),
    Param("tensor_split", "Tensor split (-ts)", "text", G_DEVICE, "-ts"),
    Param("api_key", "API key", "text", G_SERVER, "--api-key", help="masked in previews"),
)

BY_NAME: dict[str, Param] = {param.name: param for param in SCHEMA}

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(param.group for param in SCHEMA))


def aliases_of(flag: str) -> frozenset[str]:
    for group in ALIASES:
        if flag in group:
            return group
    return frozenset({flag})


def parse_extra(text: str) -> list[str]:
    """Split free-form extra arguments the way the launcher will see them."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        return text.split()


def flags_in(tokens: list[str]) -> set[str]:
    """Flag names present in a token list, ignoring any `=value` suffix."""
    return {token.split("=", 1)[0] for token in tokens if token.startswith("-")}
