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

from gui2.core import machine

#: "multi" is a closed list of which values to try, not one value to use
Kind = Literal["int", "float", "text", "bool", "choice", "slider", "multi", "devices"]
Emit = Literal["value", "presence", "absence", "composite"]

#: the four that are actually worth choosing between; the exotic types llama.cpp
#: also accepts can still be typed into the extra arguments
KV_TYPES = ("f16", "q8_0", "f8_e4m3", "q4_0")
KV_HELP = {
    "f16": "full quality, largest KV cache",
    "q8_0": "half the KV memory, quality loss rarely noticeable",
    "f8_e4m3": "like q8_0 but needs flash attention on",
    "q4_0": "smallest KV cache, only for very long contexts",
}
SPEC_TYPES = ("none", "mtp", "ngram-mod")
SPLIT_MODES = ("", "auto", "layer", "none", "row")
SPLIT_HELP = {
    "": "whatever llama.cpp does by default, which is layer",
    "auto": "let llama.cpp choose",
    "layer": "each GPU holds whole layers — the usual answer",
    "none": "no split; the whole model goes on one device",
    "row": "every GPU works on every layer — needs a fast link, rarely worth it over a network",
}

#: offered as suggestions, not as the only answers: a machine with two network
#: cards may have to name one of them, and that is still a legitimate --host
HOSTS = ("127.0.0.1", "0.0.0.0")
HOST_HELP = {
    "127.0.0.1": "this machine only — nothing on the network can connect",
    "0.0.0.0": "every network this machine is on — other machines can connect",
}

#: --threads-http is the one number better left to llama-server, which sizes it
#: from the parallel slots and the core count; 0 is how that is said in argv
THREADS_HTTP_AUTO = 0

# flags llama-server accepts under more than one name; extra arguments must
# suppress a generated flag no matter which spelling the user typed
ALIASES: tuple[frozenset[str], ...] = (
    frozenset({"-m", "--model"}),
    frozenset({"-c", "--ctx-size"}),
    frozenset({"-t", "--threads"}),
    frozenset({"-b", "--batch-size"}),
    frozenset({"-ub", "--ubatch-size"}),
    frozenset({"-np", "--parallel"}),
    frozenset({"-kvu", "--kv-unified"}),
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
    Param("host", "Host", "text", G_SERVER, "--host",
          choices=HOSTS,
          help="which network cards may reach this server"),
    Param("port", "Port", "int", G_SERVER, "--port", minimum=1, maximum=65535,
          help="any number nothing else is using; 8080 is only a habit"),
    # maximum applies only until a model is chosen; then the GGUF sets the ceiling
    Param("ctx_size", "Context", "slider", G_CONTEXT, "-c", minimum=4096, maximum=262144, step=4096,
          help="tokens the server keeps in memory; the KV cache grows with it"),
    # both maxima are this machine's core count; bounds() fills them in
    Param("threads", "CPU threads", "slider", G_CONTEXT, "-t", minimum=0, step=1,
          skip_default=True,
          help="for the work the CPU still does; irrelevant once every layer is on a GPU"),
    Param("threads_http", "HTTP threads", "slider", G_SERVER, "--threads-http",
          minimum=0, step=1, skip_default=True,
          help="threads that accept connections and move bytes — none of them do inference"),
    Param("batch_size", "Batch", "slider", G_CONTEXT, "--batch-size",
          minimum=64, maximum=8192, step=64, help="tokens submitted per prompt pass"),
    Param("ubatch_size", "Ubatch", "slider", G_CONTEXT, "--ubatch-size",
          minimum=32, maximum=2048, step=32, help="never larger than the batch"),
    Param("parallel", "Conversations at once", "slider", G_CONTEXT, "--parallel",
          minimum=1, maximum=16, step=1,
          help="llama.cpp calls these slots; each one holds one conversation"),
    Param("kv_unified", "Share one KV cache", "bool", G_CONTEXT, "--kv-unified",
          emit="presence",
          help="one pool all conversations draw from, instead of an equal share each"),
    Param("gpu_layers_all", "Offload every layer", "bool", G_DEVICE, emit="composite",
          help="emits -ngl 999; turn off to keep part of the model on the CPU"),
    Param("gpu_layers", "GPU layers", "slider", G_DEVICE, "-ngl",
          minimum=0, maximum=128, step=1, emit="composite",
          help="used only when 'offload every layer' is off"),
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
    Param("rpc_endpoints", "Worker addresses", "text", G_DEVICE, "--rpc", emit="composite",
          help="where the rpc-server processes are, as host:port, separated by commas — "
               "the order decides which one is RPC0"),
    Param("devices", "Devices", "devices", G_DEVICE, "-dev",
          help="nothing selected means every device the build finds"),
    Param("split_mode", "How to spread the model", "choice", G_DEVICE, "-sm",
          choices=SPLIT_MODES,
          help="left alone, llama.cpp fills each GPU in proportion to its free memory"),
    Param("tensor_split", "Share per device", "text", G_DEVICE, "-ts",
          help="numbers in device order, e.g. 3,2 — only needed when the automatic "
               "share puts too much on one card"),
    Param("api_key", "API key", "text", G_SERVER, "--api-key", help="masked in previews"),
)

BY_NAME: dict[str, Param] = {param.name: param for param in SCHEMA}

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(param.group for param in SCHEMA))


#: context slider steps, coarsest first. The model's own limit has to land on
#: the ladder, otherwise the top of the slider is not the top of the model.
CONTEXT_STEPS = (4096, 2048, 1024, 512, 256)

#: sliders whose ceiling is the CPU, not the model
THREAD_PARAMS = frozenset({"threads", "threads_http"})


def context_bounds(n_ctx_train: int | None) -> tuple[int, int, int]:
    """Context slider range: the model's trained length is the ceiling."""
    param = BY_NAME["ctx_size"]
    ceiling = int(n_ctx_train or param.maximum or 262144)
    step = next((size for size in CONTEXT_STEPS if ceiling % size == 0), CONTEXT_STEPS[-1])
    low = max(step, int(param.minimum or step) // step * step)
    low = min(low, ceiling // step * step)
    return low, low + (ceiling - low) // step * step, step


def bounds(param: Param, n_layers: int | None = None,
           n_ctx_train: int | None = None) -> tuple[int | float, int | float, int | float]:
    """Slider range, narrowed to what the selected model actually supports."""
    if param.name == "ctx_size":
        return context_bounds(n_ctx_train)

    low = param.minimum if param.minimum is not None else 0
    high = param.maximum if param.maximum is not None else 100
    step = param.step or 1

    if param.name in THREAD_PARAMS:
        # a slider that stops at this machine's core count cannot be set to a
        # number that would only make the CPU fight itself
        high = max(machine.cores().logical, int(low))
    if param.name == "gpu_layers" and n_layers:
        high = min(high, max(n_layers + 1, low))
    return low, high, step


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
