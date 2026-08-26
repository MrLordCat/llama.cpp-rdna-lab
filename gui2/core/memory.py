"""What a run will ask of the GPUs, before anything is started.

The point of this module is the question every OOM answers too late: will this
model, at this context, on these devices, actually load? Nothing here runs a
process or touches a driver -- the numbers come from the GGUF header and the
settings, using llama.cpp's own allocation rules.

The KV and recurrent-state formulas are the ones in `llama-kv-cache.cpp` and
`llama-hparams.cpp`, and they were checked against the sizes real runs printed:
Qwen3.8-27B at 163840 tokens reports `size = 8960.00 MiB (163840 cells, 16
layers)`, which is what `kv_bytes` returns for that header. The compute buffer
is the one term with no closed form, so it is a measured rule of thumb.
"""

from __future__ import annotations

from dataclasses import dataclass

from gui2.core.gguf import ModelFacts

MIB = 1024 * 1024
GIB = 1024 * MIB

#: Bytes per KV element. The quantized types carry a scale per 32-value block,
#: which is why q8_0 is not exactly one byte per element.
KV_ELEMENT_BYTES: dict[str, float] = {
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
    "f8_e4m3": 1.0,
    "f8_e5m2": 1.0,
}

#: Graph buffers scale with the micro-batch and the embedding width. Measured
#: across runs at ubatch 512-1024 on 5120-wide models: one f32 copy of the
#: micro-batch per device, within a few MiB.
_COMPUTE_COPIES = 1.0


@dataclass(frozen=True, slots=True)
class Term:
    """One line of the estimate."""

    label: str
    mib: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Estimate:
    terms: tuple[Term, ...] = ()
    notes: tuple[str, ...] = ()
    #: False when the header does not say enough to compute the KV cache
    complete: bool = True

    @property
    def total_mib(self) -> float:
        return sum(term.mib for term in self.terms)


def gib(mib: float) -> str:
    """A VRAM figure at the precision a person can act on.

    The tenth of a gigabyte stays: at the sizes this deals in, it is often the
    whole margin between fitting and not.
    """
    if mib >= 1024:
        return f"{mib / 1024:.1f} GiB"
    return f"{mib:.0f} MiB"


def element_bytes(cache_type: str) -> float:
    return KV_ELEMENT_BYTES.get(cache_type or "f16", 2.0)


def layer_split(facts: ModelFacts) -> tuple[int, int]:
    """Layers holding a KV cache, and layers holding a recurrent state.

    A hybrid model (`full_attention_interval`) gives full attention to every
    n-th layer only; the others run linear attention, whose state is a fixed
    size per sequence and does not grow with the context. The NextN/MTP blocks
    appended past the main stack are not part of the main pass.
    """
    total = facts.n_layers or 0
    main = max(0, total - facts.nextn_layers)
    interval = facts.full_attention_interval
    if interval > 1:
        attention = main // interval
        return attention, main - attention
    return total, 0


def kv_bytes(facts: ModelFacts, ctx: int, type_k: str = "f16", type_v: str = "f16") -> float:
    """Bytes the KV cache reserves at this context length."""
    attention, _recurrent = layer_split(facts)
    width_k, width_v = facts.n_embd_k_gqa, facts.n_embd_v_gqa
    if not attention or not width_k:
        return 0.0
    per_token = width_k * element_bytes(type_k) + width_v * element_bytes(type_v)
    return attention * max(0, ctx) * per_token


def state_bytes(facts: ModelFacts, sequences: int = 1) -> float:
    """Bytes the linear-attention layers reserve, one set per sequence."""
    _attention, recurrent = layer_split(facts)
    if not recurrent or not facts.ssm_state or not facts.ssm_inner:
        return 0.0
    conv = max(facts.ssm_conv - 1, 0) * (facts.ssm_inner + 2 * facts.ssm_group * facts.ssm_state)
    state = facts.ssm_state * facts.ssm_inner
    return recurrent * max(1, sequences) * (conv + state) * 4


def compute_bytes(facts: ModelFacts, ubatch: int, sequences: int = 1, devices: int = 1) -> float:
    """Graph and output buffers: small, but not zero at a large micro-batch."""
    if not facts.n_embd:
        return 0.0
    graph = max(1, devices) * max(1, ubatch) * facts.n_embd * 4 * _COMPUTE_COPIES
    logits = facts.n_vocab * 4 * max(1, sequences)
    return graph + logits


def weight_bytes(facts: ModelFacts, gpu_layers_all: bool, gpu_layers: int) -> float:
    """File bytes that end up on a GPU.

    Slightly high on purpose when everything is offloaded: llama.cpp keeps the
    token embedding in host memory, and erring towards "needs more" is the
    error that does not end in an OOM.
    """
    if not facts.file_bytes:
        return 0.0
    layers = facts.n_layers or 0
    if gpu_layers_all or not layers:
        return float(facts.file_bytes)
    return facts.file_bytes * min(max(gpu_layers, 0), layers + 1) / (layers + 1)


def estimate(spec, facts: ModelFacts | None, devices: int = 1,
             mmproj_bytes: int = 0) -> Estimate:
    """The VRAM bill for one RunSpec, itemised."""
    if facts is None or not facts.known:
        return Estimate(notes=("select a model to see what it will need",), complete=False)

    attention, recurrent = layer_split(facts)
    sequences = max(1, spec.parallel)
    terms = [
        Term("Weights", weight_bytes(facts, spec.gpu_layers_all, spec.gpu_layers) / MIB,
             "the whole file" if spec.gpu_layers_all
             else f"{spec.gpu_layers} of {facts.n_layers} layers"),
        Term("KV cache", kv_bytes(facts, spec.ctx_size, spec.cache_type_k, spec.cache_type_v) / MIB,
             f"{attention} attention layers × {spec.ctx_size:,} tokens".replace(",", " ")),
    ]
    state = state_bytes(facts, sequences) / MIB
    if state:
        terms.append(Term("Recurrent state", state,
                          f"{recurrent} linear-attention layers, fixed per sequence"))
    if mmproj_bytes and spec.mmproj_offload:
        terms.append(Term("Vision projector", mmproj_bytes / MIB, "mmproj on the GPU"))
    terms.append(Term("Compute buffers",
                      compute_bytes(facts, spec.ubatch_size, sequences, devices) / MIB,
                      f"ubatch {spec.ubatch_size}"))

    notes: list[str] = []
    complete = True
    if not facts.n_embd_k_gqa:
        notes.append("this header does not state the attention shape, so the KV cache "
                     "is not in the total")
        complete = False
    if facts.full_attention_interval > 1:
        notes.append(f"hybrid model: only every {facts.full_attention_interval}th layer keeps a "
                     f"KV cache, which is why the context is cheaper here than usual")
    if facts.sliding_window and facts.full_attention_interval <= 1:
        notes.append(f"some layers of this model only keep a {facts.sliding_window}-token "
                     f"window, so the real KV cache will be smaller than shown")
    if spec.spec_type == "mtp":
        notes.append("MTP adds a second, small context for the draft head, not counted here")
    return Estimate(terms=tuple(terms), notes=tuple(notes), complete=complete)


def kv_alternatives(facts: ModelFacts, ctx: int, chosen: str,
                    options: tuple[str, ...] = ("f16", "q8_0", "q4_0")) -> list[tuple[str, float]]:
    """The same KV cache at the other cache types, cheapest choice made visible."""
    return [(name, kv_bytes(facts, ctx, name, name) / MIB)
            for name in options if name != chosen]


def context_for_budget(facts: ModelFacts, room_mib: float,
                       type_k: str = "f16", type_v: str = "f16") -> int:
    """The longest context whose KV cache still fits in `room_mib`."""
    per_token = kv_bytes(facts, 1, type_k, type_v)
    if per_token <= 0 or room_mib <= 0:
        return 0
    return int(room_mib * MIB // per_token)
