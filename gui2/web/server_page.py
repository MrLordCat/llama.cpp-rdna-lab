"""Server launch page.

The form, the command and the validation all come from one schema, so a new
flag stays a one-line change in `gui2.core.params`. What this module adds is
the part a schema cannot express: which settings a newcomer meets first, and
which limits the chosen model and the discovered devices impose on them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence
from urllib.parse import urlencode

from fasthtml.common import (
    A,
    Button,
    Datalist,
    Details,
    Div,
    Form,
    H3,
    Input,
    Label,
    Option,
    Pre,
    Script,
    Select,
    Span,
    Summary,
    Textarea,
)

from gui2.config import AppConfig
from gui2.core import machine
from gui2.core.bench import BenchSpec, bench_commands
from gui2.core.devices import Device, DeviceService, Scan, pool
from gui2.core.gguf import ModelFacts, context_text, read_facts
from gui2.core.inventory import Build, discover_builds, discover_models, find_build
from gui2.core.memory import (
    Estimate,
    context_for_budget,
    estimate,
    gib,
    kv_alternatives,
    kv_bytes,
    MIB,
)
from gui2.core.measured import Measurement, notes as measured_notes
from gui2.core.memstore import MemoryStore
from gui2.core.rpc import (
    DEFAULT_PORT,
    Fleet,
    KNOWN_PROTOCOL,
    Worker,
    WorkerPlan,
    guide,
    probe_all,
)
from gui2.core.params import BY_NAME, HOST_HELP, KV_HELP, SPLIT_HELP, SCHEMA, Param, bounds
from gui2.core.runspec import (
    DEFAULTS,
    Problem,
    RunSpec,
    mask_api_key,
    parse_rpc_endpoints,
    slot_context,
    to_argv,
    validate,
)
from gui2.proc import Busy, Supervisor
from gui2.web.controls import toggle
from gui2.web import layout
from gui2.web.layout import PROBLEM_STYLE, command_lines, problem_lines, shell

#: Never echoed into the address bar, browser history or the access log.
SECRET_PARAMS = frozenset({"api_key"})

#: Form plumbing rather than run settings; not part of the shareable URL.
INTERNAL_PARAMS = frozenset({"_ceiling"})
#: fields that legitimately appear more than once in one submission
MULTI_PARAMS = frozenset({param.name for param in SCHEMA if param.kind == "devices"})
#: present only when the whole form was submitted, and so the only evidence
#: that an absent checkbox was cleared rather than never mentioned
FORM_MARKER = "_form"

#: Re-rendered when the model changes, because the model sets their limits.
BOUNDED = ("ctx_size", "gpu_layers_all", "gpu_layers")
SLIDERS = tuple(name for name in BOUNDED if BY_NAME[name].kind == "slider")
CEILING_SEPARATOR = ":"


@dataclass(frozen=True, slots=True)
class Section:
    """A block of the form. SCHEMA fixes argv order; this fixes reading order."""

    title: str
    names: tuple[str, ...]
    open: bool = False
    hint: str = ""


LAYOUT: tuple[Section, ...] = (
    Section("Model and build", ("model", "build_dir"), open=True,
            hint="Which model file to serve, and which llama-server binary serves it. "
                 "The build decides which GPUs are usable at all."),
    Section("Context and GPU", ("ctx_size", "gpu_layers_all", "gpu_layers", "devices",
                                "parallel", "kv_unified"), open=True,
            hint="The two questions that decide whether a model loads at all: how much "
                 "context to reserve, and which GPUs may hold it. The slot count belongs "
                 "here too, because it decides how that context is shared out."),
    Section("Speed and memory", ("batch_size", "ubatch_size", "threads",
                                 "cache_type_k", "cache_type_v", "flash_attn"), open=True,
            hint="A smaller KV cache type buys context at some quality; batch sizes trade "
                 "prompt speed for memory."),
    Section("Server", ("host", "port", "threads_http", "metrics", "embeddings", "api_key"),
            hint="Who may talk to this server, and on which door. None of it changes what "
                 "the model does or how much memory it takes."),
    Section("More than one machine", ("rpc_endpoints", "split_mode", "tensor_split"),
            hint="A second computer can lend its GPU over the network. It runs a small "
                 "program called rpc-server; this machine then treats its cards as "
                 "RPC0, RPC1 … in the order they are listed here."),
    Section("Speculative decoding", ("spec_type", "spec_draft_n_max",
                                     "ngram_n_min", "ngram_n_match", "ngram_n_max"),
            hint="Something cheap guesses the next few tokens and the model checks them all "
                 "at once. Faster when the guesses are right, slower when they are not, so "
                 "it is worth measuring rather than assuming."),
    Section("Prompt cache", ("conversation_cache", "cache_ram",
                             "ctx_checkpoints", "checkpoint_every_n_tokens"),
            hint="Keeps the work already done on a prompt, so the next message in a "
                 "conversation does not re-read everything before it. Costs system RAM, "
                 "not VRAM."),
    Section("Vision", ("mmproj", "mmproj_offload"),
            hint="Only for models that can look at images. The mmproj file is the part "
                 "that turns a picture into something the model can read."),
    Section("Advanced", ("no_warmup", "no_mmap", "disable_thinking", "fit"),
            hint="Rarely needed. Each one is here because some particular machine needed it."),
)


def state_query(params, refit: RunSpec | None = None,
                multi: frozenset[str] = MULTI_PARAMS) -> str:
    """Query string that reproduces the form on reload, without the secrets.

    `multi` names the fields that legitimately arrive more than once; the
    Autotune page passes its own, because its sweep axes are ticked the same
    way the device list is.
    """
    items = params.multi_items() if hasattr(params, "multi_items") else list(params.items())
    skip = SECRET_PARAMS | INTERNAL_PARAMS
    # a slider the new model just moved has to reach the address bar as it now reads
    moved = {name: str(getattr(refit, name)) for name in SLIDERS} if refit else {}

    ordered: list[tuple[str, str]] = []
    at: dict[str, int] = {}
    for key, value in items:
        if key in skip:
            continue
        pair = (key, moved.get(key, value))
        # a device list is many checkboxes under one name; anything else that
        # arrives twice is a panel that was submitted along with the form, and
        # the later value is the current one
        if key in multi or key not in at:
            at[key] = len(ordered)
            ordered.append(pair)
        else:
            ordered[at[key]] = pair
    return urlencode(ordered)


def spec_link(spec: RunSpec) -> str:
    """A whole spec as a query string, for a link between pages.

    Written the way a submitted form writes it -- a false checkbox left out,
    `_form` present to say the omission was meant -- so that reading it back
    with `spec_from_params` gives the same spec.
    """
    pairs: list[tuple[str, str]] = []
    for field in fields(RunSpec):
        if field.name in SECRET_PARAMS:
            continue
        value = getattr(spec, field.name)
        if isinstance(value, bool):
            if value:
                pairs.append((field.name, "1"))
        elif str(value) != "":
            pairs.append((field.name, str(value)))
    pairs.append((FORM_MARKER, "1"))
    return urlencode(pairs)


def spec_from_params(params) -> RunSpec:
    """A spec from the form, or from a link that names a few settings.

    An unchecked box submits nothing, so "absent means off" is only true of a
    form submission. In a link -- `/server?model=...` from the Models page --
    absent means unmentioned, and switching every default off because of it
    would hand the user a command they did not ask for.
    """
    if not params:
        return DEFAULTS
    values = {key: params[key] for key in params.keys()}
    submitted = FORM_MARKER in params
    for param in SCHEMA:
        if param.kind == "bool":
            if submitted:
                values[param.name] = param.name in params
        elif param.kind == "devices" and hasattr(params, "getlist"):
            values[param.name] = ",".join(params.getlist(param.name))
    return DEFAULTS.with_values(values)


def selected_devices(spec: RunSpec) -> set[str]:
    return {name for name in re.split(r"[,\s]+", spec.devices) if name}


def model_facts(spec: RunSpec) -> ModelFacts | None:
    return read_facts(spec.model) if spec.model and Path(spec.model).is_file() else None


def build_of(config: AppConfig, spec: RunSpec) -> Build | None:
    return find_build(discover_builds(config.builds), spec.build_dir)


def rescan(service: DeviceService, spec: RunSpec, backend: str) -> Scan:
    """Keep the device list in step with the build and the RPC list."""
    service.start(parse_rpc_endpoints(spec.rpc_endpoints), backend)
    return service.state()


# -- controls --------------------------------------------------------------


def _options(config: AppConfig, spec: RunSpec) -> dict[str, list[tuple[str, str]]]:
    models = [(f"{model.name}  ({model.size_text})", str(model.path))
              for model in discover_models(config.models) if not model.is_mmproj]
    if spec.model and spec.model not in {value for _label, value in models}:
        models.insert(0, (f"{Path(spec.model).name}  (not in models dir)", spec.model))

    builds = [(f"{build.name} · {build.backend} · "
               + (build.built_text if build.usable else "no llama-server"), build.name)
              for build in discover_builds(config.builds)]
    return {
        "model": [("— select —", "")] + models,
        "build_dir": [("— select —", "")] + builds,
    }


def _slider_caption(param: Param, spec: RunSpec, high: int, facts: ModelFacts | None) -> str:
    """What the ends of this slider mean, in words rather than in numbers.

    Written so that it stays true wherever the handle is dragged: the page is
    not re-rendered on a drag, so a caption that describes the current value
    would start lying the moment it is used.
    """
    if param.name == "ctx_size":
        if facts and facts.n_ctx_train:
            return f"max {context_text(high)} — what this model was trained for"
        return f"max {context_text(high)} — " + (
            "this file does not state a context length" if facts else "no model selected yet")
    if param.name == "gpu_layers":
        if facts and facts.n_layers:
            return f"max {high} — the model has {facts.n_layers} layers"
        return f"max {high}"
    if param.name == "threads":
        chosen = machine.cores()
        return (f"0 = automatic — llama-server counts the physical cores itself and will use "
                f"{chosen.usable} of the {chosen.logical} hardware threads here.")
    if param.name == "threads_http":
        auto = machine.auto_threads_http(spec.parallel)
        slots = "1 slot" if spec.parallel == 1 else f"{spec.parallel} slots"
        return (f"0 = automatic, which here means {auto} — llama-server takes the larger of "
                f"{slots} + 4 and one less than the {machine.cores().logical} hardware threads.")
    if param.name == "parallel":
        total, per_slot = slot_context(spec)
        if per_slot >= total:
            return (f"{spec.parallel} at once, sharing one {context_text(total)} pool."
                    if spec.parallel > 1 else
                    f"one at a time, with all {context_text(total)} to itself.")
        return (f"each of the {spec.parallel} gets {context_text(per_slot)} of the "
                f"{context_text(total)} — tick 'share one KV cache' to pool it instead.")
    return ""


def kv_line(spec: RunSpec, facts: ModelFacts | None, oob: bool = False) -> Span:
    """What the chosen context costs, said where the context is chosen.

    The other cache types are priced alongside it, because that choice is only
    meaningful once its saving is a number.
    """
    body = ""
    if facts is not None and facts.n_embd_k_gqa:
        cost = kv_bytes(facts, spec.ctx_size, spec.cache_type_k, spec.cache_type_v) / MIB
        uniform = spec.cache_type_k == spec.cache_type_v
        parts = [f"KV cache {gib(cost)}" + (f" at {spec.cache_type_k}" if uniform else "")]
        if uniform:
            parts += [f"{gib(other)} at {name}"
                      for name, other in kv_alternatives(facts, spec.ctx_size, spec.cache_type_k)]
        body = " · ".join(parts)
    return Span(body, id="kvline", cls="ceiling", hx_swap_oob="true" if oob else None)


def _slider(param: Param, spec: RunSpec, facts: ModelFacts | None):
    low, high, step = bounds(param,
                             facts.n_layers if facts else None,
                             facts.n_ctx_train if facts else None)
    value = min(max(int(getattr(spec, param.name)), int(low)), int(high))
    return Div(
        Div(
            # the range is nameless: the number box is what the form submits
            Input(type="range", value=str(value), min=low, max=high, step=step, cls="range",
                  aria_label=param.label, oninput="this.nextElementSibling.value=this.value"),
            Input(type="number", name=param.name, value=str(value), min=low, max=high, step=step,
                  cls="numberbox", oninput="this.previousElementSibling.value=this.value"),
            cls="slider",
        ),
        Span(caption,
             # "max 128K" belongs under the number box; a sentence does not
             cls="ceiling" if param.name in SLIDERS else "ceiling wide")
        if (caption := _slider_caption(param, spec, int(high), facts)) else None,
        kv_line(spec, facts) if param.name == "ctx_size" else None,
        cls="sliderbox",
    )


def _control(param: Param, spec: RunSpec, options: dict, facts: ModelFacts | None):
    value = getattr(spec, param.name)
    if param.name in options:
        # the model sets the slider limits, so it refreshes them itself; 'consume'
        # keeps the change from also reaching the form's own preview trigger,
        # which would race this request with the pre-refit values
        hooks = {"hx_post": "/server/bounds", "hx_trigger": "change consume",
                 "hx_target": "#bounded", "hx_swap": "outerHTML"} \
            if param.name == "model" else {}
        return Select(*[Option(label, value=item, selected=item == value)
                        for label, item in options[param.name]], name=param.name, **hooks)
    if param.kind == "bool":
        return toggle(param.name, param.label, bool(value), title=param.help)
    if param.kind == "slider":
        return _slider(param, spec, facts)
    if param.kind == "choice":
        return Select(*[Option(_choice_label(param, choice), value=choice, selected=choice == value)
                        for choice in param.choices], name=param.name)
    if param.kind in {"int", "float"}:
        return Input(type="number", name=param.name, value=str(value),
                     min=param.minimum, max=param.maximum, step=param.step,
                     inputmode="numeric" if param.kind == "int" else None)
    if param.name in SECRET_PARAMS:
        return Input(type="password", name=param.name, value=str(value),
                     autocomplete="new-password")
    if param.choices:
        # a suggestion list rather than a dropdown: the two answers that fit
        # almost everyone are one click away, and naming a single network card
        # by its own address is still possible for the machine that needs it
        listing = f"{param.name}_choices"
        return Div(
            Input(type="text", name=param.name, value=str(value), list=listing),
            Datalist(*[Option(HOST_HELP.get(choice, ""), value=choice)
                       for choice in param.choices], id=listing),
            cls="suggested",
        )
    return Input(type="text", name=param.name, value=str(value))


def _choice_label(param: Param, choice: str) -> str:
    if param.name.startswith("cache_type") and choice in KV_HELP:
        return f"{choice} — {KV_HELP[choice]}"
    if param.name == "split_mode" and choice in SPLIT_HELP:
        return f"{choice or 'default'} — {SPLIT_HELP[choice]}"
    return choice or "— default —"


def split_line(spec: RunSpec, facts: ModelFacts | None,
               devices: tuple[Device, ...]) -> Span:
    """What the ratio in the box works out to, per device and in gigabytes.

    "3,2" is not a quantity of anything until it is divided by five and
    multiplied by the size of the model, which is the arithmetic nobody wants
    to do in their head next to a text box.
    """
    names = [device.name for device in devices] or ["the first device", "the second"]
    shares = [value for value in re.split(r"[,;\s]+", spec.tensor_split.strip()) if value]
    if not shares:
        return Span("Empty means llama.cpp decides — with layer mode it gives each device a "
                    "share of the model in proportion to the memory it has free. Drag the "
                    "sliders above for a deliberate one.",
                    cls="hint block")
    try:
        weights = [float(value) for value in shares]
    except ValueError:
        return Span("Not numbers — a share list looks like 3,2", cls="problem err")
    total = sum(weights)
    if total <= 0:
        return Span("The shares add up to nothing, so nothing would be placed anywhere",
                    cls="problem err")
    if devices and len(weights) != len(devices):
        return Span(f"{len(weights)} share(s) for {len(devices)} device(s): llama.cpp reads "
                    f"them in device order and ignores the rest — {', '.join(names)}",
                    cls="problem warn")

    report = estimate(spec, facts, devices=max(1, len(devices)))
    parts = []
    for name, weight in zip(names, weights):
        fraction = weight / total
        size = f" ≈ {gib(report.total_mib * fraction)}" if report.terms else ""
        parts.append(f"{name} {fraction:.0%}{size}")
    return Span(" · ".join(parts), cls="hint block")


def split_balancer(spec: RunSpec, devices: tuple[Device, ...]) -> Div:
    """The share per device as one slider per card, not a list of numbers.

    The sliders always sum to the scale (ten), so dragging one rebalances the
    rest instead of drifting: 1,1 is two fives, 0.8,1 is 4.4 and 5.6. Each
    slider writes the `-ts` list it stands for into the hidden box, which is
    what the form submits; the visible box is only a face.
    """
    if not devices:
        # no cards to balance yet; the hidden box still round-trips the form
        return Div(Input(type="hidden", name="tensor_split", value=spec.tensor_split))
    shares = [value for value in re.split(r"[,;\s]+", spec.tensor_split.strip()) if value]
    try:
        weights = [max(0.0, float(value)) for value in shares]
    except ValueError:
        weights = []
    weights += [0.0] * (len(devices) - len(weights))
    total = sum(weights)
    if total <= 0 or len(weights) != len(devices):
        # no ratio yet, or a ratio for other cards: start even, and the slider
        # drag is what tells llama.cpp a deliberate split is wanted
        weights = [1.0] * len(devices)
        total = len(devices)
    scale = 10  # steps of one-tenth, matching how shares are usually written
    values = [round(weight / total * scale) for weight in weights]
    drawn = sum(values)
    if drawn != scale and values:
        # rounding did not land on the scale: hand the remainder to the first
        values[0] = max(0, min(scale, values[0] + scale - drawn))
    bars = []
    for device, weight, value in zip(devices, weights, values):
        bars.append(Div(
            Span(f"{device.name} · {device.memory_text}"),
            Input(type="range", name=f"split_{device.name}", min="0", max="10",
                  step="1", value=str(value)),
            Span(f"{weight / total:.0%}", cls="share"),
            cls="splitbar",
        ))
    return Div(
        Div(*bars, cls="splitbars"),
        Div(
            Button("Automatic", type="button", cls="small", onclick="splitAuto(this)"),
            Span("clears the shares and lets llama.cpp fill each device by its free "
                 "memory", cls="hint"),
            cls="splitactions",
        ),
        Input(type="hidden", name="tensor_split", value=spec.tensor_split),
        Script(layout.BALANCER_JS),
        id=f"split-{devices[0].name}",
        cls="splitbalance",
    )


def balancer_field(spec: RunSpec, scan: Scan, backend: str) -> Div:
    """The balancer, waiting for the device scan like the picker does.

    Until the scan answers there are no cards to draw bars for, so the field
    fills itself in once it does — the same load hook as the device list.
    """
    if not scan.ready:
        return Div(
            Div(Span("looking for devices…", cls="hint"),
                hx_get=f"/server/splitbalancer?{spec_link(spec)}",
                hx_trigger="load delay:500ms",
                hx_target="#splitbalancer",
                hx_swap="outerHTML"),
            id="splitbalancer",
        )
    return Div(split_balancer(spec, run_devices(scan, spec, backend)),
               id="splitbalancer")


def _hint(param: Param, spec: RunSpec, facts: ModelFacts | None) -> str:
    # a slider's ceiling caption already names the model's limit; no need to repeat it
    if param.name == "ctx_size":
        if facts is None:
            return f"{param.help} · the range follows the model once one is selected"
        if facts.n_embd_k_gqa:
            return ""  # the ceiling and the KV line say the same thing in numbers
    if param.name == "host":
        return _host_hint(spec)
    return param.help or (f"emits {param.flag}" if param.flag else "")


def _host_hint(spec: RunSpec) -> str:
    """The host address explained by what it lets happen, not by what it is."""
    private = HOST_HELP["127.0.0.1"]
    if spec.host not in {"0.0.0.0", "::"}:
        return f"{private}. Set 0.0.0.0 to let other machines in."
    address = machine.lan_address()
    where = f" — this machine is {address} on its network" if address else ""
    return f"{HOST_HELP['0.0.0.0']}{where}. Anyone who can reach it can use the model."


def _field(param: Param, spec: RunSpec, options: dict, facts: ModelFacts | None,
           scan: Scan | None = None, backend: str = ""):
    hint = _hint(param, spec, facts)
    if param.kind == "bool":
        # the button carries the label, so a Label around it would say it twice;
        # the sentence underneath is what the switch means, not what it emits
        return Div(_control(param, spec, options, facts),
                   Span(param.help, cls="hint") if param.help else None,
                   cls="field switch", title=hint)
    verdict = _model_verdict(spec, facts, scan, backend) if param.name == "model" else ""
    hooks = {}
    if param.name == "model" and scan is not None and not scan.ready:
        # the verdict needs the device scan, which lands after the first render;
        # the field asks for itself once, exactly like the device picker does
        hooks = {"id": "modelfield",
                 "hx_get": f"/server/modelfield?{spec_link(spec)}",
                 "hx_trigger": "load delay:700ms",
                 "hx_target": "#modelfield",
                 "hx_swap": "outerHTML"}
    return Label(
        Span(param.label),
        _control(param, spec, options, facts),
        Span(verdict, cls="problem err" if verdict.startswith("⚠") else "hint")
        if verdict else None,
        Span(hint, cls="hint") if hint else None,
        cls="field",
        title=hint,
        **hooks,
    )


def _model_verdict(spec: RunSpec, facts: ModelFacts | None, scan: Scan | None,
                   backend: str) -> str:
    """The question the model select is really asked: will this load here.

    The Memory panel answers it fully; this line exists for the moment before
    the eye reaches the right column and for the width at which the columns
    stack. Empty means there is nothing honest to say yet.
    """
    if facts is None or scan is None or not scan.ready or facts.error:
        return ""
    devices = run_devices(scan, spec, backend)
    report = estimate(spec, facts, devices=max(1, len(devices)),
                      mmproj_bytes=_file_size(spec.mmproj))
    if not report.terms:
        return ""
    budget, parts, measured = _budget(devices)
    if budget <= 0:
        return ""
    headroom = budget - report.total_mib
    source = "free" if measured else "installed"
    if headroom >= 0:
        return f"fits — {gib(report.total_mib)} of {gib(budget)} {source} ({' + '.join(parts)})"
    return f"⚠ {gib(-headroom)} over the {gib(budget)} {source} ({' + '.join(parts)})"


def _build_field(spec: RunSpec, config: AppConfig, options: dict):
    chosen = build_of(config, spec)
    when = (f"built {chosen.built_on} — {chosen.built_text}" if chosen and chosen.usable
            else "newest first, so the top one is the last thing built")
    return Label(
        Span("Build"),
        Select(*[Option(label, value=item, selected=item == spec.build_dir)
                 for label, item in options["build_dir"]], name="build_dir"),
        Span(f"supplies llama-server; capabilities are read from CMakeCache. {when}",
             cls="hint"),
        cls="field",
    )


def model_field(config: AppConfig, spec: RunSpec, scan: Scan, backend: str):
    """The Model select as its own fragment, for the refresh that waits for
    the device scan: the fit verdict is honest only once the cards answered."""
    return _field(BY_NAME["model"], spec, _options(config, spec), model_facts(spec),
                  scan, backend)


def _ceiling_of(name: str, facts: ModelFacts | None) -> int:
    _low, high, _step = bounds(BY_NAME[name],
                               facts.n_layers if facts else None,
                               facts.n_ctx_train if facts else None)
    return int(high)


def refit(spec: RunSpec, facts: ModelFacts | None, ceilings: dict[str, int]) -> RunSpec:
    """Re-aim the sliders the new model's limits invalidate, and only those.

    A value sitting on the previous model's ceiling asked for "as much as this
    model gives", so it follows the new ceiling; a value that no longer fits has
    to come down. A smaller number was chosen on purpose and is left alone.
    Growth stops at the default, so a roomier model does not silently reserve a
    KV cache nobody asked for.
    """
    updates = {}
    for name, previous in ceilings.items():
        ceiling = _ceiling_of(name, facts)
        value = getattr(spec, name, 0)
        if value >= previous or value > ceiling:
            updates[name] = min(max(getattr(DEFAULTS, name), previous), ceiling)
    return spec.with_values(updates) if updates else spec


def read_ceilings(params) -> dict[str, int]:
    ceilings: dict[str, int] = {}
    for item in (params.getlist("_ceiling") if hasattr(params, "getlist") else []):
        name, _, value = str(item).partition(CEILING_SEPARATOR)
        if name in SLIDERS and value.isdigit():
            ceilings[name] = int(value)
    return ceilings


def bounded_fields(spec: RunSpec, facts: ModelFacts | None):
    options: dict = {}
    fields = [_field(BY_NAME[name], spec, options, facts) for name in BOUNDED]
    markers = [Input(type="hidden", name="_ceiling",
                     value=f"{name}{CEILING_SEPARATOR}{_ceiling_of(name, facts)}")
               for name in SLIDERS]
    return Div(*fields, *markers, id="bounded", cls="grid")


# -- device picker ---------------------------------------------------------


def _device_query(spec: RunSpec, backend: str) -> str:
    return urlencode(
        [("backend", backend), ("rpc_endpoints", spec.rpc_endpoints)]
        + [("devices", name) for name in sorted(selected_devices(spec))]
    )


def devices_field(spec: RunSpec, scan: Scan, backend: str, oob: bool = False):
    chosen = selected_devices(spec)
    body: list = []

    if not scan.ready:
        body.append(Div(
            Span("looking for devices…", cls="hint"),
            hx_get="/server/devices?" + _device_query(spec, backend),
            hx_trigger="load delay:500ms",
            hx_target="#devicefield",
            hx_swap="outerHTML",
        ))
    else:
        found = scan.for_backend(backend)
        if found:
            body.append(Div(*[
                Label(
                    Input(type="checkbox", name="devices", value=device.name,
                          checked=device.name in chosen),
                    Span(device.name, cls="devname"),
                    Span(device.description, cls="devdesc"),
                    Span(device.memory_text, cls="devmem"),
                    cls="devrow" if device.confirmed else "devrow unconfirmed",
                    title=f"{device.source}" + ("" if device.confirmed else " · not confirmed by a run"),
                )
                for device in found
            ], cls="devlist"))
            body.append(Span(
                "Nothing checked means llama-server uses every device it finds."
                if not chosen else f"-dev {','.join(name for name in sorted(chosen))}",
                cls="hint",
            ))
        else:
            body.append(Input(type="text", name="devices", value=spec.devices,
                              placeholder="Vulkan0,RPC0"))
            body.append(Span("No device found for this build — select a build first.", cls="hint"))

    for note in scan.notes:
        body.append(Span(note, cls="hint"))

    return Div(
        Div(
            Span("Devices"),
            Button("Rescan", type="button", cls="small",
                   hx_post="/server/devices", hx_target="#devicefield", hx_swap="outerHTML"),
            cls="fieldhead",
        ),
        *body,
        id="devicefield",
        cls="field wide",
        hx_swap_oob="true" if oob else None,
    )


# -- more than one machine -------------------------------------------------

#: names for the worker-setup boxes. They configure the *other* machine, so
#: they are not RunSpec fields and never reach a llama-server command line.
WORKER_FIELDS = ("rpc_port", "rpc_devices", "rpc_open", "rpc_cache")


def worker_plan(params) -> WorkerPlan:
    """The worker command's settings, from the form or from the defaults."""
    if not params or "rpc_port" not in params:
        return WorkerPlan()
    def text(name: str) -> str:
        return str(params.get(name, "") or "").strip()
    try:
        port = int(text("rpc_port"))
    except ValueError:
        port = DEFAULT_PORT
    host = text("rpc_host")
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", host):
        host = ""
    return WorkerPlan(
        port=port if 1 <= port <= 65535 else DEFAULT_PORT,
        devices=tuple(name for name in re.split(r"[,\s]+", text("rpc_devices"))
                      if name and re.fullmatch(r"[A-Za-z0-9_.\-]+", name)),
        open_to_network="rpc_open" in params,
        cache="rpc_cache" in params,
        host=host,
    )


def _copyable(command: str) -> Div:
    """A command and a button that puts it on the clipboard, nothing more.

    Deliberately not a "run it for me": the command has to run on the other
    machine, and a GUI that reached over there would need credentials it has
    no business holding.
    """
    return Div(
        Pre(command, cls="copytext"),
        Button("Copy", type="button", cls="small",
               onclick="navigator.clipboard.writeText("
                       "this.previousElementSibling.textContent);"
                       "this.textContent='Copied';"
                       "setTimeout(()=>this.textContent='Copy',1200)"),
        cls="copyrow",
    )


def _fill_address(address: str) -> str:
    """Put `host:port` into the form's Worker addresses box and refresh the page."""
    return (f"const el=document.querySelector('input[name=\"rpc_endpoints\"]');"
            f"if(el){{el.value='{address}';"
            f"el.dispatchEvent(new Event('change',{{bubbles:true}}));}}")


def worker_panel(params, oob: bool = False) -> Div:
    """The command to run on the other machine, built from its own boxes."""
    plan = worker_plan(params)
    download = urlencode({"rpc_port": plan.port}
                         | ({"rpc_devices": ",".join(plan.devices)} if plan.devices else {})
                         | ({"rpc_cache": "on"} if plan.cache else {})
                         | ({"rpc_open": "on"} if plan.open_to_network else {}))
    return Div(
        Div(
            Label(Span("This machine sees it as"),
                  Input(type="text", name="rpc_host", value=plan.host,
                        placeholder="192.168.1.60"),
                  Span("the other machine's IP or name, used to fill the box below",
                       cls="hint"),
                  cls="field wide"),
            Label(Span("Worker port"),
                  Input(type="number", name="rpc_port", value=str(plan.port),
                        min=1, max=65535),
                  Span("both machines have to agree on this one", cls="hint"),
                  cls="field"),
            Label(Span("Devices to expose"),
                  Input(type="text", name="rpc_devices", value=",".join(plan.devices),
                        placeholder="empty = every device that machine has"),
                  Span("names as that machine sees them, e.g. Vulkan0", cls="hint"),
                  cls="field"),
            cls="grid",
        ),
        Div(Div(toggle("rpc_open", "Reachable from this machine", plan.open_to_network),
                cls="field switch"),
            Div(toggle("rpc_cache", "Cache tensors on the worker's disk", plan.cache),
                cls="field switch"),
            cls="switches"),
        Div(
            A(f"Download rpc-worker-{plan.port}.bat",
              href=f"/server/rpc/worker.bat?{download}", cls="small",
              title="run this one file on the other machine as Administrator"),
            Button("Fill the Worker addresses box", type="button", cls="small",
                   onclick=_fill_address(plan.address),
                   title=f"puts {plan.address or 'host:port'} above; the machine is "
                         "whatever the bat was run on"),
            cls="actions",
        ),
        Span("Run this on the other machine (or the bat above):", cls="hint block"),
        _copyable(plan.text()),
        id="rpcworker",
        cls="rpcworker",
        hx_post="/server/rpc/command",
        # 'consume' so this change does not also fire the form's own preview
        hx_trigger="change consume",
        hx_target="#rpcworker",
        hx_swap="outerHTML",
        hx_swap_oob="true" if oob else None,
    )


def _worker_row(name_and: tuple[str, Worker, object]) -> Div:
    name, found, device = name_and
    if not found.reachable:
        detail, cls = found.error or "no answer", "devrow bad"
    elif not found.compatible:
        detail = (f"speaks protocol {found.version_text}, this build needs "
                  f"{KNOWN_PROTOCOL[0]}.x — llama-server will refuse it")
        cls = "devrow bad"
    elif device is None:
        detail, cls = f"protocol {found.version_text}, no device reported", "devrow"
    else:
        detail = (f"protocol {found.version_text} · {gib(device.free_mib)} free of "
                  f"{gib(device.total_mib)}")
        cls = "devrow"
    return Div(
        Span(name, cls="devname"),
        Span(found.endpoint, cls="devdesc"),
        Span(detail, cls="devmem"),
        cls=cls,
    )


def rpc_status(spec: RunSpec, fleet: Fleet | None = None, oob: bool = False) -> Div:
    """What the configured addresses turned out to be, in --rpc order."""
    endpoints = parse_rpc_endpoints(spec.rpc_endpoints)
    body: list = []
    if not endpoints:
        body.append(Span("No workers configured — this machine's GPUs are all that will "
                         "be used.", cls="hint"))
    elif fleet is None:
        body.append(Span(f"{len(endpoints)} configured: {', '.join(endpoints)}. "
                         "Press Check to ask them who they are.", cls="hint"))
    else:
        body.append(Div(*[_worker_row(entry) for entry in fleet.naming()], cls="devlist"))
        total = sum(worker.free_bytes for worker in fleet.workers)
        if total:
            body.append(Span(f"{gib(total / MIB)} free on the workers, on top of this "
                             "machine's own cards. These names are what the device list "
                             "and the tensor split refer to.", cls="hint"))
    return Div(
        Div(
            Span("Workers"),
            Button("Check", type="button", cls="small",
                   hx_post="/server/rpc/check", hx_target="#rpcstatus", hx_swap="outerHTML",
                   hx_include=".paramform"),
            cls="fieldhead",
        ),
        *body,
        id="rpcstatus",
        cls="field wide",
        hx_swap_oob="true" if oob else None,
    )


def check_workers(spec: RunSpec) -> Fleet:
    """Ask the configured addresses who they are. Only ever on a button press.

    A worker serves one client at a time, so this waits behind whatever the
    worker is already doing rather than interrupting it -- which is exactly
    why it is not on a timer.
    """
    return probe_all(parse_rpc_endpoints(spec.rpc_endpoints))


def rpc_guide(plan: WorkerPlan) -> Details:
    advice = guide(plan, machine.hostname())
    return Details(
        Summary("How to set up the second machine"),
        Div(*[Div(f"{number}. {step}", cls="step")
              for number, step in enumerate(advice.steps, start=1)], cls="steps"),
        *[Div("⚠ " + warning, cls="problem warn") for warning in advice.warnings],
        cls="inline-details",
    )


# -- form ------------------------------------------------------------------


def _section(section: Section, config: AppConfig, spec: RunSpec, options: dict,
             facts: ModelFacts | None, scan: Scan, backend: str, params=None):
    named = [name for name in section.names if name != "devices"]
    # the on/off buttons sit together rather than one per column of the grid,
    # where they would be three short pills spaced like three long text boxes
    switches = [name for name in named if name in BY_NAME and BY_NAME[name].kind == "bool"]
    named = [name for name in named if name not in switches]
    fields = [_build_field(spec, config, options) if name == "build_dir"
              else _field(BY_NAME[name], spec, options, facts, scan, backend)
              for name in named]

    body: list = []
    if section.hint:
        body.append(Span(section.hint, cls="hint block"))
    if set(BOUNDED) <= set(named):
        body.append(bounded_fields(spec, facts))
        fields = [field for name, field in zip(named, fields) if name not in BOUNDED]
    if "rpc_endpoints" in named:
        # the guide and the worker command come before the boxes they fill in
        body.append(rpc_guide(worker_plan(params)))
        body.append(worker_panel(params))
    if fields:
        body.append(Div(*fields, cls="grid"))
    if switches:
        body.append(Div(*[_field(BY_NAME[name], spec, options, facts) for name in switches],
                        cls="switches"))
    if "tensor_split" in named:
        devices = run_devices(scan, spec, backend)
        body.append(balancer_field(spec, scan, backend))
        body.append(split_line(spec, facts, devices))
    if "rpc_endpoints" in named:
        body.append(rpc_status(spec))
    if "devices" in section.names:
        body.append(devices_field(spec, scan, backend))

    return Details(Summary(section.title), *body, cls="panel", open=True if section.open else None)


def form(config: AppConfig, spec: RunSpec, scan: Scan, backend: str, params=None) -> Form:
    options = _options(config, spec)
    facts = model_facts(spec)
    panels = [_section(section, config, spec, options, facts, scan, backend, params)
              for section in LAYOUT]

    panels.append(Details(
        Summary("Extra arguments"),
        Textarea(spec.extra_args, name="extra_args", rows=3,
                 placeholder="--spec-type draft-mtp --spec-draft-n-max 3"),
        Span("Anything here wins over the generated flag with the same name.", cls="hint"),
        cls="panel",
    ))

    panels.append(Div(
        # type=button: the form never submits natively, htmx owns every request
        Button("Start server", type="button", cls="primary",
               hx_post="/server/start", hx_target="#runstate", hx_swap="outerHTML"),
        Span("Runs exactly the command shown on the right.", cls="hint"),
        cls="panel runbar",
    ))

    return Form(
        *panels,
        Input(type="hidden", name=FORM_MARKER, value="1"),
        cls="paramform",
        # Validation lives in core.runspec.validate(), which reports problems in the
        # preview panel.  Browser validation would instead abort the htmx request
        # silently (a number outside min/max/step stops the preview from updating).
        novalidate=True,
        # POST keeps the API key in the request body: out of the address bar,
        # out of browser history and out of the uvicorn access log.
        enctype="application/x-www-form-urlencoded",
        hx_post="/server/preview",
        hx_target="#preview",
        # preview() renders its own #preview wrapper, so replace the node itself
        # instead of nesting a second element with the same id.
        hx_swap="outerHTML",
        hx_trigger="change, keyup changed delay:400ms",
    )


# -- memory ----------------------------------------------------------------


def run_devices(scan: Scan, spec: RunSpec, backend: str) -> tuple[Device, ...]:
    """The devices this run will use: the checked ones, or all of them."""
    found = scan.for_backend(backend) if scan.ready else ()
    chosen = selected_devices(spec)
    return tuple(device for device in found if device.name in chosen) or found


def _budget(devices: tuple[Device, ...]) -> tuple[float, list[str], bool]:
    """Device memory to spend, and whether a real run is what measured it."""
    return pool(devices)


def _file_size(path: str) -> int:
    try:
        return Path(path).stat().st_size if path else 0
    except OSError:
        return 0


def _kv_term(report: Estimate) -> float:
    return next((term.mib for term in report.terms if term.label == "KV cache"), 0.0)


def _ways_out(spec: RunSpec, facts: ModelFacts, report: Estimate, budget: float) -> list[str]:
    """Concrete ways to make it fit, priced. Two at most; more is noise."""
    tips: list[str] = []
    kv = _kv_term(report)
    room = budget - (report.total_mib - kv)
    fits = context_for_budget(facts, room, spec.cache_type_k, spec.cache_type_v)
    _low, _high, step = bounds(BY_NAME["ctx_size"], None, facts.n_ctx_train)
    if fits >= step:
        tips.append(f"{context_text(int(fits // step * step))} of context would fit as configured")
    for name, other in kv_alternatives(facts, spec.ctx_size, spec.cache_type_k):
        if kv - other > 0 and spec.cache_type_k == spec.cache_type_v:
            tips.append(f"a {name} KV cache saves {gib(kv - other)}")
            break
    return tips[:2]


def same_run(argv: Sequence[str], other: Sequence[str]) -> bool:
    """Whether two commands would put the same buffers on the same cards.

    The binary's path is dropped: rebuilding into a different directory does
    not change what a run costs. Everything else has to match, because every
    other flag can.
    """
    return bool(other) and tuple(argv[1:]) == tuple(other[1:])


def _measured_rows(measurement: Measurement) -> list:
    """One row per card, showing what it actually held."""
    rows = [
        Div(Span(device.name, cls="memlabel"),
            Span(gib(device.used_mib), cls="memnum"),
            Span(" · ".join(f"{name} {gib(mib)}" for name, mib in device.parts),
                 cls="memdetail"),
            cls="memrow")
        for device in measurement.vram
    ]
    if len(rows) > 1:
        rows.append(Div(Span("Measured total", cls="memlabel"),
                        Span(gib(measurement.vram_mib), cls="memnum"),
                        Span("", cls="memdetail"),
                        cls="memrow total"))
    return rows


def _measured_verdict(measurement: Measurement) -> list:
    """How close each card came to being full, when the card said how big it is."""
    tight = [
        f"{device.name} at {(device.used_mib + (device.overhead_mib or 0)) / device.total_mib:.0%}"
        for device in measurement.vram if device.total_mib
    ]
    if not tight:
        return []
    return [Div("Card in use: " + ", ".join(tight), cls="problem muted")]


@dataclass(frozen=True, slots=True)
class Reading:
    """What is known about this exact command's memory, and how it is known."""

    measurement: Measurement = Measurement()
    #: said out loud in the panel: a figure nobody can trace is worth little
    origin: str = ""
    #: False when a measurement of another context was rescaled to reach this one
    exact: bool = True
    #: a real measurement of some *other* command. Not an answer, but evidence
    #: that this machine has run something, and worth a footnote.
    other: Measurement = Measurement()

    def __bool__(self) -> bool:
        return bool(self.measurement.vram and self.origin)


def reading(argv: Sequence[str], supervisor: Supervisor | None,
            store: MemoryStore | None) -> Reading:
    """The best answer available for this command, newest evidence first.

    The job on the slot wins, because it is this machine right now. Failing
    that, a stored run of the same command. Failing that, the same command at
    another context, with only its KV cache moved -- and said so.
    """
    live = supervisor.measurement() if supervisor else Measurement()
    snapshot = supervisor.snapshot() if supervisor else None
    if snapshot is not None and same_run(argv, snapshot.argv) and live.vram:
        return Reading(live, "this run, as it reported its own buffers")
    aside = live if live.vram else Measurement()

    if store is None:
        return Reading(other=aside)
    measurement, record, exact = store.recall(argv)
    if record is None or not measurement.vram:
        return Reading(other=aside)
    if exact:
        return Reading(measurement, f"a run of these settings {record.age_text}", other=aside)
    return Reading(
        measurement,
        f"a run of the same settings at {context_text(record.context)}, {record.age_text}",
        exact=False,
        other=aside,
    )


def memory_panel(spec: RunSpec, facts: ModelFacts | None, scan: Scan, backend: str,
                 known: Reading = Reading()) -> Div:
    """The VRAM bill, next to the command that will run it up.

    Before the first run this is arithmetic on the model header: no process is
    started to find out, so the answer arrives before the out-of-memory rather
    than after it. Once a run of these settings has reported its own buffers,
    the arithmetic steps aside and the measurement is shown instead.
    """
    devices = run_devices(scan, spec, backend)
    report = estimate(spec, facts, devices=max(1, len(devices)),
                      mmproj_bytes=_file_size(spec.mmproj))

    if known:
        measurement = known.measurement
        headline = ("Measured, not estimated" if known.exact
                    else "Measured at another context and scaled")
        return Div(
            H3("Memory"),
            Div(f"{headline} — from {known.origin}.", cls="problem ok"),
            *_measured_rows(measurement),
            *_measured_verdict(measurement),
            *([] if known.exact else [Span(
                "Weights and compute buffers do not move with the context; the KV cache "
                "is exactly proportional to it, so it is the only figure adjusted.",
                cls="hint block")]),
            *[Span(note, cls="hint block") for note in measured_notes(measurement)],
            Span(f"the arithmetic for these settings says {gib(report.total_mib)}"
                 if report.terms else "", cls="hint block"),
            cls="panel memory",
        )

    rows = [
        Div(Span(term.label, cls="memlabel"),
            Span(gib(term.mib), cls="memnum"),
            Span(term.detail, cls="memdetail"),
            cls="memrow")
        for term in report.terms
    ]
    if rows:
        rows.append(Div(Span("Estimated total", cls="memlabel"),
                        Span(gib(report.total_mib), cls="memnum"),
                        Span("", cls="memdetail"),
                        cls="memrow total"))

    budget, parts, measured = _budget(devices)
    notes = list(report.notes)
    verdict: list = []
    if not report.terms:
        pass
    elif not scan.ready:
        verdict.append(Div("looking for devices…", cls="problem muted"))
    elif budget <= 0:
        verdict.append(Div(
            "Select a build to know which devices this would run on." if not devices
            else "No device memory known yet, so there is nothing to compare this against. "
                 "One finished run teaches the GUI the real numbers.",
            cls="problem muted"))
    else:
        headroom = budget - report.total_mib
        source = "free" if measured else "installed"
        summary = " + ".join(parts)
        if headroom >= 0:
            verdict.append(Div(f"Fits: {gib(headroom)} to spare of {gib(budget)} {source} "
                               f"({summary})", cls="problem ok"))
        else:
            verdict.append(Div(f"⚠ {gib(-headroom)} over the {gib(budget)} {source} "
                               f"({summary})", cls="problem err"))
            if facts is not None:
                verdict += [Div(tip, cls="problem muted")
                            for tip in _ways_out(spec, facts, report, budget)]
        if len(parts) > 1:
            notes.append("this is the total across the devices; how much lands on each is "
                         "the split mode's decision, and llama.cpp fills by free memory")
    if known.other.vram:
        # a measurement of some other settings is still worth more than nothing:
        # it says how far this estimate has been off on this machine before
        notes.append(f"the last run took {gib(known.other.vram_mib)} across "
                     f"{len(known.other.vram)} device(s), but it was not these settings")

    # the scan runs off the request thread, so the first render can be too early
    # to compare against anything; ask the form for one more once it has landed
    pending = bool(report.terms) and not scan.ready
    return Div(
        H3("Memory"),
        *verdict,
        *rows,
        *[Span(note, cls="hint block") for note in notes],
        cls="panel memory",
        hx_post="/server/preview" if pending else None,
        hx_trigger="load delay:700ms" if pending else None,
        hx_include=".paramform" if pending else None,
        hx_target="#preview" if pending else None,
        hx_swap="outerHTML" if pending else None,
    )


# -- preview ---------------------------------------------------------------


def _port_problems(spec: RunSpec) -> list[Problem]:
    """Whether this port is already spoken for, and which one is not.

    The probe is a loopback connect that is closed at once. It runs on every
    preview, which is why it has to stay this cheap; and it is the only way to
    turn "Error: bind failed" three seconds after a launch into an answer
    before it.
    """
    if not machine.port_taken(spec.port):
        return []
    free = machine.free_port(spec.port + 1)
    advice = f" — port {free} is free" if free else ""
    return [Problem("warn", f"Something is already listening on port {spec.port}"
                            f"{advice}. Two servers cannot share one port.")]


def preview(config: AppConfig, spec: RunSpec, scan: Scan, oob: bool = False,
            supervisor: Supervisor | None = None, store: MemoryStore | None = None) -> Div:
    build = build_of(config, spec)
    backend = build.backend if build else ""
    binary = build.server_bin if build and build.server_bin else Path("llama-server")
    facts = model_facts(spec)

    known = [device.name for device in scan.for_backend(backend)] if scan.ready else None
    problems = list(validate(spec, backend=backend,
                             supports_rpc=build.supports_rpc if build else None,
                             available_devices=known))
    if build is not None and not build.usable:
        problems.insert(0, Problem("error", f"{build.name} has no llama-server binary"))
    problems += _port_problems(spec)

    argv = to_argv(spec, binary)
    # seeded_from: one value per axis, which is a measurement of this
    # configuration -- the same command the Autotune page opens with
    _, bench_argv = bench_commands(spec, BenchSpec().seeded_from(spec),
                                   config.bench_script, binary, backend=backend)[0]
    bench_query = spec_link(spec)

    known = reading(argv, supervisor, store)

    return Div(
        Div(
            H3("Command"),
            *problem_lines(problems),
            Pre(f"# build: {build.name} ({build.backend})" if build else "# build: not selected"),
            Pre(f"# model: {facts.summary}") if facts and facts.summary else None,
            Pre(command_lines(mask_api_key(argv))),
            cls="panel",
        ),
        memory_panel(spec, facts, scan, backend, known),
        Details(
            Summary("Measure this configuration"),
            Span("The same settings, asked a fixed set of prompts and timed. Tick a second "
                 "batch size or KV type on the Autotune page and it measures each in turn.",
                 cls="hint block"),
            Pre(command_lines(mask_api_key(bench_argv))),
            A("Open in Autotune →", href=f"/autotune?{bench_query}", cls="button"),
            cls="panel",
        ),
        id="preview",
        hx_swap_oob="true" if oob else None,
    )


# -- process ---------------------------------------------------------------


def _endpoint(argv: tuple[str, ...]) -> str:
    """The URL a running server listens on, read back from its own argv."""
    host, port = "127.0.0.1", ""
    for flag, value in zip(argv, argv[1:]):
        if flag == "--host":
            host = "127.0.0.1" if value in {"0.0.0.0", "::"} else value
        elif flag == "--port":
            port = value
    return f"http://{host}:{port}" if port else ""


def run_panel(supervisor: Supervisor, message: str = "", level: str = "note") -> Div:
    snapshot = supervisor.snapshot()
    alive = bool(snapshot and snapshot.alive)

    body: list = []
    if message:
        body.append(Div(("⚠ " if level == "error" else "") + message,
                        cls="problem err" if level == "error" else "problem muted"))

    if snapshot is None:
        body.append(Div("Nothing has been started from this GUI.", cls="muted"))
    else:
        body.append(Div(
            Span(snapshot.status, cls=f"badge {snapshot.status}"),
            Span(snapshot.label, cls="label"),
            Span(f"pid {snapshot.pid}" if snapshot.pid else "", cls="muted"),
            Span(snapshot.runtime_text, cls="muted"),
            cls="runline",
        ))
        if not alive:
            body.append(Div(snapshot.outcome, cls="muted"))

    controls: list = []
    if alive and snapshot is not None:
        controls.append(Button("Stop", type="button", hx_post="/server/stop",
                               hx_target="#runstate", hx_swap="outerHTML"))
        endpoint = _endpoint(snapshot.argv)
        if endpoint:
            controls.append(A("Open web UI", href=endpoint, target="_blank", cls="button"))
        controls.append(Details(
            Summary("force stop"),
            Div("A hard kill interrupts GPU work mid-flight and can leave the driver "
                "unhappy. Use it only when a graceful stop was ignored.", cls="hint"),
            Button("Force stop", type="button", cls="danger", hx_post="/server/kill",
                   hx_target="#runstate", hx_swap="outerHTML"),
            cls="inline-details",
        ))

    return Div(
        H3("Process"),
        *body,
        Div(*controls, cls="row") if controls else None,
        id="runstate",
        cls="panel",
        # Poll only while something is alive; the last response has no trigger,
        # so an idle page makes no requests at all.
        hx_get="/server/status" if alive else None,
        hx_trigger="every 2s" if alive else None,
        hx_swap="outerHTML" if alive else None,
    )


def _poller(supervisor: Supervisor, cursor: int):
    """Self-replacing tail marker: new lines land in front of it."""
    snapshot = supervisor.snapshot()
    alive = bool(snapshot and snapshot.alive)
    unread = bool(snapshot and cursor < snapshot.log_total)
    trigger = "every 1s" if alive else ("load delay:300ms" if unread else None)
    return Span(
        # the marker carries the placeholder, so the first real line replaces it
        Div("no output yet", cls="muted") if not snapshot or not snapshot.log_total else None,
        id="logtail",
        hx_get=f"/server/log?cursor={cursor}",
        hx_trigger=trigger,
        hx_swap="outerHTML",
    )


def log_since(supervisor: Supervisor, cursor: int):
    cursor, lines = supervisor.log_since(cursor)
    return (*[Div(line, cls="logline") for line in lines], _poller(supervisor, cursor))


def log_panel(supervisor: Supervisor, oob: bool = False) -> Div:
    cursor, lines = supervisor.log_since(0)
    return Div(
        H3("Log"),
        Div(
            *[Div(line, cls="logline") for line in lines],
            _poller(supervisor, cursor),
            id="log",
            cls="logbox",
            # htmx's own event hook, not a JS dependency: keep the tail in view
            **{"hx-on::after-settle": "this.scrollTop = this.scrollHeight"},
        ),
        id="logpanel",
        cls="panel",
        hx_swap_oob="true" if oob else None,
    )


def start(config: AppConfig, supervisor: Supervisor, spec: RunSpec, scan: Scan):
    """Validate, then hand the command to the supervisor."""
    build = build_of(config, spec)
    known = [device.name for device in scan.for_backend(build.backend if build else "")] \
        if scan.ready else None
    blocking = [problem.message for problem
                in validate(spec, backend=build.backend if build else "",
                            supports_rpc=build.supports_rpc if build else None,
                            available_devices=known)
                if problem.level == "error"]
    if build is None:
        blocking.append("Select a build")
    elif not build.usable:
        blocking.append(f"{build.name} has no llama-server binary")
    if blocking:
        return run_panel(supervisor, "; ".join(dict.fromkeys(blocking)), "error")

    assert build is not None and build.server_bin is not None
    label = f"llama-server · {Path(spec.model).name} · {build.name}"
    try:
        supervisor.start("server", label, to_argv(spec, build.server_bin), cwd=build.path)
    except Busy as busy:
        return run_panel(supervisor, f"{busy.current.label} is still running", "error")
    return run_panel(supervisor), log_panel(supervisor, oob=True)


def page(config: AppConfig, spec: RunSpec, supervisor: Supervisor, scan: Scan, backend: str,
         params=None, store: MemoryStore | None = None):
    return shell(
        "Server", "/server", config,
        Div(
            form(config, spec, scan, backend, params),
            Div(preview(config, spec, scan, supervisor=supervisor, store=store),
                run_panel(supervisor), log_panel(supervisor), cls="stack"),
            cls="split",
        ),
        nav={"/autotune": spec_link(spec)},
    )
