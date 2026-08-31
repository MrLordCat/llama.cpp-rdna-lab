"""bench2 invocation, derived from the same RunSpec.

`scripts/bench2.py` measures one server per run: it loads the model once,
sized to the largest scenario asked of it, and works through every level and
session against that one server. Batch, ubatch, KV type and speculation are
therefore properties of a run rather than axes inside it -- so a search over
them is several runs, and this module produces the list, one command per
configuration, named so bench2's own index can tell them apart afterwards.

What bench2 does not own is forwarded verbatim through --server-extra, so the
server the GUI describes and the server the benchmark starts cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path

from gui2.core.gguf import ModelFacts
from gui2.core.memory import estimate
from gui2.core.params import KV_TYPES, Param, aliases_of
from gui2.core.runspec import ALL_LAYERS, Problem, RunSpec, to_argv

CONTEXT_SOURCES = ("synthetic", "repo-snapshot", "file")

#: bench2 speaks two speculation modes; llama.cpp's other ones have no flag of
#: its own to reach them
SPEC_MODES = ("none", "mtp")

#: key and value are set together here, because a run that mixes them is not
#: something anyone has wanted to measure
BENCH_KV_TYPES = KV_TYPES

#: the repository snapshot runs out at roughly this many tokens, so it cannot
#: fill a large level and the run would measure a shorter prompt than it names
REPO_SNAPSHOT_MAX_TOKENS = 53000


@dataclass(frozen=True, slots=True)
class Scenario:
    """One row of bench2's level or session table."""

    key: str
    name: str
    ctx: int
    prompt_tokens: int
    decode_tokens: int
    turns: int = 1

    @property
    def decoded(self) -> int:
        return self.decode_tokens * self.turns

    @property
    def prefilled(self) -> int:
        """Tokens actually run through prefill.

        A session re-sends the whole conversation every turn, but the server
        keeps the KV cache, so only each turn's new text is prefilled.
        """
        return self.prompt_tokens * self.turns


#: Copied from `configs/bench/levels.json`. bench2 reads that file itself; this
#: is here so a run can be sized before anything is started.
LEVELS: dict[str, Scenario] = {
    "0": Scenario("0", "smoke", 8192, 4096, 64),
    "1": Scenario("1", "test", 16384, 8192, 128),
    "2": Scenario("2", "standard", 49152, 31744, 256),
    "3": Scenario("3", "large", 98304, 66560, 256),
    "4": Scenario("4", "very-large", 131072, 97280, 256),
    "5": Scenario("5", "max", 200704, 194560, 256),
}

#: Copied from `configs/bench/sessions.json`: ten turns in one conversation,
#: context kept between them, so the KV cache grows the way an agent grows it.
SESSIONS: dict[str, Scenario] = {
    "1": Scenario("1", "light", 32768, 1024, 128, turns=10),
    "2": Scenario("2", "medium", 98304, 2048, 256, turns=10),
    "3": Scenario("3", "heavy", 131072, 4096, 512, turns=10),
}

# flags bench2 passes to llama-server itself, from options of its own
BENCH_OWNED: frozenset[str] = frozenset().union(*(
    aliases_of(flag) for flag in (
        "-m", "--host", "--port", "-c", "--batch-size", "--ubatch-size",
        "-ngl", "--parallel", "--cache-type-k", "--cache-type-v", "--flash-attn",
        "-dev", "-sm", "-ts", "-fit", "--rpc",
    )
)) | {"--no-warmup", "--seed"}

#: Flags dropped rather than forwarded.
#:
#: --api-key: bench2 sends no Authorization header, so a server started with
#: one answers 401 to every request it makes; and that server is loopback-only
#: anyway, since --host is one of the flags bench2 owns.
#:
#: --spec-type and its companions: bench2 appends its own, chosen by --spec.
#: Two would not break llama-server, but the run would be filed under one mode
#: and measured under the other.
#:
#: --cache-ram and the checkpoint flags: bench2 pins these to zero so a level
#: measures prefill rather than a cache hit. Forwarding the Server page's
#: values would quietly change what the numbers mean.
BENCH_DROPPED: frozenset[str] = frozenset().union(*(
    aliases_of(flag) for flag in (
        "--api-key", "--spec-type", "--spec-draft-n-max", "--cache-ram",
        "--ctx-checkpoints", "--checkpoint-every-n-tokens",
        "--spec-ngram-mod-n-min", "--spec-ngram-mod-n-match", "--spec-ngram-mod-n-max",
    )
))


@dataclass(frozen=True, slots=True)
class BenchSpec:
    run_name: str = ""
    #: which scenarios to run against the one server, as bench2 spells them
    levels: str = "1"
    session_levels: str = ""
    runs: int = 1
    context_source: str = "synthetic"
    context_file: str = ""
    #: server settings to try. bench2 fixes each of these for a whole run, so
    #: more than one value here is more than one run.
    batch: str = "8192"
    ubatch: str = "1024"
    kv: str = "q8_0"
    spec: str = "none"
    spec_n: str = "2"
    sweep_max: int = 12
    warmup_shot: bool = True
    warmup_tokens: int = 512
    warmup_decode: int = 16
    seed: int = 42
    temperature: float = 0.2
    top_p: float = 0.9
    health_timeout: float = 300.0
    fail_fast: bool = False

    def with_values(self, values: dict) -> "BenchSpec":
        """A copy with whatever the form named, ignoring everything else."""
        known = {field.name for field in fields(self)}
        updates = {key: _coerce(value, getattr(self, key))
                   for key, value in values.items() if key in known}
        return replace(self, **updates)

    def seeded_from(self, spec: RunSpec) -> "BenchSpec":
        """The run the Server page describes, as the smallest thing to measure.

        The context is not a setting here: bench2 sizes the server from the
        levels asked of it, so the page opens on the level that context has
        room for rather than on a context box of its own.
        """
        return replace(
            self,
            batch=str(spec.batch_size),
            ubatch=str(spec.ubatch_size),
            kv=spec.cache_type_k,
            spec=spec.spec_type if spec.spec_type in SPEC_MODES else "none",
            spec_n=str(spec.spec_draft_n_max),
            levels=level_for_context(spec.ctx_size),
        )


BENCH_DEFAULTS = BenchSpec()


def level_for_context(ctx: int) -> str:
    """The largest level the given context has room for."""
    fitting = [key for key, level in LEVELS.items() if level.ctx <= ctx]
    return max(fitting, key=lambda key: LEVELS[key].ctx) if fitting else "0"


def _coerce(value, current):
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "on", "true", "yes"} \
            if isinstance(value, str) else bool(value)
    for kind in (int, float):
        if isinstance(current, kind) and not isinstance(current, bool):
            try:
                return kind(str(value).strip())
            except (TypeError, ValueError):
                return current
    return str(value).strip()


B_WORK = "What to measure"
B_SWEEP = "Server settings to try"
B_PROMPT = "What the prompt is made of"
B_FAIR = "What the numbers are allowed to include"
B_LIMITS = "When to give up"
B_OUTPUT = "What is written down"

#: Presentation for bench2's flags, in the same shape as the llama-server
#: schema. Unlike that one it does not generate the command: `to_bench_argv`
#: spells it out, because several of these are pairs and four of them are axes
#: the GUI multiplies out itself.
BENCH_SCHEMA: tuple[Param, ...] = (
    Param("levels", "Single levels", "multi", B_WORK, choices=tuple(LEVELS),
          help="one big prompt and then a decode; the level sets both, and the "
               "largest one ticked sizes the server"),
    Param("session_levels", "Agent sessions", "multi", B_WORK, choices=tuple(SESSIONS),
          help="ten turns of one conversation with the context kept between them, "
               "which is what an agent actually does to a server"),
    Param("runs", "Repeats", "slider", B_WORK, minimum=1, maximum=10, step=1,
          help="how many times each scenario is measured — more repeats, steadier numbers"),
    Param("batch", "Batch sizes to try", "multi", B_SWEEP,
          choices=("512", "1024", "2048", "4096", "8192"),
          help="how many tokens the server reads at once"),
    Param("ubatch", "Ubatch sizes to try", "multi", B_SWEEP,
          choices=("128", "256", "512", "1024", "2048"),
          help="how much of a batch reaches the GPU in one go — never above the batch"),
    Param("kv", "KV cache types to try", "multi", B_SWEEP, choices=BENCH_KV_TYPES,
          help="what the context is stored as; key and value are set together"),
    Param("spec", "Speculation to try", "multi", B_SWEEP, choices=SPEC_MODES,
          help="whether the model's own draft head guesses ahead of itself"),
    Param("spec_n", "Draft tokens to try", "multi", B_SWEEP,
          choices=tuple(str(n) for n in range(1, 9)),
          help="how far ahead it guesses; tick several to queue one run per value — "
               "used only where speculation is on"),
    Param("sweep_max", "Refuse to start above", "slider", B_SWEEP, minimum=1, maximum=64, step=1,
          help="runs; every extra value multiplies the search rather than adding to it"),
    Param("context_source", "Prompt text", "choice", B_PROMPT, choices=CONTEXT_SOURCES,
          help="what fills the tokens the level asks for"),
    Param("context_file", "Text file", "text", B_PROMPT,
          help="only for the file source: the text put in front of every prompt"),
    Param("warmup_shot", "One unmeasured shot first", "bool", B_FAIR,
          help="a short request before anything is recorded, so the first measured "
               "answer is not paying for the first kernel launch"),
    Param("warmup_tokens", "Warm-up prompt", "slider", B_FAIR, minimum=64, maximum=4096, step=64,
          help="tokens in that unmeasured request"),
    Param("warmup_decode", "Warm-up answer", "slider", B_FAIR, minimum=4, maximum=128, step=4,
          help="tokens it is asked to generate"),
    Param("temperature", "Temperature", "slider", B_FAIR, minimum=0.0, maximum=2.0, step=0.1,
          help="low keeps answers the length they were asked for, which is what "
               "decode speed is measured over"),
    Param("top_p", "Top-p", "slider", B_FAIR, minimum=0.0, maximum=1.0, step=0.05,
          help="how much of the probability mass sampling may reach into"),
    Param("seed", "Seed", "int", B_FAIR, minimum=0,
          help="the same seed asks the model the same question twice"),
    Param("health_timeout", "Wait for the server to load for", "slider", B_LIMITS,
          minimum=60, maximum=3600, step=30,
          help="a large model, or one spread over RPC workers, can take minutes "
               "before it answers"),
    Param("fail_fast", "Stop at the first failure", "bool", B_LIMITS,
          help="otherwise a scenario that fails is recorded as failed and the rest "
               "still run"),
    Param("run_name", "Label these runs", "text", B_OUTPUT,
          help="goes in front of the folder name; what was measured is added after it "
               "either way, so attempts never write over each other"),
)

BENCH_BY_NAME: dict[str, Param] = {param.name: param for param in BENCH_SCHEMA}

#: axes ticked rather than typed: several boxes share one name, so reading the
#: form back needs every value, and none ticked means an empty axis
MULTI_NAMES: frozenset[str] = frozenset(
    param.name for param in BENCH_SCHEMA if param.kind == "multi")

#: the five settings bench2 fixes for a whole run, and so the five the GUI has
#: to run more than once to compare
SWEEP_NAMES: tuple[str, ...] = ("batch", "ubatch", "kv", "spec", "spec_n")


def items(text: str) -> list[str]:
    """One axis, split however the person happened to type it."""
    return [chunk for chunk in re.split(r"[,;\s]+", (text or "").strip()) if chunk]


def scenarios(bench: BenchSpec) -> list[Scenario]:
    """Every level and session one run works through, in bench2's order."""
    chosen = [LEVELS[key] for key in items(bench.levels) if key in LEVELS]
    chosen += [SESSIONS[key] for key in items(bench.session_levels) if key in SESSIONS]
    return chosen


def server_context(bench: BenchSpec) -> int:
    """The context the one server is started with.

    bench2 takes the largest scenario asked of it and sizes the server to that,
    so adding a small level to a large one costs no extra memory and no extra
    model load.
    """
    return max((item.ctx for item in scenarios(bench)), default=0)


@dataclass(frozen=True, slots=True)
class Configuration:
    """One server the search will measure, and what its results are called."""

    batch: int
    ubatch: int
    kv: str
    spec: str
    spec_n: int = 2

    @property
    def suffix(self) -> str:
        draft = f"-n{self.spec_n}" if self.spec != "none" else ""
        return f"b{self.batch}-u{self.ubatch}-{self.kv}-{self.spec}{draft}"

    def describe(self, varied: frozenset[str]) -> str:
        """Named by whatever tells it apart from the others."""
        parts = [f"batch {self.batch}" if "batch" in varied else "",
                 f"ubatch {self.ubatch}" if "ubatch" in varied else "",
                 self.kv if "kv" in varied else "",
                 self.spec if "spec" in varied else "",
                 f"draft {self.spec_n}" if "spec_n" in varied and self.spec != "none" else ""]
        return ", ".join(part for part in parts if part) or "one configuration"


def axis_values(bench: BenchSpec) -> dict[str, list[str]]:
    return {name: items(getattr(bench, name)) for name in SWEEP_NAMES}


def _all_configurations(bench: BenchSpec) -> list[Configuration]:
    """Every raw combination, valid or not; configurations() filters these.

    Draft tokens only multiply a run when speculation is on: with `spec none`
    they would produce the same command several times under one name.
    """
    values = axis_values(bench)
    drafts = values["spec_n"]
    found: list[Configuration] = []
    for batch, ubatch, kv, spec in product(values["batch"], values["ubatch"],
                                           values["kv"], values["spec"]):
        if not (batch.isdigit() and ubatch.isdigit()):
            continue
        # no speculation: one run whatever drafts are ticked, or the default
        # when none are — several would be the same command several times
        ns = ["2"] if spec == "none" else (drafts or ["2"])
        for draft in ns:
            if draft.isdigit():
                found.append(Configuration(int(batch), int(ubatch), kv, spec, int(draft)))
    return found


def varied(bench: BenchSpec) -> frozenset[str]:
    """The axes with more than one value, which is what a run is named after."""
    return frozenset(name for name, values in axis_values(bench).items() if len(values) > 1)


def configurations(bench: BenchSpec) -> list[Configuration]:
    """Every combination of the four that would actually run, in order.

    A ubatch above its batch is left out rather than run: llama-server
    refuses the pair, and it would refuse it once per run.
    """
    return [config for config in _all_configurations(bench)
            if config.ubatch <= config.batch]


def dropped_configs(bench: BenchSpec) -> int:
    """Combinations skipped because ubatch must never exceed its batch."""
    return len(_all_configurations(bench)) - len(configurations(bench))


def config_count(bench: BenchSpec) -> int:
    return len(configurations(bench))


def scenario_tag(bench: BenchSpec) -> str:
    """The workload a name is told apart by: `l0`, `l013`, `l0-s1`."""
    levels = "".join(key for key in items(bench.levels) if key in LEVELS)
    sessions = "".join(key for key in items(bench.session_levels) if key in SESSIONS)
    return "-".join(part for part in (f"l{levels}" if levels else "",
                                      f"s{sessions}" if sessions else "") if part)


def base_name(spec: RunSpec, bench: BenchSpec, backend: str = "") -> str:
    """What goes in front of every folder name of one search.

    Deterministic rather than stamped with the time, so the command shown is
    the command that runs. What tells one attempt from the next is added after
    it by `run_names`, not here.
    """
    if bench.run_name:
        return bench.run_name
    stem = Path(spec.model).stem.lower() if spec.model else "model"
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:32] or "model"
    return "-".join(part for part in (backend or "run", stem, scenario_tag(bench)) if part)


def run_names(spec: RunSpec, bench: BenchSpec, backend: str = "") -> list[str]:
    """One folder per configuration, each named after the configuration it is.

    The settings are in the name even when only one is being measured. Trying a
    second combination would otherwise land in the folder the first one wrote,
    and the way out of that would be typing a fresh name before every attempt --
    which is the whole of the work when the point is to try combinations.
    """
    base = base_name(spec, bench, backend)
    return [f"{base}-{config.suffix}" for config in configurations(bench)]


# -- what the run will ask of the cards -------------------------------------


@dataclass(frozen=True, slots=True)
class Weighed:
    """One configuration's VRAM bill, and what distinguishes it from the rest."""

    ctx: int
    kv: str
    ubatch: int
    mib: float


@dataclass(frozen=True, slots=True)
class Fit:
    """How much of a search the devices have room for.

    A server too big to load is not skipped: bench2 waits out the whole health
    timeout, records the failure and moves on to the next configuration. So
    each one costs that timeout and produces no measurement, which is worth
    knowing before the search rather than after.
    """

    budget_mib: float = 0.0
    #: distinct bills, lightest first
    weighed: tuple[Weighed, ...] = ()
    #: configurations each of those stands for -- the axes that cost nothing
    each: int = 1

    @property
    def total(self) -> int:
        return len(self.weighed) * self.each

    @property
    def over(self) -> tuple[Weighed, ...]:
        return tuple(item for item in self.weighed if item.mib > self.budget_mib)

    @property
    def over_count(self) -> int:
        return len(self.over) * self.each

    @property
    def heaviest(self) -> Weighed | None:
        return self.weighed[-1] if self.weighed else None

    @property
    def largest_fitting(self) -> Weighed | None:
        within = [item for item in self.weighed if item.mib <= self.budget_mib]
        return within[-1] if within else None


def fit(spec: RunSpec, facts: ModelFacts | None, bench: BenchSpec, budget_mib: float,
        devices: int = 1, mmproj_bytes: int = 0) -> Fit | None:
    """The search priced against the memory there is, or None if it cannot be.

    Only the KV type and the ubatch move the bill: the context is the same for
    every configuration, because bench2 sizes the server from the largest
    scenario and that does not change from run to run.
    """
    ctx = server_context(bench)
    if facts is None or not facts.known or budget_mib <= 0 or ctx <= 0:
        return None
    values = axis_values(bench)
    ubatches = sorted({int(value) for value in values["ubatch"] if value.isdigit()})
    kv_types = [value for value in values["kv"] if value in BENCH_KV_TYPES]
    if not (ubatches and kv_types):
        return None

    weighed: list[Weighed] = []
    for kv in kv_types:
        for ubatch in ubatches:
            probe = replace(spec, ctx_size=ctx, ubatch_size=ubatch,
                            cache_type_k=kv, cache_type_v=kv)
            report = estimate(probe, facts, devices=devices, mmproj_bytes=mmproj_bytes)
            if not report.complete:
                return None
            weighed.append(Weighed(ctx, kv, ubatch, report.total_mib))
    weighed.sort(key=lambda item: item.mib)
    # batch and speculation multiply the bill without moving it; draft tokens
    # do only where speculation is on, which is the same collapse as configurations()
    each = (max(1, len(values["batch"])) * max(1, len(values["spec"]))
            * max(1, len(values["spec_n"]) if "mtp" in values["spec"] else 1))
    return Fit(budget_mib=budget_mib, weighed=tuple(weighed), each=each)


# -- the command ------------------------------------------------------------


def server_extra_tokens(spec: RunSpec) -> list[str]:
    """Generated server flags minus the ones bench2 sets or refuses."""
    tokens = to_argv(spec)[1:]
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.split("=", 1)[0] in BENCH_OWNED | BENCH_DROPPED:
            index += 1
            if "=" not in token and index < len(tokens) and not tokens[index].startswith("-"):
                index += 1
            continue
        kept.append(token)
        index += 1
    return kept


def _flag(value: bool, on: str, off: str) -> list[str]:
    return [on] if value else [off]


def to_bench_argv(
    spec: RunSpec,
    bench: BenchSpec,
    config: Configuration,
    script: str | Path,
    server_bin: str | Path,
    run_name: str = "",
    series_id: str = "",
    backend: str = "",
    python: str = "python",
) -> list[str]:
    """One `python scripts/bench2.py run ...` command line.

    The context is absent on purpose: bench2 sizes the server from the levels
    it is given, so naming one here would describe a server it never starts.
    """
    argv = [python, str(script), "run"]
    if run_name:
        argv += ["--run-name", run_name]
    if series_id:
        argv += ["--series-id", series_id]
    if levels := [key for key in items(bench.levels) if key in LEVELS]:
        argv += ["--level", ",".join(levels)]
    if sessions := [key for key in items(bench.session_levels) if key in SESSIONS]:
        argv += ["--session-level", ",".join(sessions)]
    argv += ["--runs", str(bench.runs)]
    argv += ["--server-bin", str(server_bin), "--model", spec.model]
    if backend:
        # left unsaid, bench2 guesses from the binary's path and falls back to rocm
        argv += ["--backend", backend]
    argv += [
        "--batch-size", str(config.batch),
        "--ubatch-size", str(config.ubatch),
        "--kv-k", config.kv, "--kv-v", config.kv,
        "--spec", config.spec,
    ]
    if config.spec != "none":
        argv += ["--spec-n", str(config.spec_n)]
    argv += [
        "--gpu-layers", ALL_LAYERS if spec.gpu_layers_all else str(spec.gpu_layers),
        "--parallel", str(spec.parallel),
    ]
    argv += _flag(spec.flash_attn != "off", "--flash-attn", "--no-flash-attn")
    # the device list belongs to bench2 rather than to --server-extra: left out,
    # it falls back to the hardware profile, which names cards of its own
    if spec.rpc_endpoints:
        # bench2 owns the flag too, and puts it before -dev, which is where
        # llama-server needs it: -dev validates names as parsed, and RPC
        # devices exist only after --rpc has registered them
        argv += ["--rpc", spec.rpc_endpoints]
    argv += ["--dev", spec.devices, "--sm", spec.split_mode, "--ts", spec.tensor_split]
    argv += ["--context-source", bench.context_source]
    if bench.context_source == "file" and bench.context_file:
        argv += ["--context-file", bench.context_file]
    argv += [
        "--seed", str(bench.seed),
        "--temperature", f"{bench.temperature:g}",
        "--top-p", f"{bench.top_p:g}",
    ]
    argv += _flag(bench.warmup_shot, "--warmup-shot", "--no-warmup-shot")
    if bench.warmup_shot:
        argv += ["--warmup-tokens", str(bench.warmup_tokens),
                 "--warmup-decode", str(bench.warmup_decode)]
    argv += ["--health-timeout", str(int(bench.health_timeout))]
    if bench.fail_fast:
        argv.append("--fail-fast")

    extra = server_extra_tokens(spec)
    if extra:
        # --flag=value form: a separate value starting with '-' would be read as
        # a new option. Joined with plain spaces because bench2 splits the string
        # on whitespace and nothing else -- quoting it would arrive as quotes.
        argv.append("--server-extra=" + " ".join(extra))
    return argv


def bench_commands(
    spec: RunSpec,
    bench: BenchSpec,
    script: str | Path,
    server_bin: str | Path,
    series_id: str = "",
    backend: str = "",
    python: str = "python",
) -> list[tuple[str, list[str]]]:
    """Every run the search comes to, as (folder name, command line).

    One configuration is one command, so the page reads the same either way:
    a search over one value is a measurement of it.
    """
    return [
        (name, to_bench_argv(spec, bench, config, script, server_bin,
                             run_name=name, series_id=series_id,
                             backend=backend, python=python))
        for name, config in zip(run_names(spec, bench, backend), configurations(bench))
    ]


# -- what the run will do, counted before it does it ------------------------


@dataclass(frozen=True, slots=True)
class Plan:
    """The size of a run, in the units that decide how long it takes."""

    requests: int = 0
    decoded: int = 0
    prefilled: int = 0
    configs: int = 1
    startup_s: float = 0.0

    @property
    def loads(self) -> int:
        """Model loads. One per configuration, and they are the slow part."""
        return self.configs


def plan(bench: BenchSpec) -> Plan:
    """How much work the chosen scenarios come to, before anything is started."""
    chosen = scenarios(bench)
    each = max(1, bench.runs) * max(1, config_count(bench))
    return Plan(
        requests=sum(item.turns for item in chosen) * each,
        decoded=sum(item.decoded for item in chosen) * each,
        prefilled=sum(item.prefilled for item in chosen) * each,
        configs=max(1, config_count(bench)),
        startup_s=max(0.0, bench.health_timeout),
    )


# -- what would stop it -----------------------------------------------------


def _vocabulary_problems(bench: BenchSpec) -> list[Problem]:
    """Values on an axis that nothing downstream would accept.

    The page offers these as tick lists, so they can only arrive from a link or
    an older saved run. bench2 exits on an unknown level before starting a
    server; llama-server refuses an unknown KV type after loading one.
    """
    problems: list[Problem] = []
    for axis, label, allowed in (("levels", "Single levels", tuple(LEVELS)),
                                 ("session_levels", "Agent sessions", tuple(SESSIONS)),
                                 ("kv", "KV cache types", BENCH_KV_TYPES),
                                 ("spec", "Speculation", SPEC_MODES)):
        bad = [value for value in items(getattr(bench, axis)) if value not in allowed]
        if bad:
            problems.append(Problem(
                "error",
                f"{label}: {', '.join(bad)} — the ones that work here are "
                f"{', '.join(allowed)}"))
    for axis, label in (("batch", "Batch sizes"), ("ubatch", "Ubatch sizes")):
        bad = [value for value in items(getattr(bench, axis)) if not value.isdigit()]
        if bad:
            problems.append(Problem(
                "error",
                f"{label} to try: {', '.join(bad)} — every value on that axis has to be "
                f"a plain number of tokens"))
    return problems


def validate_bench(spec: RunSpec, bench: BenchSpec,
                   facts: ModelFacts | None = None,
                   existing: frozenset[str] = frozenset()) -> list[Problem]:
    """Everything that would stop or spoil this run, said before it is started.

    Most of these are an exit code in bench2 -- after it has been launched, and
    some of them after it has already loaded the model once.
    """
    problems: list[Problem] = []
    problems += _vocabulary_problems(bench)

    chosen = scenarios(bench)
    if not chosen:
        problems.append(Problem(
            "error",
            "Nothing ticked to measure. Given neither a level nor a session bench2 "
            "falls back to level 1, which measures something nobody asked for."))

    if bench.runs < 1:
        problems.append(Problem("error", "A repeat count below one measures nothing"))

    for name in SWEEP_NAMES:
        # draft tokens mean nothing without speculation, and an empty row then
        # is not a hole in the search
        if name == "spec_n" and "mtp" not in items(bench.spec):
            continue
        if not items(getattr(bench, name)):
            problems.append(Problem(
                "error",
                f"{BENCH_BY_NAME[name].label.removesuffix(' to try')}: nothing ticked. "
                f"One empty axis leaves the search with no configurations to run."))

    if (count := config_count(bench)) > bench.sweep_max:
        problems.append(Problem(
            "error",
            f"{count} runs against a cap of {bench.sweep_max}. Each one loads the model "
            f"again, so raise the cap deliberately or drop a value."))

    if (skipped := dropped_configs(bench)) > 0:
        problems.append(Problem(
            "warn",
            f"{skipped} of {skipped + config_count(bench)} combinations skipped: ubatch "
            f"above its batch, which llama-server refuses once per run"))

    if not configurations(bench) and all(items(getattr(bench, name)) for name in SWEEP_NAMES):
        bad = _all_configurations(bench)[0]
        problems.append(Problem(
            "error",
            f"Nothing left to run: ubatch {bad.ubatch} above batch {bad.batch}, and "
            f"llama-server refuses the pair"))

    ctx = server_context(bench)
    if facts is not None and facts.n_ctx_train and ctx > facts.n_ctx_train:
        too_big = [item.name for item in chosen if item.ctx > facts.n_ctx_train]
        problems.append(Problem(
            "error",
            f"{', '.join(too_big)} wants {ctx} tokens of context and this model was "
            f"trained for {facts.n_ctx_train}"))

    if bench.context_source == "file" and not bench.context_file:
        problems.append(Problem("error", "The file source needs a file to read"))
    if bench.context_source == "repo-snapshot":
        large = [item.name for item in chosen
                 if item.turns == 1 and item.prompt_tokens > REPO_SNAPSHOT_MAX_TOKENS]
        if large:
            problems.append(Problem(
                "warn",
                f"The repository snapshot runs out at about {REPO_SNAPSHOT_MAX_TOKENS} "
                f"tokens, so {', '.join(large)} would measure a shorter prompt than the "
                f"level names."))

    if spec.spec_type not in SPEC_MODES:
        problems.append(Problem(
            "note",
            f"The Server page's {spec.spec_type} has no flag in bench2, which knows only "
            f"{' and '.join(SPEC_MODES)}; these runs use what is ticked above."))

    if spec.fit != "off":
        problems.append(Problem(
            "note",
            "bench2 pins -fit off so that every run is given the same memory to work "
            "in. The Server page's auto fit is left out rather than measured around."))

    if spec.api_key:
        problems.append(Problem(
            "note",
            "The API key is left out of these commands. bench2 sends no Authorization "
            "header, and the server it starts only listens on this machine, so a key "
            "would lock it out of its own server."))

    if overlap := sorted(existing.intersection(run_names(spec, bench))):
        # the name carries the settings, so a clash means this exact thing was measured
        shown = ", ".join(overlap[:3]) + (f" and {len(overlap) - 3} more" if len(overlap) > 3 else "")
        problems.append(Problem(
            "warn",
            f"Measured before: {shown}. Running it again writes over those results, "
            f"which is what re-checking a build is for, and not what a new combination "
            f"needs — that one gets a folder of its own."))

    return problems
