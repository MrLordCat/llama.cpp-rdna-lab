"""Autotune invocation, derived from the same RunSpec.

`scripts/agent_workload_bench.py` is only ever run as a sweep. A sweep of one
value per axis is a plain measurement of one configuration, so the second mode
would buy nothing but a second command to keep in step with this one.

The script owns a few llama-server knobs natively and the sweep overwrites
five more per configuration; the rest of the generated server command is
forwarded verbatim via --server-extra, so server launches and measured runs
cannot drift apart.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, fields, replace
from pathlib import Path

from gui2.core.gguf import ModelFacts
from gui2.core.memory import estimate
from gui2.core.params import KV_TYPES, SPEC_TYPES, Param, aliases_of
from gui2.core.runspec import ALL_LAYERS, Problem, RunSpec, to_argv

TASK_SETS = ("quick", "full", "v2", "v2-mini", "v2-review")
REAL_CONTEXT_MODES = ("off", "repo-snapshot")
BACKGROUND_POLICIES = ("fail", "warn", "ignore")
TRACE_PRESETS = ("none", "kernel-full", "vulkan-routes", "vulkan-perf", "vulkan-q3-stats")

#: The prompts in each set, copied from the script's own task tables. They are
#: here so a run can be counted before it is started; the script is still the
#: authority, so an unrecognised set is reported as unknown rather than as zero.
#: `v2-mini` really is one prompt -- its --tasks help text says two, and the
#: code that filters `TASKS_V2` disagrees with it.
TASK_IDS: dict[str, tuple[str, ...]] = {
    "quick": ("triage_diff", "review_bug"),
    "full": ("triage_diff", "review_bug", "implementation_plan", "config_compare"),
    "v2": ("v2_code_review", "v2_write_function", "v2_debug_trace",
           "v2_refactor_plan", "v2_perf_analysis"),
    "v2-mini": ("v2_write_function",),
    "v2-review": ("v2_code_review",),
}

#: Above this the script stops with "above the active 130k benchmark policy"
#: unless the run says it meant it. The flag that says so is spelled
#: --allow-ctx-above-16k, a name left over from when the limit was 16384.
POLICY_MAX_CTX = 131072

#: The sweep axes, and the RunSpec field each one replaces. Every axis is a
#: setting the Server page also has; a sweep of one value is that setting.
SWEEP_AXES: tuple[tuple[str, str, str], ...] = (
    ("sweep_ctx", "--autotune-ctx-values", "ctx_size"),
    ("sweep_batch", "--autotune-batch-values", "batch_size"),
    ("sweep_ubatch", "--autotune-ubatch-values", "ubatch_size"),
    ("sweep_kv", "--autotune-kv-values", "cache_type_k"),
    ("sweep_spec", "--autotune-spec-values", "spec_type"),
)

#: sweep modes that need the n-gram window, and the ones that need a draft budget
NGRAM_MODES = frozenset({"ngram-mod", "ngram-mtp"})
DRAFT_MODES = frozenset({"mtp", "ngram-mtp"})

#: What may be written on the two word-shaped axes. The sweep knows one mode
#: the Server page does not -- `ngram-mtp` is assembled by the autotune loop
#: out of two flags and has no single setting behind it.
SWEEP_KV_TYPES = KV_TYPES
SWEEP_SPEC_MODES = SPEC_TYPES + ("ngram-mtp",)

# flags the bench script passes to llama-server itself
BENCH_OWNED: frozenset[str] = frozenset().union(*(
    aliases_of(flag) for flag in (
        "-m", "--host", "--port", "-c", "--batch-size", "--ubatch-size",
        "-ngl", "--parallel", "--cache-type-k", "--cache-type-v", "--flash-attn",
    )
)) | {"--no-warmup"}

#: Flags dropped rather than forwarded.
#:
#: --api-key: the script sends no Authorization header, so a server started
#: with one answers 401 to every request it makes; and the server it starts is
#: loopback-only anyway (--host is one of the flags it owns).
#:
#: --spec-type and its companions: the sweep appends its own per configuration.
#: Two would not break llama-server -- the last wins -- but the history row
#: takes its spec_mode from `infer_spec_mode`, which reads the *first*, so
#: every measurement would be filed under the mode it did not run.
BENCH_DROPPED: frozenset[str] = frozenset().union(*(
    aliases_of(flag) for flag in (
        "--api-key", "--spec-type", "--spec-draft-n-max",
        "--spec-ngram-mod-n-min", "--spec-ngram-mod-n-match", "--spec-ngram-mod-n-max",
    )
))


@dataclass(frozen=True, slots=True)
class BenchSpec:
    label: str = ""
    tasks: str = "quick"
    task_ids: str = ""
    runs: int = 1
    max_tokens: int = 16
    real_context_mode: str = "repo-snapshot"
    real_context_chars: int = 24576
    no_reuse: bool = True
    v2_prime_pass: bool = False
    disable_thinking: bool = False
    request_timeout: float = 180.0
    startup_timeout: float = 900.0
    task_hard_timeout: float = 45.0
    background_server_policy: str = "fail"
    write_diagnostics: bool = True
    trace_preset: str = "none"
    #: sweep axes, as the script wants them: comma-separated lists. The
    #: defaults are replaced by the run's own settings on arrival, so a page
    #: opened from the Server page measures exactly what it describes.
    sweep_ctx: str = "131072"
    sweep_batch: str = "512"
    sweep_ubatch: str = "128"
    sweep_kv: str = "f16"
    sweep_spec: str = "none"
    sweep_max: int = 48
    smart_prune: bool = True
    resume: bool = True

    def with_values(self, values: dict) -> "BenchSpec":
        """A copy with whatever the form named, ignoring everything else."""
        known = {field.name for field in fields(self)}
        updates = {key: _coerce(value, getattr(self, key))
                   for key, value in values.items() if key in known}
        return replace(self, **updates)

    def seeded_from(self, spec: RunSpec) -> "BenchSpec":
        """The sweep set to the one configuration the run already describes.

        Arriving from the Server page, every axis holds the value that page
        chose: the sweep is then a measurement of it, and becomes a search
        only when a second value is typed anywhere.
        """
        return replace(self, **{axis: str(getattr(spec, field))
                                for axis, _flag, field in SWEEP_AXES})


BENCH_DEFAULTS = BenchSpec()


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


B_SWEEP = "What to try"
B_WHAT = "What each one is measured with"
B_PROMPT = "How big the prompts are"
B_FAIR = "What the numbers are allowed to include"
B_LIMITS = "When to give up"
B_OUTPUT = "What is written down"

#: Presentation for the bench script's flags, in the same shape as the
#: llama-server schema. Unlike that one it does not generate the command:
#: `to_bench_argv` spells it out, because several of these flags are pairs
#: (--reuse/--no-reuse) and one of them rewrites another (--autotune-min-ctx).
BENCH_SCHEMA: tuple[Param, ...] = (
    Param("sweep_ctx", "Contexts to try", "text", B_SWEEP,
          help="one value measures it; several search it"),
    Param("sweep_batch", "Batch sizes to try", "text", B_SWEEP,
          help="how many tokens the server reads at once"),
    Param("sweep_ubatch", "Ubatch sizes to try", "text", B_SWEEP,
          help="how much of a batch reaches the GPU in one go"),
    # the vocabularies are spelled out from the constants: a hint that lists
    # fewer types than the validator accepts is how f8_e4m3 goes unnoticed
    Param("sweep_kv", "KV cache types to try", "text", B_SWEEP,
          help=f"{', '.join(SWEEP_KV_TYPES)} — smaller buys context and may cost quality"),
    Param("sweep_spec", "Speculation modes to try", "text", B_SWEEP,
          help=f"{', '.join(SWEEP_SPEC_MODES)} — the sweep sets this itself, so the "
               f"Server page's choice is replaced by whatever is listed here"),
    Param("sweep_max", "Refuse to start above", "int", B_SWEEP, minimum=1, maximum=512,
          help="configurations; every extra value multiplies the run rather than adding "
               "to it"),
    Param("smart_prune", "Abandon a direction that keeps getting slower", "bool", B_SWEEP,
          help="stops walking up batch and ubatch once the speed has dropped twice "
               "running; also what lets a sweep exceed the cap above"),
    Param("resume", "Continue an interrupted sweep", "bool", B_SWEEP,
          help="picks up from the checkpoint file if the same sweep was started before"),
    Param("tasks", "Prompt set", "choice", B_WHAT, choices=TASK_SETS,
          help="which prompts the model is asked to answer"),
    Param("task_ids", "Only these prompts", "text", B_WHAT,
          help="leave empty for the whole set; a name that is not in it stops the run"),
    Param("runs", "Repeats", "int", B_WHAT, minimum=1, maximum=50,
          help="how many times each prompt is asked — more repeats, steadier numbers"),
    Param("max_tokens", "Answer length", "int", B_WHAT, minimum=1, maximum=8192,
          help="tokens to generate per answer; this is what decode speed is measured over"),
    Param("real_context_mode", "Incoming context", "choice", B_PROMPT,
          choices=REAL_CONTEXT_MODES,
          help="whether each prompt carries a slab of real repository text in front of it"),
    Param("real_context_chars", "How much of it", "int", B_PROMPT, minimum=0, maximum=4_000_000,
          help="characters of that text; 0 lets the script fill the context it was given"),
    Param("no_reuse", "Start every prompt cold", "bool", B_FAIR,
          help="throws away the prompt cache between prompts, so prompt speed is measured "
               "rather than remembered"),
    Param("disable_thinking", "Turn thinking off", "bool", B_FAIR,
          help="asks the chat template not to emit reasoning, which otherwise counts "
               "towards the answer"),
    Param("v2_prime_pass", "One unmeasured warm-up pass", "bool", B_FAIR,
          help="only for the v2 sets with n-gram speculation and a single repeat: fills "
               "the speculative state first so the measured pass is not the cold one"),
    Param("background_server_policy", "If a server is already running", "choice", B_FAIR,
          choices=BACKGROUND_POLICIES,
          help="another llama-server shares the same GPUs and skews everything measured here"),
    Param("request_timeout", "Give up on one answer after", "float", B_LIMITS,
          minimum=1, maximum=3600, help="seconds to wait for a reply before calling it lost"),
    Param("task_hard_timeout", "Abandon the whole run after", "float", B_LIMITS,
          minimum=0, maximum=3600,
          help="seconds after which a stuck prompt also stops the server; 0 turns it off"),
    Param("startup_timeout", "Wait for the server to load for", "float", B_LIMITS,
          minimum=10, maximum=7200,
          help="seconds; a large model over RPC can take minutes before it answers"),
    Param("label", "Name this run", "text", B_OUTPUT,
          help="how it will appear in the history table; empty gets a timestamp"),
    Param("write_diagnostics", "Keep the per-run breakdown", "bool", B_OUTPUT,
          help="parses the server log into a json/markdown summary next to the results"),
    Param("trace_preset", "Backend tracing", "choice", B_OUTPUT, choices=TRACE_PRESETS,
          help="records what the backend did, at the cost of doing it slower"),
)

BENCH_BY_NAME: dict[str, Param] = {param.name: param for param in BENCH_SCHEMA}


def items(text: str) -> list[str]:
    """One sweep axis, split however the person happened to type it."""
    return [chunk for chunk in re.split(r"[,;\s]+", (text or "").strip()) if chunk]


def sweep_values(bench: BenchSpec) -> dict[str, list[str]]:
    """Each axis of the sweep, in the order the script multiplies them out."""
    return {name: items(getattr(bench, name)) for name, _flag, _field in SWEEP_AXES}


def config_count(bench: BenchSpec) -> int:
    """How many server configurations a sweep would work through.

    The script builds them with `itertools.product`, so one more value on any
    axis multiplies the whole run rather than adding to it -- which is the
    thing that turns an afternoon into a week.
    """
    total = 1
    for values in sweep_values(bench).values():
        total *= len(values)
    return total


def _sweep_contexts(bench: BenchSpec) -> list[int]:
    contexts = []
    for value in items(bench.sweep_ctx):
        try:
            contexts.append(int(value))
        except ValueError:
            continue
    return contexts


# -- what the sweep will ask of the cards ----------------------------------


@dataclass(frozen=True, slots=True)
class Weighed:
    """One configuration's VRAM bill, and what distinguishes it from the rest."""

    ctx: int
    kv: str
    ubatch: int
    mib: float


@dataclass(frozen=True, slots=True)
class Fit:
    """How much of a sweep the devices have room for.

    A configuration too big for the cards is not skipped: the server fails to
    load, the script prints CONFIG FAILED and moves on. So each one costs the
    whole startup timeout and produces no measurement, which is worth knowing
    before the sweep is started rather than after.
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
        """The most demanding configuration that still loads."""
        within = [item for item in self.weighed if item.mib <= self.budget_mib]
        return within[-1] if within else None


def fit(spec: RunSpec, facts: ModelFacts | None, bench: BenchSpec, budget_mib: float,
        devices: int = 1, mmproj_bytes: int = 0) -> Fit | None:
    """The sweep priced against the memory there is, or None if it cannot be.

    Only three axes move the bill: the context and the KV type set the cache,
    the ubatch sets the compute buffers. Batch size and speculative mode do
    not, so each distinct bill is computed once and counted for all of them.
    """
    if facts is None or not facts.known or budget_mib <= 0:
        return None
    values = sweep_values(bench)
    contexts = sorted({int(value) for value in values["sweep_ctx"] if value.isdigit()})
    ubatches = sorted({int(value) for value in values["sweep_ubatch"] if value.isdigit()})
    kv_types = [value for value in values["sweep_kv"] if value in SWEEP_KV_TYPES]
    if not (contexts and ubatches and kv_types):
        return None

    weighed: list[Weighed] = []
    for ctx in contexts:
        for kv in kv_types:
            for ubatch in ubatches:
                probe = replace(spec, ctx_size=ctx, ubatch_size=ubatch,
                                cache_type_k=kv, cache_type_v=kv)
                report = estimate(probe, facts, devices=devices, mmproj_bytes=mmproj_bytes)
                if not report.complete:
                    return None
                weighed.append(Weighed(ctx, kv, ubatch, report.total_mib))
    weighed.sort(key=lambda item: item.mib)
    each = max(1, len(values["sweep_batch"])) * max(1, len(values["sweep_spec"]))
    return Fit(budget_mib=budget_mib, weighed=tuple(weighed), each=each)


def server_extra_tokens(spec: RunSpec) -> list[str]:
    """Generated server flags minus the ones the bench script sets or refuses."""
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
    script: str | Path,
    server_bin: str | Path,
    python: str = "python",
) -> list[str]:
    """Full `python scripts/agent_workload_bench.py ...` command line.

    The context, batch, ubatch, KV type and speculative mode are absent on
    purpose: the sweep overwrites all five for every configuration it runs, so
    naming them here would describe a run that does not happen.
    """
    argv = [
        python, str(script),
        "--server-bin", str(server_bin),
        "--model", spec.model,
        "--gpu-layers", ALL_LAYERS if spec.gpu_layers_all else str(spec.gpu_layers),
        "--parallel", str(spec.parallel),
    ]
    argv += _flag(spec.flash_attn != "off", "--flash-attn", "--no-flash-attn")

    if bench.label:
        argv += ["--label", bench.label]
    argv += ["--tasks", bench.tasks]
    if chosen := items(bench.task_ids):
        # the script splits this on commas only, so a list typed with spaces
        # would reach it as one unknown id and stop the run
        argv += ["--task-ids", ",".join(chosen)]
    argv += [
        "--runs", str(bench.runs),
        "--max-tokens", str(bench.max_tokens),
        "--real-context-mode", bench.real_context_mode,
        "--real-context-chars", str(bench.real_context_chars),
    ]
    argv += _flag(bench.no_reuse, "--no-reuse", "--reuse")
    argv += _flag(bench.disable_thinking, "--disable-thinking", "--no-disable-thinking")
    argv += _flag(bench.v2_prime_pass, "--v2-prime-pass", "--no-v2-prime-pass")
    argv += _flag(bench.write_diagnostics, "--write-diagnostics", "--no-write-diagnostics")
    argv += [
        "--request-timeout", f"{bench.request_timeout:g}",
        "--startup-timeout", f"{bench.startup_timeout:g}",
        "--task-hard-timeout", f"{bench.task_hard_timeout:g}",
        "--background-server-policy", bench.background_server_policy,
    ]
    if bench.trace_preset != "none":
        argv += ["--trace-preset", bench.trace_preset]

    argv.append("--autotune")
    for name, flag, _field in SWEEP_AXES:
        argv += [flag, ",".join(items(getattr(bench, name)))]

    contexts = _sweep_contexts(bench)
    if contexts:
        # --autotune-min-ctx discards every swept context below itself and
        # defaults to 131072, so a sweep of smaller ones silently empties out.
        # Pinning it to the smallest value asked for is the only way to sweep
        # what was typed.
        argv += ["--autotune-min-ctx", str(min(contexts))]
    argv += ["--autotune-max-configs", str(bench.sweep_max)]
    # both default to on in the script, so an unticked box has to say so
    argv += _flag(bench.smart_prune, "--autotune-smart-prune", "--no-autotune-smart-prune")
    argv += _flag(bench.resume, "--autotune-resume", "--no-autotune-resume")

    # the sweep names a speculative mode but not its numbers, and takes those
    # from flags of its own; the Server page is where they were chosen
    modes = set(items(bench.sweep_spec))
    if modes & NGRAM_MODES:
        argv += ["--autotune-ngram-min", str(spec.ngram_n_min),
                 "--autotune-ngram-match", str(spec.ngram_n_match),
                 "--autotune-ngram-max", str(spec.ngram_n_max)]
    if modes & DRAFT_MODES:
        argv += ["--autotune-mtp-draft-n-max", str(spec.spec_draft_n_max)]

    # The policy gate is an exit code, not a warning: without this the script
    # refuses every context above 130 000 before it starts anything.
    if contexts and max(contexts) > POLICY_MAX_CTX:
        argv.append("--allow-ctx-above-16k")

    extra = server_extra_tokens(spec)
    if extra:
        # --flag=value form: a separate value starting with '-' would be read as a new option
        argv.append("--server-extra=" + " ".join(shlex.quote(token) for token in extra))
    return argv


# -- what the run will do, counted before it does it -----------------------


V2_SETS = frozenset({"v2", "v2-mini", "v2-review"})

#: the script raises --max-tokens for the v2 sets only when it finds this exact
#: value, so any other number is taken as a deliberate choice and left alone
V2_MAX_TOKENS_TRIGGER = 160


def selected_tasks(bench: BenchSpec) -> tuple[list[str], list[str]]:
    """The prompts that would run, and the names that match nothing.

    An unknown name is not ignored: the script prints what it does not
    recognise and exits before starting a server, which is a slow way to find
    out about a typo.
    """
    available = TASK_IDS.get(bench.tasks, ())
    asked = items(bench.task_ids)
    if not asked:
        return list(available), []
    wanted = set(asked)
    return ([name for name in available if name in wanted],
            [name for name in asked if name not in available])


@dataclass(frozen=True, slots=True)
class Plan:
    """The size of a run, in the units that decide how long it takes."""

    tasks: int = 0
    runs: int = 1
    configs: int = 1
    #: configurations that also get one unmeasured pass to fill n-gram state
    primed: int = 0
    per_request_s: float = 0.0
    startup_s: float = 0.0

    @property
    def requests(self) -> int:
        return self.tasks * (self.runs * self.configs + self.primed)

    @property
    def worst_case_s(self) -> float:
        """The longest the run may take before its own timeouts end it.

        Not an estimate of how long it will take -- a real request finishes in
        a fraction of its ceiling. It is the number that answers "can I leave
        this running overnight", which an estimate cannot.
        """
        return self.requests * self.per_request_s + self.configs * self.startup_s


def _primed_configs(bench: BenchSpec) -> int:
    """Configurations that get a priming pass, which is not all of them.

    `run_suite` decides per configuration, from the speculative mode the sweep
    gave that one -- so a sweep that tries n-gram alongside anything else
    primes only the n-gram half of it.
    """
    if not bench.v2_prime_pass or bench.tasks not in V2_SETS or bench.runs != 1:
        return 0
    modes = sweep_values(bench)["sweep_spec"]
    if not modes:
        return 0
    return config_count(bench) // len(modes) * modes.count("ngram-mod")


def plan(spec: RunSpec, bench: BenchSpec) -> Plan:
    """How many requests this configuration works out to, and at what ceiling."""
    chosen, _unknown = selected_tasks(bench)
    limits = [value for value in (bench.task_hard_timeout, bench.request_timeout) if value > 0]
    return Plan(
        tasks=len(chosen),
        runs=max(1, bench.runs),
        configs=config_count(bench),
        primed=_primed_configs(bench),
        per_request_s=min(limits) if limits else 0.0,
        startup_s=max(0.0, bench.startup_timeout),
    )


def _vocabulary_problems(bench: BenchSpec) -> list[Problem]:
    """Values on a sweep line that nothing downstream would accept.

    The numeric axes reach `parse_int_csv`, which drops what it cannot read --
    so a typo shrinks the sweep silently. The word axes reach llama-server,
    which refuses to start, once per configuration, for the whole timeout.
    """
    problems: list[Problem] = []
    for axis, label in (("sweep_ctx", "Contexts"), ("sweep_batch", "Batch sizes"),
                        ("sweep_ubatch", "Ubatch sizes")):
        bad = [value for value in items(getattr(bench, axis)) if not value.isdigit()]
        if bad:
            problems.append(Problem(
                "error",
                f"{label} to try: {', '.join(bad)} — every value on that line has to be a "
                f"plain number of tokens, and one that is not is quietly dropped"))
    for axis, label, allowed in (("sweep_kv", "KV cache types", SWEEP_KV_TYPES),
                                 ("sweep_spec", "Speculation modes", SWEEP_SPEC_MODES)):
        bad = [value for value in items(getattr(bench, axis)) if value not in allowed]
        if bad:
            problems.append(Problem(
                "error",
                f"{label} to try: {', '.join(bad)} — the ones that work here are "
                f"{', '.join(allowed)}"))
    return problems


def validate_bench(spec: RunSpec, bench: BenchSpec) -> list[Problem]:
    """Everything that would stop this run, said before it is started.

    Each of these is an exit code in the script -- after it has been launched,
    and in some cases after it has worked through part of the sweep.
    """
    problems: list[Problem] = []
    chosen, unknown = selected_tasks(bench)

    if bench.tasks not in TASK_IDS:
        problems.append(Problem("warn", f"Unknown task set {bench.tasks!r}: "
                                        "the run cannot be counted in advance"))
    elif unknown:
        problems.append(Problem(
            "error",
            f"{', '.join(unknown)} — not in the {bench.tasks} set. It has "
            f"{', '.join(TASK_IDS[bench.tasks])}."))
    elif not chosen:
        problems.append(Problem("error", "No prompts selected, so there is nothing to measure"))

    if bench.runs < 1:
        problems.append(Problem("error", "A run count below one measures nothing"))

    if bench.tasks in V2_SETS and bench.max_tokens < V2_MAX_TOKENS_TRIGGER:
        problems.append(Problem(
            "note",
            f"The v2 prompts ask for whole functions and reviews; {bench.max_tokens} tokens "
            f"stops the answer almost immediately. The script raises this by itself only "
            f"when it is left at exactly {V2_MAX_TOKENS_TRIGGER}."))

    if bench.v2_prime_pass:
        blocked = []
        if bench.tasks not in V2_SETS:
            blocked.append(f"the {bench.tasks} prompts are not one of the v2 sets")
        if bench.runs != 1:
            blocked.append(f"each prompt is measured {bench.runs} times, not once")
        if "ngram-mod" not in items(bench.sweep_spec):
            blocked.append("no configuration tries ngram-mod")
        if blocked:
            problems.append(Problem(
                "note", "No priming pass will happen: " + ", and ".join(blocked) + "."))
        elif (primed := _primed_configs(bench)) < config_count(bench):
            problems.append(Problem(
                "note",
                f"Only the {primed} ngram-mod configurations are primed. The others measure "
                "a cold start, which is what they are for."))

    if bench.trace_preset != "none":
        problems.append(Problem(
            "warn",
            f"Tracing ({bench.trace_preset}) instruments the backend and slows it down. "
            "Numbers from a traced run are for finding where time goes, not for comparing "
            "against untraced ones."))

    if spec.api_key:
        problems.append(Problem(
            "note",
            "The API key is left out of this command. The benchmark script sends no "
            "Authorization header, and the server it starts only listens on this "
            "machine, so a key would lock it out of its own server."))


    problems += _vocabulary_problems(bench)

    empty = [name for name, values in sweep_values(bench).items() if not values]
    if empty:
        problems.append(Problem(
            "error",
            f"{', '.join(name.removeprefix('sweep_') for name in empty)}: an empty line means "
            "nothing to try, and one empty axis leaves the whole sweep with no configurations"))
    elif (count := config_count(bench)) > bench.sweep_max:
        # the script's own check: an error, unless smart pruning is on, in which
        # case it prints a warning and works through the list anyway
        problems.append(Problem("warn", f"{count} configurations against a cap of "
                                        f"{bench.sweep_max}. It will start anyway, because "
                                        "abandoning a losing direction may bring it under the "
                                        "cap — but nothing promises it will.")
                        if bench.smart_prune else
                        Problem("error", f"{count} configurations against a cap of "
                                         f"{bench.sweep_max}. Raise the cap, drop a value, or "
                                         "let it abandon directions that keep getting slower; "
                                         "as set it refuses to start."))

    contexts = _sweep_contexts(bench)
    if contexts and max(contexts) > POLICY_MAX_CTX:
        problems.append(Problem("note", "Contexts above 130 000 are outside the lab's standard "
                                        "lane; the command says so explicitly."))

    return problems
