"""The benchmark command, and the size of the run it describes.

Every assertion here is about something the script would otherwise only tell
you by exiting: a policy gate, an empty sweep, an unknown prompt name, a key
that locks the script out of its own server.
"""

from __future__ import annotations

from dataclasses import replace

from gui2.core.bench import (
    BENCH_DEFAULTS,
    POLICY_MAX_CTX,
    TASK_IDS,
    BenchSpec,
    config_count,
    plan,
    selected_tasks,
    server_extra_tokens,
    to_bench_argv,
    validate_bench,
)
from gui2.core.runspec import DEFAULTS, RunSpec


def command(spec: RunSpec = DEFAULTS, bench: BenchSpec = BENCH_DEFAULTS) -> list[str]:
    return to_bench_argv(spec, bench, "scripts/agent_workload_bench.py", "llama-server")


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def levels(problems) -> set[str]:
    return {problem.level for problem in problems}


def test_the_server_flags_the_script_sets_itself_are_not_sent_twice():
    argv = command(replace(DEFAULTS, ctx_size=65536, batch_size=1024))
    extra = next(token for token in argv if token.startswith("--server-extra="))
    assert value_after(argv, "--ctx-size") == "65536"
    assert value_after(argv, "--batch-size") == "1024"
    # the script owns these; repeating them inside --server-extra would give
    # llama-server two of each and the second would win
    assert "--ctx-size" not in extra and "-c " not in extra
    assert "--batch-size" not in extra


def test_an_api_key_is_left_out_because_the_script_cannot_send_one():
    spec = replace(DEFAULTS, api_key="hunter2")
    assert "--api-key" not in server_extra_tokens(spec)
    assert "hunter2" not in " ".join(command(spec))
    assert any("API key is left out" in problem.message
               for problem in validate_bench(spec, BENCH_DEFAULTS))


def test_a_context_above_the_lane_says_so_instead_of_being_refused():
    """Without the flag the script exits 4 before starting anything."""
    ordinary = command(replace(DEFAULTS, ctx_size=POLICY_MAX_CTX))
    assert "--allow-ctx-above-16k" not in ordinary

    big = command(replace(DEFAULTS, ctx_size=POLICY_MAX_CTX + 4096))
    assert "--allow-ctx-above-16k" in big
    assert levels(validate_bench(replace(DEFAULTS, ctx_size=POLICY_MAX_CTX + 4096),
                                 BENCH_DEFAULTS)) <= {"note"}


def test_a_sweep_of_small_contexts_is_not_silently_emptied():
    """--autotune-min-ctx defaults to 131072 and drops everything below it."""
    bench = BENCH_DEFAULTS.with_values({"autotune": True, "sweep_ctx": "16384, 32768"})
    argv = command(bench=bench)
    assert value_after(argv, "--autotune-ctx-values") == "16384,32768"
    assert value_after(argv, "--autotune-min-ctx") == "16384"


def test_a_sweep_multiplies_out_and_is_counted_before_it_starts():
    bench = BENCH_DEFAULTS.with_values({
        "autotune": True, "sweep_batch": "256,512", "sweep_ubatch": "64,128,256",
        "sweep_kv": "q4_0,q8_0", "runs": 2, "tasks": "quick",
    })
    assert config_count(bench) == 2 * 3 * 2  # ctx and spec are one value each
    counted = plan(DEFAULTS, bench)
    assert counted.configs == 12
    assert counted.requests == 12 * 2 * 2  # configs x prompts x repeats
    assert not validate_bench(DEFAULTS, bench)


def test_every_boolean_the_script_defaults_to_on_is_said_either_way():
    """A flag the script turns on by itself does nothing unless its off form exists."""
    on = command(bench=BENCH_DEFAULTS.with_values({"autotune": True}))
    off = command(bench=BENCH_DEFAULTS.with_values({"autotune": True, "resume": False,
                                                    "no_reuse": False,
                                                    "write_diagnostics": False}))
    assert "--autotune-resume" in on and "--no-autotune-resume" not in on
    assert "--no-autotune-resume" in off
    assert "--reuse" in off and "--no-write-diagnostics" in off


def test_a_sweep_bigger_than_its_own_cap_is_refused_here_rather_than_there():
    bench = BENCH_DEFAULTS.with_values({"autotune": True, "sweep_max": 4})
    assert config_count(bench) == 12
    problem = next(item for item in validate_bench(DEFAULTS, bench) if item.level == "error")
    assert "12 configurations against a cap of 4" in problem.message


def test_an_empty_sweep_axis_is_an_error_not_a_missing_dimension():
    bench = BENCH_DEFAULTS.with_values({"autotune": True, "sweep_kv": "  "})
    assert any(problem.level == "error" and "empty sweep axis" in problem.message
               for problem in validate_bench(DEFAULTS, bench))


def test_a_prompt_name_that_is_not_in_the_set_is_caught_before_the_run():
    bench = BENCH_DEFAULTS.with_values({"tasks": "quick", "task_ids": "review_bug, made_up"})
    chosen, unknown = selected_tasks(bench)
    assert chosen == ["review_bug"] and unknown == ["made_up"]
    problem = next(item for item in validate_bench(DEFAULTS, bench) if item.level == "error")
    assert "made_up" in problem.message
    assert "triage_diff" in problem.message  # says what is available


def test_prompt_names_typed_with_spaces_still_reach_the_script_as_a_list():
    """It splits --task-ids on commas only, so 'a b' would be one unknown id."""
    bench = BENCH_DEFAULTS.with_values({"tasks": "quick", "task_ids": "triage_diff review_bug"})
    assert value_after(command(bench=bench), "--task-ids") == "triage_diff,review_bug"


def test_v2_mini_is_one_prompt_whatever_its_help_text_says():
    assert TASK_IDS["v2-mini"] == ("v2_write_function",)
    counted = plan(DEFAULTS, BENCH_DEFAULTS.with_values({"tasks": "v2-mini"}))
    assert counted.tasks == 1


def test_a_v2_run_with_the_default_answer_length_is_flagged_as_measuring_nothing():
    bench = BENCH_DEFAULTS.with_values({"tasks": "v2"})
    assert any("stops the answer almost immediately" in problem.message
               for problem in validate_bench(DEFAULTS, bench))


def test_the_priming_pass_is_only_counted_when_it_would_actually_happen():
    bench = BENCH_DEFAULTS.with_values({"tasks": "v2-review", "v2_prime_pass": True})
    without = plan(DEFAULTS, bench)
    assert not without.prime and without.requests == 1

    with_ngram = plan(replace(DEFAULTS, spec_type="ngram-mod"), bench)
    assert with_ngram.prime and with_ngram.requests == 2


def test_the_worst_case_is_bounded_by_the_run_s_own_timeouts():
    bench = BENCH_DEFAULTS.with_values({"tasks": "quick", "runs": 3,
                                        "task_hard_timeout": 30, "startup_timeout": 120})
    counted = plan(DEFAULTS, bench)
    # the hard timeout is shorter than the request timeout, so it is the ceiling
    assert counted.per_request_s == 30
    assert counted.requests == 6
    assert counted.worst_case_s == 6 * 30 + 120


def test_turning_the_hard_timeout_off_falls_back_to_the_request_timeout():
    counted = plan(DEFAULTS, BENCH_DEFAULTS.with_values({"task_hard_timeout": 0}))
    assert counted.per_request_s == BENCH_DEFAULTS.request_timeout
