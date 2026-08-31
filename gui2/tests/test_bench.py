"""The bench2 commands, and the size of the run they describe.

Every assertion here is about something bench2 would otherwise only tell you by
exiting -- or, worse, by measuring the wrong thing quietly: cards it chose from
a hardware profile, a prompt cache the run never asked for, a key that locks it
out of its own server.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gui2.core.bench import (
    BENCH_DEFAULTS,
    LEVELS,
    SESSIONS,
    BenchSpec,
    Configuration,
    bench_commands,
    config_count,
    configurations,
    fit,
    level_for_context,
    plan,
    run_names,
    server_context,
    server_extra_tokens,
    to_bench_argv,
    validate_bench,
)
from gui2.core.gguf import read_facts
from gui2.core.runspec import DEFAULTS, RunSpec
from gui2.tests.fixtures import QWEN35_27B, write_gguf

ONE = Configuration(8192, 1024, "q8_0", "none")


def command(spec: RunSpec = DEFAULTS, bench: BenchSpec = BENCH_DEFAULTS,
            config: Configuration = ONE, **kwargs) -> list[str]:
    return to_bench_argv(spec, bench, config, "scripts/bench2.py", "llama-server", **kwargs)


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def errors(problems) -> list[str]:
    return [problem.message for problem in problems if problem.level == "error"]


def test_the_run_is_a_subcommand_and_the_context_is_not_named():
    argv = command()
    assert argv[2] == "run"
    # bench2 sizes the server from the levels it is given; naming a context
    # here would describe a server it never starts
    for flag in ("-c", "--ctx-size", "--context"):
        assert flag not in argv


def test_what_bench2_owns_is_not_sent_to_the_server_a_second_time():
    spec = replace(DEFAULTS, ctx_size=65536, batch_size=1024, parallel=4)
    argv = command(spec)
    extra = next(token for token in argv if token.startswith("--server-extra="))

    assert value_after(argv, "--parallel") == "4"
    for flag in ("--ctx-size", "-c ", "--batch-size", "--ubatch-size", "--host", "--port",
                 "--cache-type-k", "--flash-attn", "-dev", "-sm", "-ts", "-fit"):
        assert flag not in extra


def test_the_prompt_cache_is_left_at_what_bench2_pins_it_to():
    """Forwarding these would measure a cache hit and call it prefill."""
    spec = replace(DEFAULTS, cache_ram=8192, ctx_checkpoints=8,
                   checkpoint_every_n_tokens=4096)
    assert "--cache-ram" not in server_extra_tokens(spec)
    assert "--ctx-checkpoints" not in server_extra_tokens(spec)
    assert "--checkpoint-every-n-tokens" not in " ".join(command(spec))


def test_the_cards_are_named_rather_than_left_to_the_hardware_profile():
    """Unsaid, bench2 falls back to a profile that names cards of its own."""
    spec = replace(DEFAULTS, devices="Vulkan0", split_mode="layer", tensor_split="3,2")
    argv = command(spec)
    assert value_after(argv, "--dev") == "Vulkan0"
    assert value_after(argv, "--sm") == "layer"
    assert value_after(argv, "--ts") == "3,2"

    # and "every device" has to be said out loud for the same reason
    everywhere = command(replace(DEFAULTS, devices="", split_mode="", tensor_split=""))
    assert value_after(everywhere, "--dev") == ""
    assert value_after(everywhere, "--sm") == ""


def test_bench2_index_records_gpu_order_and_proportions():
    from scripts.bench2 import METRICS_COLUMNS, _placement_fields

    assert "devices" in METRICS_COLUMNS and "tensor_split" in METRICS_COLUMNS
    assert _placement_fields({"dev": "Vulkan1,Vulkan0", "ts": "100,60"}) == {
        "devices": "Vulkan1,Vulkan0", "tensor_split": "100,60"}
    assert _placement_fields({"dev": "", "ts": ""}) == {
        "devices": "auto", "tensor_split": "auto"}


def test_the_backend_is_named_because_bench2_would_otherwise_guess_rocm():
    argv = command(backend="vk")
    assert value_after(argv, "--backend") == "vk"


def test_rpc_workers_are_bench2s_own_flag_and_come_before_the_devices():
    """-dev validates names as it parses, and an RPC device exists only after
    --rpc has registered it; the old route through --server-extra put --rpc
    after -dev and llama-server rejected RPC0."""
    spec = replace(DEFAULTS, rpc_endpoints="192.168.1.60:50052", devices="RPC0,Vulkan0")
    argv = command(spec)
    assert value_after(argv, "--rpc") == "192.168.1.60:50052"
    assert argv.index("--rpc") < argv.index("--dev"), "rpc first, then the device list"
    extra = next(token for token in argv if token.startswith("--server-extra="))
    assert "--rpc" not in extra
    assert value_after(argv, "--dev") == "RPC0,Vulkan0"


def test_arriving_from_the_server_page_measures_exactly_that_configuration():
    spec = replace(DEFAULTS, ctx_size=49152, batch_size=2048, ubatch_size=512,
                   cache_type_k="q8_0", spec_type="mtp", spec_draft_n_max=3)
    bench = BENCH_DEFAULTS.seeded_from(spec)
    assert config_count(bench) == 1
    # the context is not a setting here, it is the level that has room for it
    assert bench.levels == "2" and LEVELS["2"].ctx == 49152

    argv = command(spec, bench, configurations(bench)[0])
    assert value_after(argv, "--level") == "2"
    assert value_after(argv, "--batch-size") == "2048"
    assert value_after(argv, "--ubatch-size") == "512"
    assert value_after(argv, "--kv-k") == "q8_0" and value_after(argv, "--kv-v") == "q8_0"
    assert value_after(argv, "--spec") == "mtp"
    assert value_after(argv, "--spec-n") == "3"


@pytest.mark.parametrize("ctx, level", [
    (4096, "0"), (8192, "0"), (16384, "1"), (32768, "1"), (49152, "2"), (131072, "4"),
])
def test_the_page_opens_on_the_largest_level_the_context_has_room_for(ctx, level):
    assert level_for_context(ctx) == level


def test_a_mode_bench2_cannot_reach_is_replaced_rather_than_forwarded():
    spec = replace(DEFAULTS, spec_type="ngram-mod")
    bench = BENCH_DEFAULTS.seeded_from(spec)
    assert bench.spec == "none"
    extra = next(token for token in command(spec, bench) if token.startswith("--server-extra="))
    # llama-server would obey the last --spec-type, so the run would be filed
    # under one mode and measured under the other
    assert "--spec-type" not in extra
    assert any("has no flag in bench2" in problem.message
               for problem in validate_bench(spec, bench))


def test_an_api_key_is_left_out_because_bench2_cannot_send_one():
    spec = replace(DEFAULTS, api_key="hunter2")
    assert "--api-key" not in server_extra_tokens(spec)
    assert "hunter2" not in " ".join(command(spec))
    assert any("API key is left out" in problem.message
               for problem in validate_bench(spec, BENCH_DEFAULTS))


def test_one_server_holds_every_scenario_and_is_sized_by_the_largest():
    bench = BENCH_DEFAULTS.with_values({"levels": "0,2", "session_levels": "1"})
    argv = command(bench=bench)
    assert value_after(argv, "--level") == "0,2"
    assert value_after(argv, "--session-level") == "1"
    # L2 is the largest of the three, so it decides the memory the run needs
    assert server_context(bench) == LEVELS["2"].ctx


def test_a_session_is_ten_turns_and_is_counted_as_ten():
    single = plan(BENCH_DEFAULTS.with_values({"levels": "1", "session_levels": ""}))
    assert single.requests == 1
    assert single.decoded == LEVELS["1"].decode_tokens

    session = plan(BENCH_DEFAULTS.with_values({"levels": "", "session_levels": "1"}))
    assert session.requests == SESSIONS["1"].turns == 10
    assert session.decoded == 10 * SESSIONS["1"].decode_tokens
    assert session.prefilled == 10 * SESSIONS["1"].prompt_tokens


def test_a_second_value_on_a_row_is_a_second_run_with_its_own_name():
    bench = BENCH_DEFAULTS.with_values({"batch": "2048,8192", "kv": "q8_0,f16"})
    assert config_count(bench) == 4

    runs = bench_commands(DEFAULTS, bench, "scripts/bench2.py", "llama-server", backend="vk")
    assert len(runs) == 4
    names = [name for name, _argv in runs]
    assert len(set(names)) == 4
    assert all(name.startswith("vk-") for name in names)
    # each command carries only its own configuration
    first, argv = runs[0]
    assert "b2048-u1024-q8_0-none" in first
    assert value_after(argv, "--batch-size") == "2048"
    assert value_after(argv, "--kv-k") == "q8_0"

    counted = plan(bench)
    assert counted.configs == 4 and counted.loads == 4
    assert counted.requests == 4


def test_a_run_is_named_after_what_it_measures_even_when_it_is_the_only_one():
    """Otherwise the next combination lands in this one's folder and erases it."""
    assert run_names(DEFAULTS, BENCH_DEFAULTS, "vk") == ["vk-model-l1-b8192-u1024-q8_0-none"]
    named = BENCH_DEFAULTS.with_values({"run_name": "recheck"})
    assert run_names(DEFAULTS, named, "vk") == ["recheck-b8192-u1024-q8_0-none"]


def test_trying_one_combination_after_another_never_reuses_a_folder():
    """The whole point of the naming: attempts accumulate instead of overwriting."""
    tried = ({"ubatch": "512"}, {"ubatch": "1024"}, {"kv": "q4_0"},
             {"spec": "mtp"}, {"levels": "0"}, {"batch": "4096"})
    names = [run_names(DEFAULTS, BENCH_DEFAULTS.with_values(values), "vk")[0]
             for values in tried]
    assert len(set(names)) == len(tried), names


def test_a_folder_already_written_means_this_exact_thing_was_measured():
    """A clash can no longer be two different searches, so it is worth saying so."""
    name = run_names(DEFAULTS, BENCH_DEFAULTS)[0]
    problems = validate_bench(DEFAULTS, BENCH_DEFAULTS, existing=frozenset({name}))
    assert any(problem.level == "warn" and "Measured before" in problem.message
               for problem in problems)
    others = validate_bench(DEFAULTS, BENCH_DEFAULTS, existing=frozenset({"something-else"}))
    assert not any(problem.level == "warn" for problem in others)


def test_an_empty_axis_is_an_error_rather_than_a_missing_dimension():
    assert any("nothing ticked" in message
               for message in errors(validate_bench(DEFAULTS,
                                                    BENCH_DEFAULTS.with_values({"kv": " "}))))


def test_nothing_ticked_to_measure_is_caught_before_bench2_picks_for_us():
    """Given neither, bench2 defaults to level 1 and measures something else."""
    bench = BENCH_DEFAULTS.with_values({"levels": "", "session_levels": ""})
    assert any("falls back to level 1" in message
               for message in errors(validate_bench(DEFAULTS, bench)))


def test_a_word_no_axis_accepts_is_caught_here_rather_than_by_a_server_that_wont_start():
    bench = BENCH_DEFAULTS.with_values({"kv": "f16,q4_K", "spec": "ngram-mod", "levels": "9"})
    messages = errors(validate_bench(DEFAULTS, bench))
    assert any("q4_K" in message and "q8_0" in message for message in messages)
    assert any("ngram-mod" in message and "mtp" in message for message in messages)
    assert any("Single levels" in message and "9" in message for message in messages)


def test_a_ubatch_above_its_batch_is_refused_before_the_model_is_loaded():
    bench = BENCH_DEFAULTS.with_values({"batch": "512", "ubatch": "1024"})
    assert any("above batch 512" in message
               for message in errors(validate_bench(DEFAULTS, bench)))


def test_a_ubatch_above_one_batch_is_skipped_rather_than_fatal():
    """The pair llama-server refuses is dropped; the rest of the search runs."""
    bench = BENCH_DEFAULTS.with_values({"batch": "1024,8192", "ubatch": "128,2048"})
    # (1024, 128), (8192, 128), (8192, 2048) — (1024, 2048) cannot exist
    assert config_count(bench) == 3
    problems = validate_bench(DEFAULTS, bench)
    assert not errors(problems)
    assert any("skipped" in problem.message for problem in problems
               if problem.level == "warn")


def test_draft_tokens_are_an_axis_only_where_speculation_is_on():
    bench = BENCH_DEFAULTS.with_values({"spec": "mtp", "spec_n": "2,3"})
    assert config_count(bench) == 2
    assert value_after(command(DEFAULTS, bench, configurations(bench)[1]), "--spec-n") == "3"
    assert configurations(bench)[0].suffix.endswith("-mtp-n2")

    # with no speculation, ticked drafts collapse: they would be the same command
    plain = BENCH_DEFAULTS.with_values({"spec": "none", "spec_n": "2,3"})
    assert config_count(plain) == 1
    assert "--spec-n" not in command(DEFAULTS, plain, configurations(plain)[0])


def test_a_search_over_the_cap_is_refused_because_every_run_reloads_the_model():
    bench = BENCH_DEFAULTS.with_values({"batch": "1024,2048,4096,8192",
                                        "kv": "f16,q8_0,q4_0", "sweep_max": 6})
    assert config_count(bench) == 12
    assert any("12 runs against a cap of 6" in message
               for message in errors(validate_bench(DEFAULTS, bench)))


def test_a_level_the_model_was_not_trained_for_is_refused(tmp_path):
    facts = read_facts(write_gguf(tmp_path / "short.gguf", architecture="qwen35", layers=65,
                                  context=40960, embedding=5120, hparams=QWEN35_27B))
    bench = BENCH_DEFAULTS.with_values({"levels": "3"})
    assert any("trained for 40960" in message
               for message in errors(validate_bench(DEFAULTS, bench, facts)))
    # and the level it does have room for passes
    assert not errors(validate_bench(DEFAULTS, BENCH_DEFAULTS.with_values({"levels": "1"}),
                                     facts))


def test_the_repository_snapshot_cannot_fill_a_large_level_and_says_so():
    bench = BENCH_DEFAULTS.with_values({"levels": "3", "context_source": "repo-snapshot"})
    assert any(problem.level == "warn" and "shorter prompt" in problem.message
               for problem in validate_bench(DEFAULTS, bench))


def test_the_file_source_needs_a_file():
    bench = BENCH_DEFAULTS.with_values({"context_source": "file"})
    assert errors(validate_bench(DEFAULTS, bench)) == ["The file source needs a file to read"]


def test_the_warm_up_shot_is_said_either_way_because_bench2_defaults_it_on():
    assert "--warmup-shot" in command()
    off = command(bench=BENCH_DEFAULTS.with_values({"warmup_shot": False}))
    assert "--no-warmup-shot" in off and "--warmup-tokens" not in off


def test_the_configurations_too_big_for_the_cards_are_counted_before_the_first_load(tmp_path):
    facts = read_facts(write_gguf(tmp_path / "qwen35.gguf", architecture="qwen35", layers=65,
                                  context=262144, embedding=5120, hparams=QWEN35_27B))
    spec = replace(DEFAULTS, model=str(tmp_path / "qwen35.gguf"), gpu_layers_all=True)
    bench = BENCH_DEFAULTS.with_values({"levels": "4", "kv": "f16,q8_0,q4_0",
                                        "batch": "4096,8192"})

    report = fit(spec, facts, bench, budget_mib=4096)
    assert report is not None
    # batch does not move the bill, so three bills stand for six runs
    assert len(report.weighed) == 3 and report.total == 6
    assert report.over_count == len(report.over) * 2
    assert report.largest_fitting is not None
    assert report.largest_fitting.mib <= 4096 < report.heaviest.mib

    assert fit(spec, facts, bench, budget_mib=1.0).largest_fitting is None


def test_pricing_a_search_needs_a_model_a_budget_and_something_to_measure(tmp_path):
    assert fit(DEFAULTS, None, BENCH_DEFAULTS, budget_mib=16384) is None

    facts = read_facts(write_gguf(tmp_path / "qwen35.gguf", architecture="qwen35", layers=65,
                                  context=262144, embedding=5120, hparams=QWEN35_27B))
    assert fit(DEFAULTS, facts, BENCH_DEFAULTS, budget_mib=0) is None
    # nothing ticked to measure means no server, so there is no bill to name
    nothing = BENCH_DEFAULTS.with_values({"levels": "", "session_levels": ""})
    assert fit(DEFAULTS, facts, nothing, budget_mib=16384) is None
