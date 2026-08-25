from dataclasses import fields
from pathlib import Path

import pytest

from gui2.core.bench import BenchSpec, server_extra_tokens, to_bench_argv
from gui2.core.params import SCHEMA
from gui2.core.runspec import (
    DEFAULTS,
    RunSpec,
    errors,
    mask_api_key,
    parse_rpc_endpoints,
    to_argv,
    validate,
)


def flag_value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_schema_and_runspec_stay_in_sync():
    spec_fields = {field.name for field in fields(RunSpec)}
    schema_names = {param.name for param in SCHEMA}
    assert schema_names <= spec_fields
    assert spec_fields - schema_names == {"build_dir", "extra_args"}


def test_minimal_command_only_emits_non_default_flags():
    argv = to_argv(RunSpec(model="m.gguf"), binary="llama-server")
    assert argv[:3] == ["llama-server", "-m", "m.gguf"]
    # llama-server defaults stay implicit
    assert "--cache-type-k" not in argv
    assert "--flash-attn" not in argv
    assert "-fit" not in argv
    assert "--metrics" in argv


def test_kv_flash_and_fit_are_emitted_when_changed():
    argv = to_argv(RunSpec(model="m.gguf", cache_type_k="q4_0", cache_type_v="q4_0",
                           flash_attn="on", fit="off"))
    assert flag_value(argv, "--cache-type-k") == "q4_0"
    assert flag_value(argv, "--cache-type-v") == "q4_0"
    assert flag_value(argv, "--flash-attn") == "on"
    assert flag_value(argv, "-fit") == "off"


def test_mtp_expands_to_draft_mtp_with_draft_budget():
    argv = to_argv(RunSpec(model="m.gguf", spec_type="mtp", spec_draft_n_max=3))
    assert flag_value(argv, "--spec-type") == "draft-mtp"
    assert flag_value(argv, "--spec-draft-n-max") == "3"


def test_ngram_mod_expands_to_the_measured_profile():
    argv = to_argv(RunSpec(model="m.gguf", spec_type="ngram-mod"))
    assert flag_value(argv, "--spec-type") == "ngram-mod"
    assert flag_value(argv, "--spec-ngram-mod-n-min") == "12"
    assert flag_value(argv, "--spec-ngram-mod-n-match") == "16"
    assert flag_value(argv, "--spec-ngram-mod-n-max") == "32"


def test_rpc_is_emitted_before_dev():
    argv = to_argv(RunSpec(model="m.gguf", rpc_endpoints="10.0.0.5:50052, bad, 10.0.0.6:50052",
                           devices="ROCm1,ROCm0,RPC0,RPC1", split_mode="layer"))
    assert flag_value(argv, "--rpc") == "10.0.0.5:50052,10.0.0.6:50052"
    assert argv.index("--rpc") < argv.index("-dev")


def test_conversation_cache_replaces_the_checkpoint_knobs():
    argv = to_argv(RunSpec(model="m.gguf", conversation_cache=True))
    assert "--conversation-cache" in argv
    assert "--ctx-checkpoints" not in argv
    assert "--checkpoint-every-n-tokens" not in argv


def test_extra_arguments_win_over_generated_flags():
    spec = RunSpec(model="m.gguf", ctx_size=131072, gpu_layers=999,
                   extra_args="--ctx-size 8192 -ngl 40 --my-flag")
    argv = to_argv(spec)
    assert "-c" not in argv
    assert argv.count("-ngl") == 1
    assert flag_value(argv, "-ngl") == "40"
    assert flag_value(argv, "--ctx-size") == "8192"
    assert argv[-5:] == ["--ctx-size", "8192", "-ngl", "40", "--my-flag"]


def test_absence_flag_and_api_key_masking():
    argv = to_argv(RunSpec(model="m.gguf", mmproj_offload=False, api_key="secret"))
    assert "--no-mmproj-offload" in argv
    assert "secret" in argv
    assert "secret" not in mask_api_key(argv)


def test_api_key_masking_reaches_packed_arguments():
    packed = ["python", "bench.py", "--server-extra=-t 8 --api-key secret", "--api-key=secret"]
    masked = mask_api_key(packed)
    assert not any("secret" in token for token in masked)
    assert masked[2].startswith("--server-extra=-t 8 --api-key ")
    assert masked[3].startswith("--api-key=")


def test_with_values_coerces_form_input():
    spec = DEFAULTS.with_values({"ctx_size": "49152", "no_mmap": "on", "unknown": "x", "host": " 0.0.0.0 "})
    assert spec.ctx_size == 49152
    assert spec.no_mmap is True
    assert spec.host == "0.0.0.0"
    assert not hasattr(spec, "unknown")


def test_parse_rpc_endpoints_filters_junk():
    assert parse_rpc_endpoints("a:1, a:1, host:99999, nope, 10.0.0.2:50052") == ["a:1", "10.0.0.2:50052"]


def test_validation_blocks_impossible_runs(tmp_path: Path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")

    spec = RunSpec(model=str(model), build_dir="build-vulkan", batch_size=128, ubatch_size=256)
    assert any("Ubatch" in problem.message for problem in errors(validate(spec)))

    fp8 = RunSpec(model=str(model), build_dir="build-cpu", cache_type_k="f8_e4m3", cache_type_v="f8_e4m3")
    messages = " ".join(problem.message for problem in errors(validate(fp8, backend="cpu")))
    assert "Vulkan or ROCm" in messages and "flash attention" in messages

    mtp = RunSpec(model=str(model), build_dir="build-rocm", spec_type="mtp", parallel=4)
    assert any("--parallel 1" in problem.message for problem in errors(validate(mtp)))

    rpc = RunSpec(model=str(model), build_dir="build-rocm", rpc_endpoints="10.0.0.5:50052")
    assert any("GGML_RPC=OFF" in problem.message for problem in errors(validate(rpc, supports_rpc=False)))
    assert not errors(validate(rpc, supports_rpc=True))

    orphan = RunSpec(model=str(model), build_dir="build-rocm", devices="ROCm1,RPC0")
    assert any("RPC0" in problem.message for problem in errors(validate(orphan)))


def test_validation_notes_do_not_block(tmp_path: Path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    spec = RunSpec(model=str(model), build_dir="build-vulkan", spec_type="mtp",
                   cache_type_k="q8_0", cache_type_v="q8_0")
    problems = validate(spec, backend="vulkan")
    assert not errors(problems)
    assert any("KV layers in f16" in problem.message for problem in problems)


def test_bench_argv_reuses_the_server_command():
    spec = RunSpec(model="models/Q4.gguf", ctx_size=49152, batch_size=8192, ubatch_size=1024,
                   cache_type_k="q4_0", cache_type_v="q4_0", flash_attn="on", spec_type="mtp",
                   devices="Vulkan1,Vulkan0", split_mode="layer", tensor_split="1,1", no_mmap=True)
    argv = to_bench_argv(spec, BenchSpec(label="d131-r1", task_ids="triage_diff"),
                         script="scripts/agent_workload_bench.py",
                         server_bin="build-vulkan/bin/llama-server.exe")

    assert flag_value(argv, "--ctx-size") == "49152"
    assert flag_value(argv, "--cache-type-k") == "q4_0"
    assert "--flash-attn" in argv
    assert flag_value(argv, "--label") == "d131-r1"

    extra = next(item for item in argv if item.startswith("--server-extra="))
    # bench owns these; forwarding them again would duplicate the flag
    assert "--ctx-size" not in extra and "-ngl" not in extra and "-m " not in extra
    assert "--spec-type draft-mtp" in extra
    assert "-dev Vulkan1,Vulkan0" in extra
    assert "--no-mmap" in extra


def test_server_extra_drops_only_bench_owned_flags():
    tokens = server_extra_tokens(RunSpec(model="m.gguf", host="0.0.0.0", port=9000, no_mmap=True))
    assert "--host" not in tokens and "--port" not in tokens
    assert "--no-mmap" in tokens


@pytest.mark.parametrize("spec_type,expected", [("none", 0), ("mtp", 1), ("ngram-mod", 1)])
def test_spec_type_emits_at_most_one_spec_flag(spec_type: str, expected: int):
    argv = to_argv(RunSpec(model="m.gguf", spec_type=spec_type))
    assert argv.count("--spec-type") == expected
