from datetime import datetime
from pathlib import Path

import pytest

from gui2.core.history import (
    HistoryStore,
    RunFilter,
    apply_filter,
    facets,
    load_runs,
    past_sweeps,
    sort_runs,
    summarize,
    sync_autotune_runs,
    winning_config,
)

HEADER = (
    "timestamp,run_id,build_id,build_name,build_backend,mode,label,model,is_mtp_model,tasks,"
    "task_ids,runs,ctx,batch,ubatch,kv_k,kv_v,spec_mode,extra_preset,extra_args,no_reuse,"
    "gpu_layers,parallel,flash_attn,max_tokens,real_context_mode,real_context_chars,"
    "real_context_safe_fill,no_v2_prime_pass,temperature,top_p,aggregate_tps,mean_task_tps,"
    "prompt_eval_tps,decode_eval_tps,prompt_eval_ms,decode_eval_ms,errors,metric_scope,lane_key,"
    "best_config,jsonl_file,csv_file,summary_file,server_log_file,is_group_best"
)

ROWS = [
    # rocm, healthy, lane best
    "2026-07-08 10:00:00,run-a,bld-1,build-rocm,rocm,single-run,rocm-fast,models\\Qwen-Q4.gguf,0,quick,,1,"
    "131072,512,128,q4_0,q4_0,none,base,--spec-type none,1,999,1,on,16,repo-snapshot,24576,0.88,1,0.2,0.9,"
    "20.5,20.5,480.0,21.0,300.0,2900.0,0,cold-first,rocm|lane,,a.jsonl,a.csv,a.md,a.log,1",
    # vulkan, failed run
    "2026-07-09 11:30:00,run-b,bld-2,build-vulkan,vulkan,single-run,vk-slow,models\\Qwen-Q3.gguf,1,quick,,1,"
    "49152,256,64,q8_0,q8_0,mtp,base,,1,999,1,on,16,repo-snapshot,24576,0.88,1,0.2,0.9,"
    "5.25,5.25,100.0,6.0,900.0,5000.0,2,cold-first,vulkan|lane,,b.jsonl,b.csv,b.md,b.log,0",
    # autotune row: batch/ubatch are literal "sweep", metrics partly empty
    "2026-07-10 09:15:00,run-c,bld-1,build-rocm,rocm,autotune,at-sweep,models\\Qwen-Q4.gguf,0,quick,triage_diff,1,"
    "131072,sweep,sweep,sweep,sweep,none,base,,1,-1,1,on,16,repo-snapshot,24576,0.88,1,0.2,0.9,"
    "12.0,12.0,,,,,0,autotune,rocm|lane,ctx=131072 b=512 ub=192 kv=q4_0 spec=none,,,c.csv,c.log,0",
]


@pytest.fixture()
def history_csv(tmp_path: Path) -> Path:
    path = tmp_path / "BENCH_RUNS.csv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    return path


def test_missing_file_yields_no_runs(tmp_path: Path):
    assert load_runs(tmp_path / "absent.csv") == []


def test_rows_are_parsed_newest_first(history_csv: Path):
    runs = load_runs(history_csv)
    assert [run.run_id for run in runs] == ["run-c", "run-b", "run-a"]
    assert runs[-1].timestamp == datetime(2026, 7, 8, 10, 0, 0)
    assert runs[-1].model == "Qwen-Q4.gguf"
    assert runs[-1].is_group_best is True


def test_sweep_and_empty_metrics_do_not_break_parsing(history_csv: Path):
    autotune = next(run for run in load_runs(history_csv) if run.mode == "autotune")
    assert autotune.batch == "sweep"
    assert autotune.prompt_eval_tps is None
    assert autotune.gpu_layers == -1
    assert autotune.aggregate_tps == pytest.approx(12.0)


def test_what_a_sweep_chose_survives_only_in_best_config(history_csv: Path):
    """batch, ubatch and both KV columns of an autotune row say "sweep"."""
    autotune = next(run for run in load_runs(history_csv) if run.mode == "autotune")
    assert winning_config(autotune) == {
        "sweep_ctx": "131072", "sweep_batch": "512", "sweep_ubatch": "192",
        "sweep_kv": "q4_0", "sweep_spec": "none",
    }
    # a run that chose nothing cannot be reused, and says so by parsing to nothing
    assert winning_config(next(run for run in load_runs(history_csv)
                               if run.mode == "single-run")) == {}


def test_earlier_sweeps_are_found_by_model_name_not_by_path(history_csv: Path):
    runs = load_runs(history_csv)
    assert [run.run_id for run in past_sweeps(runs, "D:/elsewhere/Qwen-Q4.gguf")] == ["run-c"]
    # a single run of the same model is not a sweep and has no winner to offer
    assert past_sweeps(runs, "Qwen-Q3.gguf") == []
    assert past_sweeps(runs, "") == []


def test_filters_combine(history_csv: Path):
    runs = load_runs(history_csv)
    assert [r.run_id for r in apply_filter(runs, RunFilter(backend="rocm"))] == ["run-c", "run-a"]
    assert [r.run_id for r in apply_filter(runs, RunFilter(hide_errors=True))] == ["run-c", "run-a"]
    assert [r.run_id for r in apply_filter(runs, RunFilter(best_only=True))] == ["run-a"]
    assert [r.run_id for r in apply_filter(runs, RunFilter(min_tps=13.0))] == ["run-a"]
    assert [r.run_id for r in apply_filter(runs, RunFilter(spec="mtp"))] == ["run-b"]


def test_query_searches_label_and_lane(history_csv: Path):
    runs = load_runs(history_csv)
    assert [r.run_id for r in apply_filter(runs, RunFilter(query="VK-SLOW"))] == ["run-b"]
    assert [r.run_id for r in apply_filter(runs, RunFilter(query="vulkan|lane"))] == ["run-b"]
    assert apply_filter(runs, RunFilter(query="nothing-here")) == []


def test_sorting_puts_missing_values_last(history_csv: Path):
    runs = load_runs(history_csv)
    by_prompt = sort_runs(runs, "prompt", descending=True)
    assert [r.run_id for r in by_prompt] == ["run-a", "run-b", "run-c"]
    assert [r.run_id for r in sort_runs(runs, "tps", descending=False)] == ["run-b", "run-c", "run-a"]


def test_facets_are_distinct_and_sorted(history_csv: Path):
    available = facets(load_runs(history_csv))
    assert available.backends == ("rocm", "vulkan")
    assert available.modes == ("autotune", "single-run")
    assert available.specs == ("mtp", "none")


def test_summary(history_csv: Path):
    stats = summarize(load_runs(history_csv))
    assert stats.count == 3
    assert stats.with_errors == 1
    assert stats.best_tps == pytest.approx(20.5)
    assert stats.median_tps == pytest.approx(12.0)
    assert stats.best_run is not None and stats.best_run.run_id == "run-a"
    assert stats.first_time.startswith("2026-07-08")


def test_store_reloads_only_after_the_file_changes(history_csv: Path):
    store = HistoryStore(history_csv)
    assert len(store.runs()) == 3
    assert store.runs() is store.runs()

    history_csv.write_text("\n".join([HEADER, ROWS[0]]) + "\n", encoding="utf-8")
    import os

    os.utime(history_csv, (0, 0))
    assert len(store.runs()) == 1


def _write_bench_index(tmp_path: Path, rows: list[dict], meta: dict | None = None) -> Path:
    import csv
    import json

    folder = tmp_path / "build_logs" / "bench"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "index.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        keys = ["run_name", "type", "level", "timestamp", "backend", "model", "commit",
                "ctx", "prefill_tps", "decode_tps", "aggregate_tps", "mtp_draft_n",
                "status", "path"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    for row in rows:
        if not row.get("path"):
            continue
        run_dir = Path(row["path"])
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = meta or {}
        server = {"batch_size": 1024, "ubatch_size": 128, "kv_k": "f8_e4m3",
                  "kv_v": "f8_e4m3", "spec": "mtp", "gpu_layers": 64, "parallel": 1,
                  "context_source": "synthetic", "temperature": 0.2, "top_p": 0.9,
                  "runs": 1, "flash_attn": True}
        server.update(meta.get("server", {}))
        payload = {"model": meta.get("model", "D:/x/models/qwen.gguf"),
                   "type": meta.get("type", "single"),
                   "levels": meta.get("levels", ["1", "2"]), "server": server}
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_sync_carries_finished_autotune_runs_into_canonical_history(tmp_path: Path):
    """History & Analytics reads BENCH_RUNS.csv; a finished autotune run lands
    there as one row named by its fastest decode, and only once."""
    run_dir = tmp_path / "run-vk"
    index = _write_bench_index(tmp_path, [
        dict(run_name="vk-qwen-b1024-u128-f8_e4m3-mtp-n2", type="single", level="1",
             timestamp="2026-08-29T10:00:00+03:00", backend="vk", model="qwen.gguf",
             commit="abc1234", ctx="8192", prefill_tps="900", decode_tps="30",
             aggregate_tps="28", mtp_draft_n="2", status="ok", path=str(run_dir)),
        dict(run_name="vk-qwen-b1024-u128-f8_e4m3-mtp-n2", type="single", level="2",
             timestamp="2026-08-29T10:02:00+03:00", backend="vk", model="qwen.gguf",
             commit="abc1234", ctx="49152", prefill_tps="1000", decode_tps="35",
             aggregate_tps="32", mtp_draft_n="2", status="ok", path=str(run_dir)),
    ])
    runs_csv = tmp_path / "BENCH_RUNS.csv"

    assert sync_autotune_runs(runs_csv, index) == 1
    runs = load_runs(runs_csv)
    assert len(runs) == 1
    run = runs[0]
    assert run.mode == "autotune" and run.backend == "vk"
    assert run.label == "vk-qwen-b1024-u128-f8_e4m3-mtp-n2"
    assert run.batch == "1024" and run.ubatch == "128"
    assert run.kv_k == "f8_e4m3" and run.spec_mode == "mtp"
    assert run.decode_eval_tps == 35.0, "named by its fastest decode"
    assert run.raw["build_id"] == "abc1234"
    assert run.lane_key.startswith("vk|qwen.gguf|ctx")

    assert sync_autotune_runs(runs_csv, index) == 0, "idempotent"
    assert len(load_runs(runs_csv)) == 1


def test_sync_writes_a_run_that_was_rerun_as_a_new_row(tmp_path: Path):
    """The same configuration launched again is a new measurement, not a patch.

    bench2 drops the old rows when it re-runs a name, so the rerun is a later
    index with the same run_name but a new timestamp."""
    run_dir = tmp_path / "run-vk"
    index = _write_bench_index(tmp_path, [
        dict(run_name="vk-qwen-b1024-u128-q8_0-none", type="single", level="1",
             timestamp="2026-08-29T10:00:00+03:00", backend="vk", model="qwen.gguf",
             commit="abc1234", ctx="8192", prefill_tps="900", decode_tps="30",
             aggregate_tps="28", mtp_draft_n="", status="ok", path=str(run_dir)),
    ])
    runs_csv = tmp_path / "BENCH_RUNS.csv"
    assert sync_autotune_runs(runs_csv, index) == 1

    # the rerun replaces the run's rows in the index, with a later timestamp
    index = _write_bench_index(tmp_path, [
        dict(run_name="vk-qwen-b1024-u128-q8_0-none", type="single", level="1",
             timestamp="2026-08-30T10:00:00+03:00", backend="vk", model="qwen.gguf",
             commit="abc1234", ctx="8192", prefill_tps="950", decode_tps="33",
             aggregate_tps="30", mtp_draft_n="", status="ok", path=str(run_dir)),
    ])
    assert sync_autotune_runs(runs_csv, index) == 1
    runs = load_runs(runs_csv)
    assert len(runs) == 2
    assert {run.decode_eval_tps for run in runs} == {30.0, 33.0}


def test_sync_ignores_runs_without_a_single_ok_scenario(tmp_path: Path):
    index = _write_bench_index(tmp_path, [
        dict(run_name="vk-dead-b1024-u128-q8_0-none", type="single", level="1",
             timestamp="2026-08-29T10:00:00+03:00", backend="vk", model="qwen.gguf",
             commit="", ctx="8192", prefill_tps="0", decode_tps="0",
             aggregate_tps="0", mtp_draft_n="", status="error", path=""),
    ])
    runs_csv = tmp_path / "BENCH_RUNS.csv"
    assert sync_autotune_runs(runs_csv, index) == 0
    assert not runs_csv.exists()
