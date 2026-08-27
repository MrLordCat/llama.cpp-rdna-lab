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
