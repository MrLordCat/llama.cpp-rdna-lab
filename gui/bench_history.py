"""Autotune run history & best-preset persistence for the benchmark tab.

BenchHistoryMixin is mixed into BenchmarkTabWidget; methods here operate on
the host widget via self (presets_table, status_label, history_csv paths,
project_root, models_dir, _summary_sweep_cache, parent).
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QTableWidgetItem

from bench_widgets import NumericTableWidgetItem


class BenchHistoryMixin:
    """History table + preset management methods for BenchmarkTabWidget."""

    def _load_best_presets(self) -> dict[str, dict[str, str]]:
        if not self.best_presets_path.exists():
            return {}
        try:
            data = json.loads(self.best_presets_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            return {}
        return {}

    def _save_best_presets(self, presets: dict[str, dict[str, str]]) -> None:
        self.best_presets_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_presets_path.write_text(json.dumps(presets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _update_best_preset_for_model(self, model_name: str, profile_key: str = "model-best", min_ctx: int | None = None) -> None:
        if not model_name or not self.history_csv.exists():
            return

        best_row = None
        best_tps = -1.0
        try:
            with self.history_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("errors", "0")) not in ("", "0"):
                        continue
                    row_model = Path(str(row.get("model", ""))).name
                    if row_model.lower() != model_name.lower():
                        continue
                    if min_ctx is not None:
                        try:
                            row_ctx = int(str(row.get("ctx", "0") or "0"))
                        except ValueError:
                            row_ctx = 0
                        if row_ctx < min_ctx:
                            continue
                    try:
                        tps = float(str(row.get("aggregate_tps", "0") or "0"))
                    except ValueError:
                        tps = 0.0
                    if tps > best_tps:
                        best_tps = tps
                        best_row = row
        except Exception:
            return

        if not best_row:
            return

        presets = self._load_best_presets()
        store_key = model_name if profile_key == "model-best" else f"{model_name}::{profile_key}"
        presets[store_key] = {
            "model": model_name,
            "profile": profile_key,
            "best_tps": f"{best_tps:.4f}",
            "ctx": str(best_row.get("ctx", "")),
            "batch": str(best_row.get("batch", "")),
            "ubatch": str(best_row.get("ubatch", "")),
            "kv_k": str(best_row.get("kv_k", "")),
            "kv_v": str(best_row.get("kv_v", "")),
            "spec_mode": str(best_row.get("spec_mode", "")),
            "extra_preset": str(best_row.get("extra_preset", "")),
            "extra_args": str(best_row.get("extra_args", "")),
            "build_id": str(best_row.get("build_id", "")),
            "run_id": str(best_row.get("run_id", "")),
            "label": str(best_row.get("label", "")),
            "timestamp": str(best_row.get("timestamp", "")),
        }
        self._save_best_presets(presets)
        self.refresh_saved_presets_table()

    @staticmethod
    def _parse_best_config_text(best_config: str) -> dict[str, str]:
        parsed = {
            "ctx": "-",
            "batch": "-",
            "ubatch": "-",
            "kv": "-",
            "spec": "-",
            "extra_preset": "-",
            "extra_args": "-",
        }

        text = best_config.strip()
        if not text:
            return parsed

        match = re.search(
            r"ctx=(\d+)\s+b=(\d+)\s+ub=(\d+)\s+kv=([^\s,]+)\s+spec=([^\s,]+)(?:\s+extra=([^\s,]+))?(?:\s+extra_args=(.*))?$",
            text,
        )
        if not match:
            return parsed

        ctx, batch, ubatch, kv, spec_mode, extra_preset, extra_args = match.groups()
        parsed["ctx"] = ctx
        parsed["batch"] = batch
        parsed["ubatch"] = ubatch
        parsed["kv"] = kv.strip().rstrip(",;")
        parsed["spec"] = spec_mode.strip().rstrip(",;")
        parsed["extra_preset"] = (extra_preset or "base").strip().rstrip(",;")

        extra_args_text = (extra_args or "").strip()
        parsed["extra_args"] = "-" if not extra_args_text or extra_args_text == "<none>" else extra_args_text
        return parsed

    def _load_autotune_history_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen_keys: set[str] = set()
        history_candidates = [self.history_csv_v2, self.history_csv]

        for history_csv in history_candidates:
            if not history_csv.exists():
                continue
            try:
                with history_csv.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if str(row.get("mode", "")).strip().lower() != "autotune":
                            continue
                        run_id = str(row.get("run_id", "")).strip()
                        timestamp = str(row.get("timestamp", "")).strip()
                        label = str(row.get("label", "")).strip()
                        key = f"{timestamp}::{run_id}::{label}"
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        rows.append({str(k): str(v) for k, v in row.items()})
            except Exception:
                continue

        rows.sort(key=lambda item: (item.get("timestamp", ""), item.get("run_id", "")), reverse=True)
        return rows

    def _selected_history_row_data(self) -> dict[str, str] | None:
        row = self.presets_table.currentRow()
        if row < 0:
            return None

        run_time_item = self.presets_table.item(row, 0)
        model_item = self.presets_table.item(row, 1)
        run_id_item = self.presets_table.item(row, 12)
        label_item = self.presets_table.item(row, 13)

        run_time = run_time_item.text().strip() if run_time_item is not None else ""
        model_name = model_item.text().strip() if model_item is not None else ""
        run_id = run_id_item.text().strip() if run_id_item is not None else ""
        label = label_item.text().strip() if label_item is not None else ""

        for history_row in self._load_autotune_history_rows():
            history_run_id = str(history_row.get("run_id", "")).strip()
            history_time = str(history_row.get("timestamp", "")).strip()
            history_label = str(history_row.get("label", "")).strip()
            history_model = Path(str(history_row.get("model", "")).strip()).name

            if run_id and run_id != "-" and history_run_id and history_run_id == run_id:
                return history_row
            if (
                history_time == run_time
                and history_label == label
                and history_model.lower() == model_name.lower()
            ):
                return history_row

        return None

    def _extract_sweep_sets_from_summary(self, row_data: dict[str, str]) -> tuple[str, str]:
        summary_file = str(row_data.get("summary_file", "")).strip()
        if not summary_file:
            return "-", "-"

        if summary_file in self._summary_sweep_cache:
            return self._summary_sweep_cache[summary_file]

        summary_path = self.project_root / "build_logs" / "agent-workload" / summary_file
        if not summary_path.exists():
            self._summary_sweep_cache[summary_file] = ("-", "-")
            return "-", "-"

        spec_values: list[str] = []
        extra_values: list[str] = []
        seen_specs: set[str] = set()
        seen_extras: set[str] = set()

        try:
            with summary_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    spec = self._sanitize_compact_token(str(row.get("spec_mode", "")), fallback="")
                    extra = self._sanitize_compact_token(str(row.get("extra_preset", "")), fallback="")

                    if spec and spec not in seen_specs:
                        seen_specs.add(spec)
                        spec_values.append(spec)
                    if extra and extra not in seen_extras:
                        seen_extras.add(extra)
                        extra_values.append(extra)
        except Exception:
            self._summary_sweep_cache[summary_file] = ("-", "-")
            return "-", "-"

        specs_text = ",".join(spec_values) if spec_values else "-"
        extras_text = ",".join(extra_values) if extra_values else "-"
        self._summary_sweep_cache[summary_file] = (specs_text, extras_text)
        return specs_text, extras_text

    @staticmethod
    def _kv_cache_index_from_name(kv_name: str) -> int:
        kv_map = {
            "f16": 0,
            "bf16": 1,
            "f32": 2,
            "q8_0": 3,
            "q5_1": 4,
            "q5_0": 5,
            "q4_1": 6,
            "q4_0": 7,
            "iq4_nl": 8,
            "tbq4_0": 9,
            "tbq3_0": 10,
            "tq3_0": 11,
            "turbo4": 12,
            "turbo4_0": 12,
            "turbo3": 13,
            "turbo3_0": 13,
            "turbo2": 14,
            "turbo2_0": 14,
        }
        return kv_map.get(kv_name.strip().lower(), 3)

    @staticmethod
    def _bool_from_history(value: str, default: bool = True) -> bool:
        text = str(value or "").strip().lower()
        if text in {"on", "true", "1", "yes", "y"}:
            return True
        if text in {"off", "false", "0", "no", "n"}:
            return False
        return default

    @staticmethod
    def _int_from_history(value: str, default: int) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _delete_run_from_history_file(self, history_path: Path, target_row: dict[str, str]) -> int:
        if not history_path.exists():
            return 0

        target_run_id = str(target_row.get("run_id", "")).strip()
        target_time = str(target_row.get("timestamp", "")).strip()
        target_label = str(target_row.get("label", "")).strip()
        target_model = Path(str(target_row.get("model", "")).strip()).name.lower()

        try:
            with history_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                rows = [dict(row) for row in reader]
        except Exception:
            return 0

        if not fieldnames:
            return 0

        filtered_rows: list[dict[str, str]] = []
        removed = 0
        for row in rows:
            row_run_id = str(row.get("run_id", "")).strip()
            row_time = str(row.get("timestamp", "")).strip()
            row_label = str(row.get("label", "")).strip()
            row_model = Path(str(row.get("model", "")).strip()).name.lower()

            matched = False
            if target_run_id and row_run_id and row_run_id == target_run_id:
                matched = True
            elif (
                row_time == target_time
                and row_label == target_label
                and row_model == target_model
                and str(row.get("mode", "")).strip().lower() == "autotune"
            ):
                matched = True

            if matched:
                removed += 1
                continue
            filtered_rows.append(row)

        if removed <= 0:
            return 0

        try:
            with history_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)
        except Exception:
            return 0

        return removed

    def _resolve_history_model_path(self, row_data: dict[str, str]) -> Path | None:
        model_raw = str(row_data.get("model", "")).strip()
        if model_raw:
            direct_path = Path(model_raw)
            if direct_path.exists():
                return direct_path

            relative_path = self.project_root / model_raw
            if relative_path.exists():
                return relative_path

        model_name = Path(model_raw).name.strip()
        if model_name:
            fallback = self.models_dir / model_name
            if fallback.exists():
                return fallback

        return None

    def _resolve_history_artifact_path(self, artifact_value: str) -> Path | None:
        value = str(artifact_value or "").strip()
        if not value:
            return None

        candidate = Path(value)
        if candidate.exists():
            return candidate

        fallback_paths = [
            self.project_root / value,
            self.project_root / "build_logs" / "agent-workload" / value,
        ]
        for path in fallback_paths:
            if path.exists():
                return path

        return None

    def _history_log_candidates(self, row_data: dict[str, str]) -> list[str]:
        candidates: list[str] = []

        def add_candidate(value: str) -> None:
            text = str(value or "").strip()
            if not text or text in {"-", "<none>"}:
                return
            if text not in candidates:
                candidates.append(text)

        add_candidate(str(row_data.get("server_log_file", "")))
        add_candidate(str(row_data.get("server_log", "")))

        summary_file = str(row_data.get("summary_file", "")).strip()
        summary_suffix = "-autotune-summary.csv"
        if summary_file.endswith(summary_suffix):
            add_candidate(summary_file[: -len(summary_suffix)] + ".server.log")

        csv_file = str(row_data.get("csv_file", "")).strip()
        if csv_file.lower().endswith(".csv"):
            add_candidate(str(Path(csv_file).with_suffix(".server.log")))

        jsonl_file = str(row_data.get("jsonl_file", "")).strip()
        if jsonl_file.lower().endswith(".jsonl"):
            add_candidate(str(Path(jsonl_file).with_suffix(".server.log")))

        label = str(row_data.get("label", "")).strip()
        if label:
            add_candidate(f"{label}.server.log")

        return candidates

    def _discover_history_log_variants(self, log_value: str) -> list[Path]:
        value = str(log_value or "").strip()
        if not value.lower().endswith(".server.log"):
            return []

        file_name = Path(value).name
        base_name = file_name[: -len(".server.log")]
        if not base_name:
            return []

        roots = [
            self.project_root / "build_logs" / "agent-workload",
            self.project_root,
        ]
        patterns = [
            f"{base_name}.server.log",
            f"{base_name}-cfg*.server.log",
            f"{base_name}*.server.log",
        ]

        matches: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for pattern in patterns:
                for found in root.glob(pattern):
                    if not found.is_file():
                        continue
                    key = str(found)
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append(found)

        matches.sort(
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        return matches

    def _resolve_history_log_paths(self, row_data: dict[str, str]) -> tuple[list[Path], list[str]]:
        candidates = self._history_log_candidates(row_data)
        resolved: list[Path] = []
        seen_paths: set[str] = set()

        for log_value in candidates:
            log_path = self._resolve_history_artifact_path(log_value)
            if log_path is not None:
                path_key = str(log_path)
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    resolved.append(log_path)

            for variant in self._discover_history_log_variants(log_value):
                path_key = str(variant)
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                resolved.append(variant)

        return resolved, candidates

    @staticmethod
    def _clipboard_set_text(text: str) -> bool:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return False
        clipboard.setText(text)
        return True

    @staticmethod
    def _preview_candidates(candidates: list[str], limit: int = 8) -> str:
        if not candidates:
            return "(none)"
        preview = candidates[:limit]
        if len(candidates) > limit:
            preview.append("...")
        return "\n".join(preview)

    def open_selected_history_log(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Open Log", "Select a run row in history first.")
            return

        log_paths, candidates = self._resolve_history_log_paths(row_data)
        if not log_paths:
            QMessageBox.warning(
                self,
                "Open Log",
                "Log file not found for selected run.\n\nTried:\n"
                + self._preview_candidates(candidates),
            )
            return

        log_path = log_paths[0]

        try:
            if os.name == "nt":
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Log", f"Failed to open log file:\n{exc}")
            return

        self.status_label.setText(f"Opened log: {log_path.name}")

    def copy_selected_history_log_to_clipboard(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Copy Log", "Select a run row in history first.")
            return

        log_paths, candidates = self._resolve_history_log_paths(row_data)
        if not log_paths:
            QMessageBox.warning(
                self,
                "Copy Log",
                "Log file not found for selected run.\n\nTried:\n"
                + self._preview_candidates(candidates),
            )
            return

        text_chunks: list[str] = []
        if len(log_paths) == 1:
            try:
                log_text = log_paths[0].read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                QMessageBox.warning(self, "Copy Log", f"Failed to read log file:\n{exc}")
                return
        else:
            for path in log_paths:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                text_chunks.append(f"===== {path.name} =====\n{content}")
            if not text_chunks:
                QMessageBox.warning(self, "Copy Log", "No readable log files found for selected run.")
                return
            log_text = "\n\n".join(text_chunks)

        if not self._clipboard_set_text(log_text):
            QMessageBox.warning(self, "Copy Log", "Clipboard is not available.")
            return

        if len(log_paths) == 1:
            self.status_label.setText(f"Log copied to clipboard: {log_paths[0].name}")
        else:
            self.status_label.setText(f"Copied {len(log_paths)} logs to clipboard")

    def copy_selected_history_row_to_clipboard(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Copy Run Data", "Select a run row in history first.")
            return

        log_paths, candidates = self._resolve_history_log_paths(row_data)

        preferred_keys = [
            "timestamp",
            "run_id",
            "label",
            "mode",
            "model",
            "aggregate_tps",
            "best_config",
            "ctx",
            "batch",
            "ubatch",
            "kv_k",
            "kv_v",
            "spec_mode",
            "extra_preset",
            "extra_args",
            "tasks",
            "runs",
            "max_tokens",
            "errors",
            "build_id",
            "build_name",
            "build_backend",
            "summary_file",
            "server_log_file",
            "csv_file",
            "jsonl_file",
        ]

        lines: list[str] = ["# Autotune selected run"]
        seen_keys: set[str] = set()
        for key in preferred_keys:
            if key in row_data:
                value = str(row_data.get(key, "")).strip()
                lines.append(f"{key}: {value or '-'}")
                seen_keys.add(key)

        for key in sorted(row_data.keys()):
            if key in seen_keys:
                continue
            value = str(row_data.get(key, "")).strip()
            if value:
                lines.append(f"{key}: {value}")

        lines.append("")
        lines.append("resolved_server_logs:")
        if log_paths:
            for path in log_paths:
                lines.append(f"- {path}")
        else:
            lines.append("- -")
        lines.append("log_candidates:")
        for candidate in candidates:
            lines.append(f"- {candidate}")

        payload = "\n".join(lines).strip() + "\n"
        if not self._clipboard_set_text(payload):
            QMessageBox.warning(self, "Copy Run Data", "Clipboard is not available.")
            return

        self.status_label.setText("Selected run data copied to clipboard")

    def apply_selected_run_as_default_preset(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Apply Preset", "Select a run row in history first.")
            return

        model_raw = str(row_data.get("model", "")).strip()
        model_name = Path(model_raw).name
        if not model_name:
            QMessageBox.warning(self, "Apply Preset", "Selected run does not contain a model name.")
            return

        parsed_cfg = self._parse_best_config_text(str(row_data.get("best_config", "")))

        ctx = self._int_from_history(parsed_cfg["ctx"] if parsed_cfg["ctx"] != "-" else row_data.get("ctx", ""), 0)
        batch = self._int_from_history(parsed_cfg["batch"] if parsed_cfg["batch"] != "-" else row_data.get("batch", ""), 0)
        ubatch = self._int_from_history(parsed_cfg["ubatch"] if parsed_cfg["ubatch"] != "-" else row_data.get("ubatch", ""), 0)

        if ctx <= 0 or batch <= 0 or ubatch <= 0:
            QMessageBox.warning(
                self,
                "Apply Preset",
                "Selected run does not have a valid best configuration (ctx/batch/ubatch).",
            )
            return

        kv_name = parsed_cfg["kv"] if parsed_cfg["kv"] != "-" else str(row_data.get("kv_k", "")).strip()
        if not kv_name:
            kv_name = str(row_data.get("kv_v", "")).strip()

        spec_mode = parsed_cfg["spec"] if parsed_cfg["spec"] != "-" else str(row_data.get("spec_mode", "")).strip()
        extra_preset = parsed_cfg["extra_preset"] if parsed_cfg["extra_preset"] != "-" else str(row_data.get("extra_preset", "")).strip()
        extra_args = parsed_cfg["extra_args"]
        if extra_args == "-":
            extra_args = str(row_data.get("extra_args", "")).strip()
        if extra_args == "<none>":
            extra_args = ""

        spec_mode = spec_mode.strip().lower()
        if spec_mode and spec_mode not in {"-", "none", "mixed"} and "--spec-type" not in extra_args:
            spec_cli_mode = "draft-mtp" if spec_mode == "mtp" else spec_mode
            extra_args = f"--spec-type {spec_cli_mode}" if not extra_args else f"--spec-type {spec_cli_mode}\n{extra_args}"

        preset_path = self.project_root / "gui" / "model_presets.json"
        if not preset_path.exists():
            QMessageBox.warning(self, "Apply Preset", f"Preset file not found:\n{preset_path}")
            return

        try:
            data = json.loads(preset_path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.warning(self, "Apply Preset", f"Failed to read preset file:\n{exc}")
            return

        presets = data.get("presets")
        if not isinstance(presets, list):
            QMessageBox.warning(self, "Apply Preset", "Invalid model_presets.json format (missing presets array).")
            return

        run_id = str(row_data.get("run_id", "")).strip() or "-"
        run_time = str(row_data.get("timestamp", "")).strip() or "-"
        aggregate_tps = str(row_data.get("aggregate_tps", "")).strip() or "0"

        default_name = f"History Default {model_name}"
        default_preset = {
            "pattern": re.escape(model_name),
            "name": default_name,
            "ctx": ctx,
            "batch_size": batch,
            "ubatch_size": ubatch,
            "gpu_layers": self._int_from_history(row_data.get("gpu_layers", ""), 99),
            "parallel": self._int_from_history(row_data.get("parallel", ""), 1),
            "flash_attn": self._bool_from_history(row_data.get("flash_attn", ""), default=True),
            "kv_cache": self._kv_cache_index_from_name(kv_name),
            "notes": (
                "Applied from Autotune Runs History: "
                f"run_id={run_id}, tps={aggregate_tps}, time={run_time}, "
                f"spec={spec_mode or '-'}, extra={extra_preset or '-'}"
            ),
        }
        if extra_args:
            default_preset["extra_args"] = extra_args

        filtered_presets = [
            item
            for item in presets
            if not (isinstance(item, dict) and str(item.get("name", "")) == default_name)
        ]
        filtered_presets.insert(0, default_preset)
        data["presets"] = filtered_presets

        try:
            preset_path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Apply Preset", f"Failed to write preset file:\n{exc}")
            return

        applied_live = False
        if hasattr(self.parent, "server_tab") and hasattr(self.parent.server_tab, "apply_model_file_preset"):
            model_path = self._resolve_history_model_path(row_data)
            if model_path is not None and hasattr(self.parent.server_tab, "server_model_path"):
                self.parent.server_tab.server_model_path.setText(str(model_path))
            apply_result = self.parent.server_tab.apply_model_file_preset()
            applied_live = bool(isinstance(apply_result, dict) and apply_result.get("matched"))

        self.status_label.setText(f"Default preset applied from run: {run_id}")
        QMessageBox.information(
            self,
            "Apply Preset",
            "Selected run preset saved as default in gui/model_presets.json"
            + (" and applied to Launch Server tab." if applied_live else "."),
        )

    def refresh_saved_presets_table(self):
        rows = self._load_autotune_history_rows()
        self._summary_sweep_cache.clear()
        self.presets_table.setRowCount(0)

        for row_data in rows:
            run_time = str(row_data.get("timestamp", "") or "-")
            model_raw = str(row_data.get("model", "") or "-")
            model_name = Path(model_raw).name if model_raw not in {"", "-"} else "-"
            run_id = str(row_data.get("run_id", "") or "-")
            build_id = str(row_data.get("build_id", "") or "-")
            label = str(row_data.get("label", "") or "-")

            aggregate_text = str(row_data.get("aggregate_tps", "0") or "0")
            try:
                aggregate_value = float(aggregate_text)
            except ValueError:
                aggregate_value = 0.0

            parsed_cfg = self._parse_best_config_text(str(row_data.get("best_config", "")))
            if parsed_cfg["ctx"] == "-":
                parsed_cfg["ctx"] = str(row_data.get("ctx", "") or "-")
            if parsed_cfg["spec"] == "-":
                parsed_cfg["spec"] = str(row_data.get("spec_mode", "") or "-")
            if parsed_cfg["extra_preset"] == "-":
                parsed_cfg["extra_preset"] = str(row_data.get("extra_preset", "") or "-")
            if parsed_cfg["extra_args"] == "-":
                fallback_args = str(row_data.get("extra_args", "") or "").strip()
                if fallback_args:
                    parsed_cfg["extra_args"] = fallback_args

            parsed_cfg["spec"] = self._sanitize_compact_token(parsed_cfg["spec"])
            parsed_cfg["extra_preset"] = self._sanitize_compact_token(parsed_cfg["extra_preset"], fallback="base")

            swept_specs, swept_extras = self._extract_sweep_sets_from_summary(row_data)

            row = self.presets_table.rowCount()
            self.presets_table.insertRow(row)

            run_item = QTableWidgetItem(run_time)
            run_item.setData(Qt.ItemDataRole.UserRole, run_id)
            self.presets_table.setItem(row, 0, run_item)
            self.presets_table.setItem(row, 1, QTableWidgetItem(model_name or "-"))
            self.presets_table.setItem(row, 2, NumericTableWidgetItem(f"{aggregate_value:.4f}", aggregate_value))
            self.presets_table.setItem(row, 3, QTableWidgetItem(parsed_cfg["ctx"]))
            self.presets_table.setItem(row, 4, QTableWidgetItem(f"{parsed_cfg['batch']}/{parsed_cfg['ubatch']}"))
            self.presets_table.setItem(row, 5, QTableWidgetItem(parsed_cfg["kv"]))
            self.presets_table.setItem(row, 6, QTableWidgetItem(parsed_cfg["spec"]))
            self.presets_table.setItem(row, 7, QTableWidgetItem(parsed_cfg["extra_preset"]))
            self.presets_table.setItem(row, 8, QTableWidgetItem(parsed_cfg["extra_args"]))
            self.presets_table.setItem(row, 9, QTableWidgetItem(swept_specs))
            self.presets_table.setItem(row, 10, QTableWidgetItem(swept_extras))
            self.presets_table.setItem(row, 11, QTableWidgetItem(build_id or "-"))
            self.presets_table.setItem(row, 12, QTableWidgetItem(run_id or "-"))
            self.presets_table.setItem(row, 13, QTableWidgetItem(label or "-"))

    def delete_selected_preset(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Delete Run", "Select a run row in history first.")
            return

        model_name = Path(str(row_data.get("model", "")).strip()).name or "-"
        run_id = str(row_data.get("run_id", "")).strip() or "-"
        run_time = str(row_data.get("timestamp", "")).strip() or "-"
        label = str(row_data.get("label", "")).strip() or "-"

        confirm = QMessageBox.question(
            self,
            "Delete Run",
            f"Delete selected autotune run?\n\nModel: {model_name}\nRun ID: {run_id}\nTime: {run_time}\nLabel: {label}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        removed_total = 0
        removed_total += self._delete_run_from_history_file(self.history_csv_v2, row_data)
        removed_total += self._delete_run_from_history_file(self.history_csv, row_data)

        self.refresh_saved_presets_table()
        if removed_total > 0:
            self.status_label.setText(f"Run deleted from history: {run_id}")
        else:
            self.status_label.setText("Selected run was not found in history files")
            QMessageBox.warning(self, "Delete Run", "Selected run was not found in BENCH_HISTORY files.")

    def open_history_md(self):
        history_md = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY.md"
        if not history_md.exists():
            QMessageBox.warning(self, "History", "BENCH_HISTORY.md not found yet")
            return

        if os.name == "nt":
            os.startfile(str(history_md))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(history_md)])
