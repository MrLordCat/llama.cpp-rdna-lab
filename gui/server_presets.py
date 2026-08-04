"""Preset & speculative-decoding profile logic for the server launch tab.

ServerPresetsMixin is mixed into ServerTabWidget; methods operate on the host
widget via self (spec/ngram controls, model path field, preset combos, log).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path


class ServerPresetsMixin:
    """Model-preset application + speculative profile handling."""

    def on_spec_type_changed(self):
        """Handle speculative type change"""
        spec_type = self.server_spec_type_combo.currentText()
        is_ngram = spec_type == "ngram-mod"
        uses_draft_n = spec_type in ("mtp", "draft")
        if is_ngram:
            self._apply_ngram_mod_profile()
        if spec_type == "mtp":
            self._apply_mtp_profile()
        if hasattr(self, "ngram_layout_group"):
            # Enable/disable ngram widgets
            for i in range(self.ngram_layout_group.count()):
                widget = self.ngram_layout_group.itemAt(i).widget()
                if widget:
                    widget.setEnabled(is_ngram)
        if hasattr(self, "draft_layout_group"):
            for i in range(self.draft_layout_group.count()):
                widget = self.draft_layout_group.itemAt(i).widget()
                if widget:
                    widget.setEnabled(uses_draft_n)

    def _apply_mtp_profile(self):
        """Apply the measured E266 ROCm MTP profile."""
        self.server_spec_draft_n_max.setValue(self.MTP_DRAFT_N_MAX)

    def _apply_ngram_mod_profile(self):
        """Apply the measured E226 ROCm repeated/session ngram profile."""
        self.server_ngram_min.setValue(self.NGRAM_MOD_N_MIN)
        self.server_ngram_match.setValue(self.NGRAM_MOD_N_MATCH)
        self.server_ngram_max.setValue(self.NGRAM_MOD_N_MAX)

    def _ngram_mod_args(self) -> list[str]:
        return [
            "--spec-type", "ngram-mod",
            "--spec-ngram-mod-n-min", str(self.NGRAM_MOD_N_MIN),
            "--spec-ngram-mod-n-match", str(self.NGRAM_MOD_N_MATCH),
            "--spec-ngram-mod-n-max", str(self.NGRAM_MOD_N_MAX),
        ]

    @staticmethod
    def _spec_type_from_tokens(tokens: list[str]) -> str | None:
        for tok in tokens:
            if tok.startswith("--spec-type="):
                return tok.split("=", 1)[1].strip().lower()
        for i, tok in enumerate(tokens[:-1]):
            if tok == "--spec-type":
                return tokens[i + 1].strip().lower()
        return None

    def _normalize_ngram_extra_tokens(self, tokens: list[str]) -> list[str]:
        skip_value_for = {
            "--spec-type",
            "--spec-ngram-mod-n-min",
            "--spec-ngram-mod-n-match",
            "--spec-ngram-mod-n-max",
        }
        normalized = []
        skip_next = False
        for tok in tokens:
            if skip_next:
                skip_next = False
                continue
            if tok in skip_value_for:
                skip_next = True
                continue
            normalized.append(tok)
        normalized.extend(self._ngram_mod_args())
        return normalized

    def _load_model_presets(self) -> list[dict]:
        """Load model presets from JSON file used by legacy GUI."""
        preset_path = self.parent.project_root / "gui" / "model_presets.json"
        if not preset_path.exists():
            return []
        try:
            data = json.loads(preset_path.read_text(encoding="utf-8"))
            presets = data.get("presets", [])
            return [p for p in presets if isinstance(p, dict)]
        except Exception:
            return []

    def _strip_spec_from_extra_tokens(self, tokens: list[str]) -> list[str]:
        """Remove --spec-type, --spec-draft-n-max and their values from token list."""
        skip_value_for = {
            "--spec-type",
            "--spec-draft-n-max",
        }
        cleaned = []
        skip_next = False
        for tok in tokens:
            if skip_next:
                skip_next = False
                continue
            if tok in skip_value_for:
                skip_next = True
                continue
            # Handle --flag=value form
            if any(tok.startswith(flag + "=") for flag in skip_value_for):
                continue
            cleaned.append(tok)
        return cleaned

    def apply_model_file_preset(self):
        """Apply first regex-matching model preset from model_presets.json."""
        self._model_presets = self._load_model_presets()
        model_path = self.server_model_path.text().strip()
        if not model_path:
            return {"matched": False, "reason": "empty-model-path"}
        model_name = Path(model_path).name

        match = None
        for preset in self._model_presets:
            pattern = preset.get("pattern")
            if not pattern:
                continue
            try:
                if re.search(pattern, model_name, flags=re.IGNORECASE):
                    match = preset
                    break
            except re.error:
                continue

        if not match:
            # No preset for this model — clear stale spec state from a previous
            # session/preset so the user's visible UI choices take effect.
            self.server_spec_type_combo.setCurrentText("None")
            self.on_spec_type_changed()
            # Also strip stale --spec-type / --spec-draft-n-max from extra_args
            extra_text = self.server_extra_args.toPlainText().strip()
            if extra_text:
                try:
                    tokens = shlex.split(extra_text, posix=(os.name != "nt"))
                except ValueError:
                    tokens = extra_text.split()
                cleaned = self._strip_spec_from_extra_tokens(tokens)
                self.server_extra_args.setPlainText(" ".join(cleaned))
            return {"matched": False, "reason": "no-preset-match", "model": model_name}

        # Reset spec type to None before applying preset to avoid stale MTP/spec state
        self.server_spec_type_combo.setCurrentText("None")
        self.on_spec_type_changed()

        if "ctx" in match:
            self.server_context_spinbox.setValue(int(match["ctx"]))
        if "batch_size" in match:
            self.server_batch_spinbox.setValue(int(match["batch_size"]))
        if "ubatch_size" in match:
            self.server_ubatch_spinbox.setValue(int(match["ubatch_size"]))
        if "gpu_layers" in match:
            # presets use -1 (and 999) for "all layers"; the spinbox would clamp
            # -1 to 0 and silently turn a GPU preset into a CPU run
            gpu_layers = int(match["gpu_layers"])
            self.server_gpu_layers_spinbox.setValue(999 if gpu_layers < 0 else gpu_layers)
        if "parallel" in match:
            self.server_parallel_spinbox.setValue(int(match["parallel"]))
        if "flash_attn" in match:
            self.server_flash_attn_check.setChecked(bool(match["flash_attn"]))
        if "no_mmap" in match:
            self.server_no_mmap_check.setChecked(bool(match["no_mmap"]))
        if "disable_thinking" in match:
            self.server_disable_thinking_check.setChecked(bool(match["disable_thinking"]))
        if "extra_args" in match:
            self.server_extra_args.setPlainText(str(match["extra_args"]).strip())
            self._apply_spec_controls_from_extra_args(str(match["extra_args"]).strip())
        if "spec_type" in match:
            spec_type = str(match["spec_type"]).strip().lower()
            if spec_type in {"mtp", "draft-mtp"}:
                self.server_spec_type_combo.setCurrentText("mtp")
            elif spec_type == "ngram-mod":
                self.server_spec_type_combo.setCurrentText("ngram-mod")
            elif spec_type == "draft":
                self.server_spec_type_combo.setCurrentText("draft")
            elif spec_type in {"none", ""}:
                self.server_spec_type_combo.setCurrentText("None")
            self.on_spec_type_changed()
        if "spec_draft_n_max" in match:
            try:
                self.server_spec_draft_n_max.setValue(max(1, min(20, int(match["spec_draft_n_max"]))))
            except (TypeError, ValueError):
                pass

        kv_map = {
            0: "f16",
            1: "bf16",
            2: "f32",
            3: "q8_0",
            4: "q5_1",
            5: "q5_0",
            6: "q4_1",
            7: "q4_0",
            8: "iq4_nl",
        }
        kv_value = match.get("kv_cache")
        if isinstance(kv_value, int) and kv_value in kv_map:
            kv_text = kv_map[kv_value]
            idx = self.server_kv_type_combo.findText(kv_text)
            if idx >= 0:
                self.server_kv_type_combo.setCurrentIndex(idx)

        if "notes" in match:
            self.server_log.append(f"[INFO] Applied model preset: {match.get('name', 'Unnamed')} - {match['notes']}")

        return {
            "matched": True,
            "model": model_name,
            "preset_name": match.get("name", "Unnamed"),
            "context": self.server_context_spinbox.value(),
            "batch": self.server_batch_spinbox.value(),
            "ubatch": self.server_ubatch_spinbox.value(),
            "parallel": self.server_parallel_spinbox.value(),
            "kv": self.server_kv_type_combo.currentText(),
            "flash_attn": self.server_flash_attn_check.isChecked(),
            "extra_args": self.server_extra_args.toPlainText().strip(),
        }

    def _apply_spec_controls_from_extra_args(self, extra_args: str):
        """Parse known speculative flags from preset extra args and sync visible controls."""
        if not extra_args:
            return

        try:
            tokens = shlex.split(extra_args, posix=(os.name != "nt"))
        except ValueError:
            tokens = extra_args.split()

        spec_type = None
        ngram_min = None
        ngram_max = None
        ngram_match = None
        mtp_draft_n_max = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if tok.startswith("--spec-type="):
                spec_type = tok.split("=", 1)[1].strip().lower()
                i += 1
                continue
            if tok == "--spec-type" and nxt is not None:
                spec_type = nxt.strip().lower()
                i += 2
                continue
            if tok == "--spec-ngram-mod-n-min" and nxt is not None:
                ngram_min = nxt
                i += 2
                continue
            if tok == "--spec-ngram-mod-n-max" and nxt is not None:
                ngram_max = nxt
                i += 2
                continue
            if tok == "--spec-ngram-mod-n-match" and nxt is not None:
                ngram_match = nxt
                i += 2
                continue
            if tok == "--spec-draft-n-max" and nxt is not None:
                mtp_draft_n_max = nxt
                i += 2
                continue
            i += 1

        if spec_type == "ngram-mod":
            self.server_spec_type_combo.setCurrentText("ngram-mod")
            self._apply_ngram_mod_profile()
        elif spec_type in {"mtp", "draft-mtp"}:
            self.server_spec_type_combo.setCurrentText("mtp")
        elif spec_type == "draft":
            self.server_spec_type_combo.setCurrentText("draft")
        elif spec_type == "none":
            self.server_spec_type_combo.setCurrentText("None")

        if spec_type == "ngram-mod":
            ngram_min = ngram_max = ngram_match = None

        if ngram_min is not None:
            try:
                self.server_ngram_min.setValue(max(1, min(512, int(ngram_min))))
            except ValueError:
                pass
        if ngram_max is not None:
            try:
                self.server_ngram_max.setValue(max(1, min(512, int(ngram_max))))
            except ValueError:
                pass
        if ngram_match is not None:
            try:
                self.server_ngram_match.setValue(max(1, min(512, int(ngram_match))))
            except ValueError:
                pass
        if mtp_draft_n_max is not None:
            try:
                self.server_spec_draft_n_max.setValue(max(1, min(20, int(mtp_draft_n_max))))
            except ValueError:
                pass

        self.on_spec_type_changed()
