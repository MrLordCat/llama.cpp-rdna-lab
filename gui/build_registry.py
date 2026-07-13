"""Build version registry for persistent GUI build tracking."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import csv
from pathlib import Path
from typing import Any


class BuildVersionRegistry:
    """Persistent registry for build versions and metadata."""

    def __init__(self, project_root: Path, registry_path: Path | None = None):
        self.project_root = Path(project_root)
        self.registry_path = registry_path or (self.project_root / "gui" / "build_versions.json")
        self._records: list[dict[str, Any]] = []
        self._load()

    @property
    def history_csv_path(self) -> Path:
        return self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY.csv"

    @property
    def history_csv_paths(self) -> list[Path]:
        history_dir = self.project_root / "build_logs" / "agent-workload"
        return [
            history_dir / "BENCH_HISTORY.csv",
            history_dir / "BENCH_HISTORY_V2.csv",
        ]

    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                records = data.get("builds", []) if isinstance(data, dict) else []
                if isinstance(records, list):
                    self._records = [self._normalize_record(r) for r in records if isinstance(r, dict)]
            except Exception:
                # Do not overwrite a potentially valid registry on transient read/parse errors.
                self._records = []
            return

        # Initialize registry file on first run only.
        self._save()

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "builds": self._records,
        }
        self.registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _now() -> str:
        return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _timestamp_from_mtime(timestamp: float) -> str:
        return dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _make_id(build_dir: Path) -> str:
        stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        digest = hashlib.sha1(str(build_dir).encode("utf-8", errors="replace")).hexdigest()[:8]
        return f"bld-{stamp}-{digest}"

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        build_dir = str(record.get("build_dir", "")).strip()
        if build_dir:
            build_dir = str(Path(build_dir))
        normalized = {
            "id": str(record.get("id", "")).strip(),
            "name": str(record.get("name", "")).strip() or Path(build_dir).name,
            "backend": str(record.get("backend", "cpu")).strip().lower(),
            "source_type": str(record.get("source_type", "fork")).strip() or "fork",
            "source_ref": str(record.get("source_ref", "")).strip(),
            "build_dir": build_dir,
            "server_bin": str(record.get("server_bin", "")).strip(),
            "toolchain": record.get("toolchain", {}) if isinstance(record.get("toolchain"), dict) else {},
            "created_at": str(record.get("created_at", "")).strip() or self._now(),
            "updated_at": str(record.get("updated_at", "")).strip() or self._now(),
            "status": str(record.get("status", "ready")).strip() or "ready",
            "notes": str(record.get("notes", "")).strip(),
            "bench_best_non_mtp_tps": record.get("bench_best_non_mtp_tps"),
            "bench_best_mtp_tps": record.get("bench_best_mtp_tps"),
            "bench_last_run_at": str(record.get("bench_last_run_at", "")).strip(),
        }

        if not normalized["id"]:
            normalized["id"] = self._make_id(Path(normalized["build_dir"]) if normalized["build_dir"] else Path("build"))
        return normalized

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _detect_backend_from_cache(build_dir: Path) -> str:
        cache = build_dir / "CMakeCache.txt"
        if not cache.exists():
            return "cpu"
        try:
            content = cache.read_text(errors="ignore")
            if "GGML_HIP:BOOL=ON" in content or "GGML_ROCM:BOOL=ON" in content:
                return "rocm"
            if "GGML_VULKAN:BOOL=ON" in content:
                return "vulkan"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _find_server_bin(build_dir: Path) -> str:
        candidates = [
            build_dir / "bin" / "llama-server.exe",
            build_dir / "bin" / "Release" / "llama-server.exe",
            build_dir / "bin" / "Debug" / "llama-server.exe",
            build_dir / "bin" / "llama-server",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return ""

    def _get_repo_short_ref(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def list_builds(self) -> list[dict[str, Any]]:
        return list(self._records)

    def get_effective_build_timestamp(self, record: dict[str, Any]) -> str:
        server_bin_text = str(record.get("server_bin", "")).strip()
        if server_bin_text:
            server_bin = Path(server_bin_text)
            if server_bin.exists():
                try:
                    return self._timestamp_from_mtime(server_bin.stat().st_mtime)
                except OSError:
                    pass

        build_dir_text = str(record.get("build_dir", "")).strip()
        if build_dir_text:
            build_dir = Path(build_dir_text)
            if build_dir.exists():
                candidate = self._find_server_bin(build_dir)
                if candidate:
                    candidate_path = Path(candidate)
                    if candidate_path.exists():
                        try:
                            return self._timestamp_from_mtime(candidate_path.stat().st_mtime)
                        except OSError:
                            pass

                cache = build_dir / "CMakeCache.txt"
                if cache.exists():
                    try:
                        return self._timestamp_from_mtime(cache.stat().st_mtime)
                    except OSError:
                        pass

        return str(record.get("created_at", "")).strip() or str(record.get("updated_at", "")).strip()

    def detect_build_id_from_server_bin(self, server_bin: str) -> str:
        server_bin_path = Path(server_bin).resolve()
        for record in self._records:
            build_dir_text = str(record.get("build_dir", "")).strip()
            if not build_dir_text:
                continue
            build_dir = Path(build_dir_text)
            if not build_dir.exists():
                continue
            try:
                if server_bin_path.is_relative_to(build_dir.resolve()):
                    return str(record.get("id", ""))
            except Exception:
                # Fallback for Python versions without is_relative_to behavior in edge paths.
                if str(server_bin_path).lower().startswith(str(build_dir.resolve()).lower()):
                    return str(record.get("id", ""))
        return ""

    def get_by_id(self, build_id: str) -> dict[str, Any] | None:
        for record in self._records:
            if record.get("id") == build_id:
                return record
        return None

    def get_by_dir(self, build_dir: Path) -> dict[str, Any] | None:
        target = str(Path(build_dir))
        for record in self._records:
            if str(record.get("build_dir", "")) == target:
                return record
        return None

    def upsert(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_record(record)
        existing = self.get_by_id(normalized["id"])
        if existing is None and normalized.get("build_dir"):
            existing = self.get_by_dir(Path(normalized["build_dir"]))

        if existing is None:
            self._records.append(normalized)
            self._save()
            return normalized

        existing.update(normalized)
        existing["updated_at"] = self._now()
        self._save()
        return existing

    def remove_by_id(self, build_id: str) -> None:
        self._records = [r for r in self._records if r.get("id") != build_id]
        self._save()

    def sync_with_existing_builds(self) -> int:
        imported = 0
        changed = False
        source_ref = self._get_repo_short_ref()

        for item in sorted(self.project_root.iterdir()):
            if not item.is_dir():
                continue
            if not (item.name == "build" or item.name.startswith("build-")):
                continue
            if not (item / "CMakeCache.txt").exists():
                continue

            if self.get_by_dir(item) is not None:
                continue

            backend = self._detect_backend_from_cache(item)
            self._records.append(
                self._normalize_record(
                    {
                        "id": self._make_id(item),
                        "name": item.name,
                        "backend": backend,
                        "source_type": "fork",
                        "source_ref": source_ref,
                        "build_dir": str(item),
                        "server_bin": self._find_server_bin(item),
                        "toolchain": {},
                        "created_at": self._now(),
                        "updated_at": self._now(),
                        "status": "ready",
                        "notes": "Imported from existing build directory",
                        "bench_best_non_mtp_tps": None,
                        "bench_best_mtp_tps": None,
                        "bench_last_run_at": "",
                    }
                )
            )
            imported += 1
            changed = True

        # Refresh server binary/status for existing records.
        for record in self._records:
            build_dir_text = str(record.get("build_dir", ""))
            if not build_dir_text:
                continue
            build_dir = Path(build_dir_text)
            old_server_bin = str(record.get("server_bin", ""))
            old_status = str(record.get("status", ""))

            if build_dir.exists():
                new_server_bin = self._find_server_bin(build_dir)
                if new_server_bin:
                    new_status = "archived" if old_status == "archived" and old_server_bin else "ready"
                else:
                    new_status = "archived"
            else:
                new_server_bin = ""
                new_status = "archived"

            if old_server_bin != new_server_bin or old_status != new_status:
                record["server_bin"] = new_server_bin
                record["status"] = new_status
                if new_server_bin:
                    record["backend"] = self._detect_backend_from_cache(build_dir)
                    record["source_ref"] = source_ref
                record["updated_at"] = self._now()
                changed = True

        if changed:
            self._save()
        return imported

    def update_benchmark_stats_from_history(self, persist: bool = True) -> int:
        """Refresh per-build benchmark maxima from available history CSV files."""
        history_paths = [path for path in self.history_csv_paths if path.exists()]
        if not history_paths:
            return 0

        history_rows: list[dict[str, str]] = []
        for history_path in history_paths:
            try:
                with history_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if isinstance(row, dict):
                            history_rows.append({k: str(v) for k, v in row.items()})
            except Exception:
                continue

        if not history_rows:
            return 0

        updated = 0
        for record in self._records:
            build_id = str(record.get("id", ""))
            best_non = None
            best_mtp = None
            last_ts = ""
            matched_history = False

            for row in history_rows:
                if str(row.get("build_id", "")) != build_id:
                    continue
                if str(row.get("errors", "0")) not in ("", "0"):
                    continue
                matched_history = True

                tps = self._to_float(row.get("aggregate_tps", 0.0), 0.0)
                model_is_mtp = str(row.get("is_mtp_model", "0")) == "1"
                if model_is_mtp:
                    if best_mtp is None or tps > best_mtp:
                        best_mtp = tps
                else:
                    if best_non is None or tps > best_non:
                        best_non = tps

                ts = str(row.get("timestamp", ""))
                if ts > last_ts:
                    last_ts = ts

            if not matched_history:
                continue

            new_non = f"{best_non:.4f}" if best_non is not None else None
            new_mtp = f"{best_mtp:.4f}" if best_mtp is not None else None
            if (
                str(record.get("bench_best_non_mtp_tps")) != str(new_non)
                or str(record.get("bench_best_mtp_tps")) != str(new_mtp)
                or str(record.get("bench_last_run_at", "")) != last_ts
            ):
                record["bench_best_non_mtp_tps"] = new_non
                record["bench_best_mtp_tps"] = new_mtp
                record["bench_last_run_at"] = last_ts
                updated += 1

        if updated and persist:
            self._save()
        return updated

    def rename_build(self, build_id: str, new_name: str, new_dir: Path) -> bool:
        record = self.get_by_id(build_id)
        if record is None:
            return False
        record["name"] = new_name
        record["build_dir"] = str(new_dir)
        record["server_bin"] = self._find_server_bin(new_dir)
        record["updated_at"] = self._now()
        self._save()
        return True
