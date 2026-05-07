"""
Background worker threads for llama.cpp GUI

Provides thread implementations for:
- Running llama-server in background
- Running inference operations
- Updating fork from upstream
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal


class ServerThread(QThread):
    """Thread for running llama-server in background"""
    output_ready = pyqtSignal(str)
    server_ready = pyqtSignal(str)  # Server URL
    finished_signal = pyqtSignal(int)  # exit code
    error_signal = pyqtSignal(str)
    
    def __init__(self, command: list, working_dir: str, port: int = 8080, env: dict = None):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.process = None
        self.port = port
        self.env = env
        self.user_stopped = False
        self._output_line_count = 0
        
    def run(self):
        try:
            # Build environment
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)
            
            # On Linux: lower CPU scheduler priority so desktop stays responsive under GPU load
            _preexec = (lambda: os.nice(10)) if os.name != 'nt' else None

            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.working_dir,
                bufsize=1,
                universal_newlines=True,
                env=process_env,
                preexec_fn=_preexec
            )
            
            server_started = False
            for line in self.process.stdout:
                self._output_line_count += 1
                self.output_ready.emit(line)
                
                # Detect when server is ready
                if not server_started and ("HTTP server listening" in line or "server is listening" in line):
                    self.server_ready.emit(f"http://localhost:{self.port}")
                    server_started = True
                
            self.process.wait()
            self.finished_signal.emit(self.process.returncode)
            
        except Exception as e:
            self.error_signal.emit(str(e))
            
    def stop(self):
        """Stop the server thread"""
        self.user_stopped = True
        if self.process:
            self.process.terminate()
            self.process.wait()


class InferenceThread(QThread):
    """Thread for running inference in background"""
    output_ready = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, command: list, working_dir: str):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.process = None
        
    def run(self):
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.working_dir,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in self.process.stdout:
                self.output_ready.emit(line)
                
            self.process.wait()
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))
            
    def stop(self):
        """Stop the inference thread"""
        if self.process:
            self.process.terminate()
            self.process.wait()


class UpdateForkThread(QThread):
    """Thread for updating the fork from upstream in background"""
    output = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # success, summary message

    def __init__(self, repo_path: Path):
        super().__init__()
        self.repo_path = repo_path

    def _run_git(self, args: list) -> Tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)"""
        result = subprocess.run(
            ["git"] + args,
            cwd=self.repo_path,
            capture_output=True, text=True, timeout=120
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def run(self):
        try:
            # Check upstream remote
            self.output.emit("🔍 Checking upstream remote...\n")
            rc, out, err = self._run_git(["remote", "-v"])
            if "upstream" not in out:
                self.output.emit("⚙️ Adding upstream remote (ggml-org/llama.cpp)...\n")
                rc, out, err = self._run_git(["remote", "add", "upstream", "https://github.com/ggml-org/llama.cpp.git"])
                if rc != 0:
                    self.finished_signal.emit(False, f"Failed to add upstream: {err}")
                    return

            # Fetch upstream
            self.output.emit("📡 Fetching upstream/master...\n")
            rc, out, err = self._run_git(["fetch", "upstream", "master"])
            if rc != 0:
                self.finished_signal.emit(False, f"Fetch failed: {err}")
                return

            # Check how far behind
            rc, out, err = self._run_git(["rev-list", "--count", "HEAD..upstream/master"])
            commits_behind = int(out) if out.isdigit() else 0
            if commits_behind == 0:
                self.output.emit("✅ Already up to date!\n")
                self.finished_signal.emit(True, "Already up to date — no new commits.")
                return

            self.output.emit(f"📦 {commits_behind} new commits to merge...\n")

            # Merge
            self.output.emit("🔀 Merging upstream/master...\n")
            rc, out, err = self._run_git(["merge", "upstream/master", "--no-edit"])
            merge_output = out + "\n" + err
            self.output.emit(merge_output + "\n")

            if rc != 0:
                # Check for conflicts
                rc2, conflicts, _ = self._run_git(["diff", "--name-only", "--diff-filter=U"])
                if conflicts:
                    self.output.emit(f"\n⚠️ Merge conflicts in:\n{conflicts}\n")
                    self.output.emit("\n🔧 Resolving: removing .github/workflows conflicts...\n")
                    # Auto-resolve workflow file conflicts (we don't use them in our fork)
                    conflict_files = conflicts.split("\n")
                    workflow_conflicts = [f for f in conflict_files if f.startswith(".github/")]
                    other_conflicts = [f for f in conflict_files if not f.startswith(".github/")]

                    for wf in workflow_conflicts:
                        self._run_git(["rm", wf])
                        self.output.emit(f"  🗑️ Removed {wf}\n")

                    if other_conflicts:
                        self.output.emit(f"\n❌ {len(other_conflicts)} conflict(s) need manual resolution:\n")
                        for f in other_conflicts:
                            self.output.emit(f"  ⚠️ {f}\n")
                        self.output.emit("\nResolve them manually, then run 'git add' and 'git commit'.\n")
                        self.finished_signal.emit(False,
                            f"Merged {commits_behind} commits but {len(other_conflicts)} conflict(s) need manual fix.")
                        return

                    # All conflicts were workflows — commit
                    self._run_git(["commit", "--no-edit"])
                    self.output.emit("✅ Auto-resolved workflow conflicts.\n")
                else:
                    self.finished_signal.emit(False, f"Merge failed: {err}")
                    return

            # Push
            self.output.emit("📤 Pushing to origin...\n")
            rc, out, err = self._run_git(["push", "origin", "master"])
            if rc != 0:
                self.output.emit(f"⚠️ Push failed: {err}\n")
                self.output.emit("You can push manually later with: git push origin master\n")
                self.finished_signal.emit(True,
                    f"Merged {commits_behind} commits but push failed. Push manually.")
                return

            self.output.emit("✅ Successfully pushed to origin!\n")
            self.finished_signal.emit(True, f"Updated: {commits_behind} commits merged and pushed.")

        except subprocess.TimeoutExpired:
            self.finished_signal.emit(False, "Git command timed out. Check your network.")
        except FileNotFoundError:
            self.finished_signal.emit(False, "Git not found. Install Git and add it to PATH.")
        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}")
