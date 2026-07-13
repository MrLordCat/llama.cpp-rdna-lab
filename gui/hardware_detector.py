"""Hardware detection for the supported CPU, Vulkan and ROCm backends."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import psutil


class HardwareDetector:
    """Collect the hardware facts used by the GUI build recommendations."""

    def __init__(self) -> None:
        self.os_type = platform.system()

    @staticmethod
    def _run(command: List[str], timeout: int = 5) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def get_cpu_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "name": platform.processor() or "Unknown CPU",
            "cores": psutil.cpu_count(logical=False) or 1,
            "threads": psutil.cpu_count(logical=True) or 1,
            "frequency": None,
        }

        try:
            frequency = psutil.cpu_freq()
            if frequency:
                info["frequency"] = f"{frequency.max:.0f} MHz"
        except (OSError, AttributeError):
            pass

        if self.os_type == "Windows":
            result = self._run([
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).Name",
            ])
            if result and result.returncode == 0 and result.stdout.strip():
                info["name"] = result.stdout.strip()
        elif self.os_type == "Linux":
            try:
                for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                    if line.startswith("model name"):
                        info["name"] = line.split(":", 1)[1].strip()
                        break
            except OSError:
                pass

        return info

    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "total_gb": memory.total / (1024 ** 3),
            "available": memory.available,
            "available_gb": memory.available / (1024 ** 3),
            "percent_used": memory.percent,
        }

    @staticmethod
    def _gpu_entry(name: str) -> Dict[str, Any]:
        normalized = name.upper()
        is_amd = any(marker in normalized for marker in ("AMD", "ATI", "RADEON", "NAVI"))
        entry: Dict[str, Any] = {
            "name": name,
            "type": "AMD" if is_amd else "Other",
            "backend": "ROCm or Vulkan" if is_amd else "Vulkan",
        }
        if is_amd and any(marker in normalized for marker in ("9070", "9080", "9050", "NAVI 4")):
            entry["is_rdna4"] = True
            entry["is_9070xt"] = "9070" in normalized
        return entry

    def get_gpu_info(self) -> List[Dict[str, Any]]:
        names: List[str] = []

        if self.os_type == "Windows":
            result = self._run([
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name",
            ], timeout=10)
            if result and result.returncode == 0:
                names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        elif self.os_type == "Linux":
            result = self._run(["lspci"])
            if result and result.returncode == 0:
                for line in result.stdout.splitlines():
                    if any(kind in line for kind in ("VGA", "3D", "Display")):
                        names.append(line.split(": ", 1)[-1].strip())

        unique_names = list(dict.fromkeys(names))
        return [self._gpu_entry(name) for name in unique_names]

    @staticmethod
    def recommend_backend(gpu_info: List[Dict[str, Any]]) -> str:
        if any(gpu.get("type") == "AMD" for gpu in gpu_info):
            return "Vulkan or ROCm (recommended for AMD GPU)"
        if gpu_info:
            return "Vulkan"
        return "CPU (GPU not detected)"

    def get_hardware_info(self) -> Dict[str, Any]:
        gpu_info = self.get_gpu_info()
        return {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "gpu": gpu_info,
            "recommended_backend": self.recommend_backend(gpu_info),
            "rocm_available": self._check_rocm(),
            "vulkan_available": self._check_vulkan_sdk(),
        }

    def check_dependencies(self, backend: str) -> Dict[str, bool]:
        checks = {
            "cmake": self._check_command("cmake"),
            "git": self._check_command("git"),
        }
        backend_lower = backend.lower()
        if "vulkan" in backend_lower:
            checks["vulkan_sdk"] = self._check_vulkan_sdk()
        elif "rocm" in backend_lower or "hip" in backend_lower:
            checks["rocm"] = self._check_rocm()
        return checks

    def _check_command(self, command: str) -> bool:
        result = self._run([command, "--version"], timeout=3)
        return bool(result and result.returncode == 0)

    def _check_vulkan_sdk(self) -> bool:
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk:
            return Path(vulkan_sdk).exists()
        if self.os_type == "Windows":
            program_files = os.environ.get("PROGRAMFILES", "C:/Program Files")
            return (Path(program_files) / "VulkanSDK").exists()
        if self.os_type == "Linux":
            return Path("/usr/share/vulkan").exists()
        return False

    def _check_rocm(self) -> bool:
        candidates = [os.environ.get("ROCM_PATH", ""), os.environ.get("HIP_PATH", "")]
        if self.os_type == "Windows":
            candidates.extend(["C:/Program Files/AMD/ROCm", "C:/Program Files (x86)/AMD/ROCm"])
        elif self.os_type == "Linux":
            candidates.extend(["/opt/rocm", "/usr"])
        return any(path and Path(path).exists() for path in candidates)

    def is_rocm_available(self) -> bool:
        return self._check_rocm()

    def is_vulkan_available(self) -> bool:
        return self._check_vulkan_sdk()


if __name__ == "__main__":
    import json

    print(json.dumps(HardwareDetector().get_hardware_info(), indent=2, ensure_ascii=False))
