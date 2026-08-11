#!/usr/bin/env python3
"""
Build script for creating a standalone RDNA LLM Studio executable
Uses PyInstaller to create a single .exe file
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True

def build_exe(windowed=False):
    """Build the executable"""
    project_root = Path(__file__).parent
    gui_dir = project_root / "gui"
    dist_dir = project_root / "dist"

    # Main script
    main_script = project_root / "run.py"

    if not main_script.exists():
        print(f"❌ Main script not found: {main_script}")
        return False

    mode = "RELEASE (no console)" if windowed else "DEBUG (with console)"
    print("=" * 60)
    print(f"🔨 Building RDNA LLM Studio executable - {mode}")
    print("=" * 60)

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "RDNA-LLM-Studio",
        "--onefile",  # Single executable
        "--noconfirm",  # Overwrite without asking
    ]

    # Add windowed flag for release builds
    if windowed:
        cmd.append("--windowed")
    else:
        cmd.append("--console")

    # Add paths so Python can find the modules
    cmd.extend(["--paths", str(project_root), "--paths", str(gui_dir)])

    # Collect all submodules for huggingface_hub
    cmd.extend(["--collect-all", "huggingface_hub"])

    # Bundle theme SVG assets (spinbox/combobox chevrons) next to the modules so
    # gui_theme's __file__-relative asset path resolves inside the frozen app
    assets_dir = gui_dir / "assets"
    if assets_dir.is_dir():
        cmd.extend(["--add-data", f"{assets_dir}{os.pathsep}assets"])

    # Hidden imports that PyInstaller might miss
    hidden_imports = [
        "PyQt6.QtWidgets",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "main_window",
        "gui_theme",
        "server_tab",
        "inference_tab",
        "download_tab",
        "build_tab",
        "benchmark_tab",
        "builds_info_tab",
        "hardware_tab",
        "threads",
        "project_utils",
        "build_registry",
        "gui_api",
        "requests",
        "huggingface_hub",
        "psutil",
        "tqdm",
        "model_downloader",
        "hardware_detector",
        "build_manager",
        "dependency_installer",
        "dependency_checker",
    ]

    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # Output directories
    cmd.extend([
        "--distpath", str(dist_dir),
        "--workpath", str(project_root / "build_temp"),
        "--specpath", str(project_root),
    ])

    # Main script
    cmd.append(str(main_script))

    print(f"📦 Running PyInstaller...")
    print()

    try:
        subprocess.check_call(cmd)

        exe_path = dist_dir / "RDNA-LLM-Studio.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print()
            print("=" * 60)
            print(f"✅ Build successful!")
            print(f"📁 Executable: {exe_path}")
            print(f"📊 Size: {size_mb:.1f} MB")
            print("=" * 60)
            print()

            if not windowed:
                print("📝 NOTE: Built with console window for debugging.")
                print("   Run 'python build_exe.py --release' for windowless version.")
                print()

            # Copy exe to project root for convenience
            root_exe = project_root / "RDNA-LLM-Studio.exe"
            shutil.copy2(exe_path, root_exe)
            print(f"📋 Copied to: {root_exe}")

            return True
        else:
            print("❌ Build failed - executable not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False

def clean_build():
    """Clean build artifacts"""
    project_root = Path(__file__).parent

    dirs_to_clean = [
        project_root / "build_temp",
        project_root / "dist",
    ]

    files_to_clean = [
        project_root / "RDNA-LLM-Studio.spec",
    ]

    for d in dirs_to_clean:
        if d.exists():
            print(f"🧹 Removing {d}")
            shutil.rmtree(d)

    for f in files_to_clean:
        if f.exists():
            print(f"🧹 Removing {f}")
            f.unlink()

    print("✅ Clean complete")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RDNA LLM Studio executable")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--release", action="store_true", help="Build release version (no console)")
    args = parser.parse_args()

    if args.clean:
        clean_build()
    else:
        check_pyinstaller()
        build_exe(windowed=args.release)
