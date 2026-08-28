@echo off
REM Build RDNA LLM Studio as a standalone executable

echo ============================================================
echo Building RDNA LLM Studio Executable
echo ============================================================

python scripts\build_gui_exe.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build complete! You can find RDNA-LLM-Studio.exe in the project folder.
    echo.
) else (
    echo.
    echo Build failed. Check the output above for errors.
    echo.
)

pause
