@echo off
REM Launch GUI 2.0 - the local web UI (http://127.0.0.1:8770 by default)

python -c "import fasthtml, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing GUI 2.0 dependencies...
    pip install -r gui2\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Could not install dependencies
        pause
        exit /b 1
    )
)

python -m gui2
if errorlevel 1 (
    echo.
    echo [ERROR] GUI 2.0 failed to start
    pause
)
