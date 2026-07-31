@echo off
title CairnIQ Console Launcher
cls

echo.
echo   =============================================================
echo                    C A I R N I Q   L A U N C H E R
echo                    Private Portfolio Intelligence
echo   =============================================================
echo.

echo [SYSTEM CHECK] Initializing...

rem Check if virtual environment exists
if not exist .venv (
    echo [ERROR] Virtual environment not found.
    echo Please run the installation steps first.
    echo.
    pause
    exit /b 1
)

echo Virtual environment: Active
call .venv\Scripts\activate

rem Get python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo Python: v%PY_VER%

rem Check FAISS
python -c "import faiss" >nul 2>&1
if %errorlevel% equ 0 (
    echo FAISS Vector Search: Enabled
) else (
    echo FAISS Vector Search: Fallback Mode
)

echo.
echo [LAUNCH] Starting CairnIQ...
echo -------------------------------------------------------------
echo.
echo   [*] Server:      http://localhost:8000
echo   [*] Dashboard:   http://localhost:8000/dashboard
echo   [*] Chat:        http://localhost:8000/
echo.
echo   Press Ctrl+C to stop the server
echo.

rem Reset environment overrides to run in Production mode
set DEMO_MODE=
set CAIRNIQ_FORCE_DEMO=
set DEMO_PROFILE=
set DEMO_RESET=

echo Mode: Production (Your Account)
echo.

rem Auto-open browser
start "" http://localhost:8000

rem Start the server
python server.py

echo.
echo [SHUTDOWN] Server stopped
pause
