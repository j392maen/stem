@echo off
rem stemapp server start (double-click to run). Stop with Ctrl+C or close the window.
cd /d "%~dp0.."
where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Run scripts\setup.ps1 first.
    pause
    exit /b 1
)
echo Starting stemapp ... http://127.0.0.1:8000/
uv run stemapp serve
if errorlevel 1 pause
