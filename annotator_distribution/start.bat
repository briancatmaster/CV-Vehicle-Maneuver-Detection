@echo off
REM Vehicle Annotator launcher (Windows)
REM Double-click this file to start.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo ============================================================
  echo Python is not installed.
  echo Please install it from https://www.python.org/downloads/
  echo During install, check "Add Python to PATH".
  echo Then double-click this file again.
  echo ============================================================
  pause
  exit /b 1
)

echo Starting Vehicle Annotator...
python launch_annotator.py

pause
