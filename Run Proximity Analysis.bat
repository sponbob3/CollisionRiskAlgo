@echo off
rem Windows: double-click in File Explorer to start the proximity-risk
rem analysis. Opens a console window and asks which dataset and outputs to
rem run (same as "python run_proximity.py" with no arguments). Uses the
rem .venv environment created in README step 3 when it exists.
cd /d "%~dp0"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (where py >nul 2>nul && set "PY=py -3.12")
if not defined PY set "PY=python"
%PY% run_proximity.py
if errorlevel 1 pause
