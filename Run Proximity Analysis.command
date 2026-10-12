#!/bin/bash
# Double-click in Finder to start the proximity-risk analysis: opens
# Terminal and asks which airport and which outputs (same as running
# `python run_proximity.py` with no arguments).
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
    PY="python3.12"
else
    PY="python3"
fi
"$PY" run_proximity.py
