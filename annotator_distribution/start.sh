#!/usr/bin/env bash
# Vehicle Annotator launcher (macOS / Linux)
# Double-click this file, or run: ./start.sh

set -e
cd "$(dirname "$0")"

# Find a Python 3 interpreter
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "============================================================"
  echo "Python is not installed."
  echo "Please install it from https://www.python.org/downloads/"
  echo "Then double-click this file again."
  echo "============================================================"
  read -p "Press Enter to close..."
  exit 1
fi

echo "Starting Vehicle Annotator..."
"$PYTHON" launch_annotator.py

read -p "Press Enter to close..."
