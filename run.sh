#!/usr/bin/env bash
# Launch the MLB scoreboard server. Reuses .venv if present, else creates it.
# Binds 0.0.0.0 so the ESP32 (and other hosts) can reach it — localhost would not.
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet 'fastapi==0.115.*' 'uvicorn==0.34.*' 'httpx==0.28.*' 'Pillow==12.*'

exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}"
