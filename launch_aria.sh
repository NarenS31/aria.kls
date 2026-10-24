#!/bin/bash
set -e

ARIA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ARIA_DIR"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/aria-matplotlib-cache"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/aria-cache"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME/fontconfig"

PYTHON="${ARIA_PYTHON:-$ARIA_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON="${ARIA_PYTHON:-python3}"; fi
if [ "${1:-}" = "--legacy" ]; then
  shift
  echo "Launching the legacy/research interface on port 7860."
  exec "$PYTHON" main.py "$@"
fi
echo "Launching the ARIA math workspace on http://127.0.0.1:7861"
exec "$PYTHON" -m ui.demo_app --no-log "$@"
