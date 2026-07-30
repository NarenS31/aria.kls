#!/bin/bash
set -e

ARIA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ARIA_DIR"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/aria-matplotlib-cache"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/aria-cache"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME/fontconfig"

echo "ARIA — Adaptive Reasoning & Intelligence Assistant"
echo "Launching on http://localhost:7860 ..."
exec python3.11 main.py
