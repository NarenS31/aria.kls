#!/bin/bash
set -e

ARIA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ARIA_DIR"

echo "ARIA — Adaptive Reasoning & Intelligence Assistant"
echo "Launching on http://localhost:7860 ..."
exec python3.11 main.py
