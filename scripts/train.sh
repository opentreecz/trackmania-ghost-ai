#!/usr/bin/env bash
# =============================================================================
# Train the Trackmania Ghost AI agent (all-in-one local mode)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$PROJECT_DIR/configs/training.yaml}"

echo "=== Trackmania Ghost AI - Training ==="
echo "Config: $CONFIG"
echo "======================================="

# Activate venv if it exists
if [ -d "$PROJECT_DIR/.venv" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

exec tm-train --config "$CONFIG" --verbose
