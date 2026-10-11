#!/usr/bin/env bash
# =============================================================================
# First-time setup for tmrl runtime data (~~/TmrlData)
# =============================================================================
set -euo pipefail

echo "=== Trackmania Ghost AI - tmrl Setup ==="

# Initialise TmrlData directory
python -c "
import tmrl
print('tmrl version:', tmrl.__version__)
print('Initialising TmrlData ...')
tmrl.init()
print('Done. TmrlData is ready.')
"

echo ""
echo "Next steps:"
echo "  1. Make sure TrackMania (Nations or 2020) is installed and running."
echo "  2. For TM2020: install the OpenPlanet plugin and the tmrl plugin."
echo "  3. Start training:  tm-train --config configs/training.yaml"
echo "========================================="
