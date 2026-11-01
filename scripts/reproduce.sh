#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen
uv run --frozen python scripts/download.py
uv run --frozen python scripts/prepare_data.py --training-config "${1:-configs/mvp.yaml}"
uv run --frozen pytest
uv run --frozen python -m battery_shift.run --config "${1:-configs/mvp.yaml}"
