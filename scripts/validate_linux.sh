#!/usr/bin/env bash
# Run in a separate Linux Python environment; never reuses the Windows .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${BATTERY_LINUX_PYTHON:-$HOME/.cache/battery-shift-linux/bin/python}"
run_dir="${BATTERY_LINUX_OUTPUT:-runs/linux-$(date -u +%Y%m%dT%H%M%S)}"
"$python_bin" -m pytest -q --basetemp=/tmp/battery-linux-pytest --junitxml=results/linux_test_results.xml
"$python_bin" scripts/prepare_data.py
"$python_bin" -m battery_shift.run --config configs/mvp.yaml --output "$run_dir"
"$python_bin" scripts/verify_c.py --run "$run_dir" --output results/linux-c-inference
