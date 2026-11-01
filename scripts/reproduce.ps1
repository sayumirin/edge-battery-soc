param([string]$Config = "configs/mvp.yaml")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m uv sync --frozen
if ($LASTEXITCODE -ne 0) { throw "Environment setup failed" }
python -m uv run --frozen python scripts/download.py
if ($LASTEXITCODE -ne 0) { throw "Download failed" }
python -m uv run --frozen python scripts/prepare_data.py --training-config $Config
if ($LASTEXITCODE -ne 0) { throw "Preparation failed" }
python -m uv run --frozen pytest
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
python -m uv run --frozen python -m battery_shift.run --config $Config
if ($LASTEXITCODE -ne 0) { throw "Experiments failed" }
