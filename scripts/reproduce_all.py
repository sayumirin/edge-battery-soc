"""One entry point for all currently runnable experiments (requires a C compiler).

uv sync --frozen --group report
uv run --group report python scripts/reproduce_all.py
Raw data are cached and verified. Every experiment run receives a fresh directory.
Official LG and Panasonic ZIPs must be present; see README.md for the browser download.
The legacy MVP/extension stages reproduce the first-round (v1) protocol; current results come from
the protocol-v3 modular pipeline.
MCU and user-deferred remote CI are excluded.
"""

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from omegaconf import OmegaConf


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "runs" / ("delivery-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    for archive in ["data/raw/lg/lg.zip", "data/raw/panasonic-official.zip"]:
        if not (root / archive).exists():
            raise FileNotFoundError(f"Missing official archive {archive}; see README.md")

    def run(*args):
        print("Running:", " ".join(args), flush=True)
        subprocess.run([sys.executable, *args], cwd=root, check=True)

    def publish(src, dest):
        dest.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_file() and p.suffix not in {".pt"}:
                shutil.copy2(p, dest / p.name)

    run("scripts/download.py")
    for config in ["configs/prepare.yaml", "configs/prepare_lg.yaml", "configs/prepare_nasa.yaml"]:
        run("scripts/prepare_data.py", "--config", config)
    run("-m", "pytest", "-q", "--junitxml=results/test_results.xml")
    run("scripts/audit_official.py")
    baseline = folder / "mvp"
    candidate = folder / "no-temperature"
    extensions = folder / "extensions"
    run("-m", "battery_shift.run", "--config", "configs/mvp.yaml", "--output", str(baseline))
    publish(baseline, root / "results/mvp-v1")
    run("scripts/diagnose.py", "--baseline", str(baseline))
    run("-m", "battery_shift.run", "--config", "configs/no_temperature.yaml", "--output", str(candidate))
    publish(candidate, root / "results/no-temperature-v1")
    run("scripts/compare_ablation.py", "--baseline", str(baseline), "--candidate", str(candidate))
    run("scripts/physics_baseline.py")
    run("scripts/coulomb_sensitivity.py")
    run("scripts/cnn_baseline.py")
    run("scripts/soh.py", "--config", "configs/e5.yaml", "--output", "results/soh-v1")
    cfg = OmegaConf.load(root / "configs/extensions.yaml")
    cfg.base_run = str(baseline)
    config_path = folder / "extensions.yaml"
    OmegaConf.save(cfg, config_path)
    run("scripts/extensions.py", "--config", str(config_path), "--output", str(extensions))
    publish(extensions, root / "results/extensions-v1")
    run("scripts/extension_figures.py", "--base", str(baseline))
    # Protocol v3 (fixed temperature rule, fixed adaptation steps): retrains sources, runs E1-E9.
    modular = folder / "modular"
    run("scripts/run_pipeline.py", "--output", str(modular))
    shutil.copytree(
        modular,
        root / "results/modular-v3",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.pt", "*.onnx"),
    )
    source = json.loads((modular / "pipeline.json").read_text())["source"]
    run("scripts/verify_c.py", "--run", source, "--output", "results/c-inference-v3")
    run("scripts/build_report.py")
    (folder / "COMPLETE").write_text(
        json.dumps(
            dict(
                status="completed",
                not_included=[
                    "real MCU deployment",
                    "remote CI (deferred by user)",
                    "independent physical-machine validation",
                ],
                capacity_scope="Official Panasonic archive has only a 25 C C/20 reference; "
                "nominal labels and capacity sensitivity are explicitly reported.",
            ),
            indent=2,
        )
    )
    (root / "results/full_pipeline_verification.json").write_text(
        json.dumps(
            dict(
                status="completed",
                run=str(folder),
                entrypoint="scripts/reproduce_all.py",
                e7_included=True,
                raw_archives_cached=True,
            ),
            indent=2,
        )
    )
    print("Delivery run:", folder)


if __name__ == "__main__":
    main()
