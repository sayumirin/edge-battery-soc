"""Run independent E1-E9 stages, optionally reusing a saved source-training artifact."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from omegaconf import OmegaConf
from battery_shift.artifacts import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--reuse-source", help="Skip preparation/training and consume checked saved artifacts"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    out = Path(args.output or cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    records = []

    def run(stage, *command):
        print("Stage:", stage, flush=True)
        subprocess.run([sys.executable, *command], check=True)
        records.append(dict(stage=stage, command=list(command), status="completed"))

    if args.reuse_source:
        source = Path(args.reuse_source)
        if not (source / "COMPLETE").exists():
            raise ValueError("Cannot reuse incomplete source artifact")
    else:
        for config in cfg.preparation:
            run("prepare:" + config, "scripts/prepare_data.py", "--config", config)
        source = Path("runs") / ("source-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
        run("train", "-m", "battery_shift.experiments", "--config", cfg.training, "--output", str(source))
    for stage in cfg.stages:
        conf = OmegaConf.load(stage.config)
        destination = out / conf.experiment
        if conf.experiment == "E5":
            command = ["scripts/soh.py", "--config", stage.config, "--output", str(destination)]
        elif conf.experiment == "E7":
            command = [
                "scripts/cross_cell.py",
                "--config",
                stage.config,
                "--base",
                str(source),
                "--output",
                str(destination),
            ]
        else:
            command = [
                "-m",
                "battery_shift.experiments",
                "--config",
                stage.config,
                "--source",
                str(source),
                "--output",
                str(destination),
            ]
        run(conf.experiment, *command)
        if not (destination / "COMPLETE").exists():
            raise AssertionError("Stage missing completion marker")
    run("figures", "scripts/modular_figures.py", "--results", str(out))
    (out / "pipeline.json").write_text(
        json.dumps(
            dict(
                status="completed",
                source=str(source),
                reuse_source=bool(args.reuse_source),
                stages=records,
                pipeline_config_sha256=digest(args.config),
            ),
            indent=2,
        )
    )
    (out / "COMPLETE").write_text("All configured independent stages completed\n")
    run("validate", "scripts/validate_modular.py", "--results", str(out))


if __name__ == "__main__":
    main()
