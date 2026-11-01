"""Explicit preprocessing entry. Subsequent stages never parse the original archives."""

import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from battery_shift.data import prepare
from battery_shift.lg import prepare_lg
from battery_shift.artifacts import save_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prepare.yaml")
    parser.add_argument("--training-config")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.training_config:
        cfg.training_config = args.training_config
    if cfg.dataset == "panasonic":
        prepare(".", OmegaConf.load(cfg.training_config), output=cfg.output)
    elif cfg.dataset == "lg":
        frames = prepare_lg(cfg.archive, cfg.output)
        save_frames(frames, cfg.output, dict(dataset="lg", capacity_ah=3.0, sample_period_s=1.0))
    elif cfg.dataset == "nasa":
        from soh import prepare_nasa

        prepare_nasa(cfg)
    else:
        raise ValueError("Unknown dataset")
    out = Path(cfg.output)
    OmegaConf.save(cfg, out / "prepare_config.yaml")
    (out / "PREPARED").write_text(json.dumps(dict(dataset=cfg.dataset, status="prepared")))
    print("Prepared:", out)


if __name__ == "__main__":
    main()
