"""Export one trained source model as a versioned parameter file for the C kernel."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from battery_shift.models import GRU
from battery_shift.params import float_count, identity, save, from_state_dict

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/source-v3")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--name", default="gru-source-v3")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    stats = json.loads((run / "normalization.json").read_text())
    checkpoint = run / str(args.seed) / "float32.pt"
    model = GRU(cfg.hidden).eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    splits = json.loads((ROOT / "data/splits/manifest.json").read_text())
    provenance = dict(
        model="GRU, 3 inputs at 1 Hz, reset-after gates",
        trained_on="Panasonic 18650PF drive cycles at 25 degC",
        run=run.name,
        seed=args.seed,
        checkpoint_sha256=sha256(checkpoint),
        config_sha256=sha256(run / "config.yaml"),
        split_repository=splits["repository"],
        split_revision=splits["revision"],
    )
    params = from_state_dict(model.state_dict(), cfg.hidden, stats["mean"], stats["std"],
                             cfg.get("feature_mask", [1, 1, 1]), args.name, args.version, provenance)
    save(params, args.output)
    print(f"Wrote {args.output}: {identity(params)}, {float_count(params)} floats")


if __name__ == "__main__":
    main()
