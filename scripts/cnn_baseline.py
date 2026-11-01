"""Three-seed source-only 1D-CNN comparison on the fixed MVP split."""

import json
from pathlib import Path
import pandas as pd
import torch
from omegaconf import OmegaConf
from battery_shift.models import CNN
from battery_shift.data import Windows, statistics
from battery_shift.artifacts import load_panasonic
from battery_shift.train import fit, seed_all
from battery_shift.evaluate import metrics, predict


def main():
    # Same training configuration and input scaling as the GRU source models.
    cfg = OmegaConf.load("configs/training.yaml")
    torch.set_num_threads(cfg.threads)
    out = Path("results/cnn-v2")
    out.mkdir(parents=True, exist_ok=True)
    frames = load_panasonic("data/interim", cfg)
    train = [f for f in frames if f.split.iloc[0] == "train"]
    val = [f for f in frames if f.split.iloc[0] == "validation"]
    mean, std = statistics(train, cfg.get("temperature_scale"))

    def windows(fs, training=False):
        return Windows(fs, mean, std, cfg.window, cfg.train_stride if training else cfg.eval_stride)

    rows = []
    model_dir = Path("runs/cnn-v2")
    model_dir.mkdir(exist_ok=True)
    for seed in cfg.seeds:
        seed_all(seed)
        model, history = fit(CNN(cfg.hidden), windows(train, True), cfg, windows(val))
        torch.save(model.state_dict(), model_dir / f"{seed}.pt")
        (out / f"{seed}_training.json").write_text(json.dumps(history, indent=2))
        for temp in [25, 10, 0, -10, -20]:
            fs = [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp]
            y, p = predict(model, windows(fs), cfg.batch_size)
            for scope, values in metrics(y, p).items():
                rows.append(
                    dict(
                        seed=seed,
                        ambient_c=temp,
                        scope=scope,
                        parameters=sum(p.numel() for p in model.parameters()),
                        **values,
                    )
                )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(rows).groupby(["ambient_c", "scope"]).rmse.agg(["mean", "std"]).to_csv(out / "summary.csv")
    print("CNN comparison completed")


if __name__ == "__main__":
    main()
