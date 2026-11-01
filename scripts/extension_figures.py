"""Render measured E4/E8/E9 summaries and empirical interval calibration curves."""

import argparse
import json
from pathlib import Path
from battery_shift.artifacts import load_frames
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from battery_shift.data import Windows
from battery_shift.models import GRU
from battery_shift.evaluate import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="runs/mvp-v1")
    args = parser.parse_args()
    out = Path("results/extensions-v1")
    data = pd.read_csv(out / "e4_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, kind in zip(axes.flat, ["voltage", "current", "temperature", "dropout", "hold", "crop"]):
        for name in ["clean_training", "augmented_training"]:
            g = (
                data[
                    (data.ambient_c == 25)
                    & (data.scope == "common")
                    & (data.noise == kind)
                    & (data.model == name)
                ]
                .groupby("level")
                .rmse.agg(["mean", "std"])
            )
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", label=name)
        ax.set(title=kind, xlabel="Configured strength", ylabel="RMSE (pp)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "e4_robustness.png", dpi=160)
    plt.close(fig)
    online = pd.read_csv(out / "e9_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, domain in zip(axes, ["target", "source"]):
        for supervision in ["true_label", "coulomb_pseudo"]:
            g = (
                online[
                    (online.target_temp == -10)
                    & (online.scope == "common")
                    & (online.evaluation_domain == domain)
                    & (online.supervision == supervision)
                ]
                .groupby("update")
                .rmse.agg(["mean", "std"])
            )
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", label=supervision)
        ax.set(title=domain + " evaluation: -10C adaptation", xlabel="Update number", ylabel="RMSE (pp)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "e9_forgetting.png", dpi=160)
    plt.close(fig)
    base = Path(args.base)
    cfg = OmegaConf.load(base / "config.yaml")
    torch.set_num_threads(cfg.threads)
    stats = json.loads((base / "normalization.json").read_text())
    frames = load_frames("data/interim")

    def windows(fs):
        return Windows(
            fs,
            np.array(stats["mean"], dtype="float32"),
            np.array(stats["std"], dtype="float32"),
            cfg.window,
            cfg.eval_stride,
            feature_mask=cfg.get("feature_mask"),
        )

    models = []
    for seed in cfg.seeds:
        m = GRU(cfg.hidden).eval()
        m.load_state_dict(torch.load(base / str(seed) / "float32.pt", weights_only=True))
        models.append(m)
    val = windows([f for f in frames if f.split.iloc[0] == "validation"])
    yv, pv = predict(models[0], val)
    preds = [pv] + [predict(m, val)[1] for m in models[1:]]
    residual = abs(yv - np.mean(preds, axis=0))
    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for temp in [25, 10, 0, -10, -20]:
        ds = windows([f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp])
        y, p = predict(models[0], ds)
        ps = [p] + [predict(m, ds)[1] for m in models[1:]]
        errors = abs(y - np.mean(ps, axis=0))
        coverage = []
        for level in [0.5, 0.8, 0.9, 0.95, 0.99]:
            q = float(np.quantile(residual, level, method="higher"))
            rate = float(np.mean(errors <= q))
            coverage.append(rate)
            rows.append(dict(ambient_c=temp, nominal=level, full_range_coverage=rate, width_pp=200 * q))
        ax.plot([0.5, 0.8, 0.9, 0.95, 0.99], coverage, marker="o", label=f"{temp} C")
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.set(
        xlabel="Validation residual quantile",
        ylabel="Empirical coverage: full SoC range",
        title="Descriptive calibration; validation reused for early stopping",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "e8_calibration.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "e8_calibration.csv", index=False)


if __name__ == "__main__":
    main()
