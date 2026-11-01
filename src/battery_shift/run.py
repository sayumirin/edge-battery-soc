"""Execute E1/E2/E3/E6 and produce traceable, seed-aggregated artifacts."""

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from .data import Windows, statistics
from .artifacts import load_panasonic
from .evaluate import metrics, predict
from .export import export_models
from .models import GRU, coulomb_count
from .train import fit, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path.cwd()
    cfg = OmegaConf.load(args.config)
    out = Path(args.output or ("runs/" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")))
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, out / "config.yaml")
    tracked = sorted(
        p
        for folder in ["src", "scripts", "configs", "data/splits"]
        for p in Path(folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    provenance = dict(
        commit=revision.stdout.strip() or None,
        dirty=bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout),
        source_sha256={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
        torch=torch.__version__,
        label_basis=cfg.capacity_basis,
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    torch.set_num_threads(cfg.threads)
    frames = load_panasonic(root / "data/interim", cfg)
    train = [f for f in frames if f.split.iloc[0] == "train"]
    val = [f for f in frames if f.split.iloc[0] == "validation"]
    mean, std = statistics(train)
    (out / "normalization.json").write_text(json.dumps(dict(mean=mean.tolist(), std=std.tolist())))

    def windows(fs, stats=(mean, std), training=False):
        return Windows(
            fs,
            *stats,
            cfg.window,
            cfg.train_stride if training else cfg.eval_stride,
            feature_mask=cfg.get("feature_mask"),
        )

    ds, vs = windows(train, training=True), windows(val)
    rows, sizes = [], []
    temperatures = [25, 10, 0, -10, -20]

    def record(seed, temp, method, y, p, seconds=0):
        for scope, values in metrics(y, p).items():
            rows.append(
                dict(seed=seed, ambient_c=temp, method=method, scope=scope, adapt_seconds=seconds, **values)
            )

    for seed in cfg.seeds:
        seed_all(seed)
        sd = out / str(seed)
        sd.mkdir()
        model, history = fit(GRU(cfg.hidden), ds, cfg, vs)
        count = sum(p.numel() for p in model.parameters())
        if count >= 50000:
            raise ValueError("Parameter budget exceeded")
        (sd / "training.json").write_text(json.dumps(history, indent=2))
        x = torch.from_numpy(np.stack([ds[i][0] for i in range(min(4, len(ds)))]))
        quantized, size = export_models(model, x, sd)
        sizes.append(dict(seed=seed, parameters=count, **size))
        for temp in temperatures:
            test = [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp]
            td = windows(test)
            y, p = predict(model, td, cfg.batch_size)
            record(seed, temp, "GRU direct", y, p)
            _, qp = predict(quantized, td, cfg.batch_size)
            record(seed, temp, "GRU int8", y, qp)
            baselines = [
                coulomb_count(f.time_s, f.current_a, float(f.soc.iloc[0]) + 0.05, cfg.capacity_ah)
                for f in test
            ]
            bp = np.array([baselines[j][end] for j, end in td.index])
            record(seed, temp, "Coulomb +5pp", y, bp)
            pd.DataFrame(
                dict(
                    file_id=[test[j].file_id.iloc[0] for j, _ in td.index],
                    time_s=[float(test[j].time_s.iloc[e]) for j, e in td.index],
                    soc=y,
                    direct=p,
                    int8=qp,
                    coulomb=bp,
                )
            ).to_csv(sd / f"predictions_{temp}.csv", index=False)
            if temp != 25:
                adapt = [
                    f.loc[f.time_s < cfg.adapt_seconds].copy()
                    for f in frames
                    if f.split.iloc[0] == "adapt" and f.ambient_c.iloc[0] == temp
                ]
                seconds = sum(float(f.time_s.iloc[-1] - f.time_s.iloc[0]) for f in adapt)
                target_stats = statistics(adapt)
                (sd / f"normalization_{temp}.json").write_text(
                    json.dumps(dict(mean=target_stats[0].tolist(), std=target_stats[1].tolist()))
                )
                yy, pp = predict(model, windows(test, target_stats), cfg.batch_size)
                record(seed, temp, "GRU target stats", yy, pp, seconds)
                for mode in ["head", "input", "all"]:
                    seed_all(seed)
                    tuned, hist = fit(copy.deepcopy(model), windows(adapt, training=True), cfg, mode=mode)
                    yy, pp = predict(tuned, td, cfg.batch_size)
                    record(seed, temp, "GRU tune " + mode, yy, pp, seconds)
                    torch.save(tuned.state_dict(), sd / f"adapt_{temp}_{mode}.pt")
                    (sd / f"adapt_{temp}_{mode}.json").write_text(json.dumps(hist))
            print(f"Seed {seed}, temperature {temp}: complete", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "metrics.csv", index=False)
    summary = raw.groupby(["method", "ambient_c", "scope"])[["rmse", "mae", "max_error"]].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    pd.DataFrame(sizes).to_csv(out / "model_sizes.csv", index=False)
    table = [
        "# Results: " + str(args.config),
        "",
        "Label: " + cfg.capacity_basis,
        "",
        "RMSE in percentage points, SoC 40–100%, mean ± sample SD over seeds. Adaptation at 25°C is not applicable.",
        "",
        "| Method | 25°C | 10°C | 0°C | -10°C | -20°C |",
        "|---|---|---|---|---|---|",
    ]
    for method in raw.method.unique():
        cells = []
        for temp in temperatures:
            v = raw.loc[(raw.method == method) & (raw.ambient_c == temp) & (raw.scope == "common"), "rmse"]
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else "—")
        table.append("| " + method + " | " + " | ".join(cells) + " |")
    (out / "results.md").write_text("\n".join(table), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, group in raw[raw.scope == "common"].groupby("method"):
        g = group.groupby("ambient_c").rmse.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", label=method)
    ax.set(xlabel="Ambient temperature (°C)", ylabel="SoC RMSE (percentage points)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "temperature_shift.png", dpi=160)
    plt.close(fig)
    prediction = pd.read_csv(out / str(cfg.seeds[0]) / "predictions_-10.csv")
    prediction = prediction[prediction.file_id == prediction.file_id.iloc[0]]
    fig, ax = plt.subplots(figsize=(9, 4))
    for column in ["soc", "direct", "int8", "coulomb"]:
        ax.plot(prediction.time_s / 60, prediction[column] * 100, label=column)
    ax.set(xlabel="Time (minutes)", ylabel="SoC (%)", title="-10°C: one held-out drive cycle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "soc_prediction.png", dpi=160)
    plt.close(fig)
    (out / "COMPLETE").write_text("All configured seeds and temperatures finished.\n")
    print("Results:", out.resolve())


if __name__ == "__main__":
    main()
