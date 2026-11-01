"""Reproducible source/input audit, capacity sensitivity, and all-temperature EDA.

Run from project root: uv run python scripts/diagnose.py
Capacity sensitivity changes label definition only; it is not a recalibrated model evaluation.
"""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from omegaconf import OmegaConf

from battery_shift.data import FEATURES, statistics
from battery_shift.evaluate import metrics

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = "Panasonic 18650PF Data/25degC/C20 OCV and 1C discharge tests_start_of_tests/05-08-17_13.26 C20 OCV Test_C20_25dC.mat"
SHA256 = "c52c1941bc6ef8897578efea5316521297a05d5f6c042d07e66003bd5427f7fa"
REV = "6d720489c9b5b95624e448d44e8e2c0e48bdacaf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="runs/mvp-v1")
    parser.add_argument("--output", default="results/diagnostics-v1")
    args = parser.parse_args()
    baseline = ROOT / args.baseline
    config = OmegaConf.load(baseline / "config.yaml")
    if config.capacity_ah != 2.9:
        raise ValueError("This audit requires the original nominal 2.9 Ah reference")
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/splits/manifest.json").read_text())
    frames = [pd.read_parquet(ROOT / "data/interim" / (e["file_id"] + ".parquet")) for e in manifest["files"]]
    for frame in frames:
        np.testing.assert_allclose(frame.soc, 1 - frame.ah / 2.9, atol=1e-7)
    mean, std = statistics([f for f in frames if f.split.iloc[0] == "train"])
    rows, coverage = [], []
    for f in frames:
        for j, feature in enumerate(FEATURES):
            z = (f[feature].to_numpy() - mean[j]) / std[j]
            rows.append(
                dict(
                    file_id=f.file_id.iloc[0],
                    ambient_c=int(f.ambient_c.iloc[0]),
                    cycle=f.cycle.iloc[0],
                    split=f.split.iloc[0],
                    feature=feature,
                    train_mean=float(mean[j]),
                    train_std=float(std[j]),
                    mean=float(f[feature].mean()),
                    min=float(f[feature].min()),
                    max=float(f[feature].max()),
                    absolute_z_p99=float(np.quantile(abs(z), 0.99)),
                    fraction_abs_z_gt5=float(np.mean(abs(z) > 5)),
                )
            )
        selected = f[f.time_s < 1800] if f.split.iloc[0] == "adapt" else f
        coverage.append(
            dict(
                file_id=f.file_id.iloc[0],
                ambient_c=int(f.ambient_c.iloc[0]),
                cycle=f.cycle.iloc[0],
                split=f.split.iloc[0],
                duration_s=float(f.time_s.iloc[-1]),
                selected_rows=len(selected),
                soc_min=float(selected.soc.min()),
                soc_max=float(selected.soc.max()),
                rest_fraction=float((abs(f.current_a) < 0.05).mean()),
            )
        )
    pd.DataFrame(rows).to_csv(out / "feature_shift.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    for temp in [25, 10, 0, -10, -20]:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for f in frames:
            if f.ambient_c.iloc[0] != temp:
                continue
            label = f"{f.cycle.iloc[0]} ({f.split.iloc[0]}) {f.file_id.iloc[0][:5]}"
            for ax, col in zip(axes, ["voltage_v", "current_a", "soc"]):
                ax.plot(f.time_s.iloc[::10] / 60, f[col].iloc[::10], linewidth=0.7, alpha=0.75, label=label)
        for ax, ylabel in zip(axes, ["Voltage (V)", "Discharge current (A)", "Nominal-reference SoC"]):
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
        axes[0].set_title(f"{temp}°C: all selected files; rest periods retained; 10 s plot stride")
        axes[0].legend(fontsize=7, ncol=2)
        axes[-1].set_xlabel("Time from file start (minutes)")
        fig.tight_layout()
        fig.savefig(out / f"eda_{temp}.png", dpi=150)
        plt.close(fig)
    p = ROOT / "data/raw" / REF_PATH
    url = f"https://raw.githubusercontent.com/fededc88/Panasonic-18650PF-Data/{REV}/" + urllib.parse.quote(
        REF_PATH
    )
    if not p.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Reference checksum mismatch")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    if hashlib.sha256(p.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Reference checksum mismatch")
    m = loadmat(p, simplify_cells=True)["meas"]
    segments = np.split(np.arange(len(m["Current"])), np.where(np.diff(m["Current"] < -0.05))[0] + 1)
    discharge = [s for s in segments if len(s) > 30 and np.median(m["Current"][s]) < -0.05]
    if len(discharge) != 1 or discharge[0][0] == 0:
        raise ValueError("Ambiguous full discharge reference")
    indices = discharge[0]
    start, end = int(indices[0] - 1), int(indices[-1])
    capacity = float(m["Ah"][start] - m["Ah"][end])
    evidence = dict(
        url=url,
        sha256=SHA256,
        capacity_ah=capacity,
        definition="Ah before discharge minus Ah at discharge cutoff; retained pre-discharge counter offset",
        start_index=start,
        end_index=end,
        start_ah=float(m["Ah"][start]),
        end_ah=float(m["Ah"][end]),
        cutoff_v=float(m["Voltage"][end]),
        median_discharge_current_a=float(np.median(m["Current"][indices])),
        scope="25C reference only; not per-temperature calibration; measurement dated May 8, later than March source drive cycles",
    )
    (out / "capacity_reference.json").write_text(json.dumps(evidence, indent=2))
    sensitivity = []
    for f in frames:
        difference = 100 * ((1 - f.ah / capacity) - f.soc)
        sensitivity.append(
            dict(
                file_id=f.file_id.iloc[0],
                ambient_c=int(f.ambient_c.iloc[0]),
                cycle=f.cycle.iloc[0],
                label_delta_mae_pp=float(abs(difference).mean()),
                label_delta_max_pp=float(abs(difference).max()),
            )
        )
    pd.DataFrame(sensitivity).to_csv(out / "capacity_label_sensitivity.csv", index=False)
    # Existing checkpoints/predictions stay fixed; this measures label sensitivity, not retraining.
    metric_rows = []
    for seed in [17, 42, 2026]:
        for temp in [25, 10, 0, -10, -20]:
            predictions = pd.read_csv(baseline / str(seed) / f"predictions_{temp}.csv")
            for basis, q in [("nominal", 2.9), ("25C_reference_sensitivity", capacity)]:
                y = 1 - (1 - predictions.soc.to_numpy()) * 2.9 / q
                for scope, values in metrics(y, predictions.direct.to_numpy()).items():
                    metric_rows.append(
                        dict(seed=seed, ambient_c=temp, label_basis=basis, scope=scope, **values)
                    )
    pd.DataFrame(metric_rows).to_csv(out / "fixed_predictions_label_sensitivity.csv", index=False)
    print(json.dumps(evidence, indent=2))
    print("Diagnostics saved:", out)


if __name__ == "__main__":
    main()
