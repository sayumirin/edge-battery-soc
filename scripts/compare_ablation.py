"""Compare paired seeds on identical endpoints, including active-current endpoints."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_shift.evaluate import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="runs/mvp-v1")
    parser.add_argument("--candidate", default="runs/no-temperature-v1")
    parser.add_argument("--output", default="results/diagnostics-v1")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("data/splits/manifest.json").read_text())
    frames = {
        e["file_id"]: pd.read_parquet(Path("data/interim") / (e["file_id"] + ".parquet"))
        for e in manifest["files"]
    }
    rows = []
    for seed in [17, 42, 2026]:
        for temp in [25, 10, 0, -10, -20]:
            a = pd.read_csv(Path(args.baseline) / str(seed) / f"predictions_{temp}.csv")
            b = pd.read_csv(Path(args.candidate) / str(seed) / f"predictions_{temp}.csv")
            pd.testing.assert_frame_equal(a[["file_id", "time_s", "soc"]], b[["file_id", "time_s", "soc"]])
            currents = []
            for file_id, time_s in zip(a.file_id, a.time_s):
                f = frames[file_id]
                index = np.searchsorted(f.time_s.to_numpy(), time_s)
                if index >= len(f) or f.time_s.iloc[index] != time_s:
                    raise ValueError("Prediction timestamp not in prepared data")
                currents.append(float(f.current_a.iloc[index]))
            active = np.abs(currents) > 0.05
            for name, p in [("VIT", a.direct.to_numpy()), ("VI_only", b.direct.to_numpy())]:
                for subset, mask in [
                    ("all_endpoints", np.ones(len(a), dtype=bool)),
                    ("active_current_endpoints", active),
                ]:
                    for scope, values in metrics(a.soc.to_numpy()[mask], p[mask]).items():
                        rows.append(
                            dict(seed=seed, ambient_c=temp, model=name, subset=subset, scope=scope, **values)
                        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "ablation_metrics.csv", index=False)
    grouped = result.groupby(["model", "ambient_c", "subset", "scope"]).rmse.agg(["mean", "std"])
    grouped.to_csv(out / "ablation_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, subset in zip(axes, ["all_endpoints", "active_current_endpoints"]):
        for model in ["VIT", "VI_only"]:
            g = (
                result[(result.model == model) & (result.subset == subset) & (result.scope == "common")]
                .groupby("ambient_c")
                .rmse.agg(["mean", "std"])
            )
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, label=model)
        ax.set(title=subset.replace("_", " "), xlabel="Ambient temperature (°C)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("SoC RMSE (percentage points), 40–100%")
    fig.tight_layout()
    fig.savefig(out / "temperature_ablation.png", dpi=160)
    plt.close(fig)
    (out / "ablation_protocol.json").write_text(
        json.dumps(
            dict(
                baseline=args.baseline,
                candidate=args.candidate,
                seeds=[17, 42, 2026],
                scope="exploratory after observing baseline test results; requires independent confirmation",
                active_definition="abs(current_a) > 0.05 A at window endpoint; full history including preceding rest retained",
                paired_endpoints_verified=True,
            ),
            indent=2,
        )
    )
    print(grouped.to_string())


if __name__ == "__main__":
    main()
