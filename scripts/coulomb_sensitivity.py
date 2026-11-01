"""Deterministic initial-SoC and current-bias sensitivity on real held-out files."""

from pathlib import Path
from battery_shift.artifacts import load_frames
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_shift.models import coulomb_count
from battery_shift.evaluate import metrics


def main():
    frames = load_frames("data/interim")
    out = Path("results/coulomb-sensitivity")
    out.mkdir(exist_ok=True, parents=True)
    rows = []
    for temp in [25, 10, 0, -10, -20]:
        files = [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp]
        for initial in [-0.05, 0, 0.05]:
            for bias in [-0.05, 0, 0.05]:
                ys = []
                ps = []
                for f in files:
                    pred = coulomb_count(f.time_s, f.current_a + bias, float(f.soc.iloc[0]) + initial, 2.9)
                    ix = np.arange(299, len(f), 10)
                    ys.extend(f.soc.iloc[ix])
                    ps.extend(pred[ix])
                for scope, values in metrics(np.array(ys), np.array(ps)).items():
                    rows.append(
                        dict(
                            ambient_c=temp,
                            initial_error_pp=initial * 100,
                            current_bias_a=bias,
                            scope=scope,
                            **values,
                        )
                    )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    f = next(
        f
        for f in frames
        if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == 25 and f.cycle.iloc[0] == "US06"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    for bias in [-0.05, 0, 0.05]:
        pred = coulomb_count(f.time_s, f.current_a + bias, float(f.soc.iloc[0]) + 0.05, 2.9)
        ax.plot(f.time_s / 60, (pred - f.soc) * 100, label=f"{bias:+.2f} A bias")
    ax.set(
        xlabel="Time (minutes)",
        ylabel="SoC error (percentage points)",
        title="25 C US06: initial error +5 pp; no clipping",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "drift.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
