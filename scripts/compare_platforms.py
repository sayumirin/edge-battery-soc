"""Record Windows/Linux numerical differences without claiming bitwise identity."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    windows, linux = Path("runs/mvp-v1"), Path("runs/linux-full-v1")
    a, b = [pd.read_csv(p / "metrics.csv") for p in [windows, linux]]
    keys = ["seed", "ambient_c", "method", "scope", "adapt_seconds", "n"]
    pd.testing.assert_frame_equal(a[keys], b[keys])
    delta = abs(a[["rmse", "mae", "max_error"]] - b[["rmse", "mae", "max_error"]])
    deltas = delta.assign(method=a.method).groupby("method").max()
    pred_max = dict(direct=0.0, int8=0.0, coulomb=0.0)
    for seed in [17, 42, 2026]:
        for temp in [25, 10, 0, -10, -20]:
            aa, bb = [pd.read_csv(p / str(seed) / f"predictions_{temp}.csv") for p in [windows, linux]]
            pd.testing.assert_frame_equal(aa[["file_id", "time_s", "soc"]], bb[["file_id", "time_s", "soc"]])
            for method in pred_max:
                pred_max[method] = max(pred_max[method], float(np.max(abs(aa[method] - bb[method]))))
    c = json.loads(Path("results/linux-c-inference/verification.json").read_text())
    assert sum(x["real_stream_steps"] for x in c) == 45000
    assert max(x["max_abs_error"] for x in c) < 1e-4
    result = dict(
        status="completed",
        platforms=["Windows/MSVC", "WSL2 Ubuntu 24.04/GCC"],
        independent_physical_machine=False,
        file_endpoints_and_labels_equal=True,
        metrics_max_delta_pp=deltas.to_dict(orient="index"),
        predictions_max_abs_delta=pred_max,
        c_max_abs_error=max(x["max_abs_error"] for x in c),
        interpretation="Float32 results closely agree; dynamic int8 is backend-dependent and not bitwise portable. "
        "These are observed differences, not a tuned pass threshold or remote CI result.",
    )
    lg_linux = Path("runs/linux-cross-cell-v1/metrics.csv")
    if lg_linux.exists():
        lg_windows = pd.read_csv("results/cross-cell-v1/metrics.csv")
        lg_other = pd.read_csv(lg_linux)
        keys = ["seed", "ambient_c", "method", "scope", "n"]
        pd.testing.assert_frame_equal(lg_windows[keys], lg_other[keys])
        result["lg_cross_cell"] = dict(
            status="completed_on_both_platforms",
            metric_rows=len(lg_windows),
            max_rmse_delta_pp=float((lg_windows.rmse - lg_other.rmse).abs().max()),
        )
    Path("results/cross_platform_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
