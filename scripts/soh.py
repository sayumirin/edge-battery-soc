"""E5: NASA four-cell early-discharge SoH, random cycles versus held-out cells."""

import argparse
import hashlib
import json
import urllib.request
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.io import loadmat
from omegaconf import OmegaConf

URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.%20Battery%20Data%20Set.zip"
SHA = "82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e"


def ridge(x, y, test, alpha=1.0):
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-8)
    a = np.column_stack([np.ones(len(x)), (x - mean) / std])
    b = np.column_stack([np.ones(len(test)), (test - mean) / std])
    penalty = alpha * np.eye(a.shape[1])
    penalty[0, 0] = 0
    return b @ np.linalg.solve(a.T @ a + penalty, a.T @ y)


def prepare_nasa(cfg):
    raw = Path(cfg.archive).parent
    raw.mkdir(parents=True, exist_ok=True)
    path = Path(cfg.archive)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=120) as r:
            path.write_bytes(r.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA:
        raise ValueError("NASA checksum mismatch")
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ZipFile(path) as z:
        nested = next(n for n in z.namelist() if "FY08Q4" in n)
        with ZipFile(BytesIO(z.read(nested))) as archive:
            (raw / "README.txt").write_bytes(archive.read("README.txt"))
            for cell in cfg.cells:
                cycles = loadmat(BytesIO(archive.read(cell + ".mat")), simplify_cells=True)[cell]["cycle"]
                for index, cycle in enumerate(cycles):
                    if cycle["type"] != "discharge":
                        continue
                    d = cycle["data"]
                    t = np.asarray(d["Time"])
                    keep = t <= cfg.horizon_seconds
                    if t[-1] < cfg.horizon_seconds:
                        raise ValueError("Early-discharge interval unavailable")
                    v = d["Voltage_measured"]
                    temp = d["Temperature_measured"]
                    current = d["Current_measured"]
                    rows.append(
                        dict(
                            cell=cell,
                            cycle=index,
                            voltage_start=float(v[0]),
                            **{
                                f"voltage_{int(cfg.horizon_seconds)}": float(
                                    np.interp(cfg.horizon_seconds, t, v)
                                )
                            },
                            voltage_mean=float(np.mean(v[keep])),
                            temperature_rise=float(np.interp(cfg.horizon_seconds, t, temp) - temp[0]),
                            current_mean=float(np.mean(current[keep])),
                            soh=float(d["Capacity"]) / cfg.capacity_ah,
                        )
                    )
    data = pd.DataFrame(rows)
    data.to_parquet(out / "cycles.parquet", index=False)
    (out / "dataset.json").write_text(
        json.dumps(
            dict(
                source_url=URL,
                archive_sha256=SHA,
                capacity_ah=cfg.capacity_ah,
                horizon_seconds=cfg.horizon_seconds,
                sha256=hashlib.sha256((out / "cycles.parquet").read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e5.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    prepared = Path(cfg.prepared)
    meta = json.loads((prepared / "dataset.json").read_text())
    if hashlib.sha256((prepared / "cycles.parquet").read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Prepared NASA checksum mismatch")
    data = pd.read_parquet(prepared / "cycles.parquet")
    out = Path(args.output or cfg.output)
    cfg.output = str(out)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "experiment.yaml")
    data.to_csv(out / "cycles.csv", index=False)
    features = list(cfg.features)
    x = data[features].to_numpy()
    y = data.soh.to_numpy()
    metrics = []
    splits = []

    def evaluate(seed, protocol, fold, train, test):
        if set(train) & set(test):
            raise AssertionError("Row leakage")
        if protocol == "leave_one_cell_out" and set(data.cell.iloc[train]) & set(data.cell.iloc[test]):
            raise AssertionError("Cell leakage")
        pred = ridge(x[train], y[train], x[test], cfg.ridge_alpha)
        err = pred - y[test]
        metrics.append(
            dict(
                seed=seed,
                protocol=protocol,
                fold=fold,
                n=len(test),
                rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(abs(err))),
            )
        )
        splits.append(
            dict(seed=seed, protocol=protocol, fold=fold, train_rows=train.tolist(), test_rows=test.tolist())
        )

    for seed in cfg.seeds:
        order = np.random.default_rng(seed).permutation(len(data))
        cut = int(cfg.train_fraction * len(data))
        evaluate(seed, "random_cycles", "all", order[:cut], order[cut:])
        for cell in data.cell.unique():
            evaluate(
                seed,
                "leave_one_cell_out",
                cell,
                np.flatnonzero(data.cell != cell),
                np.flatnonzero(data.cell == cell),
            )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(metrics).groupby(["protocol", "fold"]).rmse.agg(["mean", "std"]).to_csv(out / "summary.csv")
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                source_url=URL,
                sha256=SHA,
                features=features,
                capacity_reference_ah=meta["capacity_ah"],
                feature_horizon_seconds=meta["horizon_seconds"],
                model=f"ridge alpha={cfg.ridge_alpha} with train-only scaling",
                prepared_sha256=meta["sha256"],
                limitations="Four cells only; different cutoff voltages/protocols confound cell generalization. LOCO deterministic across seeds, so SD=0 is not independent evidence. Random-cycle comparison intentionally allows same-cell leakage, but no repeated rows.",
            ),
            indent=2,
        )
    )
    (out / "COMPLETE").write_text("E5 completed from persisted NASA feature table\n")
    print(pd.DataFrame(metrics).groupby(["protocol", "fold"]).rmse.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
