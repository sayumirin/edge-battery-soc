"""E7: Panasonic source -> LG HG2; fixed file splits, nominal and C/20 label audit."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from battery_shift.data import Windows, statistics
from battery_shift.evaluate import metrics, predict
from battery_shift.artifacts import load_frames, digest
from battery_shift.models import GRU
from battery_shift.train import fit, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e7.yaml")
    parser.add_argument("--base")
    parser.add_argument("--prepared")
    parser.add_argument("--output")
    args = parser.parse_args()
    experiment = OmegaConf.load(args.config)
    base, out = Path(args.base or experiment.source), Path(args.output or experiment.output)
    if not (base / "COMPLETE").exists():
        raise ValueError("Source training is incomplete")
    out.mkdir(parents=True, exist_ok=True)
    prepared = Path(args.prepared or experiment.prepared)
    experiment.source, experiment.prepared, experiment.output = str(base), str(prepared), str(out)
    OmegaConf.save(experiment, out / "experiment.yaml")
    cfg = OmegaConf.load(base / "config.yaml")
    cfg.seeds = experiment.seeds
    cfg.adapt_seconds = experiment.adapt.seconds
    fitting = copy.deepcopy(cfg)
    fitting.batch_size, fitting.adapt_steps = experiment.adapt.batch_size, experiment.adapt.steps
    fitting.learning_rate = experiment.adapt.learning_rate
    torch.set_num_threads(cfg.threads)
    fs = load_frames(prepared, dict(dataset="lg", capacity_ah=3.0, sample_period_s=1.0))
    (out / "data_audit.json").write_bytes((prepared / "data_audit.json").read_bytes())
    norms = json.loads((base / "normalization.json").read_text())
    source_stats = (np.array(norms["mean"]), np.array(norms["std"]))
    rows, sensitivity, models = [], [], []

    def windows(frames, stats=source_stats, stride=None):
        return Windows(frames, *stats, cfg.window, stride or cfg.eval_stride)

    for seed in cfg.seeds:
        model = GRU(cfg.hidden)
        path = base / str(seed) / "float32.pt"
        if (base / "models.json").exists():
            recorded = json.loads((base / "models.json").read_text())[str(seed)]["sha256"]
            if digest(path) != recorded:
                raise ValueError("Source checkpoint checksum mismatch")
        model.load_state_dict(torch.load(path, weights_only=True))
        models.append(dict(seed=seed, source_model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for temp in experiment.temperatures:
            tests = [f for f in fs if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == temp]
            adapt = [
                f.loc[f.time_s < cfg.adapt_seconds].copy()
                for f in fs
                if f.split.iloc[0] == "adapt" and f.ambient_c.iloc[0] == temp
            ]
            ds = windows(tests)
            y, direct = predict(model, ds, cfg.batch_size)
            outputs = {"direct": direct}
            target = statistics(adapt, cfg.get("temperature_scale"))
            _, outputs["target stats"] = predict(model, windows(tests, target), cfg.batch_size)
            adapt_windows = windows(adapt, stride=experiment.adapt.stride)
            for mode in experiment.adapt.modes:
                seed_all(seed)
                tuned, _ = fit(model, adapt_windows, fitting, mode=mode)
                _, outputs["tune " + mode] = predict(tuned, ds, cfg.batch_size)
            c20 = np.array([tests[j].soc_c20.iloc[e] for j, e in ds.index])
            for method, prediction in outputs.items():
                for scope, values in metrics(y, prediction).items():
                    rows.append(dict(seed=seed, ambient_c=temp, method=method, scope=scope, **values))
                # Same endpoints, predictions unchanged: label-definition sensitivity only.
                sensitivity.append(
                    dict(
                        seed=seed,
                        ambient_c=temp,
                        method=method,
                        nominal_rmse=metrics(y, prediction)["full"]["rmse"],
                        c20_rmse=metrics(c20, prediction)["full"]["rmse"],
                        label_max_delta_pp=float(np.max(abs(y - c20)) * 100),
                    )
                )
            pd.DataFrame(
                dict(
                    file_id=[tests[j].file_id.iloc[0] for j, _ in ds.index],
                    time_s=[float(tests[j].time_s.iloc[e]) for j, e in ds.index],
                    soc=y,
                    soc_c20=c20,
                    **outputs,
                )
            ).to_csv(out / f"predictions_{seed}_{temp}.csv", index=False)
            print(f"LG seed {seed}, temperature {temp}: complete", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "capacity_sensitivity.csv", index=False)
    summary = raw.groupby(["method", "ambient_c", "scope"])[["rmse", "mae", "max_error"]].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                base_run=str(base),
                prepared_sha256=digest(prepared / "dataset.json"),
                models=models,
                adaptation=OmegaConf.to_container(experiment.adapt),
                test_cycles=["UDDS", "HWFET", "LA92", "US06"],
                scope="Single cell per dataset; chemistry, instruments and protocols also differ",
                capacity="Primary LG nominal 3 Ah. C/20 is retrospective sensitivity, no retraining.",
            ),
            indent=2,
        )
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, g in raw[raw.scope == "common"].groupby("method"):
        s = g.groupby("ambient_c").rmse.agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", label=method)
    ax.set(
        xlabel="LG ambient temperature (C)",
        ylabel="SoC RMSE (percentage points)",
        title="Panasonic -> LG HG2: common SoC 40-100%",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cross_cell.png", dpi=160)
    plt.close(fig)
    (out / "COMPLETE").write_text("E7 completed from persisted LG data; see experiment.yaml for scope.\n")


if __name__ == "__main__":
    main()
