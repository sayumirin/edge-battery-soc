"""Run E4 input robustness, E8 empirical intervals and E9 online-update forgetting."""

import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from battery_shift.data import Windows, statistics
from battery_shift.artifacts import load_panasonic
from battery_shift.models import GRU, coulomb_count
from battery_shift.evaluate import predict, metrics
from battery_shift.train import fit, seed_all
from battery_shift.noise import augment, perturb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extensions.yaml")
    parser.add_argument("--output", default="runs/extensions-v1")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    base = Path(cfg.base_run)
    training = OmegaConf.load(base / "config.yaml")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, out / "config.yaml")
    torch.set_num_threads(training.threads)
    frames = load_panasonic("data/interim", training)
    train = [f for f in frames if f.split.iloc[0] == "train"]
    val = [f for f in frames if f.split.iloc[0] == "validation"]
    mean, std = statistics(train)

    def windows(fs, train_mode=False):
        return Windows(
            fs,
            mean,
            std,
            training.window,
            training.train_stride if train_mode else training.eval_stride,
            feature_mask=training.get("feature_mask"),
        )

    test = {
        t: [f for f in frames if f.split.iloc[0] == "test" and f.ambient_c.iloc[0] == t]
        for t in [25, 10, 0, -10, -20]
    }
    rows = []
    online = []
    models = []

    def row(seed, temp, kind, level, model_name, y, p):
        for scope, v in metrics(y, p).items():
            rows.append(
                dict(seed=seed, ambient_c=temp, noise=kind, level=level, model=model_name, scope=scope, **v)
            )

    for seed in cfg.seeds:
        seed_all(seed)
        model = GRU(training.hidden).eval()
        model.load_state_dict(torch.load(base / str(seed) / "float32.pt", weights_only=True))
        models.append(model)
        seed_all(seed)  # Match source model initialization before testing augmentation.
        augmented, history = fit(
            GRU(training.hidden), windows(augment(train, seed), True), training, windows(val)
        )
        torch.save(augmented.state_dict(), out / f"augmented_{seed}.pt")
        (out / f"augmented_{seed}.json").write_text(json.dumps(history))
        for temp, fs in test.items():
            for kind, levels in cfg.noise.items():
                for level in levels:
                    perturbed = [perturb(f, kind, float(level), seed + j) for j, f in enumerate(fs)]
                    for name, m in [("clean_training", model), ("augmented_training", augmented)]:
                        robust_windows = Windows(
                            perturbed,
                            mean,
                            std,
                            training.window,
                            cfg.robust_eval_stride,
                            feature_mask=training.get("feature_mask"),
                        )
                        y, p = predict(m, robust_windows, training.batch_size)
                        row(seed, temp, kind, level, name, y, p)
            print(f"E4 seed {seed} temperature {temp} complete", flush=True)
        online_cfg = copy.deepcopy(training)
        online_cfg.learning_rate = cfg.online_learning_rate
        online_cfg.adapt_epochs = cfg.online_epochs
        for temp in [10, 0, -10, -20]:
            f = next(f for f in frames if f.split.iloc[0] == "adapt" and f.ambient_c.iloc[0] == temp)
            if abs(float(f.soc.iloc[0]) - 1) > 0.001:
                raise ValueError("Online full-charge anchor not supported")
            pseudo = coulomb_count(f.time_s, f.current_a, 1, training.capacity_ah)
            for supervision in ["true_label", "coulomb_pseudo"]:
                updated = copy.deepcopy(model)
                for step in range(cfg.online_chunks + 1):
                    if step:
                        lo = (step - 1) * cfg.online_chunk_seconds
                        hi = step * cfg.online_chunk_seconds
                        chunk = f.loc[(f.time_s >= lo) & (f.time_s < hi)].copy()
                        if supervision == "coulomb_pseudo":
                            chunk["soc"] = pseudo[chunk.index]
                        seed_all(seed + step)
                        updated, _ = fit(updated, windows([chunk], True), online_cfg)
                    for domain, eval_temp in [("source", 25), ("target", temp)]:
                        y, p = predict(updated, windows(test[eval_temp]), training.batch_size)
                        for scope, v in metrics(y, p).items():
                            online.append(
                                dict(
                                    seed=seed,
                                    target_temp=temp,
                                    supervision=supervision,
                                    update=step,
                                    elapsed_seconds=step * cfg.online_chunk_seconds,
                                    evaluation_domain=domain,
                                    scope=scope,
                                    **v,
                                )
                            )
                torch.save(updated.state_dict(), out / f"online_{seed}_{temp}_{supervision}.pt")
        print(f"E9 seed {seed} complete", flush=True)
    pd.DataFrame(rows).to_csv(out / "e4_metrics.csv", index=False)
    pd.DataFrame(rows).groupby(["ambient_c", "noise", "level", "model", "scope"]).rmse.agg(
        ["mean", "std"]
    ).to_csv(out / "e4_summary.csv")
    pd.DataFrame(online).to_csv(out / "e9_metrics.csv", index=False)
    pd.DataFrame(online).groupby(
        ["target_temp", "supervision", "update", "evaluation_domain", "scope"]
    ).rmse.agg(["mean", "std"]).to_csv(out / "e9_summary.csv")
    # A three-seed ensemble. Validation is reused after early stopping: descriptive,
    # not an independent conformal calibration set; no nominal coverage guarantee.
    yv, pv = predict(models[0], windows(val))
    vp = [pv] + [predict(m, windows(val))[1] for m in models[1:]]
    q = float(np.quantile(abs(yv - np.mean(vp, axis=0)), cfg.interval_coverage, method="higher"))
    interval_rows = []
    for temp in [25, 10, 0, -10, -20]:
        y, p = predict(models[0], windows(test[temp]))
        pred = [p] + [predict(m, windows(test[temp]))[1] for m in models[1:]]
        center = np.mean(pred, axis=0)
        spread = np.std(pred, axis=0, ddof=1)
        for method, halfwidth in [
            ("raw_ensemble_1.645sd", 1.645 * spread),
            ("validation_residual_90pct", np.full_like(center, q)),
        ]:
            for scope, mask in [("full", np.ones(len(y), bool)), ("common", (y >= 0.4) & (y <= 1))]:
                interval_rows.append(
                    dict(
                        ambient_c=temp,
                        method=method,
                        scope=scope,
                        n=int(mask.sum()),
                        empirical_coverage=float(np.mean(abs(y[mask] - center[mask]) <= halfwidth[mask])),
                        mean_width_pp=float(np.mean(2 * halfwidth[mask]) * 100),
                    )
                )
    pd.DataFrame(interval_rows).to_csv(out / "e8_intervals.csv", index=False)
    (out / "limitations.json").write_text(
        json.dumps(
            dict(
                e4="Positive current/temperature offsets; augmentation combines 5mV, +50mA, +2C. Hold/forward fill are causal within each file. Crop changes endpoints and is evaluated separately.",
                e8="3 seed ensemble, not three independent ensemble repeats. Validation reused for early stopping. Empirical coverage only; temporal dependence and domain shift invalidate iid guarantees.",
                e9="Updates consume disjoint chronological chunks from adaptation Cycle 1 only. Test data never updates model. Pseudo labels assume verified full-charge anchor and nominal capacity; no replay regularizer. Source error across updates measures forgetting.",
            ),
            indent=2,
        )
    )
    (out / "COMPLETE").write_text("E4 E8 E9 complete\n")


if __name__ == "__main__":
    main()
