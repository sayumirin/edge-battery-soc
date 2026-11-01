"""Confirmatory re-run of the temperature-transfer experiment with two fixes.

All settings below were fixed before any of these runs were evaluated:

  A  original     : z-score of V, I, T on 25 C training data (loaded from the
                    reproduced run; same models as results/mvp-v1)
  B  phys_T       : V and I z-scored on 25 C training data; temperature scaled
                    by a fixed physical rule (T - 25) / 20, no data-fitted stats
  C  tuning       : first 1,800 s of target Cycle 1, windows of 300 s at
                    stride 10 s (151 windows), batch 32, exactly 300 Adam
                    steps at lr 1e-3, no early stopping, no test data used.
                    The original protocol ran 3 steps (51 windows, batch 128,
                    3 epochs).
  D  multi_temp   : leave-one-temperature-out. Train on 25 C Cycle 1-4 + NN
                    plus the Cycle 1 recordings of the other three cold
                    temperatures; the target temperature is never seen.
                    Temperature uses the physical rule of B.

Usage (repository root): uv run python scripts/fix_check.py [original_run_dir]
then: uv run python scripts/fix_check_summary.py

Metrics: RMSE in percentage points on SoC 40-100 % ("common", the original
main metric) and the same set restricted to endpoints with |I| > 0.05 A
("active", which drops long rests at 100 % SoC).
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from battery_shift.artifacts import load_panasonic
from battery_shift.data import FEATURES, Windows
from battery_shift.evaluate import predict
from battery_shift.models import GRU
from battery_shift.train import fit, seed_all

ROOT = Path(__file__).resolve().parents[1]
# Source models of the original protocol (identical to results/mvp-v1).
ORIG = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/clean-env-full-v1")
OUT = ROOT / "results/fix-check-v1"
TEMPS = [25, 10, 0, -10, -20]
COLD = [10, 0, -10, -20]
PHYS_T = (25.0, 20.0)  # (offset, scale) for temperature, fixed a priori

cfg = OmegaConf.load(ROOT / "configs/mvp.yaml")
torch.set_num_threads(2)
OUT.mkdir(parents=True, exist_ok=True)
frames = load_panasonic(ROOT / "data/interim", cfg)


def pick(split, temp=None, cycle=None):
    return [
        f
        for f in frames
        if f.split.iloc[0] == split
        and (temp is None or f.ambient_c.iloc[0] == temp)
        and (cycle is None or f.cycle.iloc[0] == cycle)
    ]


train25, val25 = pick("train"), pick("validation")
tests = {t: pick("test", t) for t in TEMPS}
adapt_full = {t: pick("adapt", t) for t in COLD}
adapt_1800 = {t: [f.loc[f.time_s < cfg.adapt_seconds].copy() for f in adapt_full[t]] for t in COLD}


def zstats(fs):
    x = np.concatenate([f[FEATURES].to_numpy() for f in fs])
    return x.mean(0).astype("float32"), np.maximum(x.std(0), 1e-6).astype("float32")


def phys_stats(fs):
    mean, std = zstats(fs)
    mean[2], std[2] = PHYS_T
    return mean, std


def win(fs, stats, stride):
    return Windows(fs, *stats, cfg.window, stride)


def endpoint_current(fs, ds):
    return np.array([fs[j].current_a.iloc[e] for j, e in ds.index])


rows = []


def record(seed, variant, method, temp, model, stats, extra=None):
    ds = win(tests[temp], stats, cfg.eval_stride)
    y, p = predict(model, ds, cfg.batch_size)
    i = endpoint_current(tests[temp], ds)
    common = (y >= 0.4) & (y <= 1)
    for scope, mask in [("common", common), ("active", common & (np.abs(i) > 0.05))]:
        err = 100 * (p[mask] - y[mask])
        rows.append(
            dict(
                seed=seed,
                variant=variant,
                method=method,
                ambient_c=temp,
                scope=scope,
                n=int(mask.sum()),
                rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(np.abs(err))),
                pred_min=float(p[mask].min()),
                pred_max=float(p[mask].max()),
                **(extra or {}),
            )
        )


def tune(model, stats, temp, mode, seed):
    tcfg = copy.deepcopy(cfg)
    tcfg.batch_size, tcfg.adapt_epochs = 32, 60  # 151 windows -> 5 batches x 60 = 300 steps
    ds = win(adapt_1800[temp], stats, 10)
    steps = int(np.ceil(len(ds) / tcfg.batch_size)) * tcfg.adapt_epochs
    seed_all(seed)
    tuned, hist = fit(copy.deepcopy(model), ds, tcfg, mode=mode)
    return tuned, dict(adapt_windows=len(ds), adapt_steps=steps, adapt_final_mse=hist[-1]["loss"])


def train(fs, stats, seed):
    seed_all(seed)
    model, hist = fit(
        GRU(cfg.hidden), win(fs, stats, cfg.train_stride), cfg, win(val25, stats, cfg.eval_stride)
    )
    return model, hist


start = time.time()
orig_norm = json.loads((ORIG / "normalization.json").read_text())
orig_stats = (np.array(orig_norm["mean"], "float32"), np.array(orig_norm["std"], "float32"))
log = {}
for seed in cfg.seeds:
    # A: original models, reloaded from the reproduced run
    a = GRU(cfg.hidden)
    a.load_state_dict(torch.load(ORIG / str(seed) / "float32.pt"))
    a.eval()
    for t in TEMPS:
        record(seed, "A_original_zscoreT", "direct", t, a, orig_stats)
    for t in COLD:
        tuned, info = tune(a, orig_stats, t, "all", seed)
        record(seed, "A_original_zscoreT", "tune_all_300", t, tuned, orig_stats, info)
    print(f"[{time.time() - start:6.0f}s] seed {seed} A done", flush=True)

    # B/C: physical temperature scaling, source-only training
    stats_b = phys_stats(train25)
    b, hist = train(train25, stats_b, seed)
    log[f"B_{seed}"] = hist
    for t in TEMPS:
        record(seed, "B_phys_T", "direct", t, b, stats_b)
    for t in COLD:
        for mode in ["head", "all"]:
            tuned, info = tune(b, stats_b, t, mode, seed)
            record(seed, "B_phys_T", f"tune_{mode}_300", t, tuned, stats_b, info)
    print(f"[{time.time() - start:6.0f}s] seed {seed} B/C done", flush=True)

    # D: leave-one-temperature-out multi-temperature training
    for t in COLD:
        fs = train25 + [f for u in COLD if u != t for f in adapt_full[u]]
        stats_d = phys_stats(fs)
        d, hist = train(fs, stats_d, seed)
        log[f"D_{seed}_{t}"] = hist
        record(seed, "D_multi_temp_phys_T", "direct", t, d, stats_d)
    print(f"[{time.time() - start:6.0f}s] seed {seed} D done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

(OUT / "training_history.json").write_text(json.dumps(log, indent=1))
df = pd.DataFrame(rows)
df.to_csv(OUT / "metrics.csv", index=False)
summary = df.groupby(["scope", "variant", "method", "ambient_c"]).rmse.agg(["mean", "std"]).reset_index()
summary.to_csv(OUT / "summary.csv", index=False)
print(f"done in {time.time() - start:.0f}s", file=sys.stderr)
