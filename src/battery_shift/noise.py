"""Deterministic input perturbations; labels and original frames stay unchanged."""

import numpy as np
from .data import FEATURES


def perturb(frame, kind, strength, seed):
    if not np.isfinite(strength):
        raise ValueError("Noise strength must be finite")
    if strength < 0 and kind not in {"current", "temperature"}:
        raise ValueError("Noise strength must be nonnegative")
    f = frame.copy(deep=True)
    rng = np.random.default_rng(seed)
    if kind == "voltage":
        f["voltage_v"] += rng.normal(0, 0.005 * strength, len(f))
    elif kind == "current":
        f["current_a"] += 0.05 * strength
    elif kind == "temperature":
        f["temp_c"] += 2 * strength
    elif kind == "dropout":
        if strength > 1:
            raise ValueError("Dropout must be <= 1")
        mask = rng.random(len(f)) < strength
        mask[0] = False
        f.loc[f.index[mask], FEATURES] = np.nan
        f[FEATURES] = f[FEATURES].ffill()
    elif kind == "hold":
        step = int(strength)
        if step < 1 or step != strength:
            raise ValueError("Hold interval must be a positive integer")
        f[FEATURES] = f[FEATURES].iloc[(np.arange(len(f)) // step) * step].to_numpy()
    elif kind == "crop":
        if strength >= 1:
            raise ValueError("Crop fraction must be < 1")
        f = f.iloc[int(len(f) * strength) :].copy()
    else:
        raise ValueError("Unknown noise type")
    return f


def augment(frames, seed):
    return [
        perturb(
            perturb(perturb(f, "voltage", 1, seed + j), "current", 1, seed + j), "temperature", 1, seed + j
        )
        for j, f in enumerate(frames)
    ]
