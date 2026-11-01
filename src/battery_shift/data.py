"""Strict file-level preparation; positive current/Ah denotes discharge."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import Dataset

FEATURES = ["voltage_v", "current_a", "temp_c"]


def labels(ah, capacity):
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("Capacity must be positive")
    soc = 1 - np.asarray(ah) / capacity
    if not np.isfinite(soc).all() or soc.min() < -0.05 or soc.max() > 1.05:
        raise ValueError("Implausible SoC; inspect sign, counter reset, and capacity")
    return soc  # Do not clip away label problems.


def validate_manifest(entries):
    ids = [e["file_id"] for e in entries]
    paths = [e["path"] for e in entries]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Duplicate file across splits")
    for e in entries:
        if e["split"] not in {"train", "validation", "test", "adapt"}:
            raise ValueError("Unknown split")
        if e["split"] in {"train", "validation"} and e["ambient_c"] != 25:
            raise ValueError("Target data in source training/validation")


def validate_intervals(intervals):
    previous_end = None
    for start, end, file_id in sorted(intervals):
        if previous_end is not None and start <= previous_end:
            raise ValueError(f"Overlapping absolute recording intervals: {file_id}")
        previous_end = end


def prepare(root, cfg, output=None):
    root = Path(root)
    entries = json.loads((root / "data/splits/manifest.json").read_text())["files"]
    validate_manifest(entries)
    frames, intervals, audit = [], [], []
    mapping = dict(
        Time="time_s", Voltage="voltage_v", Current="current_a", Battery_Temp_degC="temp_c", Ah="ah"
    )
    for e in entries:
        p = root / "data/raw" / e["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != e["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {p}")
        m = loadmat(p, simplify_cells=True)["meas"]
        raw = pd.DataFrame({dst: np.asarray(m[src], dtype=float) for src, dst in mapping.items()})
        if not np.isfinite(raw.to_numpy()).all():
            raise ValueError(f"Non-finite measurements: {p}")
        stamp_text = [s + " 12:00:00 AM" if " " not in s else s for s in m["TimeStamp"]]
        stamps = pd.to_datetime(stamp_text, format="%m/%d/%Y %I:%M:%S %p")
        intervals.append((stamps.min(), stamps.max(), e["file_id"]))
        if (np.diff(raw.time_s) < 0).any():
            raise ValueError(f"Backward relative time: {p}")
        e = {**e, "duplicate_times_removed": int(raw.time_s.duplicated().sum())}
        raw = raw.drop_duplicates("time_s", keep="last")
        t = raw.time_s.to_numpy()
        if (np.diff(t) <= 0).any():
            raise ValueError(f"Non-increasing relative time: {p}")
        grid = np.arange(0, t[-1] - t[0] + 1e-8, cfg.sample_period_s)
        frame = pd.DataFrame(
            {c: np.interp(grid, t - t[0], raw[c]) for c in mapping.values() if c != "time_s"}
        )
        frame["time_s"] = grid
        frame["current_a"] *= -1
        frame["ah"] *= -1
        frame["soc"] = labels(frame.ah, cfg.capacity_ah)
        for k in ["file_id", "ambient_c", "cycle", "split"]:
            frame[k] = e[k]
        frames.append(frame)
        audit.append(
            {
                **e,
                "rows": len(frame),
                "start": str(stamps.min()),
                "end": str(stamps.max()),
                "soc_min": float(frame.soc.min()),
                "soc_max": float(frame.soc.max()),
                "max_gap_s": float(np.diff(t).max()),
            }
        )
    validate_intervals(intervals)
    out = Path(output) if output else root / "data/interim"
    out.mkdir(parents=True, exist_ok=True)
    from .artifacts import save_frames

    save_frames(
        frames,
        out,
        dict(
            dataset="panasonic",
            capacity_ah=float(cfg.capacity_ah),
            sample_period_s=float(cfg.sample_period_s),
            source_manifest_sha256=hashlib.sha256(
                (root / "data/splits/manifest.json").read_bytes()
            ).hexdigest(),
        ),
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    return frames


def statistics(frames, temperature_scale=None):
    x = np.concatenate([f[FEATURES].to_numpy() for f in frames])
    mean, std = x.mean(0).astype("float32"), np.maximum(x.std(0), 1e-6).astype("float32")
    if temperature_scale is not None:
        # Fixed physical rule (T - offset) / scale: a single-temperature source has almost no
        # temperature spread, so a fitted z-score would place cold data tens of SD away.
        mean[2], std[2] = temperature_scale
    return mean, std


class Windows(Dataset):
    def __init__(self, frames, mean, std, length, stride, feature_mask=None):
        if length < 1 or stride < 1:
            raise ValueError("Invalid window length/stride")
        self.length = length
        self.x = [((f[FEATURES].to_numpy(dtype="float32") - mean) / std).astype("float32") for f in frames]
        if feature_mask is not None:
            mask = np.asarray(feature_mask, dtype="float32")
            if mask.shape != (3,) or not np.isin(mask, [0, 1]).all():
                raise ValueError("feature_mask must contain three binary entries")
            self.x = [x * mask for x in self.x]
        self.y = [f.soc.to_numpy(dtype="float32") for f in frames]
        self.index = [(j, end) for j, f in enumerate(frames) for end in range(length - 1, len(f), stride)]
        if not self.index:
            raise ValueError("No complete windows available")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j, end = self.index[i]
        return self.x[j][end - self.length + 1 : end + 1], self.y[j][end]
