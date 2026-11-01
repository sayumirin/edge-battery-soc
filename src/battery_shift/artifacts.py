"""Checked, explicit disk boundaries between preparation, fitting and evaluation."""

import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_frames(frames, folder, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    ids = set()
    for frame in frames:
        file_id = str(frame.file_id.iloc[0])
        if file_id in ids or not all(c.isalnum() or c in "_-" for c in file_id):
            raise ValueError("Duplicate or unsafe prepared file ID")
        ids.add(file_id)
        path = folder / (file_id + ".parquet")
        frame.to_parquet(path, index=False)
        entries.append(dict(file_id=file_id, file=path.name, rows=len(frame), sha256=digest(path)))
    (folder / "dataset.json").write_text(
        json.dumps(dict(schema=1, metadata=metadata, files=entries), indent=2)
    )


def load_frames(folder, expected=None):
    folder = Path(folder)
    manifest = json.loads((folder / "dataset.json").read_text())
    if manifest["schema"] != 1 or not manifest["files"]:
        raise ValueError("Unknown or empty prepared dataset")
    for key, value in (expected or {}).items():
        if manifest["metadata"].get(key) != value:
            raise ValueError(f"Prepared dataset mismatch: {key}")
    frames, seen = [], set()
    for entry in manifest["files"]:
        path = (folder / entry["file"]).resolve()
        if path.parent != folder.resolve() or entry["file_id"] in seen:
            raise ValueError("Duplicate or unsafe prepared file")
        if digest(path) != entry["sha256"]:
            raise ValueError("Prepared file checksum mismatch: " + entry["file"])
        frame = pd.read_parquet(path)
        if len(frame) != entry["rows"] or not frame.file_id.eq(entry["file_id"]).all():
            raise ValueError("Prepared rows/identity mismatch")
        seen.add(entry["file_id"])
        frames.append(frame)
    return frames


def load_panasonic(folder, cfg):
    return load_frames(
        folder,
        dict(
            dataset="panasonic",
            capacity_ah=float(cfg.capacity_ah),
            sample_period_s=float(cfg.sample_period_s),
        ),
    )
