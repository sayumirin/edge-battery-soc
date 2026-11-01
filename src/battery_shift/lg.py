"""Original LG HG2 MAT adapter: archive members, not duplicate CSV exports."""

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .data import labels, validate_intervals

ARCHIVE_SHA256 = "a5d1f057d02bf15e37bfd222ebcdfe0a6c13ab34bec82a2419755913b7412959"
SOURCE = "https://data.mendeley.com/datasets/b5mj79w5w9/2"
TEMPERATURES = {"25degC": 25, "10degC": 10, "0degC": 0, "n10degC": -10, "n20degC": -20}


def resample(meas, capacity=3.0):
    mapping = dict(
        Time="time_s", Voltage="voltage_v", Current="current_a", Battery_Temp_degC="temp_c", Ah="ah"
    )
    raw = pd.DataFrame({v: np.asarray(meas[k], dtype=float) for k, v in mapping.items()})
    if not np.isfinite(raw.to_numpy()).all() or (np.diff(raw.time_s) < 0).any():
        raise ValueError("Nonfinite LG measurement or backward time")
    duplicates = int(raw.time_s.duplicated().sum())
    raw = raw.drop_duplicates("time_s", keep="last")
    if len(raw) < 2 or abs(raw.ah.iloc[0]) > 0.01:
        raise ValueError("LG drive must begin at a reset Ah counter after full charge")
    t = raw.time_s.to_numpy() - raw.time_s.iloc[0]
    grid = np.arange(0, t[-1] + 1e-8, 1.0)
    frame = pd.DataFrame({c: np.interp(grid, t, raw[c]) for c in mapping.values() if c != "time_s"})
    frame["time_s"] = grid
    frame["current_a"] *= -1
    frame["ah"] *= -1
    frame["soc"] = labels(frame.ah, capacity)
    return frame, {"duplicate_times_removed": duplicates, "max_gap_s": float(np.diff(t).max())}


def prepare_lg(archive, output):
    archive, output = Path(archive), Path(output)
    if not archive.exists():
        raise FileNotFoundError(f"Download All from {SOURCE} and save the original ZIP to {archive}")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("LG archive checksum mismatch; audit a new archive before use")
    output.mkdir(parents=True, exist_ok=True)
    frames, audit, capacities, intervals, excluded = [], [], [], [], []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for folder, temp in TEMPERATURES.items():
            (cap_name,) = [
                n for n in names if n.startswith(folder + "/") and "C20DisCh_" in n and n.endswith(".mat")
            ]
            cm = loadmat(io.BytesIO(z.read(cap_name)), simplify_cells=True)["meas"]
            # Original Ah counter starts at zero before discharge. The first logged sample
            # is already one minute into discharge; retain its accumulated charge.
            discharge = (np.asarray(cm["Current"]) < -0.05) & (np.asarray(cm["Current"]) > -0.3)
            capacity = float((-np.asarray(cm["Ah"])[discharge]).max())
            if not 1 < capacity < 3.5:
                raise ValueError("Implausible LG C/20 capacity")
            capacities.append(
                dict(
                    ambient_c=temp,
                    capacity_ah=capacity,
                    path=cap_name,
                    sha256=hashlib.sha256(z.read(cap_name)).hexdigest(),
                )
            )
            for cycle in ["Mixed1", "UDDS", "HWFET", "LA92", "US06"]:
                (name,) = [
                    n
                    for n in names
                    if n.startswith(folder + "/") and f"_{cycle}_" in n and n.endswith(".mat")
                ]
                payload = z.read(name)
                meas = loadmat(io.BytesIO(payload), simplify_cells=True)["meas"]
                if temp == 10 and cycle == "LA92":
                    # Official archive has interleaved dates, four backward time jumps,
                    # and three nonfinite measurement rows. Do not sort/stitch a new test.
                    excluded.append(
                        dict(
                            path=name,
                            sha256=hashlib.sha256(payload).hexdigest(),
                            reason="4 backward relative-time jumps; 3 nonfinite rows; interleaved Nov 25/26 records",
                        )
                    )
                    continue
                frame, details = resample(meas)
                # MAT exports switch between US and day-first dates at lower temperatures.
                # Use original cycler CSV timestamps (US format), without mixing duplicate
                # measurement records into the learning dataset.
                number = Path(name).name.split(" ")[1].split("_")[0]
                csv_name = f"{folder}/{number}_{cycle}.csv"
                csv_bytes = z.read(csv_name)
                dates = [
                    line.split(",")[0]
                    for line in csv_bytes.decode("cp1252").splitlines()
                    if re.match(r"\d+/\d+/\d+ ", line)
                ]
                stamps = pd.to_datetime([dates[0], dates[-1]], format="%m/%d/%Y %I:%M:%S %p")
                if (
                    abs((stamps[-1] - stamps[0]).total_seconds() - float(meas["Time"][-1] - meas["Time"][0]))
                    > 2
                ):
                    raise ValueError("Original CSV and MAT duration disagree")
                file_id = f"LG_{temp}_{cycle}"
                intervals.append((stamps.min(), stamps.max(), file_id))
                split = "adapt" if cycle == "Mixed1" else "test"
                for key, value in dict(file_id=file_id, ambient_c=temp, cycle=cycle, split=split).items():
                    frame[key] = value
                frame["soc_c20"] = labels(frame.ah, capacity)
                frames.append(frame)
                audit.append(
                    dict(
                        file_id=file_id,
                        path=name,
                        sha256=hashlib.sha256(payload).hexdigest(),
                        timestamp_source=csv_name,
                        timestamp_sha256=hashlib.sha256(csv_bytes).hexdigest(),
                        ambient_c=temp,
                        split=split,
                        rows=len(frame),
                        start=str(stamps.min()),
                        end=str(stamps.max()),
                        soc_min=float(frame.soc.min()),
                        **details,
                    )
                )
    validate_intervals(intervals)
    (output / "data_audit.json").write_text(
        json.dumps(
            dict(
                source=SOURCE,
                license="CC BY 4.0",
                archive_sha256=ARCHIVE_SHA256,
                nominal_capacity_ah=3.0,
                files=audit,
                c20_capacities=capacities,
                excluded=excluded,
            ),
            indent=2,
        )
    )
    return frames
