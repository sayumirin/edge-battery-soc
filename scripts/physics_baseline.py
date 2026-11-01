"""Fit source-temperature HPPC ECM, transfer unchanged to five temperatures."""

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares
from battery_shift.physics import polarization, ekf
from battery_shift.evaluate import metrics


def main():
    out = Path("results/physics-v1")
    out.mkdir(exist_ok=True, parents=True)
    path = "Panasonic 18650PF Data/25degC/5 pulse disch/03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat"
    sha = "21f5d94812f5059bc484ce44cccf1485c3b0dc02e36f36d80f9d24173ac1a8e6"
    raw = Path("data/raw") / path
    url = (
        "https://raw.githubusercontent.com/fededc88/Panasonic-18650PF-Data/6d720489c9b5b95624e448d44e8e2c0e48bdacaf/"
        + urllib.parse.quote(path)
    )
    if not raw.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            b = r.read()
        if hashlib.sha256(b).hexdigest() != sha:
            raise ValueError("HPPC checksum mismatch")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b)
    if hashlib.sha256(raw.read_bytes()).hexdigest() != sha:
        raise ValueError("HPPC checksum mismatch")
    hppc = loadmat(raw, simplify_cells=True)["meas"]
    ref = json.loads(Path("results/diagnostics-v1/capacity_reference.json").read_text())
    ocv_path = next(Path("data/raw").rglob("*C20_25dC.mat"))
    if hashlib.sha256(ocv_path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError("OCV checksum mismatch")
    d = loadmat(ocv_path, simplify_cells=True)["meas"]
    start = ref["start_index"]
    end = ref["end_index"]
    s = 1 - (d["Ah"][start] - d["Ah"][start : end + 1]) / 2.9
    v = d["Voltage"][start : end + 1]
    keep = s >= 0
    s = s[keep][::-1]
    v = v[keep][::-1]
    sg = np.linspace(0, 1, 101)
    og = np.maximum.accumulate(np.interp(sg, s, v))
    # Fixed source calibration only, no test-temperature parameter identification.
    t = hppc["Time"][::10]
    i = -hppc["Current"][::10]
    voltage = hppc["Voltage"][::10]
    soc = 1 + hppc["Ah"][::10] / 2.9

    def residual(params):
        r0, r1, tau = params
        return np.interp(soc, sg, og) - r0 * i - polarization(i, t, r1, tau) - voltage

    fitted = least_squares(
        residual, [0.03, 0.02, 30], bounds=([0.0001, 0.0001, 0.5], [0.5, 0.5, 1000]), max_nfev=100
    )
    if not fitted.success:
        raise RuntimeError(fitted.message)
    r0, r1, tau = fitted.x
    pd.DataFrame(dict(soc=sg, quasi_ocv_v=og)).to_csv(out / "ocv.csv", index=False)
    (out / "parameters.json").write_text(
        json.dumps(
            dict(
                r0_ohm=r0,
                r1_ohm=r1,
                tau_s=tau,
                fit_voltage_rmse_v=float(np.sqrt(np.mean(fitted.fun**2))),
                source_url=url,
                sha256=sha,
                limitations="One constant RC branch fitted to 25C HPPC, gaps >30s reset polarization. Quasi-OCV from later 25C C/20 discharge, on nominal 2.9Ah scale; ageing/hysteresis not corrected. No target calibration. Deterministic baseline; repeat seeds would not measure uncertainty.",
            ),
            indent=2,
        )
    )
    frames = [pd.read_parquet(p) for p in Path("data/interim").glob("*.parquet")]
    rows = []
    for temp in [25, 10, 0, -10, -20]:
        ys = []
        ps = []
        for f in frames:
            if f.split.iloc[0] != "test" or f.ambient_c.iloc[0] != temp:
                continue
            pred = ekf(f, sg, og, r0, r1, tau)
            indices = np.arange(299, len(f), 10)
            ys.extend(f.soc.iloc[indices])
            ps.extend(pred[indices])
        for scope, v in metrics(np.array(ys), np.array(ps)).items():
            rows.append(dict(ambient_c=temp, scope=scope, **v))
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
