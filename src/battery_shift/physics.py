"""One-RC ECM and EKF with Joseph covariance update, discharge-positive current."""

import numpy as np


def polarization(current, time_s, r1, tau):
    result = np.zeros(len(current))
    for k in range(1, len(current)):
        dt = time_s[k] - time_s[k - 1]
        if dt > 30:
            continue  # HPPC omitted intervals cannot be reconstructed.
        a = np.exp(-dt / tau)
        result[k] = a * result[k - 1] + r1 * (1 - a) * current[k - 1]
    return result


def ekf(frame, soc_grid, ocv_grid, r0, r1, tau, capacity=2.9):
    t = frame.time_s.to_numpy()
    i = frame.current_a.to_numpy()
    v = frame.voltage_v.to_numpy()
    x = np.array([float(frame.soc.iloc[0]) + 0.05, 0.0])
    p = np.diag([0.05**2, 0.02**2])
    out = []
    gradient = np.gradient(ocv_grid, soc_grid)
    for k in range(len(t)):
        if k:
            dt = t[k] - t[k - 1]
            a = np.exp(-dt / tau)
            transition = np.diag([1, a])
            x = np.array([x[0] - i[k - 1] * dt / (3600 * capacity), a * x[1] + r1 * (1 - a) * i[k - 1]])
            p = transition @ p @ transition.T + np.diag([1e-8 * dt, 1e-6 * dt])
        h = np.array([np.interp(x[0], soc_grid, gradient), -1.0])
        voltage = np.interp(x[0], soc_grid, ocv_grid) - r0 * i[k] - x[1]
        noise = 0.02**2
        k_gain = p @ h / (h @ p @ h + noise)
        x = x + k_gain * (v[k] - voltage)
        x[0] = np.clip(x[0], 0, 1)
        a = np.eye(2) - np.outer(k_gain, h)
        p = a @ p @ a.T + noise * np.outer(k_gain, k_gain)
        out.append(x[0])
    return np.array(out)
