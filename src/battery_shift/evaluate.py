import numpy as np
import torch
from torch.utils.data import DataLoader


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Invalid prediction arrays")
    result = {}
    for name, mask in [("full", np.ones(len(y), dtype=bool)), ("common", (y >= 0.4) & (y <= 1))]:
        err = 100 * (pred[mask] - y[mask])
        result[name] = dict(
            n=len(err),
            rmse=float(np.sqrt(np.mean(err**2))) if len(err) else None,
            mae=float(np.mean(abs(err))) if len(err) else None,
            max_error=float(np.max(abs(err))) if len(err) else None,
        )
    return result


def predict(model, dataset, batch_size=128):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=batch_size):
            ps.append(model(x).numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)
