import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from battery_shift.data import Windows, labels, statistics, validate_intervals, validate_manifest
from battery_shift.evaluate import metrics
from battery_shift.export import export_models
from battery_shift.models import GRU, coulomb_count


def frame(n=12, offset=0):
    return pd.DataFrame(
        dict(
            voltage_v=np.arange(n) + offset,
            current_a=np.ones(n),
            temp_c=np.ones(n) * 25,
            soc=np.linspace(1, 0.5, n),
        )
    )


def test_pinned_manifest_disjoint():
    m = json.loads(Path("data/splits/manifest.json").read_text())
    validate_manifest(m["files"])
    assert len(m["files"]) == 26
    assert {e["ambient_c"] for e in m["files"]} == {-20, -10, 0, 10, 25}
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(m["files"] + m["files"][:1])


def test_time_overlap_rejected_even_different_filenames():
    with pytest.raises(ValueError, match="Overlapping"):
        validate_intervals([(1, 10, "a"), (9, 20, "b")])
    validate_intervals([(1, 10, "a"), (11, 20, "b")])


def test_labels_signed_and_capacity():
    np.testing.assert_allclose(labels([0, 1.45, 2.9], 2.9), [1, 0.5, 0])
    with pytest.raises(ValueError):
        labels([0], 0)
    with pytest.raises(ValueError):
        labels([-2.9], 2.9)


def test_integral_nonuniform_times_and_bias():
    t = np.array([0, 600, 1800, 3600])
    np.testing.assert_allclose(coulomb_count(t, np.ones(4) * 2.9, 1, 2.9), 1 - t / 3600)
    nominal = coulomb_count(t, np.zeros(4), 1, 2.9)
    biased = coulomb_count(t, np.ones(4) * 0.05, 1, 2.9)
    assert nominal[-1] - biased[-1] == pytest.approx(0.05 / 2.9)


def test_windows_never_cross_file_and_stats_train_only():
    train, target = frame(), frame(offset=1000)
    mean, std = statistics([train])
    ds = Windows([train, target], mean, std, 5, 1)
    assert len(ds) == 16
    assert ds[7][1] == pytest.approx(0.5)
    assert ds[8][0].min(axis=0)[0] > 100
    np.testing.assert_allclose(statistics([train])[0], mean)


def test_fixed_temperature_scale_keeps_fitted_voltage_and_current():
    train = frame()
    fitted = statistics([train])
    mean, std = statistics([train], (25.0, 20.0))
    np.testing.assert_allclose(mean[:2], fitted[0][:2])
    np.testing.assert_allclose(std[:2], fitted[1][:2])
    assert (mean[2], std[2]) == (25.0, 20.0)
    cold = frame()
    cold["temp_c"] = -10.0
    assert Windows([cold], mean, std, 5, 1)[0][0][0, 2] == pytest.approx(-1.75)


def test_common_range_and_units():
    result = metrics(np.array([0.2, 0.5, 1]), np.array([0.4, 0.55, 1.05]))
    assert result["common"]["rmse"] == pytest.approx(5)
    assert result["common"]["n"] == 2
    assert metrics(np.array([0.1]), np.array([0.1]))["common"]["rmse"] is None


def test_streaming_matches_same_history():
    torch.manual_seed(7)
    model = GRU(16).eval()
    x = torch.randn(2, 30, 3)
    state = None
    with torch.no_grad():
        for i in range(x.shape[1]):
            p, state = model.step(x[:, i : i + 1], state)
        torch.testing.assert_close(p, model(x), atol=1e-6, rtol=1e-5)
    assert sum(p.numel() for p in GRU(32).parameters()) < 50000


def test_export_and_int8_inference(tmp_path):
    torch.manual_seed(7)
    model = GRU(16).eval()
    x = torch.randn(3, 30, 3)
    quantized, info = export_models(model, x, tmp_path)
    assert info["onnx_max_abs_error"] < 1e-5
    with torch.no_grad():
        assert torch.max(abs(model(x) - quantized(x))) < 0.05
