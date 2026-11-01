import json
import numpy as np
import pandas as pd
import pytest
import torch
from battery_shift.artifacts import load_frames, save_frames
from battery_shift.experiments import streaming_predictions, train_source, evaluate_source
from battery_shift.models import GRU
from battery_shift.noise import perturb
from omegaconf import OmegaConf


def frame():
    return pd.DataFrame(
        dict(
            file_id=["test_file"] * 4,
            voltage_v=[4.0, 3.9, 3.8, 3.7],
            current_a=[1.0, 2.0, 1.0, 0.0],
            temp_c=[25.0] * 4,
            soc=[1.0, 0.9, 0.8, 0.7],
        )
    )


def test_prepared_roundtrip_needs_no_raw_files(tmp_path):
    original = frame()
    save_frames([original], tmp_path, dict(dataset="unit", capacity_ah=3.0))
    pd.testing.assert_frame_equal(load_frames(tmp_path, dict(dataset="unit"))[0], original)
    with pytest.raises(ValueError, match="mismatch"):
        load_frames(tmp_path, dict(capacity_ah=2.9))


def test_prepared_rejects_corruption_and_duplicate_files(tmp_path):
    save_frames([frame()], tmp_path, {})
    path = tmp_path / "test_file.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_frames(tmp_path)
    save_frames([frame()], tmp_path, {})
    manifest = json.loads((tmp_path / "dataset.json").read_text())
    manifest["files"] *= 2
    (tmp_path / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        load_frames(tmp_path)


@pytest.mark.parametrize("kind,unit", [("current", 0.05), ("temperature", 2.0)])
def test_signed_bias_changes_only_input(kind, unit):
    f = frame()
    column = "current_a" if kind == "current" else "temp_c"
    for sign in [-4, -1, 0, 1, 4]:
        p = perturb(f, kind, sign, 1)
        np.testing.assert_allclose(p[column] - f[column], sign * unit, atol=1e-7)
        pd.testing.assert_series_equal(p.soc, f.soc)
    with pytest.raises(ValueError):
        perturb(f, kind, float("nan"), 1)


def test_chunked_streaming_carries_state_and_resets_per_call():
    torch.manual_seed(9)
    model = GRU(8).eval()
    x = np.random.default_rng(9).normal(size=(47, 3)).astype("float32")
    state, expected = None, []
    with torch.inference_mode():
        for row in x:
            y, state = model.step(torch.from_numpy(row).reshape(1, 1, 3), state)
            expected.append(y.item())
    for chunk in [1, 7, 512]:
        np.testing.assert_allclose(streaming_predictions(model, x, chunk), expected, atol=1e-6)
    streaming_predictions(model, x * 10, 7)
    np.testing.assert_allclose(streaming_predictions(model, x, 7), expected, atol=1e-6)


def test_streaming_and_window_histories_are_not_assumed_equivalent():
    torch.manual_seed(4)
    model = GRU(4).eval()
    x = np.random.default_rng(4).normal(size=(30, 3)).astype("float32")
    full = streaming_predictions(model, x, 5)
    short = streaming_predictions(model, x[-1:], 1)
    assert abs(full[-1] - short[-1]) > 1e-6


def test_fit_and_evaluate_from_disk_without_raw_loader(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A downstream stage tried to parse raw data")

    monkeypatch.setattr("battery_shift.data.loadmat", forbidden)
    prepared = tmp_path / "prepared"
    frames = []
    for split in ["train", "validation", "test"]:
        f = frame()
        f["file_id"], f["split"], f["ambient_c"], f["time_s"] = split, split, 25, np.arange(len(f))
        frames.append(f)
    save_frames(frames, prepared, dict(dataset="panasonic", capacity_ah=2.9, sample_period_s=1.0))
    training = OmegaConf.create(
        dict(
            hidden=4,
            window=2,
            train_stride=1,
            eval_stride=1,
            batch_size=2,
            epochs=1,
            patience=1,
            learning_rate=0.001,
            capacity_ah=2.9,
            sample_period_s=1.0,
            threads=1,
        )
    )
    training_path = tmp_path / "training.yaml"
    OmegaConf.save(training, training_path)
    cfg = OmegaConf.create(dict(prepared=str(prepared), training_config=str(training_path), seeds=[7]))
    source = tmp_path / "source"
    source.mkdir()
    train_source(cfg, source)
    norms = json.loads((source / "normalization.json").read_text())
    stats = tuple(np.array(norms[k], dtype="float32") for k in ["mean", "std"])
    output = tmp_path / "evaluation"
    output.mkdir()
    evaluation = OmegaConf.create(dict(experiment="E1", seeds=[7], temperatures=[25], initial_soc_error_pp=5))
    evaluate_source(evaluation, output, source, training, load_frames(prepared), stats)
    assert len(pd.read_csv(output / "metrics.csv")) == 4
