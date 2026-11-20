import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from battery_shift.models import GRU
from battery_shift.params import (
    digest,
    float_count,
    from_state_dict,
    load,
    normalize,
    render_header,
    save,
    to_model,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]


def make(hidden=4):
    torch.manual_seed(0)
    model = GRU(hidden).eval()
    params = from_state_dict(model.state_dict(), hidden, [3.6, 0.8, 25.0], [0.27, 2.1, 20.0],
                             name="test-model", version="1.2.3")
    return model, params


def test_export_round_trips_through_file(tmp_path):
    _, params = make()
    save(params, tmp_path / "p.json")
    assert load(tmp_path / "p.json") == params
    assert params["sha256"] == digest(params)


def test_reference_model_matches_source_model_exactly():
    model, params = make()
    raw = np.random.default_rng(1).normal([3.7, 1.0, 10.0], [0.1, 1.0, 10.0], (50, 3)).astype(np.float32)
    x = torch.from_numpy(normalize(params, raw))[None]
    with torch.inference_mode():
        expected, _ = model.gru(model.input(x))
        got, _ = to_model(params).gru(to_model(params).input(x))
    assert torch.equal(expected, got)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p: p["tensors"]["head_bias"].__setitem__(0, 0.5), "Checksum"),
        (lambda p: p["tensors"]["head_weight"].pop(), "head_weight must hold"),
        (lambda p: p["tensors"]["input_bias"].__setitem__(1, float("nan")), "non-finite"),
        (lambda p: p["tensors"]["feature_std"].__setitem__(2, 0.0), "positive"),
        (lambda p: p["tensors"]["feature_mask"].__setitem__(0, 0.5), "0 or 1"),
        (lambda p: p["tensors"].pop("gru_bias_hh_l0"), "missing"),
        (lambda p: p.update(version="v1"), "Version"),
        (lambda p: p.update(inputs=["current_a", "voltage_v", "temp_c"]), "Inputs"),
        (lambda p: p.update(hidden=8), "must hold"),
    ],
)
def test_validation_rejects_unusable_files(change, message):
    _, params = make()
    broken = copy.deepcopy(params)
    change(broken)
    # The checksum is checked last, so every other case fails on its own rule first.
    with pytest.raises(ValueError, match=message):
        validate(broken)


def test_header_carries_identity_and_every_tensor():
    _, params = make(hidden=4)
    header = render_header(params)
    assert "#define HIDDEN 4\n" in header
    assert f'#define PARAMS_ID "test-model 1.2.3 sha256:{params["sha256"][:16]}"' in header
    assert header.count("static const float") == len(params["tensors"])
    assert f"head_weight[4] = {{{params['tensors']['head_weight'][0]:.9e}f," in header


def test_released_parameter_file_is_valid():
    path = ROOT / "params/gru-source-v3.json"
    params = load(path)
    assert params["hidden"] == 32
    # 3,597 model parameters plus mean, std and mask (README: 14,424 bytes as float32).
    assert float_count(params) == 3597 + 9
    assert json.loads(path.read_text())["sha256"] == digest(params)
