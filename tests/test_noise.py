import numpy as np
import pandas as pd
import pytest
from battery_shift.noise import perturb


def sample():
    return pd.DataFrame(
        dict(
            voltage_v=np.arange(20.0),
            current_a=np.ones(20),
            temp_c=np.arange(20.0),
            soc=np.linspace(1, 0.5, 20),
        )
    )


def test_noise_keeps_labels_and_does_not_mutate():
    f = sample()
    original = f.copy()
    for kind, strength in [("voltage", 2), ("current", 1), ("temperature", 2), ("dropout", 0.9), ("hold", 5)]:
        a = perturb(f, kind, strength, 17)
        b = perturb(f, kind, strength, 17)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_series_equal(a.soc, f.soc)
        pd.testing.assert_frame_equal(f, original)


def test_sample_hold_is_causal():
    f = sample()
    held = perturb(f, "hold", 5, 17)
    np.testing.assert_array_equal(held.voltage_v, np.repeat([0, 5, 10, 15], 5))
    dropped = perturb(f, "dropout", 1, 17)
    np.testing.assert_array_equal(dropped.voltage_v, 0)
    assert perturb(f, "crop", 0.5, 17).index[0] == 10


@pytest.mark.parametrize(
    "kind,strength", [("dropout", 1.1), ("hold", 0), ("hold", 1.2), ("crop", 1), ("voltage", -1)]
)
def test_invalid_noise_rejected(kind, strength):
    with pytest.raises(ValueError):
        perturb(sample(), kind, strength, 17)
