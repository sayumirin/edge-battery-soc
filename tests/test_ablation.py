import numpy as np
import pandas as pd
import pytest
from battery_shift.data import Windows, statistics


def test_temperature_mask_removes_only_temperature_and_preserves_samples():
    frame = pd.DataFrame(
        dict(voltage_v=[3, 4, 5, 6], current_a=[1, 2, 3, 4], temp_c=[-20, 0, 10, 25], soc=[1, 0.9, 0.8, 0.7])
    )
    original = frame.copy(deep=True)
    mean, std = statistics([frame])
    direct = Windows([frame], mean, std, 2, 1)
    masked = Windows([frame], mean, std, 2, 1, feature_mask=[1, 1, 0])
    assert direct.index == masked.index
    for i in range(len(direct)):
        np.testing.assert_array_equal(direct[i][0][:, :2], masked[i][0][:, :2])
        np.testing.assert_array_equal(masked[i][0][:, 2], 0)
        assert direct[i][1] == masked[i][1]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("mask", [[1, 0], [1, 1, 2], [1, float("nan"), 0]])
def test_invalid_mask_rejected(mask):
    frame = pd.DataFrame(dict(voltage_v=[3, 4], current_a=[1, 2], temp_c=[20, 21], soc=[1, 0.9]))
    mean, std = statistics([frame])
    with pytest.raises(ValueError, match="feature_mask"):
        Windows([frame], mean, std, 2, 1, feature_mask=mask)
