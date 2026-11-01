import numpy as np
import pytest
from battery_shift.lg import resample


def measurements():
    return dict(
        Time=[0, 1, 1, 2],
        Voltage=[4.2, 4.1, 4.0, 3.9],
        Current=[-3.0, -3.0, -3.0, -3.0],
        Battery_Temp_degC=[25.0] * 4,
        Ah=[0.0, -0.001, -0.001, -0.002],
    )


def test_lg_sign_duplicate_and_label():
    f, audit = resample(measurements())
    assert audit["duplicate_times_removed"] == 1
    assert f.voltage_v.iloc[1] == 4.0
    np.testing.assert_allclose(f.current_a, 3.0)
    assert f.soc.iloc[-1] == pytest.approx(1 - 0.002 / 3.0)


@pytest.mark.parametrize(
    "key,value", [("Time", [0, 2, 1, 3]), ("Ah", [-0.2, -0.3, -0.3, -0.4]), ("Voltage", [4, np.nan, 4, 4])]
)
def test_lg_rejects_invalid_recordings(key, value):
    m = measurements()
    m[key] = value
    with pytest.raises(ValueError):
        resample(m)
