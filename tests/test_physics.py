import numpy as np
import pandas as pd
from battery_shift.physics import polarization, ekf
from battery_shift.models import CNN
import torch


def test_rc_matches_analytic_constant_current_response():
    t = np.arange(30.0)
    np.testing.assert_allclose(
        polarization(np.ones(30) * 2, t, 0.1, 10), 0.2 * (1 - np.exp(-t / 10)), atol=1e-12
    )


def test_ekf_finite_and_bounded_on_constant_ocv():
    f = pd.DataFrame(
        dict(
            time_s=np.arange(100.0),
            current_a=np.zeros(100),
            voltage_v=np.full(100, 3.6),
            soc=np.full(100, 0.5),
        )
    )
    p = ekf(f, np.array([0.0, 1.0]), np.array([3.0, 4.2]), 0.03, 0.02, 30)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert abs(p[-1] - 0.5) < 0.01


def test_cnn_shape_and_parameter_budget():
    m = CNN(32)
    p = m(torch.randn(4, 300, 3))
    assert p.shape == (4,) and sum(p.numel() for p in m.parameters()) < 50000
