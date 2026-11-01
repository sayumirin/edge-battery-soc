import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from battery_shift.models import GRU
from battery_shift.train import fit, seed_all


def test_head_adaptation_preserves_source_and_frozen_gru():
    seed_all(17)
    source = GRU(8)
    original = {k: v.clone() for k, v in source.state_dict().items()}
    ds = TensorDataset(torch.randn(16, 5, 3), torch.ones(16))
    cfg = OmegaConf.create(dict(learning_rate=0.01, batch_size=8, adapt_epochs=2))
    tuned, _ = fit(source, ds, cfg, mode="head")
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, original[name])
        if not name.startswith("head."):
            torch.testing.assert_close(tuned.state_dict()[name], original[name])
    assert not torch.equal(tuned.head.weight, source.head.weight)


def test_adaptation_step_budget_overrides_epochs():
    seed_all(17)
    ds = TensorDataset(torch.randn(23, 5, 3), torch.ones(23))
    for budget in [3, 7, 12]:
        cfg = OmegaConf.create(dict(learning_rate=0.01, batch_size=5, adapt_epochs=1, adapt_steps=budget))
        _, history = fit(GRU(8), ds, cfg)
        assert history[-1]["steps"] == budget
    _, history = fit(GRU(8), ds, OmegaConf.create(dict(learning_rate=0.01, batch_size=5, adapt_epochs=2)))
    assert history[-1]["steps"] == 10


def test_seed_reproducibility():
    seed_all(42)
    a = GRU(8)
    x = np.random.randn(2, 5, 3).astype("float32")
    seed_all(42)
    b = GRU(8)
    y = np.random.randn(2, 5, 3).astype("float32")
    np.testing.assert_array_equal(x, y)
    torch.testing.assert_close(a(torch.from_numpy(x)), b(torch.from_numpy(y)))
