import numpy as np
import pytest

from brume import data
from brume.diffusion import Diffusion
from brume.metrics import sliced_wasserstein
from brume.schedule import Schedule
from brume.train import evaluate, train


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_schedule_is_sane(kind):
    s = Schedule(200, kind)
    assert np.all(np.diff(s.abar) < 0)
    assert 0 < s.abar[-1] < 0.05 and s.abar[0] > 0.99
    assert np.all((s.betas > 0) & (s.betas <= 0.999))
    assert np.all(s.post_var <= s.betas + 1e-12)


class PointOracle:
    """The exact optimal noise predictor when the data is a single point c.

    If the sampler math is right, both DDPM and DDIM must collapse onto c.
    """

    dim = 2

    def __init__(self, c, sched):
        self.c, self.sched = np.asarray(c), sched

    def forward(self, x, t):
        ab = self.sched.abar[t][:, None]
        return (x - np.sqrt(ab) * self.c) / np.sqrt(1 - ab)


def test_ddpm_sampler_recovers_point_mass():
    s = Schedule(100)
    diff = Diffusion(PointOracle([1.5, -0.5], s), s)
    x, snaps = diff.sample(500, np.random.default_rng(0), keep=(100, 0))
    assert np.allclose(x, [1.5, -0.5], atol=1e-2)
    assert abs(snaps[100].std() - 1) < 0.1  # started from pure noise


def test_ddim_sampler_recovers_point_mass():
    s = Schedule(100)
    diff = Diffusion(PointOracle([-1.0, 2.0], s), s)
    x = diff.sample_ddim(200, np.random.default_rng(0), steps=10)
    assert np.allclose(x, [-1.0, 2.0], atol=1e-6)


def test_forward_process_ends_in_noise():
    s = Schedule(200)
    diff = Diffusion(None, s)
    rng = np.random.default_rng(0)
    x0 = data.make("rings", 20000)
    xt = diff.q_sample(x0, np.full(len(x0), 199), rng.standard_normal(x0.shape))
    assert abs(xt.mean()) < 0.05 and abs(xt.std() - 1) < 0.05


@pytest.mark.parametrize("name", sorted(data.DATASETS))
def test_datasets_are_standardised(name):
    x = data.make(name, 4000, seed=2)
    assert x.shape == (4000, 2)
    assert np.allclose(x.mean(0), 0, atol=1e-9)
    assert abs(x.std() - 1) < 1e-9


def test_sliced_wasserstein_separates_distributions():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((2000, 2)), rng.standard_normal((2000, 2))
    assert sliced_wasserstein(a, b) < 0.06
    assert sliced_wasserstein(a, b + 1.0) > 0.5


def test_short_training_beats_gaussian_baseline():
    # small budget so the suite stays quick; even this should clearly beat
    # a moment-matched gaussian on the moons
    diff, ref = train("moons", steps=800, batch=256, T=100, hidden=64, log=None)
    scores, samples = evaluate(diff, ref, n=1500)
    assert np.isfinite(samples).all()
    assert scores["swd_model"] < 0.7 * scores["swd_gauss"]
    assert scores["mmd_model"] < scores["mmd_gauss"]
