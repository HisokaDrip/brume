import time

import numpy as np

from . import data
from .diffusion import Diffusion
from .metrics import mmd_rbf, sliced_wasserstein
from .nn import Adam, NoiseNet
from .schedule import Schedule


class EMA:
    """Exponential moving average of the weights. Sampling from the EMA copy
    is noticeably cleaner than sampling from the raw, still-jittering weights."""

    def __init__(self, params, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.copy() for k, v in params.items()}

    def update(self, params):
        d = self.decay
        for k, v in params.items():
            self.shadow[k] *= d
            self.shadow[k] += (1 - d) * v


def train(dataset="moons", steps=6000, batch=512, lr=2e-3, T=200, hidden=128,
          depth=3, n_freq=4, seed=0, log_every=1000, log=print):
    rng = np.random.default_rng(seed)
    x_all = data.make(dataset, 20000, seed=seed)
    net = NoiseNet(hidden=hidden, depth=depth, n_freq=n_freq, seed=seed)
    diff = Diffusion(net, Schedule(T))
    opt = Adam(net.params, lr=lr)
    ema = EMA(net.params)

    t0 = time.time()
    running = None
    for step in range(1, steps + 1):
        # cosine decay with a short warmup
        warm = min(1.0, step / 200)
        cur_lr = lr * warm * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * step / steps)))
        batch_x = x_all[rng.integers(0, len(x_all), batch)]
        loss, grads = diff.loss_and_grads(batch_x, rng)
        opt.step(grads, lr=cur_lr)
        ema.update(net.params)
        running = loss if running is None else 0.98 * running + 0.02 * loss
        if log and (step % log_every == 0 or step == steps):
            log(f"step {step:>6}  loss {running:.4f}  lr {cur_lr:.2e}  {time.time() - t0:5.1f}s")

    net.params.update({k: v.copy() for k, v in ema.shadow.items()})
    return diff, x_all


def evaluate(diff, x_ref, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    samples, _ = diff.sample(n, rng)
    # baseline: a gaussian with the same mean/cov. If we can't beat this the
    # model hasn't learned anything a single matrix couldn't.
    gauss = rng.multivariate_normal(x_ref.mean(0), np.cov(x_ref.T), n)
    return {
        "swd_model": sliced_wasserstein(samples, x_ref),
        "swd_gauss": sliced_wasserstein(gauss, x_ref),
        "mmd_model": mmd_rbf(samples, x_ref),
        "mmd_gauss": mmd_rbf(gauss, x_ref),
    }, samples
