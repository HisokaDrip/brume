import numpy as np

from .schedule import Schedule


class Diffusion:
    """DDPM training + ancestral sampling, and DDIM for the fast path.

    `model` is anything with forward(x, t) -> eps_hat; for the training step it
    also needs backward(grad) and a params dict (see NoiseNet).
    """

    def __init__(self, model, schedule=None):
        self.model = model
        self.sched = schedule or Schedule()

    @property
    def T(self):
        return self.sched.T

    def q_sample(self, x0, t, noise):
        ab = self.sched.abar[t][:, None]
        return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise

    def loss_and_grads(self, x0, rng):
        n = len(x0)
        t = rng.integers(0, self.T, n)
        noise = rng.standard_normal(x0.shape)
        xt = self.q_sample(x0, t, noise)
        pred = self.model.forward(xt, t)
        diff = pred - noise
        loss = float((diff ** 2).mean())
        grads = self.model.backward(2 * diff / diff.size)
        return loss, grads

    def predict_x0(self, xt, t, eps, clip=None):
        ab = self.sched.abar[t]
        x0 = (xt - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
        return x0 if clip is None else np.clip(x0, -clip, clip)

    def sample(self, n, rng, keep=()):
        """Ancestral DDPM sampling over all T steps.

        Returns final samples and a dict {t: x_t} for any timesteps in `keep`
        (handy for watching the cloud condense out of the noise).
        """
        s = self.sched
        x = rng.standard_normal((n, self.model_dim()))
        snaps = {}
        if self.T in keep:
            snaps[self.T] = x.copy()
        for t in range(self.T - 1, -1, -1):
            eps = self.model.forward(x, np.full(n, t))
            mean = (x - s.betas[t] / np.sqrt(1 - s.abar[t]) * eps) / np.sqrt(s.alphas[t])
            if t > 0:
                x = mean + np.sqrt(s.post_var[t]) * rng.standard_normal(x.shape)
            else:
                x = mean
            if t in keep:
                snaps[t] = x.copy()
        return x, snaps

    def sample_ddim(self, n, rng, steps=50, eta=0.0, clip=4.0):
        """DDIM (Song et al. 2020). eta=0 is fully deterministic given x_T.

        The x0 estimate gets clipped: with the cosine schedule abar at the
        last step is ~1e-5, so x0 = (x - sqrt(1-abar) eps) / sqrt(abar) blows
        a tiny eps error up by ~300x and the first big jump flings points
        all over the place. Data is standardised so |x0| < 4 is safe.
        """
        s = self.sched
        ts = np.linspace(self.T - 1, 0, steps).round().astype(int)
        ts = np.unique(ts)[::-1]
        x = rng.standard_normal((n, self.model_dim()))
        for i, t in enumerate(ts):
            eps = self.model.forward(x, np.full(n, t))
            x0 = self.predict_x0(x, t, eps, clip=clip)
            # re-derive eps from the clipped x0 so the two stay consistent
            eps = (x - np.sqrt(s.abar[t]) * x0) / np.sqrt(1 - s.abar[t])
            ab_prev = s.abar[ts[i + 1]] if i + 1 < len(ts) else 1.0
            sigma = eta * np.sqrt((1 - ab_prev) / (1 - s.abar[t]) * (1 - s.abar[t] / ab_prev))
            dir_xt = np.sqrt(max(1 - ab_prev - sigma ** 2, 0.0)) * eps
            x = np.sqrt(ab_prev) * x0 + dir_xt
            if sigma > 0:
                x = x + sigma * rng.standard_normal(x.shape)
        return x

    def model_dim(self):
        return getattr(self.model, "dim", 2)
