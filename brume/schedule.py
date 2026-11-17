import numpy as np


class Schedule:
    """Noise schedule plus every derived quantity the sampler needs.

    Cosine is the default (Nichol & Dhariwal 2021). With only a couple hundred
    steps the linear schedule destroys the signal too early and most of the
    timesteps end up wasted on pure noise.
    """

    def __init__(self, T=200, kind="cosine", s=0.008):
        self.T = T
        if kind == "cosine":
            steps = np.arange(T + 1) / T
            f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
            abar = f / f[0]
            betas = 1 - abar[1:] / abar[:-1]
            betas = np.clip(betas, 1e-5, 0.999)
        elif kind == "linear":
            # rescale the DDPM 1e-4..0.02 @ T=1000 range to whatever T we have
            scale = 1000 / T
            betas = np.linspace(scale * 1e-4, scale * 0.02, T)
        else:
            raise ValueError(kind)

        self.betas = betas
        self.alphas = 1 - betas
        self.abar = np.cumprod(self.alphas)
        self.abar_prev = np.r_[1.0, self.abar[:-1]]
        # variance of q(x_{t-1} | x_t, x_0)
        self.post_var = betas * (1 - self.abar_prev) / (1 - self.abar)
