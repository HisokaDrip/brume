import numpy as np


def time_embedding(t, dim):
    """Transformer-style sinusoidal embedding of the integer timestep."""
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    ang = np.asarray(t, dtype=np.float64)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def fourier_features(x, n_freq):
    """[sin(2^k pi x), cos(2^k pi x)] per coordinate.

    A plain MLP on raw (x, y) is biased towards smooth functions and can't
    resolve thin structures - the spiral's inner arms just melt into a blob.
    Feeding it a few octaves of sinusoids fixes that (same trick as NeRF).
    """
    if n_freq == 0:
        return np.zeros((len(x), 0))
    freqs = np.pi * 2.0 ** np.arange(n_freq)
    ang = x[:, :, None] * freqs  # (n, dim, n_freq)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=2).reshape(len(x), -1)


def _sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # stable for big |z|, no overflow warnings


class NoiseNet:
    """MLP that predicts the noise eps from (x_t, t).

    Input is [x, fourier(x), emb(t)], then a few SiLU layers. The time embedding is also
    re-injected into every hidden layer through its own projection - without
    that the net tends to blur timesteps together and the samples come out
    fuzzy around thin structures like the spiral arms.
    """

    def __init__(self, dim=2, hidden=128, depth=3, temb=32, n_freq=4, seed=0):
        rng = np.random.default_rng(seed)
        self.dim, self.hidden, self.depth, self.temb, self.n_freq = dim, hidden, depth, temb, n_freq
        n_in = dim * (1 + 2 * n_freq) + temb
        p = {}
        p["W0"] = rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden))
        p["b0"] = np.zeros(hidden)
        for i in range(1, depth):
            p[f"W{i}"] = rng.normal(0, np.sqrt(2 / hidden), (hidden, hidden))
            p[f"b{i}"] = np.zeros(hidden)
            p[f"U{i}"] = rng.normal(0, np.sqrt(1 / temb), (temb, hidden)) * 0.5
        # small output init so the untrained net predicts ~0 noise
        p["Wout"] = rng.normal(0, 0.01, (hidden, dim))
        p["bout"] = np.zeros(dim)
        self.params = p

    def forward(self, x, t):
        p = self.params
        e = time_embedding(t, self.temb)
        h_in = np.concatenate([x, fourier_features(x, self.n_freq), e], axis=1)
        cache = {"e": e, "inputs": [h_in], "pre": []}

        z = h_in @ p["W0"] + p["b0"]
        cache["pre"].append(z)
        h = z * _sigmoid(z)
        for i in range(1, self.depth):
            cache["inputs"].append(h)
            z = h @ p[f"W{i}"] + p[f"b{i}"] + e @ p[f"U{i}"]
            cache["pre"].append(z)
            h = z * _sigmoid(z)
        cache["inputs"].append(h)
        self._cache = cache
        return h @ p["Wout"] + p["bout"]

    def backward(self, gout):
        p, c = self.params, self._cache
        g = {}
        h = c["inputs"][-1]
        g["Wout"] = h.T @ gout
        g["bout"] = gout.sum(0)
        gh = gout @ p["Wout"].T

        for i in reversed(range(self.depth)):
            z = c["pre"][i]
            s = _sigmoid(z)
            gz = gh * (s * (1 + z * (1 - s)))  # d/dz of z*sigmoid(z)
            inp = c["inputs"][i]
            g[f"W{i}"] = inp.T @ gz
            g[f"b{i}"] = gz.sum(0)
            if i > 0:
                g[f"U{i}"] = c["e"].T @ gz
            gh = gz @ p[f"W{i}"].T
        self.grads = g
        return g

    def copy_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def save(self, path, **extra):
        meta = np.array([self.dim, self.hidden, self.depth, self.temb, self.n_freq])
        np.savez(path, __meta=meta, **self.params, **{f"__x_{k}": v for k, v in extra.items()})

    @classmethod
    def load(cls, path):
        f = np.load(path)
        dim, hidden, depth, temb, n_freq = (int(v) for v in f["__meta"])
        net = cls(dim, hidden, depth, temb, n_freq)
        for k in net.params:
            net.params[k] = f[k].copy()
        extra = {k[4:]: f[k] for k in f.files if k.startswith("__x_")}
        return net, extra


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, clip=1.0):
        self.params, self.lr, self.eps, self.clip = params, lr, eps, clip
        self.b1, self.b2 = betas
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        # global-norm clipping; the loss is spiky at small t early on
        norm = np.sqrt(sum((g * g).sum() for g in grads.values()))
        scale = min(1.0, self.clip / (norm + 1e-12)) if self.clip else 1.0
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k, g in grads.items():
            g = g * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            self.params[k] -= lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)
        return norm
