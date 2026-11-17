import numpy as np


def sliced_wasserstein(a, b, n_proj=256, seed=0):
    """Average 1D Wasserstein-1 distance over random projections.

    Cheap, no bandwidth to tune, and in 2D with a few hundred directions it's
    a pretty faithful "how different do these clouds look" number.
    """
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    a = a[rng.permutation(len(a))[:n]]
    b = b[rng.permutation(len(b))[:n]]
    dirs = rng.standard_normal((a.shape[1], n_proj))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    pa = np.sort(a @ dirs, axis=0)
    pb = np.sort(b @ dirs, axis=0)
    return float(np.abs(pa - pb).mean())


def mmd_rbf(a, b, bandwidths=(0.1, 0.3, 1.0), max_n=1500, seed=0):
    """Unbiased squared MMD with a sum of RBF kernels."""
    rng = np.random.default_rng(seed)
    a = a[rng.permutation(len(a))[:max_n]]
    b = b[rng.permutation(len(b))[:max_n]]

    def k(x, y):
        d2 = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T
        return sum(np.exp(-d2 / (2 * h * h)) for h in bandwidths)

    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    m, n = len(a), len(b)
    np.fill_diagonal(kaa, 0)
    np.fill_diagonal(kbb, 0)
    return float(kaa.sum() / (m * (m - 1)) + kbb.sum() / (n * (n - 1)) - 2 * kab.mean())
