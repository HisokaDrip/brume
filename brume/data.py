import numpy as np


def _moons(n, rng):
    k = n // 2
    a = rng.uniform(0, np.pi, k)
    b = rng.uniform(0, np.pi, n - k)
    top = np.c_[np.cos(a), np.sin(a)]
    bottom = np.c_[1 - np.cos(b), 0.5 - np.sin(b)]
    return np.r_[top, bottom] + rng.normal(0, 0.06, (n, 2))


def _spiral(n, rng):
    # two interleaved arms, the classic "can your model do this" shape
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    arm = rng.integers(0, 2, n) * np.pi
    pts = np.c_[t * np.cos(t + arm), t * np.sin(t + arm)]
    return pts + rng.normal(0, 0.25, (n, 2))


def _rings(n, rng):
    r = rng.choice([1.0, 2.2], n)
    th = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(th), r * np.sin(th)] + rng.normal(0, 0.07, (n, 2))


def _checker(n, rng):
    # 4x4 board, only the "black" squares get points
    out = []
    while sum(len(o) for o in out) < n:
        p = rng.uniform(-2, 2, (n, 2))
        keep = (np.floor(p[:, 0]) + np.floor(p[:, 1])) % 2 == 0
        out.append(p[keep])
    return np.concatenate(out)[:n]


def _gmm(n, rng):
    centers = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]) * 2.5
    idx = rng.integers(0, len(centers), n)
    return centers[idx] + rng.normal(0, 0.15, (n, 2))


DATASETS = {
    "moons": _moons,
    "spiral": _spiral,
    "rings": _rings,
    "checker": _checker,
    "gmm8": _gmm,
}


def make(name, n, seed=0):
    """Sample n points and standardise them so every dataset lives at roughly unit scale."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, try one of {sorted(DATASETS)}")
    rng = np.random.default_rng(seed)
    x = DATASETS[name](n, rng)
    x = x - x.mean(0)
    return x / x.std()
