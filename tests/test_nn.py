import numpy as np

from brume.nn import Adam, NoiseNet, time_embedding


def test_time_embedding_shape_and_range():
    e = time_embedding(np.arange(10), 16)
    assert e.shape == (10, 16)
    assert np.all(np.abs(e) <= 1)
    # t=0 -> sin part all zero, cos part all one
    assert np.allclose(e[0, :8], 0) and np.allclose(e[0, 8:], 1)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    net = NoiseNet(hidden=16, depth=3, temb=8, seed=1)
    x = rng.standard_normal((5, 2))
    t = rng.integers(0, 100, 5)
    target = rng.standard_normal((5, 2))

    def loss():
        return 0.5 * ((net.forward(x, t) - target) ** 2).sum()

    loss()
    grads = net.backward(net.forward(x, t) - target)
    eps = 1e-6
    for name, p in net.params.items():
        # spot-check a handful of entries per tensor
        flat = p.reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            up = loss()
            flat[i] = old - eps
            down = loss()
            flat[i] = old
            num = (up - down) / (2 * eps)
            ana = grads[name].reshape(-1)[i]
            assert abs(num - ana) <= 1e-5 + 1e-4 * abs(num), (name, i, num, ana)


def test_adam_minimises_quadratic():
    params = {"w": np.array([3.0, -2.0])}
    opt = Adam(params, lr=0.1, clip=None)
    for _ in range(300):
        opt.step({"w": 2 * params["w"]})
    assert np.abs(params["w"]).max() < 1e-2


def test_save_load_roundtrip(tmp_path):
    net = NoiseNet(hidden=8, depth=2, temb=4, seed=5)
    path = tmp_path / "w.npz"
    net.save(path, T=np.array(123))
    net2, extra = NoiseNet.load(path)
    x = np.ones((3, 2))
    t = np.array([0, 5, 9])
    assert np.allclose(net.forward(x, t), net2.forward(x, t))
    assert int(extra["T"]) == 123
