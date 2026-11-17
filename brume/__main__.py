import argparse
import sys
import time

import numpy as np

from . import data
from .diffusion import Diffusion
from .metrics import sliced_wasserstein
from .nn import NoiseNet
from .plot import density, side_by_side
from .schedule import Schedule
from .train import evaluate, train


def cmd_train(args):
    print(f"training on '{args.data}' for {args.steps} steps (T={args.T})")
    diff, x_ref = train(args.data, steps=args.steps, T=args.T, seed=args.seed,
                        hidden=args.hidden, log_every=max(1, args.steps // 8))
    scores, samples = evaluate(diff, x_ref, n=args.n)

    print()
    print(side_by_side([density(x_ref[:args.n]), density(samples)], ["data", "brume samples"]))
    print()
    print(f"sliced W1   model {scores['swd_model']:.4f}   gaussian baseline {scores['swd_gauss']:.4f}")
    print(f"MMD^2       model {scores['mmd_model']:.4f}   gaussian baseline {scores['mmd_gauss']:.4f}")

    if args.show_steps:
        rng = np.random.default_rng(7)
        T = diff.T
        keep = [T, int(T * 0.6), int(T * 0.3), int(T * 0.1), 0]
        _, snaps = diff.sample(args.n, rng, keep=keep)
        panels = [density(snaps[t], width=22, height=11, lim=3.0) for t in keep]
        print()
        print(side_by_side(panels, [f"t={t}" for t in keep], gap=" | "))

    if args.save:
        diff.model.save(args.save, T=np.array(diff.T))
        print(f"\nsaved weights to {args.save}")


def cmd_sample(args):
    net, extra = NoiseNet.load(args.weights)
    diff = Diffusion(net, Schedule(int(extra.get("T", 200))))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    if args.ddim:
        xs = diff.sample_ddim(args.n, rng, steps=args.ddim)
        how = f"DDIM, {args.ddim} steps"
    else:
        xs, _ = diff.sample(args.n, rng)
        how = f"DDPM, {diff.T} steps"
    dt = time.time() - t0
    print(f"{args.n} samples ({how}) in {dt:.2f}s")
    print("\n".join(density(xs)))
    if args.data:
        ref = data.make(args.data, 20000, seed=0)
        print(f"sliced W1 vs '{args.data}': {sliced_wasserstein(xs, ref):.4f}")
    if args.out:
        np.savetxt(args.out, xs, delimiter=",", fmt="%.5f")
        print(f"wrote {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="brume", description="tiny denoising diffusion in numpy")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default="spiral", choices=sorted(data.DATASETS))
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--T", type=int, default=200)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--n", type=int, default=3000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save")
    t.add_argument("--show-steps", action="store_true", help="print the reverse process as it denoises")
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("sample")
    s.add_argument("weights")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--ddim", type=int, default=0, help="use DDIM with this many steps")
    s.add_argument("--data", choices=sorted(data.DATASETS), help="score against this dataset")
    s.add_argument("--out", help="write samples to csv")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
