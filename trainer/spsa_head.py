"""SPSA tuning of the output-head parameters against PLAYING STRENGTH.

Every other trainer here minimises cross-entropy against gnubg's labels. That is
a proxy for what we care about, and it is ceilinged: perfect distillation makes
us gnubg, never better. SPSA optimises the thing we actually measure — points per
game — so it is not bounded by the teacher, in the same way search-parameter SPSA
is not bounded by the eval it tunes around.

Why the heads and not the body: the 12 class-aware heads are trained on very
uneven data (crashed ~0.8M rows against contact's ~2.9M in the 22.5M set), so a
systematic per-head offset is exactly the residue cross-entropy on imbalanced
data leaves behind — and `heads.bias` is only 72 numbers. SPSA's cost per
iteration is dimension-free but its convergence is not: the full head layer is
128*72+72 = 9,288 parameters, far past what a noisy objective can resolve. Start
at 72, widen only on evidence.

Objective: the two perturbed nets play EACH OTHER (paired, mirrored dice), so one
match yields y(+) - y(-) directly instead of two matches against a third party,
halving the cost and cancelling most of the dice variance. PPG is the score, not
win rate — it counts gammons and is the more sensitive statistic.

Run:
  .venv/Scripts/python trainer/spsa_head.py --net net_22p5M_256x128.pt \
      --iters 300 --games 20000 --out net_spsa.pt
"""
from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import NUM_INPUTS, net_bucketed_from_state

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PPG = re.compile(r"PPG\s+([+-][\d.]+)")


class BucketedSoftmax(nn.Module):
    """[N,198] -> [N, heads*6] probabilities, as export_bucketed.py writes them."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1).reshape(x.shape[0], -1)


def export(net, path: Path) -> None:
    """Lean ONNX export: no parity check, and it must NOT touch parity.json —
    export_bucketed.py rewrites that fixture on every run, which would leave the
    Rust live-net test failing after a few hundred tuning iterations."""
    torch.onnx.export(
        BucketedSoftmax(net).eval(), torch.zeros(1, NUM_INPUTS), str(path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}}, opset_version=13)


def match_ppg(a: Path, b: Path, games: int, ply: int, workers: int, seed: int) -> float:
    """PPG of `a` against `b`. Both arms see identical dice (mirrored, seeded)."""
    out = subprocess.run(
        [str(PY), str(Path(__file__).parent / "nply_h2h.py"),
         "--a", a.name, "--b", b.name, "--ply", str(ply), "--games", str(games),
         "--workers", str(workers), "--seed", str(seed)],
        cwd=str(ROOT), capture_output=True, text=True)
    m = PPG.findall(out.stdout)
    if not m:
        raise RuntimeError(f"could not parse PPG from nply_h2h:\n{out.stdout[-500:]}\n{out.stderr[-500:]}")
    return float(m[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="net_22p5M_256x128.pt", help="starting checkpoint")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=20000, help="games per iteration (~21s at 0-ply)")
    ap.add_argument("--ply", type=int, default=0, help="search depth for the objective matches")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--param", choices=["bias", "head"], default="bias",
                    help="bias: 72 head biases. head: the full 9,288-parameter head layer "
                         "(likely too many dimensions for a noisy objective)")
    # Standard Spall gains. `c` must be big enough that the perturbation moves PPG
    # further than the objective's own noise (~0.007 PPG at 20k games) or every
    # gradient estimate is noise; probe it before committing to a long run.
    ap.add_argument("--a", type=float, default=0.02, help="step gain")
    ap.add_argument("--c", type=float, default=0.05, help="perturbation size")
    ap.add_argument("--alpha", type=float, default=0.602)
    ap.add_argument("--gamma", type=float, default=0.101)
    ap.add_argument("--eval-every", type=int, default=25,
                    help="games against the UNPERTURBED start net, to see real progress")
    ap.add_argument("--eval-games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="net_spsa.pt")
    args = ap.parse_args()

    ck = torch.load(MODELS / args.net, map_location="cpu")
    net = net_bucketed_from_state(ck["model"], ck["hidden"], ck.get("act", "relu"))
    key = "heads.bias" if args.param == "bias" else "heads.weight"
    theta = net.state_dict()[key].detach().clone()
    shape = theta.shape
    n = theta.numel()
    rng = np.random.default_rng(args.seed)

    # The fixed reference: a frozen copy of the starting net. Progress is measured
    # against this, never against the current iterate, or the tuner can drift
    # while every local comparison still looks like an improvement.
    ref = MODELS / "_spsa_ref.onnx"
    export(net, ref)
    plus_p, minus_p = MODELS / "_spsa_plus.onnx", MODELS / "_spsa_minus.onnx"

    print(f"SPSA on {key} ({n} params) of {args.net} | {args.iters} iters x {args.games} "
          f"games at {args.ply}-ply | a={args.a} c={args.c}", flush=True)
    t0 = time.time()

    def with_theta(t):
        sd = net.state_dict()
        sd[key] = t.reshape(shape)
        m = copy.deepcopy(net)
        m.load_state_dict(sd)
        return m

    for k in range(args.iters):
        ak = args.a / (args.iters / 10 + k + 1) ** args.alpha
        ckk = args.c / (k + 1) ** args.gamma
        delta = torch.from_numpy(rng.choice([-1.0, 1.0], size=n).astype(np.float32))

        export(with_theta(theta + ckk * delta), plus_p)
        export(with_theta(theta - ckk * delta), minus_p)
        # Vary the dice base per iteration: a fixed base would tune to those exact
        # streams. Within the iteration both arms still share them (paired).
        y = match_ppg(plus_p, minus_p, args.games, args.ply, args.workers,
                      seed=1000 + 7919 * (k + 1))
        # delta is +/-1, so 1/delta == delta.
        ghat = (y / (2.0 * ckk)) * delta
        theta = theta + ak * ghat                     # ascent: maximise PPG

        line = f"iter {k+1:4d} | c {ckk:.4f} a {ak:.5f} | paired PPG {y:+.4f}"
        if (k + 1) % args.eval_every == 0 or k + 1 == args.iters:
            cur = MODELS / "_spsa_cur.onnx"
            export(with_theta(theta), cur)
            gain = match_ppg(cur, ref, args.eval_games, args.ply, args.workers, seed=555)
            line += f" || vs start net: PPG {gain:+.4f}"
            torch.save({"model": with_theta(theta).state_dict(), "hidden": ck["hidden"],
                        "act": ck.get("act", "relu"), "bucketed": True,
                        "n_buckets": ck.get("n_buckets", 12), "class_aware": True,
                        "spsa_iters": k + 1, "spsa_param": key, "iter": k + 1},
                       MODELS / args.out)
        print(line + f" | {time.time()-t0:.0f}s", flush=True)

    print(f"\nsaved {MODELS / args.out}")


if __name__ == "__main__":
    main()
