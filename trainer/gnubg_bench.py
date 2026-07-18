"""Benchmark our champion vs gnubg at 0-ply (task #5).

The metric is a millipoint ERROR RATE, not a win/loss scoreline — far more
sensitive, and the standard way backgammon strength is measured. For decisions
sampled from champion self-play, our engine picks a move and gnubg (0-ply
"static", cubeless) evaluates every resulting position. Our error on a decision =
gnubg's best-move equity minus the equity of the move WE chose. Averaged over many
decisions, that's how much equity per move we bleed against gnubg as a 0-ply
oracle. Move-agreement % = how often we pick gnubg's top move.

Bridged via the GNU Position ID (our Board.position_id()), which gnubg loads with
`set board`. gnubg's 0-ply static includes its exact bearoff databases and
contact/crashed/race routing, so this is our-0-ply vs gnubg-0-ply, warts and all.

Run: .venv/Scripts/python trainer/gnubg_bench.py --net td.onnx --decisions 400
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np

import bgcore

GNUBG = r"C:/Users/chris/AppData/Local/gnubg/gnubg-cli.exe"
MODELS = Path(__file__).resolve().parent.parent / "models"
STATIC = re.compile(
    r"static:\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+)")


def gnubg_static_equities(pos_ids):
    """gnubg 0-ply cubeless equity for each position id (player-on-roll's view)."""
    cmds = ["set evaluation chequerplay eval plies 0", "new game"]
    for pid in pos_ids:
        cmds += [f"set board {pid}", "eval"]
    cmds.append("quit")
    out = subprocess.run([GNUBG, "-t", "-q"], input="\n".join(cmds) + "\n",
                         capture_output=True, text=True).stdout
    eqs = [float(m.group(1)) for m in STATIC.finditer(out)]
    if len(eqs) != len(pos_ids):
        raise RuntimeError(f"gnubg parse mismatch: {len(eqs)} equities for {len(pos_ids)} positions")
    return eqs


def sample_decisions(net, n, eps, rng):
    """(board, d1, d2) decisions from champion self-play (eps-greedy for variety)."""
    out = []
    while len(out) < n:
        b = bgcore.Board.starting()
        for _ in range(4000):
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            kids = bgcore.legal_moves(b, d1, d2)
            out.append((b, d1, d2))
            term = [k.winner_points() for k in kids]
            if rng.random() < eps:
                i = rng.randrange(len(kids))
            else:
                eqs = [float(t) if t is not None else -net.equity(k.swap_perspective())
                       for k, t in zip(kids, term)]
                i = max(range(len(kids)), key=lambda j: eqs[j])
            if term[i] is not None or len(out) >= n:
                break
            b = kids[i].swap_perspective()
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="td.onnx")
    ap.add_argument("--decisions", type=int, default=400)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    net = bgcore.Neural(str(MODELS / args.net), 0, 0)
    t0 = time.time()
    decisions = sample_decisions(net, args.decisions, args.eps, rng)

    # Our chosen move per decision + collect every non-terminal child's id.
    all_ids, recs = [], []
    for (b, d1, d2) in decisions:
        kids = bgcore.legal_moves(b, d1, d2)
        term = [k.winner_points() for k in kids]
        oure, slots = [], []
        for k, t in zip(kids, term):
            if t is not None:
                oure.append(float(t)); slots.append(None)
            else:
                s = k.swap_perspective()
                oure.append(-net.equity(s))
                slots.append(len(all_ids)); all_ids.append(s.position_id())
        our_choice = max(range(len(kids)), key=lambda j: oure[j])
        recs.append((slots, our_choice, term))

    print(f"evaluating {len(all_ids)} child positions through gnubg 0-ply...", flush=True)
    gnu = gnubg_static_equities(all_ids)   # opponent equity per non-terminal child

    errors, agree = [], 0
    for (slots, our_choice, term) in recs:
        gmov = [float(term[j]) if sl is None else -gnu[sl] for j, sl in enumerate(slots)]
        best = max(range(len(gmov)), key=lambda j: gmov[j])
        errors.append(gmov[best] - gmov[our_choice])
        agree += (best == our_choice)
    errors = np.array(errors)
    n = len(errors)

    print(f"\n=== gnubg 0-ply benchmark | net {args.net} | {n} decisions "
          f"| {time.time()-t0:.0f}s ===")
    print(f"  move agreement (we pick gnubg's top move) : {100*agree/n:5.1f}%")
    print(f"  mean equity error / move                  : {errors.mean()*1000:6.1f} millipoints")
    print(f"  median equity error / move                : {np.median(errors)*1000:6.1f} millipoints")
    print(f"  decisions within 5 mEMG of gnubg's best    : {100*(errors < 0.005).mean():5.1f}%")
    print(f"  worst single decision                     : {errors.max()*1000:6.1f} millipoints")


if __name__ == "__main__":
    main()
