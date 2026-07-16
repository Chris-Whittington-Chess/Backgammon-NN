"""Calibrate pip-count bucket edges for even population.

Uniform 42-pip slices pile the opening (and clamped hit-scrambles) into the top
bucket and starve the bearoff buckets. Instead, sample the total-pip distribution
over real games and cut it at the octiles, so each of the 8 buckets holds ~1/8 of
the positions — every head gets adequate data.

Sampled from champion self-play (native, fast): the pip distribution is
policy-robust (games progress at ~8 pips/roll regardless of skill), so this is a
faithful proxy for the games the bucketed net will see.

Prints the 7 edges to paste into PIP_BUCKET_EDGES in model.py AND the matching
array in crates/bgcore/src/eval/nn.rs — they must stay in sync.

Run: .venv/Scripts/python trainer/calibrate_buckets.py [games]
"""

from __future__ import annotations

import random
import sys

import numpy as np

import bgcore

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def main():
    champ = bgcore.Neural("models/td.onnx", 0, 0)
    rng = random.Random(0)
    totals = []
    for _ in range(GAMES):
        b = bgcore.Board.starting()
        for _ply in range(600):
            totals.append(b.pip_count(0) + b.pip_count(1))
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            children = bgcore.legal_moves(b, d1, d2)
            if not children:
                b = b.swap_perspective()
                continue
            scores = champ.scores(b, d1, d2)
            best = children[max(range(len(scores)), key=lambda k: scores[k])]
            if best.winner_points() is not None:
                break
            b = best.swap_perspective()

    totals = np.array(totals)
    edges = [int(round(np.percentile(totals, 100 * k / 8))) for k in range(1, 8)]
    print(f"{len(totals):,} positions over {GAMES} games")
    print(f"total pips: min {totals.min()}  median {int(np.median(totals))}  max {totals.max()}")
    print(f"\nPIP_BUCKET_EDGES = {edges}")

    # Show resulting population per bucket (should be ~even).
    def bucket(t):
        return sum(1 for e in edges if t >= e)
    pop = np.bincount([bucket(t) for t in totals], minlength=8)
    print(f"population per bucket: {pop.tolist()}")
    print(f"  (ratio max/min = {pop.max() / max(pop.min(), 1):.1f}x; uniform was ~90x)")


if __name__ == "__main__":
    main()
