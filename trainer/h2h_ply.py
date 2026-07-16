"""Head-to-head between two exported nets at a chosen search depth.

compare_nets.py settles 0-ply strength, but the app plays with search, and a
better static evaluator is only *usually* a better searched one. This plays the
two nets against each other through the identical bgcore search, so the only
difference is the evaluation.

Dice are mirrored: both seats play the same rolls, which removes most of the luck
and makes a few hundred games worth something.

Run: .venv/Scripts/python trainer/h2h_ply.py models/td.onnx models/td_deep3.onnx --ply 1 --games 200
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bgcore


def play(net_a, net_b, seed, a_first):
    """One game, net_a vs net_b, dice fixed by `seed`. Returns points to A."""
    rng = random.Random(seed)
    board = bgcore.Board.starting()
    a_to_move = a_first
    for _ in range(400):
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        engine = net_a if a_to_move else net_b
        scores = engine.scores(board, d1, d2)
        moves = bgcore.legal_moves(board, d1, d2)
        if moves:
            best = max(range(len(scores)), key=lambda i: scores[i])
            board = moves[best]
            pts = board.winner_points()
            if pts is not None and pts > 0:
                return pts if a_to_move else -pts
        board = board.swap_perspective()
        a_to_move = not a_to_move
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ply", type=int, default=1)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=0, help="prune at 2-ply+")
    args = ap.parse_args()

    cand = args.candidates or (4 if args.ply >= 2 else 0)
    a = bgcore.Neural(args.a, args.ply, cand)
    b = bgcore.Neural(args.b, args.ply, cand)
    print(f"A = {args.a}\nB = {args.b}\n{args.games} games at {args.ply}-ply, "
          f"mirrored dice, seats swapped.\n")

    wins_b = pts_b = 0
    for g in range(args.games):
        # Same seed for both seatings: identical dice, swapped sides.
        seed = 1000 + g // 2
        a_first = (g % 2 == 0)
        p = play(a, b, seed, a_first)
        pts_b -= p
        if p < 0:
            wins_b += 1
        if (g + 1) % 40 == 0:
            print(f"  {g+1:4d} games: B wins {100*wins_b/(g+1):.1f}%", flush=True)

    wr = wins_b / args.games
    se = math.sqrt(0.25 / args.games)
    z = (wr - 0.5) / se if se else 0.0
    print(f"\nB wins {100*wr:.1f}%  (z = {z:+.2f})   PPG {pts_b/args.games:+.3f}")
    verdict = "B STRONGER" if z > 1.96 else ("A STRONGER" if z < -1.96 else "TOO CLOSE TO CALL")
    print(f"=> {verdict} at {args.ply}-ply")


if __name__ == "__main__":
    main()
