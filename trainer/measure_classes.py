"""Measure how often race / crashed / contact positions occur in champion self-play.

Step 1 of the class-aware bucketing plan. Pure Python, no engine changes: uses the
board accessors already exposed to Python (point/off/no_contact/pip_count) to apply
gnubg's exact position classification, then cross-tabulates class x pip-bucket over
positions sampled from the live bucketed net's self-play.

The numbers decide the head layout (how many heads per class) before any Rust change.

Run:
  .venv/Scripts/python trainer/measure_classes.py --games 2000
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch

import bgcore
from model import net_bucketed_from_state, pip_bucket, PIP_BUCKET_EDGES, N_BUCKETS
from train import roll
from train_bucketed import choose_next_bucketed, total_pips

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

CHECKERS = 15
CRASH_N = 6  # gnubg's N: at most 6 checkers not buried on the 1- and 2-points

CLASSES = ("race", "crashed", "contact")


def side_crashed(board, side: int) -> bool:
    """gnubg's per-side crashed test. `tot` = checkers still on board (15 - off);
    ace/two = checkers on that side's 1- and 2-points. Board is mover-relative, so
    the mover's ace/2 are points 1/2 and the opponent's are the negated points 24/23.
    """
    tot = CHECKERS - board.off(side)
    if side == 0:  # mover
        ace = max(board.point(1), 0)
        two = max(board.point(2), 0)
    else:          # opponent, stored negated in the mover's frame
        ace = max(-board.point(24), 0)
        two = max(-board.point(23), 0)

    if tot <= CRASH_N:
        return True
    if ace > 1:
        if tot <= CRASH_N + ace:
            return True
        if two > 1 and (1 + tot - ace - two) <= CRASH_N:
            return True
    else:
        if tot <= CRASH_N + (two - 1):
            return True
    return False


def classify(board) -> str:
    """gnubg's three contact/race classes. Crashed is a sub-class of contact:
    a raced position (armies passed) is never crashed."""
    if board.no_contact():
        return "race"
    if side_crashed(board, 0) or side_crashed(board, 1):
        return "crashed"
    return "contact"


def sample_boards(net, n_games, epsilon, rng):
    """Yield every decision board (mover-relative) from champion self-play."""
    for _ in range(n_games):
        board = bgcore.Board.starting()
        for _ in range(4000):
            yield board
            d1, d2 = roll(rng)
            nb, pts = choose_next_bucketed(net, board, d1, d2, epsilon, rng)
            if pts is not None:
                break
            board = nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="td_latest.pt", help="bucketed checkpoint to self-play")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--race", type=int, default=3, help="race sub-buckets (heads)")
    ap.add_argument("--crashed", type=int, default=2, help="crashed sub-buckets (heads)")
    ap.add_argument("--contact", type=int, default=7, help="contact sub-buckets (heads)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ck = torch.load(MODELS_DIR / args.net, map_location="cpu")
    if not ck.get("bucketed"):
        raise SystemExit(f"{args.net} is not a bucketed checkpoint")
    net = net_bucketed_from_state(ck["model"], ck["hidden"], ck.get("act", "relu"))
    print(f"Self-playing {args.games} games with {args.net} "
          f"(hidden={ck['hidden']} act={ck.get('act','relu')}, eps={args.epsilon})\n")

    class_counts = Counter()
    grid = {c: Counter() for c in CLASSES}          # class -> pip-bucket -> count
    pip_extent = {c: [10**9, 0] for c in CLASSES}    # class -> [min, max] total pips
    pips = {c: [] for c in CLASSES}                  # class -> list of total pips
    total = 0

    for board in sample_boards(net, args.games, args.epsilon, rng):
        cls = classify(board)
        tp = total_pips(board)
        pb = pip_bucket(tp)
        class_counts[cls] += 1
        grid[cls][pb] += 1
        pips[cls].append(tp)
        lo_hi = pip_extent[cls]
        lo_hi[0] = min(lo_hi[0], tp)
        lo_hi[1] = max(lo_hi[1], tp)
        total += 1
        if total % 50000 == 0:
            print(f"  ...{total} boards")

    print(f"\n=== {total} decision boards ===\n")
    print("Class shares:")
    for c in CLASSES:
        n = class_counts[c]
        lo, hi = pip_extent[c]
        extent = f"total-pips {lo}-{hi}" if n else "-"
        print(f"  {c:8s} {n:8d}  {100*n/max(total,1):5.1f}%   {extent}")

    print(f"\nClass x pip-bucket grid (edges {list(PIP_BUCKET_EDGES)}):")
    header = "class     " + "".join(f"b{b:<7d}" for b in range(N_BUCKETS)) + "   total"
    print("  " + header)
    for c in CLASSES:
        row = grid[c]
        cells = "".join(f"{row.get(b,0):<8d}" for b in range(N_BUCKETS))
        print(f"  {c:8s}  {cells}  {class_counts[c]:8d}")
    col_tot = [sum(grid[c].get(b, 0) for c in CLASSES) for b in range(N_BUCKETS)]
    print(f"  {'TOTAL':8s}  " + "".join(f"{t:<8d}" for t in col_tot))

    # Even-population total-pip edges within each class, for the chosen head layout.
    # Convention matches pip_bucket: sub-bucket = count of edges with total >= edge.
    layout = {"race": args.race, "crashed": args.crashed, "contact": args.contact}

    def even_edges(values, k):
        s = sorted(values)
        n = len(s)
        return [s[min(n - 1, round(j * n / k))] for j in range(1, k)]

    print("\nCalibrated within-class total-pip edges (even population):")
    for c in CLASSES:
        k = layout[c]
        edges = even_edges(pips[c], k)
        # report resulting sub-bucket populations
        counts = Counter(sum(1 for e in edges if tp >= e) for tp in pips[c])
        pop = [counts.get(i, 0) for i in range(k)]
        print(f"  {c:8s} {k} heads  edges={edges}  pop={pop}")


if __name__ == "__main__":
    main()
