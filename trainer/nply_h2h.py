"""Head-to-head between two nets at a chosen search ply — promotion verify.

The training benchmarks are all 0-ply; 0-ply edges have repeatedly washed out
under search in this project, so a promotion candidate must be verified at the
ply the app actually plays. Both nets play at --ply; mirrored dice; multiprocessed
(Neural.scores holds the GIL, so parallelise across processes).

Run: .venv/Scripts/python trainer/nply_h2h.py --a td_2ply_full.onnx --b td.onnx --ply 1
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import bgcore

MODELS = Path(__file__).resolve().parent.parent / "models"

_A = _B = None


def _init(a_path, b_path, ply, cand):
    global _A, _B
    _A = bgcore.Neural(a_path, ply, cand)
    _B = bgcore.Neural(b_path, ply, cand)


def _best(net, board, d1, d2):
    moves = bgcore.legal_moves(board, d1, d2)
    scores = net.scores(board, d1, d2)  # mover-frame equity per move at net's ply
    i = max(range(len(moves)), key=lambda k: scores[k])
    return moves[i]


def _play(job):
    _, seed, a_first = job
    rng = random.Random(seed)
    board = bgcore.Board.starting()
    a_to_move = a_first
    for _ in range(300):
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        chosen = _best(_A if a_to_move else _B, board, d1, d2)
        pts = chosen.winner_points()
        if pts is not None:
            return pts if a_to_move else -pts  # points to A
        board = chosen.swap_perspective()
        a_to_move = not a_to_move
    # Ply cap on a crawling race: resolve by pip count (fewer pips wins).
    a_pip = board.pip_count(0 if a_to_move else 1)
    b_pip = board.pip_count(1 if a_to_move else 0)
    return 1 if a_pip < b_pip else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="candidate net (.onnx)")
    ap.add_argument("--b", required=True, help="baseline/champion net (.onnx)")
    ap.add_argument("--ply", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=0, help="prune 2-ply+ nodes; 0 = full")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()

    jobs = [(g, 1000 + g // 2, g % 2 == 0) for g in range(args.games)]  # mirrored dice
    print(f"A={args.a} vs B={args.b} at {args.ply}-ply | {args.games} games, mirrored "
          f"dice, {args.workers} workers\n", flush=True)
    t0 = time.time()
    a_path, b_path = str(MODELS / args.a), str(MODELS / args.b)
    results = []
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(a_path, b_path, args.ply, args.candidates)) as pool:
        for i, r in enumerate(pool.imap_unordered(_play, jobs, chunksize=4)):
            results.append(r)
            if (i + 1) % 50 == 0:
                n = len(results)
                w = sum(1 for x in results if x > 0)
                print(f"  {n}/{args.games} ({n/max(time.time()-t0,1):.1f}/s): "
                      f"A win {100*w/n:.1f}%  ppg {sum(results)/n:+.3f}", flush=True)

    n = len(results)
    w = sum(1 for x in results if x > 0)
    pts = sum(results)
    wr = w / n
    z = (wr - 0.5) / math.sqrt(0.25 / n)
    print(f"\nA ({args.a}) wins {100*wr:.1f}%  (z={z:+.2f})  PPG {pts/n:+.3f}  vs B "
          f"({args.b}) at {args.ply}-ply | {n} games in {time.time()-t0:.0f}s")
    print("=>", "A STRONGER" if z > 1.96 else "B STRONGER" if z < -1.96 else "TOO CLOSE TO CALL")


if __name__ == "__main__":
    main()
