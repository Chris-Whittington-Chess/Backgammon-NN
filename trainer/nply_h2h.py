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


def _init(a_path, b_path, ply_a, ply_b, cand_a, cand_b):
    global _A, _B
    # Separate depths so a net can be played against ITSELF at a different ply —
    # that is the only way to measure what a rung of search is worth, and it is
    # what the Elo ladder's search steps are built from. Separate CANDIDATE
    # widths for the same reason: pruning is a strength setting like depth, and
    # the two sides have to differ for it to be measurable at all.
    _A = bgcore.Neural(a_path, ply_a, cand_a if ply_a >= 2 else 0)
    _B = bgcore.Neural(b_path, ply_b, cand_b if ply_b >= 2 else 0)


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
    ap.add_argument("--ply", type=int, default=1, help="depth for BOTH sides unless overridden")
    ap.add_argument("--ply-a", type=int, default=None, help="A's depth (default: --ply)")
    ap.add_argument("--ply-b", type=int, default=None, help="B's depth (default: --ply)")
    ap.add_argument("--candidates", type=int, default=0, help="prune 2-ply+ nodes; 0 = full")
    ap.add_argument("--candidates-a", type=int, default=None, help="A's width (default: --candidates)")
    ap.add_argument("--candidates-b", type=int, default=None, help="B's width (default: --candidates)")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=1000,
                    help="dice-stream base. Fixed by default so runs are comparable; "
                         "vary it per iteration when tuning, or the tuner fits THESE dice.")
    args = ap.parse_args()

    ply_a = args.ply if args.ply_a is None else args.ply_a
    ply_b = args.ply if args.ply_b is None else args.ply_b
    cand_a = args.candidates if args.candidates_a is None else args.candidates_a
    cand_b = args.candidates if args.candidates_b is None else args.candidates_b
    depth = f"{ply_a}-ply" if ply_a == ply_b else f"A {ply_a}-ply vs B {ply_b}-ply"
    if cand_a != cand_b:
        depth += f" | width A={cand_a or 'full'} B={cand_b or 'full'}"
    jobs = [(g, args.seed + g // 2, g % 2 == 0) for g in range(args.games)]  # mirrored dice
    print(f"A={args.a} vs B={args.b} at {depth} | {args.games} games, mirrored "
          f"dice, {args.workers} workers\n", flush=True)
    t0 = time.time()
    a_path, b_path = str(MODELS / args.a), str(MODELS / args.b)
    results = []
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(a_path, b_path, ply_a, ply_b, cand_a, cand_b)) as pool:
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
          f"({args.b}) at {depth} | {n} games in {time.time()-t0:.0f}s")
    print("=>", "A STRONGER" if z > 1.96 else "B STRONGER" if z < -1.96 else "TOO CLOSE TO CALL")


if __name__ == "__main__":
    main()
