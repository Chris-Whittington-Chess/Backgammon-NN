"""Parallel positions-only generator (no labels) -- feeds gen_distil_data.py.

Self-play with the champion net across W worker processes (distinct seeds), each
collecting decision positions by GNU id with signed game outcome + route bucket.
Same schema gen_distil_data.py expects as --source (probs is a zero placeholder
the distiller overwrites).

Use a distinct --seed-base per machine so the two Threadrippers generate DIFFERENT
positions (seeds = seed_base*1000 + worker_id).

Run (Machine A):
  .venv/Scripts/python trainer/gen_positions_par.py --positions 500000 \
      --seed-base 0 --workers 16 --out posA.npz
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import bgcore
from gen_rollout_data import champion_move

MODELS = Path(__file__).resolve().parent.parent / "models"

_POL = None
_EPS = 0.10


def _init(net_path, eps):
    global _POL, _EPS
    _POL = bgcore.Neural(net_path, 0, 0)
    _EPS = eps


def _gen_chunk(task):
    import random
    n, seed = task
    rng = random.Random(seed)
    ids, outs, bks = [], [], []
    while len(ids) < n:
        b = bgcore.Board.starting()
        boards = []
        res = None
        for _ in range(4000):
            boards.append(b)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            nb, pts = champion_move(_POL, b, d1, d2, rng, _EPS)
            if pts is not None:
                res = pts
                break
            b = nb
        if res is None:
            continue
        for i, bd in enumerate(boards):
            plies_from_end = len(boards) - 1 - i
            signed = res if plies_from_end % 2 == 0 else -res
            ids.append(bd.position_id())
            outs.append(signed)
            bks.append(bd.route_bucket())
            if len(ids) >= n:
                break
    return ids, outs, bks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="td.onnx")
    ap.add_argument("--positions", type=int, default=500000)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--seed-base", type=int, default=0, help="distinct per machine")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="posA.npz")
    args = ap.parse_args()

    base = args.positions // args.workers
    rem = args.positions % args.workers
    tasks = [(base + (1 if i < rem else 0), args.seed_base * 1000 + i)
             for i in range(args.workers)]
    net_path = str(MODELS / args.net)

    t0 = time.time()
    ids, outs, bks = [], [], []
    with mp.Pool(args.workers, initializer=_init, initargs=(net_path, args.eps)) as pool:
        for cid, co, cb in pool.imap_unordered(_gen_chunk, tasks):
            ids += cid; outs += co; bks += cb
    dt = time.time() - t0

    out = MODELS / args.out
    np.savez_compressed(
        out, pos_ids=np.array(ids),
        probs=np.zeros((len(ids), 6), dtype=np.float32),
        outcomes=np.asarray(outs, dtype=np.int8),
        buckets=np.asarray(bks, dtype=np.int8),
        trials=0, truncate=0, net=args.net)
    pop = np.bincount(np.asarray(bks, dtype=int), minlength=12)
    print(f"saved {out} | {len(ids)} positions in {dt:.1f}s "
          f"({len(ids)/dt:.0f} pos/sec, {args.workers} workers)")
    print(f"per-bucket population: {pop.tolist()}")


if __name__ == "__main__":
    main()
