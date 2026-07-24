"""Measure 3-ply search_dist cost on THIS machine: single-thread seconds/label at
a few candidate widths, plus self-play games/sec. Positions are realistic (short
champion self-play), not just the opening."""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np
import bgcore
from gen_rollout_data import champion_move

MODELS = Path(__file__).resolve().parent.parent / "models"

def gen_positions(pol, n, eps, rng):
    pos = []
    games = 0
    t0 = time.time()
    while len(pos) < n:
        b = bgcore.Board.starting(); boards = []
        res = None
        for _ in range(4000):
            boards.append(b)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            nb, pts = champion_move(pol, b, d1, d2, rng, eps)
            if pts is not None:
                res = pts; break
            b = nb
        if res is None:
            continue
        games += 1
        pos.extend(boards)
    dt = time.time() - t0
    print(f"self-play: {games/dt:6.1f} games/sec | {len(pos)/dt:7.1f} pos/sec "
          f"(single process, {games} games, {dt:.1f}s)", flush=True)
    return pos[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="td.onnx")
    ap.add_argument("--sample", type=int, default=25, help="positions to time per candidate")
    ap.add_argument("--gen", type=int, default=400, help="positions to self-play for the pool")
    ap.add_argument("--cands", default="1,2,3", help="candidate widths to time")
    args = ap.parse_args()

    rng = random.Random(0)
    pol = bgcore.Neural(str(MODELS / args.net), 0, 0)
    pool = gen_positions(pol, args.gen, 0.10, rng)
    # a fixed random sample of positions (mix of game phases)
    idx = random.Random(1).sample(range(len(pool)), min(args.sample, len(pool)))
    sample = [pool[i] for i in idx]

    for c in [int(x) for x in args.cands.split(",")]:
        net = bgcore.Neural(str(MODELS / args.net), 3, c)
        # warm up
        net.search_dist(sample[0])
        times = []
        for b in sample:
            t = time.time()
            net.search_dist(b)
            times.append(time.time() - t)
        times = np.array(times)
        print(f"3-ply candidates={c}: mean {times.mean():6.2f}s  median {np.median(times):6.2f}s  "
              f"min {times.min():5.2f}s  max {times.max():6.2f}s  (n={len(times)}, 1 thread)", flush=True)

if __name__ == "__main__":
    main()
