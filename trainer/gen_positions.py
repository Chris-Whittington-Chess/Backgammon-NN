"""Generate a *positions-only* dataset (no rollout/search labels) for benchmarking
and for feeding the n-ply distiller (gen_distil_data.py relabels these positions).

Self-play with the champion net (0-ply greedy + eps exploration), storing each
decision position by GNU id with its signed game outcome and route bucket -- the
exact schema gen_distil_data.py expects as its --source, minus the `probs` (which
the distiller fills in). Also reports self-play throughput (games/sec, pos/sec).

Run:
  .venv/Scripts/python trainer/gen_positions.py --net td.onnx --positions 6000 \
      --out bench_positions.npz
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

import bgcore
from gen_rollout_data import champion_move

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="td.onnx")
    ap.add_argument("--positions", type=int, default=6000)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench_positions.npz")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pol = bgcore.Neural(str(MODELS / args.net), 0, 0)

    pos_ids, outcomes, buckets = [], [], []
    games = 0
    t0 = time.time()
    while len(pos_ids) < args.positions:
        boards, b = [], bgcore.Board.starting()
        result = None
        for _ in range(4000):
            boards.append(b)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            nb, pts = champion_move(pol, b, d1, d2, rng, args.eps)
            if pts is not None:
                result = pts
                break
            b = nb
        if result is None:
            continue
        games += 1
        for i, bd in enumerate(boards):
            plies_from_end = len(boards) - 1 - i
            signed = result if plies_from_end % 2 == 0 else -result
            pos_ids.append(bd.position_id())
            outcomes.append(signed)
            buckets.append(bd.route_bucket())
            if len(pos_ids) >= args.positions:
                break

    dt = time.time() - t0
    out = MODELS / args.out
    np.savez_compressed(
        out, pos_ids=np.array(pos_ids),
        probs=np.zeros((len(pos_ids), 6), dtype=np.float32),  # placeholder; distiller overwrites
        outcomes=np.asarray(outcomes, dtype=np.int8),
        buckets=np.asarray(buckets, dtype=np.int8),
        trials=0, truncate=0, net=args.net)

    pop = np.bincount(np.asarray(buckets, dtype=int), minlength=12)
    print(f"saved {out} | {len(pos_ids)} positions from {games} games")
    print(f"self-play throughput: {games / dt:6.1f} games/sec | "
          f"{len(pos_ids) / dt:7.1f} pos/sec | {dt:.1f}s (single process)")
    print(f"per-bucket population: {pop.tolist()}")


if __name__ == "__main__":
    main()
