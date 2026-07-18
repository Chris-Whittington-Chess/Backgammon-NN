"""Generate a rollout-labeled dataset for supervised training (task #4).

For each sampled position we store BOTH signals so the soft/hard blend is tunable
later without re-rolling:
  - a SOFT label: the rollout's 6-outcome distribution [ws,wg,wbg,ls,lg,lbg]
    (distillation-from-search, low variance but biased by our own leaf eval);
  - a HARD label: the actual game outcome from that position's mover perspective
    (unbiased ground truth, high variance).

Positions come from champion self-play (0-ply greedy + a little exploration), so
they are the kind of positions the net actually faces, and naturally spread across
the 12 class x pip buckets. Stored by GNU position id so the set is
architecture-agnostic (re-encodable with any feature set).

Run:
  .venv/Scripts/python trainer/gen_rollout_data.py --net td.onnx --positions 200 \
      --trials 180 --truncate 11 --out rollout_data.npz
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

import bgcore

MODELS = Path(__file__).resolve().parent.parent / "models"


def dist6_from5(d):
    """[win, win_g, win_bg, lose_g, lose_bg] -> the 6 mutually-exclusive outcome
    probabilities [ws, wg, wbg, ls, lg, lbg]."""
    win, wg, wbg, lg, lbg = d
    lose = 1.0 - win
    return [win - wg, wg - wbg, wbg, lose - lg, lg - lbg, lbg]


def champion_move(pol, board, d1, d2, rng, eps):
    """0-ply greedy move (with prob eps a random legal move). Returns
    (next_board_opponent_to_move, None) or (None, terminal_points)."""
    moves = bgcore.legal_moves(board, d1, d2)
    term = [m.winner_points() for m in moves]
    if rng.random() < eps:
        i = rng.randrange(len(moves))
    else:
        eqs = [float(t) if t is not None else -pol.equity(m.swap_perspective())
               for m, t in zip(moves, term)]
        i = max(range(len(moves)), key=lambda k: eqs[k])
    if term[i] is not None:
        return None, term[i]
    return moves[i].swap_perspective(), None


def _save(out, pos_ids, probs, outcomes, buckets, trials, truncate, net):
    np.savez_compressed(
        out, pos_ids=np.array(pos_ids),
        probs=np.asarray(probs, dtype=np.float32),
        outcomes=np.asarray(outcomes, dtype=np.int8),
        buckets=np.asarray(buckets, dtype=np.int8),
        trials=trials, truncate=truncate, net=net)


def generate(net_file, n_positions, trials, truncate, eps, seed, out, save_every):
    rng = random.Random(seed)
    pol = bgcore.Neural(str(MODELS / net_file), 0, 0)
    ro = bgcore.Rollouts(str(MODELS / net_file), trials, truncate, 0, 0x5EED, 0, 0)

    pos_ids, probs, outcomes, buckets = [], [], [], []
    t0 = time.time()
    last_save = 0
    while len(pos_ids) < n_positions:
        boards, b = [], bgcore.Board.starting()
        result = None
        for _ in range(4000):
            boards.append(b)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            nb, pts = champion_move(pol, b, d1, d2, rng, eps)
            if pts is not None:
                result = pts  # from the final mover's perspective (a win, >0)
                break
            b = nb
        if result is None:
            continue
        # Label every decision position of this game.
        for i, bd in enumerate(boards):
            plies_from_end = len(boards) - 1 - i
            signed = result if plies_from_end % 2 == 0 else -result
            pos_ids.append(bd.position_id())
            probs.append(dist6_from5(ro.dist(bd)))
            outcomes.append(signed)
            buckets.append(bd.route_bucket())
            if len(pos_ids) >= n_positions:
                break
        # Periodic progress + incremental checkpoint (a crash keeps most of the run).
        if len(pos_ids) - last_save >= save_every:
            last_save = len(pos_ids)
            _save(out, pos_ids, probs, outcomes, buckets, trials, truncate, net_file)
            rate = len(pos_ids) / max(time.time() - t0, 1e-9)
            eta_h = (n_positions - len(pos_ids)) / max(rate, 1e-9) / 3600
            print(f"  {len(pos_ids):7d}/{n_positions} | {rate:6.1f} pos/sec | "
                  f"ETA {eta_h:4.1f}h | saved", flush=True)

    _save(out, pos_ids, probs, outcomes, buckets, trials, truncate, net_file)
    return (np.array(pos_ids), np.asarray(probs, dtype=np.float32),
            np.asarray(outcomes, dtype=np.int8), np.asarray(buckets, dtype=np.int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="td.onnx", help="ONNX net for policy + rollout leaf")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--trials", type=int, default=180)
    ap.add_argument("--truncate", type=int, default=11)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20000,
                    help="incrementally checkpoint the .npz every N positions")
    ap.add_argument("--out", default="rollout_data.npz")
    args = ap.parse_args()

    out = MODELS / args.out
    print(f"Labeling {args.positions} positions with {args.net} rollouts "
          f"(trials={args.trials}, truncate={args.truncate}, eps={args.eps}) -> {args.out}")
    pos_ids, probs, outcomes, buckets = generate(
        args.net, args.positions, args.trials, args.truncate, args.eps, args.seed,
        out, args.save_every)

    # Sanity summary.
    pop = np.bincount(buckets.astype(int), minlength=12)
    eq = probs @ np.array([1, 2, 3, -1, -2, -3], dtype=np.float32)
    print(f"\nsaved {out} | {len(pos_ids)} positions")
    print(f"per-bucket population: {pop.tolist()}")
    print(f"rollout equity: mean {eq.mean():+.3f}  min {eq.min():+.3f}  max {eq.max():+.3f}")
    print(f"prob rows sum to ~1: min {probs.sum(1).min():.4f} max {probs.sum(1).max():.4f}")


if __name__ == "__main__":
    main()
