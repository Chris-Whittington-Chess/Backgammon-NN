"""1-ply distillation labels (task #6, ceiling-breaking attempt).

The rollout experiments all capped at champion parity because their labels
bottomed out at the champion's own 0-ply eval. Here the label is a STRONGER
estimate: the position's value after ONE lookahead under the champion's own
greedy policy --

    label(P) = E over the mover's 21 dice rolls [ dist of the champion's
               0-ply-best move's resulting position, in the mover's frame ]

i.e. one Bellman backup of the net's value. Distilling the net toward this
pulls its static eval up by ~one ply (policy improvement). Needs only 0-ply
evals, so it's pure Python on the existing engine -- no distribution-returning
search required (that's only needed for 2-ply+).

Relabels the positions of an existing rollout .npz (same positions, new soft
labels) so the comparison against the rollout runs is apples-to-apples.

Run:
  .venv/Scripts/python trainer/gen_distil_data.py --source rollout_lambda1.npz \
      --net td.onnx --limit 500000 --out rollout_1ply.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import bgcore
from gen_rollout_data import dist6_from5

MODELS = Path(__file__).resolve().parent.parent / "models"


def win6(pts):
    """Mover wins `pts` (1/2/3) -> the mutually-exclusive one-hot [ws,wg,wbg,ls,lg,lbg]:
    single win [1,0,0,..], gammon [0,1,0,..], backgammon [0,0,1,..]."""
    d = [0.0] * 6
    d[pts - 1] = 1.0
    return d


def swap_winlose(d6):
    """Opponent-frame 6-dist -> mover frame: swap the win and lose triples."""
    return [d6[3], d6[4], d6[5], d6[0], d6[1], d6[2]]


def label_nply(net, board):
    """The n-ply expectiminimax value distribution of `board`, for the side to
    move, as the mutually-exclusive 6-outcome [ws,wg,wbg,ls,lg,lbg]. The net's
    `lookahead`/`candidates` set the depth; the dice-averaging and PV propagation
    run in Rust (search_dist). 1-ply here matches the earlier Python 1-ply label.
    With selective deepening (`_SELECTIVE`) the all-rank-1 frontier is extended one
    ply via `search_dist_ext`."""
    if _SELECTIVE:
        d5, _total, _ext = net.search_dist_ext(board)
        return dist6_from5(d5)
    return dist6_from5(net.search_dist(board))


# --- Multiprocessing: scores/dist hold the GIL, so parallelise across PROCESSES,
#     each with its own Neural (like the gnubg h2h workers). ---
_NET = None
_SELECTIVE = False


def _init(net_path, lookahead, candidates, cand_schedule, selective):
    global _NET, _SELECTIVE
    _NET = bgcore.Neural(net_path, lookahead, candidates, cand_schedule=cand_schedule)
    _SELECTIVE = selective


def _label_chunk(pos_id_chunk):
    out = np.zeros((len(pos_id_chunk), 6), dtype=np.float32)
    for i, pid in enumerate(pos_id_chunk):
        out[i] = label_nply(_NET, bgcore.Board.from_id(str(pid)))
    return out


def _atomic_savez(out, **arrays):
    """Checkpoint write that survives a mid-write crash: write a sibling temp
    file, then os.replace() it over `out` (atomic on the same filesystem, incl.
    Windows). A reboot during the save leaves the previous checkpoint intact."""
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp.npz")  # ends in .npz -> savez won't re-suffix
    np.savez_compressed(tmp, **arrays)
    for attempt in range(10):  # os.replace can transiently fail on Windows (AV/indexer lock)
        try:
            os.replace(tmp, out)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rollout_lambda1.npz",
                    help="npz whose positions (pos_ids/outcomes/buckets) to relabel")
    ap.add_argument("--net", default="td.onnx")
    ap.add_argument("--lookahead", type=int, default=1, help="search depth in half-moves")
    ap.add_argument("--candidates", type=int, default=0,
                    help="prune deep (2-ply+) nodes to the best N moves; 0 = full width")
    ap.add_argument("--cand-schedule", type=int, nargs="+", default=None,
                    help="per-ply candidate counts from the root inward (e.g. 3 3); "
                         "overrides --candidates")
    ap.add_argument("--selective", action="store_true",
                    help="selective deepening: extend the all-rank-1 frontier by one ply")
    ap.add_argument("--limit", type=int, default=0, help="relabel a random N-subset")
    ap.add_argument("--limit-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--chunk", type=int, default=2000, help="positions per work unit")
    ap.add_argument("--save-every", type=int, default=50000)
    ap.add_argument("--out", default="rollout_1ply.npz")
    ap.add_argument("--resume", action="store_true",
                    help="if --out already exists, keep its labeled prefix and only "
                         "label the remaining positions (use the SAME --source/--limit/"
                         "--limit-seed as the interrupted run)")
    args = ap.parse_args()

    src = np.load(MODELS / args.source)
    pos_ids = src["pos_ids"]
    outcomes = src["outcomes"].astype(np.int8)
    buckets = src["buckets"].astype(np.int8)
    if args.limit and args.limit < len(pos_ids):
        sel = np.random.default_rng(args.limit_seed).choice(
            len(pos_ids), size=args.limit, replace=False)
        pos_ids, outcomes, buckets = pos_ids[sel], outcomes[sel], buckets[sel]
    n = len(pos_ids)

    out = MODELS / args.out
    mode = "selective-deepen " if args.selective else ""
    cand_desc = args.cand_schedule if args.cand_schedule is not None else args.candidates
    print(f"{args.lookahead}-ply {mode}distilling {n} positions from {args.source} with "
          f"{args.net} (candidates={cand_desc}) | {args.workers} workers -> {args.out}",
          flush=True)

    probs = np.zeros((n, 6), dtype=np.float32)
    # --resume: reuse the labeled prefix of an existing checkpoint. imap preserves
    # source order and each checkpoint is pos_ids[:done], so a valid checkpoint is a
    # prefix of this run's pos_ids -- verify that before trusting it.
    start = 0
    if args.resume and out.exists():
        # `with` closes the file handle before the loop os.replace()s over `out`
        # (Windows refuses to replace a file with an open handle).
        with np.load(out) as prev:
            ppid = prev["pos_ids"]
            start = len(ppid)
            if start > n or not np.array_equal(ppid, pos_ids[:start]):
                raise SystemExit(
                    f"--resume: {args.out} ({start} positions) is not a prefix of the "
                    f"current source order -- re-run with the SAME --source/--limit/"
                    f"--limit-seed as the interrupted run, or drop --resume to relabel.")
            probs[:start] = prev["probs"]
        print(f"resuming: {start}/{n} already labeled in {args.out}, "
              f"{n - start} to go", flush=True)

    chunks = [pos_ids[i:i + args.chunk] for i in range(start, n, args.chunk)]
    t0 = time.time()
    done, last_save = start, start
    net_path = str(MODELS / args.net)
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(net_path, args.lookahead, args.candidates,
                           args.cand_schedule, args.selective)) as pool:
        for res in pool.imap(_label_chunk, chunks):  # imap preserves input order
            probs[done:done + len(res)] = res
            done += len(res)
            if done - last_save >= args.save_every or done == n:
                last_save = done
                _atomic_savez(out, pos_ids=pos_ids[:done], probs=probs[:done],
                              outcomes=outcomes[:done], buckets=buckets[:done],
                              trials=0, truncate=1, net=args.net)
                rate = max(done - start, 1) / max(time.time() - t0, 1e-9)
                eta_h = (n - done) / max(rate, 1e-9) / 3600
                print(f"  {done:7d}/{n} | {rate:6.1f} pos/sec | ETA {eta_h:4.1f}h | saved",
                      flush=True)

    eq = probs @ np.array([1, 2, 3, -1, -2, -3], dtype=np.float32)
    print(f"\nsaved {out} | {n} positions")
    print(f"{args.lookahead}-ply equity: mean {eq.mean():+.3f}  min {eq.min():+.3f}  max {eq.max():+.3f}")
    print(f"prob rows sum to ~1: min {probs.sum(1).min():.4f} max {probs.sum(1).max():.4f}")


if __name__ == "__main__":
    main()
