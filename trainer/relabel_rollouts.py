"""Relabel an existing position set with UNTRUNCATED rollouts (Monte-Carlo truth).

Unlike gen_distil_data (n-ply *search*, whose leaf is the net's own eval, so the
label is capped at the net's knowledge), this rolls each position to the actual
game end and uses the real outcome frequencies -- an UNBIASED target that can
exceed the teacher. `truncate=0` => roll to the end.

Single process on purpose: `Rollouts.dist` already parallelises its `trials`
across all cores, so there's nothing to gain from worker processes (and much to
lose -- they'd oversubscribe the cores). Checkpoints atomically every
--save-every and resumes from --out, like gen_distil_data.

Run:
  .venv/Scripts/python trainer/relabel_rollouts.py --source posA2.npz --limit 100000 \
      --net td.onnx --trials 300 --truncate 0 --out labels_A2_roll100k.npz --resume
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

# Positions are parallelised across worker PROCESSES; each worker's Rollouts uses
# `threads` internal rollout threads. workers*threads ~ cores keeps the box full
# (a single all-core rollout leaves cores idle to game-length load imbalance).
_RO = None


def _init(net_path, trials, truncate, candidates, seed, threads, playout_plies, playout_cands):
    global _RO
    _RO = bgcore.Rollouts(net_path, trials, truncate, candidates, seed, 0, threads,
                          playout_plies, playout_cands)


def _label_chunk(chunk):
    return np.array(
        [dist6_from5(_RO.dist(bgcore.Board.from_id(str(p)))) for p in chunk],
        dtype=np.float32,
    )


def _atomic_savez(out, **arrays):
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    for attempt in range(10):
        try:
            os.replace(tmp, out)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="npz whose positions to relabel")
    ap.add_argument("--net", default="td.onnx")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--truncate", type=int, default=0, help="0 = roll to the game end (truth)")
    ap.add_argument("--candidates", type=int, default=0, help="rollout move filter; 0 = full width")
    ap.add_argument("--playout-plies", type=int, default=0,
                    help="playout policy strength: 0 = greedy 0-ply (cheap), 1 = 1-ply search (stronger play, ~74x slower)")
    ap.add_argument("--playout-cands", type=int, default=3,
                    help="with --playout-plies 1, 1-ply-search only the top-K moves per ply (explosion guard)")
    ap.add_argument("--start", type=int, default=0,
                    help="relabel the window [start : start+limit] of the source (for splitting a run across machines)")
    ap.add_argument("--limit", type=int, default=0, help="window length from --start; 0 = to the end")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0x5EED)
    ap.add_argument("--workers", type=int, default=15, help="position-parallel worker processes")
    ap.add_argument("--threads", type=int, default=4, help="rollout threads per worker (workers*threads ~ cores)")
    ap.add_argument("--chunk", type=int, default=8, help="positions per work unit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    src = np.load(MODELS / args.source)
    pos_ids = src["pos_ids"]
    outcomes = src["outcomes"].astype(np.int8)
    buckets = src["buckets"].astype(np.int8)
    # Window [start : start+limit] so two machines can each label a disjoint slice
    # (each writes its own --out; merge the outputs afterwards).
    lo = args.start
    hi = len(pos_ids) if not args.limit else min(len(pos_ids), lo + args.limit)
    pos_ids, outcomes, buckets = pos_ids[lo:hi], outcomes[lo:hi], buckets[lo:hi]
    n = len(pos_ids)

    out = MODELS / args.out
    kind = "full games (truth)" if args.truncate == 0 else f"{args.truncate}-ply trunc"
    policy = f"{args.playout_plies}-ply playout" + (f" (top-{args.playout_cands})" if args.playout_plies else "")
    print(f"rollout-relabelling {n} positions [{lo}:{hi}] from {args.source} | {args.trials} trials, {kind}, {policy} "
          f"| {args.workers} workers x {args.threads} threads -> {args.out}", flush=True)

    probs = np.zeros((n, 6), dtype=np.float32)
    start = 0
    if args.resume and out.exists():
        with np.load(out) as prev:
            ppid = prev["pos_ids"]
            start = len(ppid)
            if start > n or not np.array_equal(ppid, pos_ids[:start]):
                raise SystemExit("--resume: existing --out is not a prefix of this source/limit.")
            probs[:start] = prev["probs"]
        print(f"resuming: {start}/{n} already labelled, {n - start} to go", flush=True)

    chunks = [pos_ids[i:i + args.chunk] for i in range(start, n, args.chunk)]
    t0 = time.time()
    done, last_save = start, start
    net_path = str(MODELS / args.net)
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(net_path, args.trials, args.truncate, args.candidates,
                           args.seed, args.threads, args.playout_plies, args.playout_cands)) as pool:
        for res in pool.imap(_label_chunk, chunks):  # imap preserves order for the prefix checkpoint
            probs[done:done + len(res)] = res
            done += len(res)
            if done - last_save >= args.save_every or done == n:
                last_save = done
                _atomic_savez(out, pos_ids=pos_ids[:done], probs=probs[:done],
                              outcomes=outcomes[:done], buckets=buckets[:done],
                              trials=args.trials, truncate=args.truncate, net=args.net,
                              playout_plies=args.playout_plies, playout_cands=args.playout_cands)
                rate = max(done - start, 1) / max(time.time() - t0, 1e-9)
                eta_h = (n - done) / max(rate, 1e-9) / 3600
                print(f"  {done:7d}/{n} | {rate:5.1f} pos/sec | ETA {eta_h:4.1f}h | saved", flush=True)

    print(f"\nsaved {out} | {n} positions (untruncated rollout truth labels)", flush=True)


if __name__ == "__main__":
    main()
