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


def _init(net_path, trials, truncate, candidates, seed, threads):
    global _RO
    _RO = bgcore.Rollouts(net_path, trials, truncate, candidates, seed, 0, threads)


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
    ap.add_argument("--limit", type=int, default=0, help="relabel the FIRST N positions (contiguous)")
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
    if args.limit and args.limit < len(pos_ids):
        pos_ids, outcomes, buckets = pos_ids[:args.limit], outcomes[:args.limit], buckets[:args.limit]
    n = len(pos_ids)

    out = MODELS / args.out
    kind = "full games (truth)" if args.truncate == 0 else f"{args.truncate}-ply trunc"
    print(f"rollout-relabelling {n} positions from {args.source} | {args.trials} trials, {kind} "
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
                           args.seed, args.threads)) as pool:
        for res in pool.imap(_label_chunk, chunks):  # imap preserves order for the prefix checkpoint
            probs[done:done + len(res)] = res
            done += len(res)
            if done - last_save >= args.save_every or done == n:
                last_save = done
                _atomic_savez(out, pos_ids=pos_ids[:done], probs=probs[:done],
                              outcomes=outcomes[:done], buckets=buckets[:done],
                              trials=args.trials, truncate=args.truncate, net=args.net)
                rate = max(done - start, 1) / max(time.time() - t0, 1e-9)
                eta_h = (n - done) / max(rate, 1e-9) / 3600
                print(f"  {done:7d}/{n} | {rate:5.1f} pos/sec | ETA {eta_h:4.1f}h | saved", flush=True)

    print(f"\nsaved {out} | {n} positions (untruncated rollout truth labels)", flush=True)


if __name__ == "__main__":
    main()
