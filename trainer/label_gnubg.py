"""Relabel a position set with gnubg's evaluation — an EXTERNAL teacher.

Every label we have generated from our own engine (n-ply search, rollout truth at
0-ply and 1-ply playout) came back at parity: a student distilled from its own
teacher cannot exceed it. gnubg is outside that loop. Measured on this box over
1000 mirrored-dice games:

    our champion vs gnubg 0-ply : 49.0% (z -0.28)  -> parity, useless as a teacher
    our champion vs gnubg 2-ply : 43.9% (z -3.86)  -> clearly stronger

So gnubg at 2 ply is a teacher whose labels can actually break the ceiling, and it
is nearly free: gnubg computes the full static/1-ply/2-ply table on EVERY `eval`
regardless of settings (`set evaluation chequerplay eval plies N` does not change
it), so the deep evaluation costs the same ~0.04s as the shallow one.

Output schema matches gen_rollout_data / relabel_rollouts (pos_ids, probs,
outcomes, buckets) so it drops straight into train_rollout.py.

Run:
  .venv/Scripts/python trainer/label_gnubg.py --source posA2.npz --limit 60000 \
      --workers 60 --out labels_gnubg2.npz --resume
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import bgcore
from gen_rollout_data import dist6_from5

GNUBG = os.environ.get("GNUBG_CLI", r"C:/Users/chris/AppData/Local/gnubg/gnubg-cli.exe")
MODELS = Path(__file__).resolve().parent.parent / "models"

# One `eval` prints a block: a header, then `static:`, ` 1 ply:`, ` 2 ply:` (only
# the depths it actually computed), then a `N-ply cubeless equity` summary.
# The five columns are the NESTED probabilities [Win, W(g), W(bg), L(g), L(bg)] —
# the same convention as bgcore's Value, so dist6_from5 converts them directly.
ROW = re.compile(
    r"^\s*(static:|(\d+) ply:)((?:\s+[\d.]+){5})\s+[+-][\d.]+")
END = re.compile(r"\d+-ply cubeless equity")


def row_depth(m: re.Match) -> int:
    """0 for the `static:` row, N for an ` N ply:` row."""
    return 0 if m.group(2) is None else int(m.group(2))


class GnubgLabeller:
    """Persistent gnubg-cli process returning the 5-vector for each position.

    Reads the DEEPEST row present in each eval block, capped at `max_plies`.
    Taking "the ` 2 ply:` row" directly would be wrong: positions gnubg answers
    from its bearoff databases print `static:` and no deeper rows, so a parser
    counting ` 2 ply:` lines silently comes up short and then blocks until it
    times out. Those static rows are database-exact, so falling back to them
    loses nothing. Blocks are delimited by the always-present summary line.
    """

    def __init__(self, max_plies: int = 2, timeout: float = 30.0):
        self.max_plies = max_plies
        self.timeout = timeout
        self._start()

    def _start(self):
        self.p = subprocess.Popen(
            [GNUBG, "-t", "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.p.stdin.write("new game\n")
        self.p.stdin.flush()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put("")

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()

    def label(self, pos_ids) -> np.ndarray:
        """`[n, 5]` nested probabilities, one row per position, in input order."""
        self.p.stdin.write("".join(f"set board {pid}\neval\n" for pid in pos_ids))
        self.p.stdin.flush()
        out = np.zeros((len(pos_ids), 5), dtype=np.float32)
        got = 0
        best = None          # deepest row seen in the current block
        best_depth = -1
        while got < len(pos_ids):
            try:
                line = self.q.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError(f"gnubg timed out after {got}/{len(pos_ids)} positions")
            if line == "":
                raise RuntimeError("gnubg exited unexpectedly")
            m = ROW.match(line)
            if m:
                d = row_depth(m)
                if d <= self.max_plies and d > best_depth:
                    best_depth = d
                    best = [float(x) for x in m.group(3).split()]
                continue
            if END.search(line) and best is not None:
                out[got] = best
                got += 1
                best, best_depth = None, -1
        return out


_LAB = None


def _init(max_plies, timeout):
    global _LAB
    _LAB = GnubgLabeller(max_plies, timeout)


def _label_chunk(chunk):
    """Label a chunk, restarting gnubg and retrying on a hang.

    Never silently drops a position: a chunk that cannot be labelled after the
    retries raises, which aborts the run with everything checkpointed so far
    intact (rerun with --resume). Silently dropping would bias the label set.
    """
    global _LAB
    for attempt in range(3):
        try:
            return _LAB.label([str(p) for p in chunk])
        except Exception:
            try:
                _LAB.close()
            except Exception:
                pass
            _init(_LAB.max_plies, _LAB.timeout)
    raise RuntimeError(f"gnubg failed 3x on chunk starting {chunk[0]}")


def _atomic_savez(out: Path, **arrays):
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
    ap.add_argument("--source", default="posA2.npz", help="npz whose positions to relabel")
    ap.add_argument("--start", type=int, default=0, help="window start into the source")
    ap.add_argument("--limit", type=int, default=0, help="window length; 0 = to the end")
    ap.add_argument("--plies", type=int, default=2,
                    help="deepest gnubg row to read (2 = its strongest; 0 = static)")
    ap.add_argument("--workers", type=int, default=60,
                    help="gnubg processes (Windows mp.Pool tops out near 60)")
    ap.add_argument("--chunk", type=int, default=64, help="positions per work unit")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds to wait per gnubg line")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    src = np.load(MODELS / args.source)
    pos_ids = src["pos_ids"]
    outcomes = src["outcomes"].astype(np.int8)
    buckets = src["buckets"].astype(np.int8)
    lo = args.start
    hi = len(pos_ids) if not args.limit else min(len(pos_ids), lo + args.limit)
    pos_ids, outcomes, buckets = pos_ids[lo:hi], outcomes[lo:hi], buckets[lo:hi]
    n = len(pos_ids)

    out = MODELS / args.out
    print(f"gnubg-labelling {n} positions [{lo}:{hi}] from {args.source} | teacher "
          f"gnubg {args.plies}-ply | {args.workers} workers -> {args.out}", flush=True)

    probs = np.zeros((n, 6), dtype=np.float32)
    start = 0
    if args.resume and out.exists():
        with np.load(out) as prev:
            ppid = prev["pos_ids"]
            start = len(ppid)
            if start > n or not np.array_equal(ppid, pos_ids[:start]):
                raise SystemExit("--resume: existing --out is not a prefix of this window.")
            probs[:start] = prev["probs"]
        print(f"resuming: {start}/{n} already labelled, {n - start} to go", flush=True)

    chunks = [pos_ids[i:i + args.chunk] for i in range(start, n, args.chunk)]
    t0 = time.time()
    done, last_save = start, start
    with mp.Pool(args.workers, initializer=_init, initargs=(args.plies, args.timeout)) as pool:
        for res in pool.imap(_label_chunk, chunks):   # imap preserves order
            probs[done:done + len(res)] = np.array([dist6_from5(r) for r in res],
                                                   dtype=np.float32)
            done += len(res)
            if done - last_save >= args.save_every or done == n:
                last_save = done
                _atomic_savez(out, pos_ids=pos_ids[:done], probs=probs[:done],
                              outcomes=outcomes[:done], buckets=buckets[:done],
                              teacher="gnubg", plies=args.plies, source=args.source)
                rate = max(done - start, 1) / max(time.time() - t0, 1e-9)
                eta_h = (n - done) / max(rate, 1e-9) / 3600
                print(f"  {done:7d}/{n} | {rate:6.1f} pos/sec | ETA {eta_h:4.1f}h | saved",
                      flush=True)

    eq = probs @ np.array([1, 2, 3, -1, -2, -3], dtype=np.float32)
    print(f"\nsaved {out} | {n} positions labelled by gnubg {args.plies}-ply")
    print(f"equity mean {eq.mean():+.4f}  min {eq.min():+.3f}  max {eq.max():+.3f}")
    print(f"prob rows sum to ~1: min {probs.sum(1).min():.4f} max {probs.sum(1).max():.4f}")


if __name__ == "__main__":
    main()
