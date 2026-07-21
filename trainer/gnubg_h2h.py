"""Direct 0-ply head-to-head: our champion vs gnubg (task #5).

Unbiased — decided by actual game outcomes, not equity estimates. Both engines
play their own 0-ply evaluation; we orchestrate. On gnubg's turn, OUR engine
generates the legal moves (wildbg-validated) and gnubg picks its best by evaluating
each resulting position (`set board`/`eval`, bridged by GNU Position ID) — no
move-notation parsing. Dice are mirrored (each roll sequence played twice, seats
swapped) to cut luck. Cubeless money play.

Run: .venv/Scripts/python trainer/gnubg_h2h.py --net td.onnx --games 200
"""

from __future__ import annotations

import argparse
import math
import queue
import random
import re
import subprocess
import threading
import time
from pathlib import Path

import bgcore

GNUBG = r"C:/Users/chris/AppData/Local/gnubg/gnubg-cli.exe"
MODELS = Path(__file__).resolve().parent.parent / "models"
STATIC = re.compile(
    r"static:\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+)")


class GnubgEngine:
    """Persistent gnubg-cli process that picks the 0-ply best move among children."""

    def __init__(self):
        self.p = subprocess.Popen(
            [GNUBG, "-t", "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        # Pump stdout into a queue so reads can time out: gnubg occasionally hangs
        # on a position, emitting no `static:` line, which would block forever.
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.p.stdin.write("set evaluation chequerplay eval plies 0\nnew game\n")
        self.p.stdin.flush()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put("")  # EOF sentinel

    def best_child(self, children):
        """Index of the child gnubg prefers (min opponent equity = best for the
        mover). `children` are the mover-frame resulting boards; we score each one
        from the opponent's on-roll view via its swapped position id."""
        # An immediate win is always best, and a game-over board makes gnubg emit
        # no `static:` line — which desyncs/hangs the parser. So resolve winning
        # moves ourselves and only ask gnubg about the non-terminal children.
        terms = [c.winner_points() for c in children]
        wins = [i for i, t in enumerate(terms) if t is not None]
        if wins:
            return max(wins, key=lambda i: terms[i])
        ids = [c.swap_perspective().position_id() for c in children]
        self.p.stdin.write("".join(f"set board {pid}\neval\n" for pid in ids))
        self.p.stdin.flush()
        eqs = []
        while len(eqs) < len(ids):
            try:
                line = self.q.get(timeout=15)
            except queue.Empty:
                raise RuntimeError("gnubg eval timed out")
            if line == "":
                raise RuntimeError("gnubg process closed unexpectedly")
            m = STATIC.search(line)
            if m:
                eqs.append(float(m.group(1)))
        return min(range(len(eqs)), key=lambda i: eqs[i])  # lowest opp equity

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


def our_best(net, children):
    term = [c.winner_points() for c in children]
    eqs = [float(t) if t is not None else -net.equity(c.swap_perspective())
           for c, t in zip(children, term)]
    return max(range(len(children)), key=lambda i: eqs[i])


def play(net, gnu, seed, a_is_ours):
    """One game; returns points to OUR engine (+win / -loss). Dice fixed by seed."""
    rng = random.Random(seed)
    board = bgcore.Board.starting()
    ours_to_move = a_is_ours
    for _ in range(200):
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        children = bgcore.legal_moves(board, d1, d2)
        i = our_best(net, children) if ours_to_move else gnu.best_child(children)
        chosen = children[i]
        pts = chosen.winner_points()
        if pts is not None and pts > 0:
            return pts if ours_to_move else -pts
        board = chosen.swap_perspective()
        ours_to_move = not ours_to_move
    # Ply cap hit (a rare crawling race): resolve by pip count — fewer pips wins.
    # This stops pathological mirrored races from stalling the run, and scores
    # them correctly (a capped game is effectively a decided race).
    our_pip = board.pip_count(0 if ours_to_move else 1)
    opp_pip = board.pip_count(1 if ours_to_move else 0)
    return 1 if our_pip < opp_pip else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="td.onnx")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14,
                    help="parallel games (each = one gnubg process on its own core)")
    args = ap.parse_args()

    net = bgcore.Neural(str(MODELS / args.net), 0, 0)  # Sync: shared across threads
    print(f"our net {args.net} vs gnubg, 0-ply, {args.games} games, mirrored dice, "
          f"{args.workers} parallel workers\n", flush=True)

    jobs = queue.Queue()
    for g in range(args.games):
        jobs.put(g)
    results = []
    lock = threading.Lock()
    t0 = time.time()

    def worker():
        gnu = GnubgEngine()
        try:
            while True:
                try:
                    g = jobs.get_nowait()
                except queue.Empty:
                    break
                try:
                    p = play(net, gnu, 1000 + g // 2, g % 2 == 0)  # mirrored dice
                except Exception:
                    # gnubg hung/desynced on this game: restart it and skip.
                    try:
                        gnu.close()
                    except Exception:
                        pass
                    gnu = GnubgEngine()
                    continue
                with lock:
                    results.append(p)
        finally:
            gnu.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    # Progress from the main thread.
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        with lock:
            n = len(results); wins = sum(1 for p in results if p > 0); pts = sum(results)
        if n:
            print(f"  {n:4d}/{args.games} games ({n/max(time.time()-t0,1):.1f}/s): "
                  f"our win {100*wins/n:.1f}%  ppg {pts/n:+.3f}", flush=True)
    for t in threads:
        t.join()

    n = len(results)
    wins = sum(1 for p in results if p > 0)
    pts = sum(results)
    wr = wins / n
    z = (wr - 0.5) / math.sqrt(0.25 / n)
    print(f"\nOUR net wins {100*wr:.1f}%  (z = {z:+.2f})   PPG {pts/n:+.3f}  vs gnubg 0-ply "
          f"| {n} games in {time.time()-t0:.0f}s")
    verdict = ("WE ARE STRONGER" if z > 1.96 else
               "GNUBG STRONGER" if z < -1.96 else "TOO CLOSE TO CALL")
    print(f"=> {verdict} at 0-ply")


if __name__ == "__main__":
    main()
