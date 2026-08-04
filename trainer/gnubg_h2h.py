"""Direct head-to-head: our champion vs gnubg (task #5).

Unbiased — decided by actual game outcomes, not equity estimates. Our net plays
at `--our-ply` (default 0 = static eval), gnubg at `--plies` (default 0); set them
equal for a like-for-like comparison. On gnubg's turn, OUR engine
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
from label_gnubg import END, ROW, LabelSink, row_depth

import os
# Path to gnubg-cli.exe. Override per-machine with the GNUBG_CLI env var so this
# doesn't have to be edited on each box (e.g. box B installs gnubg elsewhere).
GNUBG = os.environ.get("GNUBG_CLI", r"C:/Users/chris/AppData/Local/gnubg/gnubg-cli.exe")
MODELS = Path(__file__).resolve().parent.parent / "models"
STATIC = re.compile(
    r"static:\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+)")


# gnubg's `eval` ALWAYS computes the full table (static / 1 ply / 2 ply) and ends
# with a `N-ply cubeless equity` summary. `set evaluation chequerplay eval plies N`
# does NOT change it — verified: N=0, 2 and 3 give byte-identical output. So gnubg's
# playing strength here is decided purely by WHICH ROW WE READ, and 2 ply is the
# deepest available.
DEEP = re.compile(r"\d+-ply cubeless equity\s+([+-][\d.]+)")
ONE_PLY = re.compile(r"^\s*1 ply:" + r"\s+[\d.]+" * 5 + r"\s+([+-][\d.]+)")


def equity_re(plies: int):
    """The regex selecting how strong gnubg plays: 0 = its static eval, 1 = its
    1-ply row, 2+ = its deepest (2-ply) evaluation.

    The deep case reads the summary line rather than the ` 2 ply:` row on purpose.
    Positions gnubg answers from its bearoff databases print `static:` but no
    deeper rows, so counting ` 2 ply:` rows silently comes up short — the read then
    blocks until the 15s timeout and the worker DISCARDS the whole game. The
    summary line is emitted exactly once per eval, whatever the position.
    """
    if plies <= 0:
        return STATIC
    return ONE_PLY if plies == 1 else DEEP


class GnubgEngine:
    """Persistent gnubg-cli process picking the best move among children at `plies`.

    With a `sink`, the five probability columns gnubg prints for every child are
    also captured instead of discarded — see `label_gnubg.LabelSink`. Move choice
    is unaffected: it still uses the same equity from the same row as before, so
    results stay comparable with runs made before capture existed.
    """

    def __init__(self, plies: int = 0, sink=None):
        self.plies = plies
        self.sink = sink
        self.re = equity_re(plies)
        self.p = subprocess.Popen(
            [GNUBG, "-t", "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        # Pump stdout into a queue so reads can time out: gnubg occasionally hangs
        # on a position, emitting no `static:` line, which would block forever.
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.p.stdin.write(f"set evaluation chequerplay eval plies {plies}\nnew game\n")
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
        # Separate counter: at 0 ply the equity row IS the `static:` row, so
        # `len(eqs)` has already advanced by the time the block terminator
        # arrives and would misalign every captured label by one position.
        cap = 0
        best, best_depth = None, -1      # deepest 5-column row in the current block
        while len(eqs) < len(ids):
            try:
                line = self.q.get(timeout=15)
            except queue.Empty:
                raise RuntimeError("gnubg eval timed out")
            if line == "":
                raise RuntimeError("gnubg process closed unexpectedly")
            if self.sink is not None:
                # Capture alongside, without touching the move-choice parse.
                rm = ROW.match(line)
                if rm:
                    d = row_depth(rm)
                    if d > best_depth:
                        best, best_depth = [float(x) for x in rm.group(3).split()], d
                elif END.search(line) and best is not None and cap < len(ids):
                    self.sink.add(ids[cap], best)
                    cap += 1
                    best, best_depth = None, -1
            m = self.re.search(line)
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


def our_best_searched(net, board, d1, d2):
    """Our move when searching (`--our-ply >= 1`): argmax of the per-move equities
    the search returns, aligned with genmoves order.

    This is deliberately `scores`, not the Rust `SearchEngine::choose`: the app's
    NativeNeuralEngine also picks by sorting `scores`, so this measures what the
    shipped engine actually plays. The two can differ — a pruned move keeps its
    static value in `scores`, which can top a searched move's deep value.
    """
    eqs = net.scores(board, d1, d2)
    return max(range(len(eqs)), key=lambda i: eqs[i])


def play(net, gnu, seed, a_is_ours, our_ply=0):
    """One game; returns points to OUR engine (+win / -loss). Dice fixed by seed."""
    rng = random.Random(seed)
    board = bgcore.Board.starting()
    ours_to_move = a_is_ours
    for _ in range(200):
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        children = bgcore.legal_moves(board, d1, d2)
        if ours_to_move:
            i = (our_best_searched(net, board, d1, d2) if our_ply
                 else our_best(net, children))
        else:
            i = gnu.best_child(children)
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
    ap.add_argument("--plies", type=int, default=0,
                    help="gnubg's search depth (0 = static)")
    ap.add_argument("--our-ply", type=int, default=0,
                    help="OUR net's search depth (0 = static eval, the historic default). "
                         "Set equal to --plies for an equal-depth comparison.")
    ap.add_argument("--dump-labels", default=None,
                    help="capture gnubg's 2-ply evaluation of every child it scores "
                         "(~500 labelled positions per game, otherwise discarded) and "
                         "write them to this npz under models/. Free DAgger data: it is "
                         "the against-gnubg distribution, not champion self-play.")
    ap.add_argument("--our-candidates", type=int, default=4,
                    help="candidate filter for our 2-ply+ search; 0 = full width. "
                         "4 matches NativeNeuralEngine.CANDIDATES, i.e. what the app plays. "
                         "Ignored below 2 ply, which is full width anyway.")
    args = ap.parse_args()

    # The search depth lives in the CONSTRUCTOR, not in which scoring function is
    # called: `scores()` on a net built at 0 ply returns static values however it
    # is invoked. This was hardcoded to (0, 0), so every `--our-ply 2` run before
    # 2026-08-04 actually played our net at 0 ply against a searching gnubg, and
    # `--our-candidates` was parsed but never reached the engine.
    cand = args.our_candidates if args.our_ply >= 2 else 0
    net = bgcore.Neural(str(MODELS / args.net), args.our_ply, cand)  # shared across threads
    print(f"our net {args.net} ({args.our_ply}-ply) vs gnubg {args.plies}-ply, "
          f"{args.games} games, mirrored dice, {args.workers} parallel workers\n",
          flush=True)

    sink = LabelSink() if args.dump_labels else None
    jobs = queue.Queue()
    for g in range(args.games):
        jobs.put(g)
    results = []
    skipped = []          # games gnubg hung/desynced on: MUST be reported, not hidden
    lock = threading.Lock()
    t0 = time.time()

    def worker():
        gnu = GnubgEngine(args.plies, sink)
        try:
            while True:
                try:
                    g = jobs.get_nowait()
                except queue.Empty:
                    break
                try:
                    p = play(net, gnu, 1000 + g // 2, g % 2 == 0, args.our_ply)  # mirrored dice
                except Exception:
                    # gnubg hung/desynced on this game: restart it and skip.
                    try:
                        gnu.close()
                    except Exception:
                        pass
                    gnu = GnubgEngine(args.plies, sink)
                    with lock:
                        skipped.append(g)
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
    if skipped:
        print(f"\nWARNING: {len(skipped)} of {args.games} games DISCARDED (gnubg hung/"
              f"desynced). Survivors are not a random subset — treat the result as "
              f"unreliable until this is zero.")
    wins = sum(1 for p in results if p > 0)
    pts = sum(results)
    wr = wins / n
    z = (wr - 0.5) / math.sqrt(0.25 / n)
    print(f"\nOUR net wins {100*wr:.1f}%  (z = {z:+.2f})   PPG {pts/n:+.3f}  vs gnubg {args.plies}-ply "
          f"| {n} games in {time.time()-t0:.0f}s")
    verdict = ("WE ARE STRONGER" if z > 1.96 else
               "GNUBG STRONGER" if z < -1.96 else "TOO CLOSE TO CALL")
    print(f"=> {verdict} vs gnubg {args.plies}-ply")
    if sink is not None:
        sink.save(MODELS / args.dump_labels, f"gnubg_h2h:{args.net}")


if __name__ == "__main__":
    main()
