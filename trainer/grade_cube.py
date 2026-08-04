"""Grade our cube decisions against gnubg's, one decision at a time.

Everything in `cube.py` and `match.py` has only ever been checked against our own
tests. The cube has never faced an opponent: `gnubg_h2h.py` is cubeless money
play, so the doubling logic is the largest untested surface in the project.

The obvious test — play matches against gnubg — is a bad instrument. A 7-point
match is ~15 games of large variance and contains only a handful of cube
decisions, so it takes thousands of matches to see anything, and the result still
confounds cube errors with checker errors. Instead this asks gnubg to analyse the
cube decision at a position directly::

    Cube analysis
    2-ply cubeless equity +0.012 (Money: +0.076)
      0.525 0.149 0.007 - 0.475 0.124 0.005
    Cubeful equities:
    1. No double           +0.227
    2. Double, pass        +1.000  (+0.773)
    3. Double, take        +0.009  (-0.218)
    Proper cube action: No double, take (22.0%)

which prices every alternative, so a disagreement costs a measurable number of
millipoints rather than an unknown fraction of a match. It also works at any
score, and gnubg answers ~53 positions/sec/process.

We feed our model **gnubg's own probabilities**, not our net's. That is deliberate:
it isolates the cube model from the net's evaluation error, which is what we want
when fitting the cube. Grading the shipped stack end to end is a separate run.

Two phases, because gnubg's answer does not depend on our efficiency parameter:

    collect -> run gnubg once, cache (probabilities, equities, verdict)
    score   -> replay the cache against our model at any `x`, instantly

so sweeping `x` costs no gnubg time at all.

    .venv/Scripts/python trainer/grade_cube.py collect --mode money --n 20000
    .venv/Scripts/python trainer/grade_cube.py score  --records cube_money.npz
    .venv/Scripts/python trainer/grade_cube.py sweep  --records cube_money.npz

Limitation: the cube is centred in every position, so redouble decisions (cube
owned) are not covered.
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cube as cubemod
import match as matchmod

GNUBG = os.environ.get("GNUBG_CLI", r"C:/Users/chris/AppData/Local/gnubg/gnubg-cli.exe")
MODELS = Path(__file__).resolve().parent.parent / "models"

# `show cube` after every `hint` gives a terminator that is present even when the
# hint produced nothing. Without it an "Illegal position." (gnubg keeps the old
# board and re-analyses it) shifts every later record by one, silently.
END = "The cube is at"
ILLEGAL = "Illegal position"

PROBS = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+-\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
EQUITY = re.compile(r"^\s*\d+\.\s+(.+?)\s{2,}([+-][\d.]+)")
VERDICT = re.compile(r"Proper cube action:\s*(.+?)\s*$")

# Away-score pairs to sample in match mode. `1` on either side means the Crawford
# game is gone and the cube is back — post-Crawford, the case the equity table was
# split for. Symmetric pairs are included in both directions because the cube
# decision is not symmetric: leading 2-away is very different from trailing it.
# The mover being 1-away is deliberately absent: it wins the match with a single
# point at the current cube, so there is no cube decision and gnubg declines every
# such position. Including them cost 16% of each collection for nothing.
MATCH_SCORES = [(2, 2), (2, 4), (4, 2), (3, 5), (5, 3), (2, 6), (6, 2),
                (4, 4), (7, 7), (5, 7), (7, 5), (3, 3), (6, 6),
                (2, 1), (3, 1), (4, 1)]
MATCH_LEN = 7


class CubeGrader:
    """Persistent gnubg-cli process returning one cube analysis per position."""

    def __init__(self, mode: str, timeout: float = 60.0):
        self.mode = mode
        self.timeout = timeout
        self._start()

    def _start(self):
        self.p = subprocess.Popen(
            [GNUBG, "-t", "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        setup = ["set output mwc off",       # equities in EMG, so errors are mEMG
                 "set player 0 human", "set player 1 human"]
        if self.mode == "money":
            setup += ["new session", "new game"]
        else:
            setup += [f"new match {MATCH_LEN}"]
        self._send(setup)

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put("")

    def _send(self, cmds):
        self.p.stdin.write("".join(c + "\n" for c in cmds))
        self.p.stdin.flush()

    def close(self):
        try:
            self._send(["quit"])
        except Exception:
            pass
        self.p.terminate()

    def analyse(self, items):
        """items: (pos_id, away_a, away_b). Returns one dict per item, in order.

        A dict with ``ok=False`` means gnubg declined that position; the caller
        drops it. Never silently reuses a previous analysis.
        """
        cmds = []
        for pid, a, b in items:
            if self.mode == "match":
                cmds.append(f"set score {MATCH_LEN - a} {MATCH_LEN - b} {MATCH_LEN}")
                # Always off: a Crawford game has no cube, so there is no decision
                # to grade. A 1-away side therefore means post-Crawford — the case
                # the equity table was split for.
                cmds.append("set crawford off")
            cmds += [f"set board {pid}", "set turn 0", "hint", "show cube"]
        # `show cube` prints no trailing newline, so its line only completes when
        # something else prints. Without a flush the LAST item never terminates
        # and the batch hangs until the timeout.
        cmds.append("show board")
        self._send(cmds)

        out = []
        cur = self._blank()
        while len(out) < len(items):
            try:
                line = self.q.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError(f"gnubg timed out after {len(out)}/{len(items)}")
            if line == "":
                raise RuntimeError("gnubg exited unexpectedly")
            # Every pattern is tested against every line, with no short-circuit:
            # because of the missing newline the terminator arrives fused to
            # whatever printed next, so a `continue` on an earlier match would
            # swallow it and shift every subsequent record by one.
            if ILLEGAL in line:
                cur["illegal"] = True
            m = PROBS.match(line)
            if m and cur["dist"] is None:
                w, wg, wbg, _l, lg, lbg = (float(x) for x in m.groups())
                cur["dist"] = [w, wg, wbg, lg, lbg]
            m = EQUITY.match(line)
            if m:
                cur["eq"][m.group(1).strip()] = float(m.group(2))
            m = VERDICT.search(line)
            if m:
                cur["verdict"] = m.group(1)
            if END in line:
                cur["ok"] = (not cur["illegal"] and cur["dist"] is not None
                             and cur["verdict"] is not None
                             and len(cur["eq"]) >= 3)
                out.append(cur)
                cur = self._blank()
        return out

    @staticmethod
    def _blank():
        return {"dist": None, "eq": {}, "verdict": None, "illegal": False, "ok": False}


def _eq(rec_eq: dict, *names):
    """Pick the first present label — gnubg says 'Double' with a centred cube and
    'Redouble' when it is owned, and 'Too good to double' replaces 'No double'."""
    for n in names:
        if n in rec_eq:
            return rec_eq[n]
    return None


def collect(args):
    src = np.load(args.source)
    pos = src["pos_ids"]
    if "kind" in src.files and not args.all_kinds:
        pos = pos[src["kind"] == 1]     # trajectory states: positions real games reached
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(pos), size=min(args.n, len(pos)), replace=False)
    pos = pos[idx]

    if args.mode == "money":
        items = [(str(p), 0, 0) for p in pos]
    else:
        items = [(str(p), *MATCH_SCORES[i % len(MATCH_SCORES)])
                 for i, p in enumerate(pos)]

    print(f"grading {len(items)} {args.mode} cube decisions with gnubg "
          f"({args.workers} workers)", flush=True)

    chunks = [items[i::args.workers] for i in range(args.workers)]
    results: list = [None] * args.workers

    errors: list = []

    def run(w):
        g = CubeGrader(args.mode)
        try:
            got = []
            for i in range(0, len(chunks[w]), 200):
                got += g.analyse(chunks[w][i:i + 200])
                if w == 0:
                    print(f"  worker0 {len(got)}/{len(chunks[w])}", end="\r", flush=True)
            results[w] = got
        except Exception as e:
            # A worker dying used to show up only as a low graded count, which
            # reads like gnubg being fussy rather than the harness being broken.
            errors.append(f"worker{w}: {type(e).__name__}: {e}")
        finally:
            g.close()

    threads = [threading.Thread(target=run, args=(w,)) for w in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise SystemExit("\n".join(["gnubg workers failed:"] + errors))

    rows = []
    for w in range(args.workers):
        for item, r in zip(chunks[w], results[w] or []):
            if not r["ok"]:
                continue
            nd = _eq(r["eq"], "No double", "Too good to double")
            dp = _eq(r["eq"], "Double, pass", "Redouble, pass")
            dt = _eq(r["eq"], "Double, take", "Redouble, take")
            if None in (nd, dp, dt):
                continue
            rows.append((item[0], item[1], item[2], r["dist"], nd, dp, dt, r["verdict"]))

    n_fail = len(items) - len(rows)
    out = MODELS / (args.out or f"cube_{args.mode}.npz")
    np.savez_compressed(
        out,
        pos_ids=np.array([r[0] for r in rows], dtype="<U14"),
        away_a=np.array([r[1] for r in rows], dtype=np.int8),
        away_b=np.array([r[2] for r in rows], dtype=np.int8),
        dist=np.array([r[3] for r in rows], dtype=np.float32),
        eq_nodouble=np.array([r[4] for r in rows], dtype=np.float32),
        eq_double_pass=np.array([r[5] for r in rows], dtype=np.float32),
        eq_double_take=np.array([r[6] for r in rows], dtype=np.float32),
        verdict=np.array([r[7] for r in rows], dtype="<U48"),
        mode=args.mode, match_len=MATCH_LEN)
    print(f"\nsaved {out} | {len(rows)} graded, {n_fail} declined by gnubg")


def our_decision(dist, a, b, x):
    """(we_double, we_take) from our model, using gnubg's probabilities."""
    if a == 0:
        d = cubemod.cube_action(dist, owner=cubemod.CENTER, x=x)
        return d.action.startswith("double"), d.take
    d = matchmod.match_cube_action(dist, a, b, cube=1, owner=cubemod.CENTER,
                                   post_crawford=(1 in (a, b)), x=x)
    return d.action.startswith("double"), d.take


def evaluate(rec, x):
    """Error in EMG for the doubler's decision and for the taker's response."""
    dist, a, b = rec["dist"], rec["away_a"], rec["away_b"]
    nd, dp, dt = rec["eq_nodouble"], rec["eq_double_pass"], rec["eq_double_take"]
    n = len(dist)
    dbl_err = np.zeros(n)
    take_err = np.zeros(n)
    dbl_agree = np.zeros(n, dtype=bool)
    take_agree = np.zeros(n, dtype=bool)
    for i in range(n):
        we_double, we_take = our_decision(list(dist[i]), int(a[i]), int(b[i]), x)
        # The opponent answers a double with whichever is worse for the doubler.
        double_value = min(dp[i], dt[i])
        best = max(nd[i], double_value)
        ours = double_value if we_double else nd[i]
        dbl_err[i] = best - ours
        dbl_agree[i] = (double_value > nd[i]) == we_double
        # The response is graded on its own: taking is right when it holds the
        # doubler below what a pass banks.
        right_take = dt[i] < dp[i]
        take_agree[i] = right_take == we_take
        take_err[i] = 0.0 if take_agree[i] else abs(dp[i] - dt[i])
    return dbl_err, take_err, dbl_agree, take_agree


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def default_x(rec) -> float:
    """What the shipped code uses for these records — the two modes are fitted
    separately and do not share a value."""
    return (cubemod.DEFAULT_EFFICIENCY if str(rec["mode"]) == "money"
            else matchmod.DEFAULT_EFFICIENCY_MATCH)


def score(args):
    rec = load(args.records)
    args.x = default_x(rec) if args.x is None else args.x
    dbl_err, take_err, dbl_ok, take_ok = evaluate(rec, args.x)
    n = len(dbl_err)
    print(f"{rec['mode']} | {n} decisions | efficiency x = {args.x}")
    print(f"  double/no-double : {dbl_ok.mean()*100:5.1f}% agree | "
          f"mean error {dbl_err.mean()*1000:6.2f} mEMG")
    print(f"  take/pass        : {take_ok.mean()*100:5.1f}% agree | "
          f"mean error {take_err.mean()*1000:6.2f} mEMG")
    blunders = dbl_err > 0.08
    print(f"  blunders (>80 mEMG on the double): {blunders.sum()} "
          f"({blunders.mean()*100:.2f}%)")
    if args.worst:
        order = np.argsort(-dbl_err)[:args.worst]
        print("\n  worst double decisions:")
        for i in order:
            a, b = int(rec["away_a"][i]), int(rec["away_b"][i])
            sc = "money" if a == 0 else f"{a}-away/{b}-away"
            we, _ = our_decision(list(rec["dist"][i]), a, b, args.x)
            print(f"    {rec['pos_ids'][i]} {sc:16} we {'double ' if we else 'hold   '}"
                  f"| gnubg: {rec['verdict'][i]:28} | -{dbl_err[i]*1000:.0f} mEMG")


def sweep(args):
    rec = load(args.records)
    print(f"{rec['mode']} | {len(rec['dist'])} decisions | sweeping efficiency x")
    print(f"  {'x':>6}  {'double agree':>12}  {'mEMG':>8}")
    best = None
    for x in np.arange(args.lo, args.hi + 1e-9, args.step):
        dbl_err, _te, dbl_ok, _tk = evaluate(rec, float(x))
        m = dbl_err.mean() * 1000
        print(f"  {x:6.3f}  {dbl_ok.mean()*100:11.1f}%  {m:8.2f}")
        if best is None or m < best[1]:
            best = (float(x), m)
    print(f"\nbest x = {best[0]:.3f} at {best[1]:.2f} mEMG "
          f"(current default {cubemod.DEFAULT_EFFICIENCY})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="run gnubg and cache its cube analyses")
    c.add_argument("--source", default=str(MODELS / "harvest_gnubg_10k.npz"))
    c.add_argument("--mode", choices=["money", "match"], default="money")
    c.add_argument("--n", type=int, default=20000)
    c.add_argument("--workers", type=int, default=60,
                   help="gnubg processes (Windows tops out near 60)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--all-kinds", action="store_true",
                   help="include child positions, not just trajectory states")
    c.add_argument("--out")
    c.set_defaults(fn=collect)

    s = sub.add_parser("score", help="grade our model against a cached run")
    s.add_argument("--records", required=True)
    s.add_argument("--x", type=float, default=None,
                   help="override; default is whatever the shipped code uses")
    s.add_argument("--worst", type=int, default=0)
    s.set_defaults(fn=score)

    w = sub.add_parser("sweep", help="find the efficiency that minimises the error")
    w.add_argument("--records", required=True)
    w.add_argument("--lo", type=float, default=0.40)
    w.add_argument("--hi", type=float, default=0.95)
    w.add_argument("--step", type=float, default=0.025)
    w.set_defaults(fn=sweep)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
