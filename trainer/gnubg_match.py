"""Full matches against gnubg, with the cube — the end-to-end test.

`gnubg_h2h.py` measures checker play in cubeless money games. This plays actual
matches: score, doubling cube, Crawford, both sides free to double and to take.
It is the only harness that exercises `cube.py` and `match.py` against an
opponent rather than against our own tests.

It is deliberately NOT the instrument for measuring cube quality — a match is ~15
games of large variance holding a handful of cube decisions, so `grade_cube.py`
(which prices every decision individually against gnubg) will always see a small
change sooner. This answers a different question: does the whole thing hold
together and win matches.

We own the game state; gnubg is consulted for its decisions only, exactly as in
`gnubg_h2h.py`:

  - **checker play** — our engine generates the legal moves and gnubg picks by
    evaluating each resulting position (`set board` / `eval`).
  - **the cube** — `set score` / `set cube` / `hint`, reading gnubg's "Proper cube
    action". The same line also carries its take/pass advice, which is what it
    plays when *we* double.

Two gnubg processes per worker, not one: `hint` prints a `2-ply cubeless equity`
line that the move parser's own regex matches, so sharing a process would let a
cube analysis desync a move read.

Run: .venv/Scripts/python trainer/gnubg_match.py --matches 200 --length 7
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import random
import re
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import bgcore

import cube as cubemod
import match as matchmod
from gnubg_h2h import GnubgEngine, our_best, our_best_searched

GNUBG = os.environ.get("GNUBG_CLI", r"C:/Users/chris/AppData/Local/gnubg/gnubg-cli.exe")
MODELS = Path(__file__).resolve().parent.parent / "models"

VERDICT = re.compile(r"Proper cube action:\s*(.+?)\s*$")
CUBE_END = "The cube is at"
NO_DOUBLE = "You cannot double"


class GnubgCube:
    """gnubg's cube decision for a position at a match score.

    Returns ``(doubles, takes)``: whether gnubg would double here, and what it
    advises as the response — which is what it does when we double, since its
    "Proper cube action" line carries both halves ("Double, take").
    """

    def __init__(self, match_len: int, timeout: float = 60.0):
        self.match_len = match_len
        self.timeout = timeout
        self.p = subprocess.Popen(
            [GNUBG, "-t", "-q"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._send(["set output mwc off", "set player 0 human", "set player 1 human",
                    f"new match {match_len}"])

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

    def action(self, pos_id: str, mover_away: int, opp_away: int, cube: int,
               owner: int, crawford: bool) -> tuple[bool, bool]:
        L = self.match_len
        owner_cmd = {cubemod.CENTER: "centre", cubemod.MOVER: "0", cubemod.OPP: "1"}[owner]
        # Order matters: `set turn` last, because it is what puts the position in
        # the pre-roll state where a cube decision exists at all.
        self._send([f"set score {L - mover_away} {L - opp_away} {L}",
                    "set crawford " + ("on" if crawford else "off"),
                    f"set cube value {cube}", f"set cube owner {owner_cmd}",
                    f"set board {pos_id}", "set turn 0", "hint", "show cube",
                    # `show cube` prints no trailing newline, so its line only
                    # completes once something else prints — without this the read
                    # blocks until the timeout every single time.
                    "show board"])
        verdict = None
        refused = False
        while True:
            try:
                line = self.q.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError("gnubg cube hint timed out")
            if line == "":
                raise RuntimeError("gnubg cube process closed")
            if NO_DOUBLE in line:
                refused = True
            m = VERDICT.search(line)
            if m:
                verdict = m.group(1)
            if CUBE_END in line:
                break
        if refused or verdict is None:
            # gnubg refuses a cube that is already worth the match, and gives no
            # verdict for a Crawford game. Either way there is nothing to do.
            return False, True
        v = verdict.lower()
        doubles = v.startswith("double") or v.startswith("redouble")
        takes = ", take" in v or ", beaver" in v
        return doubles, takes


class MatchState:
    """Score, cube and Crawford bookkeeping for one match, from our seat."""

    def __init__(self, length: int):
        self.length = length
        self.score = [0, 0]          # [ours, theirs]
        self.crawford_used = False
        self.next_crawford = False

    def away(self, ours: bool) -> int:
        i = 0 if ours else 1
        return max(1, self.length - self.score[i])

    def over(self) -> bool:
        return max(self.score) >= self.length

    def record(self, points: int, ours_won: bool, was_crawford: bool):
        self.score[0 if ours_won else 1] += points
        if was_crawford:
            self.crawford_used = True
        if not self.crawford_used and max(self.score) == self.length - 1:
            self.next_crawford = True


def our_cube_decision(net, board, state: MatchState, ours_to_move: bool,
                      cube_value: int, cube_owner, crawford: bool):
    """Our model's decision for the side on roll, in that side's frame."""
    owner = (cubemod.CENTER if cube_owner is None else
             cubemod.MOVER if cube_owner == ours_to_move else cubemod.OPP)
    return matchmod.match_cube_action(
        net.dist(board), state.away(ours_to_move), state.away(not ours_to_move),
        cube_value, owner, crawford=crawford,
        post_crawford=state.crawford_used)


def play_game(net, gnu_move, gnu_cube, state: MatchState, rng, ours_first: bool,
              crawford: bool, our_ply: int, stats: Counter):
    """One game. Returns (points, ours_won)."""
    board = bgcore.Board.starting()
    ours_to_move = ours_first
    cube_value = 1
    cube_owner = None            # None = centred, True = ours, False = theirs

    for _ in range(400):
        may_double = (not crawford and cube_value < 64
                      and cube_owner in (None, ours_to_move))
        if may_double:
            pid = board.position_id()
            if ours_to_move:
                d = our_cube_decision(net, board, state, True, cube_value,
                                      cube_owner, crawford)
                doubles = d.action.startswith("double")
            else:
                doubles, _ = gnu_cube.action(
                    pid, state.away(False), state.away(True), cube_value,
                    cubemod.CENTER if cube_owner is None else cubemod.MOVER,
                    crawford)
            if doubles:
                stats["doubles_ours" if ours_to_move else "doubles_theirs"] += 1
                if ours_to_move:
                    # gnubg answers. Its verdict at this position carries the
                    # take advice it would follow.
                    _, takes = gnu_cube.action(
                        pid, state.away(True), state.away(False), cube_value,
                        cubemod.CENTER if cube_owner is None else cubemod.MOVER,
                        crawford)
                else:
                    # Our response, evaluated in the doubler's frame: `take` is
                    # exactly the opponent's (our) decision.
                    d = our_cube_decision(net, board, state, False, cube_value,
                                          cube_owner, crawford)
                    takes = d.take
                if not takes:
                    stats["passes_ours" if not ours_to_move else "passes_theirs"] += 1
                    return cube_value, ours_to_move
                stats["takes"] += 1
                cube_value *= 2
                cube_owner = not ours_to_move

        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        children = bgcore.legal_moves(board, d1, d2)
        if ours_to_move:
            i = (our_best_searched(net, board, d1, d2) if our_ply
                 else our_best(net, children))
        else:
            i = gnu_move.best_child(children)
        chosen = children[i]
        pts = chosen.winner_points()
        if pts is not None and pts > 0:
            return pts * cube_value, ours_to_move
        board = chosen.swap_perspective()
        ours_to_move = not ours_to_move

    # Ply cap: resolve a crawling race by pip count, as gnubg_h2h does.
    our_pip = board.pip_count(0 if ours_to_move else 1)
    opp_pip = board.pip_count(1 if ours_to_move else 0)
    return cube_value, (our_pip < opp_pip) == ours_to_move


def play_match(net, gnu_move, gnu_cube, length, seed, ours_first, our_ply, stats):
    """One match. Returns True if we won it."""
    rng = random.Random(seed)
    state = MatchState(length)
    first = ours_first
    while not state.over():
        crawford = state.next_crawford
        state.next_crawford = False
        pts, ours_won = play_game(net, gnu_move, gnu_cube, state, rng, first,
                                  crawford, our_ply, stats)
        state.record(pts, ours_won, crawford)
        stats["games"] += 1
        first = not first          # alternate the seat that opens
    return state.score[0] > state.score[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="td.onnx")
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--length", type=int, default=7)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--plies", type=int, default=2, help="gnubg's search depth")
    ap.add_argument("--our-ply", type=int, default=2, help="our net's search depth")
    args = ap.parse_args()

    net = bgcore.Neural(str(MODELS / args.net), args.our_ply, 4)
    print(f"MATCH PLAY: our net {args.net} ({args.our_ply}-ply) vs gnubg "
          f"{args.plies}-ply | {args.matches} matches to {args.length} | "
          f"cube live, Crawford on | {args.workers} workers\n", flush=True)

    jobs: queue.Queue = queue.Queue()
    for m in range(args.matches):
        jobs.put(m)
    results: list[bool] = []
    skipped: list[int] = []
    stats: Counter = Counter()
    reasons: Counter = Counter()
    lock = threading.Lock()
    t0 = time.time()

    def worker():
        gm = GnubgEngine(args.plies)
        gc_ = GnubgCube(args.length)
        try:
            while True:
                try:
                    m = jobs.get_nowait()
                except queue.Empty:
                    break
                local: Counter = Counter()
                try:
                    won = play_match(net, gm, gc_, args.length, 5000 + m // 2,
                                     m % 2 == 0, args.our_ply, local)
                except Exception as e:
                    # A hung gnubg poisons only this match; restart both processes.
                    # Keep the reason: a silently discarded match reads as gnubg
                    # being fussy when it is usually the harness being wrong.
                    reasons[f"{type(e).__name__}: {e}"] += 1
                    for proc in (gm, gc_):
                        try:
                            proc.close()
                        except Exception:
                            pass
                    gm = GnubgEngine(args.plies)
                    gc_ = GnubgCube(args.length)
                    with lock:
                        skipped.append(m)
                    continue
                with lock:
                    results.append(won)
                    stats.update(local)
        finally:
            for proc in (gm, gc_):
                try:
                    proc.close()
                except Exception:
                    pass

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(20)
        with lock:
            n, w = len(results), sum(results)
        if n:
            print(f"  {n:4d}/{args.matches} matches "
                  f"({n/max(time.time()-t0,1)*60:.1f}/min): our match win "
                  f"{100*w/n:.1f}%", flush=True)
    for t in threads:
        t.join()

    n = len(results)
    if skipped:
        print(f"\nWARNING: {len(skipped)} of {args.matches} matches DISCARDED "
              f"(gnubg hung/desynced). Survivors are not a random subset.")
    if not n:
        raise SystemExit("no matches completed")
    wr = sum(results) / n
    z = (wr - 0.5) / math.sqrt(0.25 / n)
    print(f"\nOUR net wins {100*wr:.1f}% of matches  (z = {z:+.2f})  "
          f"| {n} matches, {stats['games']} games, {time.time()-t0:.0f}s")
    print(f"  cube: we doubled {stats['doubles_ours']}, gnubg doubled "
          f"{stats['doubles_theirs']}, takes {stats['takes']}, "
          f"we passed {stats['passes_ours']}, gnubg passed {stats['passes_theirs']}")
    verdict = ("WE ARE STRONGER" if z > 1.96 else
               "GNUBG STRONGER" if z < -1.96 else "TOO CLOSE TO CALL")
    print(f"=> {verdict} over {args.length}-point matches")


if __name__ == "__main__":
    main()
