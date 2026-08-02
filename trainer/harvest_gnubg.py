"""Harvest gnubg-2-ply-labelled positions from games our net plays AGAINST gnubg.

Every label we have trained on so far came from champion SELF-PLAY positions. All
the scaling experiments therefore varied label *volume* while holding the
*distribution* fixed — which is a plausible reason 2.5M -> 17.5M flattened
(+0.114 -> +0.131 PPG vs the champion, direct h2h z +1.91, unresolved).

This changes the distribution instead: positions from games against a stronger
opponent, i.e. the states our net actually reaches when its weaknesses are being
punished. That is the DAgger argument — label the states the *student* visits,
using the *expert*.

The labels are nearly free. On gnubg's turn it already evaluates every legal
child at 2 ply to choose its move; `gnubg_h2h.py` parses one equity out of that
and discards the rest. Here we keep them: ~20 children per gnubg decision, ~30
such decisions per game, ~6 games/sec => ~3,600 labelled positions/sec, faster
than the dedicated labeller and in the distribution we actually want.

Two kinds of row are emitted, tagged in `kind`, because they are NOT the same
distribution and the training mix should be a decision rather than an accident:

  kind=1  TRAJECTORY — a state actually reached in play (~60 per game). The true
          DAgger target. `outcomes` is the real game result from that state's
          mover's perspective.
  kind=0  CHILD — a position gnubg evaluated while choosing (~600 per game).
          Far broader coverage, but weighted toward moves it considered,
          including ones nobody would play.

          WARNING: a child's `outcomes` entry is COUNTERFACTUAL — that line was
          not played, so the recorded game result did not follow from it. Only
          use child rows with `--alpha 1.0` (pure soft labels), which is what
          every run so far has used. Filter on `kind` if you need honest hard
          labels.

Run:
  .venv/Scripts/python trainer/harvest_gnubg.py --net net_17p5M_512x256.onnx \
      --games 2000 --workers 60 --out harvest_gnubg_2k.npz
"""
from __future__ import annotations

import argparse
import os
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np

import bgcore
from gen_rollout_data import dist6_from5
from label_gnubg import GnubgLabeller

MODELS = Path(__file__).resolve().parent.parent / "models"

# Points per outcome, matching Value::equity / OUTCOME_POINTS.
PTS5 = np.array([1.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float32)


def equity5(v):
    """Cubeless equity of a nested [win, win_g, win_bg, lose_g, lose_bg] row."""
    win, wg, wbg, lg, lbg = (float(x) for x in v)
    return (win - (1.0 - win)) + (wg - lg) + (wbg - lbg)


class Harvester:
    """Plays one game at a time, our net vs gnubg, recording everything."""

    def __init__(self, net, our_ply, plies, timeout):
        self.net = net
        self.our_ply = our_ply
        self.lab = GnubgLabeller(plies, timeout)

    def close(self):
        self.lab.close()

    def gnubg_move(self, children, rows):
        """gnubg's choice, capturing the 2-ply evaluation of every child it saw.

        Children are scored from the OPPONENT's on-roll view (their swapped id),
        so gnubg's pick is the minimum equity — and the vector we record belongs
        to that swapped position, which is the one we store.
        """
        terms = [c.winner_points() for c in children]
        wins = [i for i, t in enumerate(terms) if t is not None]
        if wins:
            return max(wins, key=lambda i: terms[i])
        swapped = [c.swap_perspective() for c in children]
        ids = [s.position_id() for s in swapped]
        vecs = self.lab.label(ids)                      # [n, 5], gnubg 2-ply
        for s, pid, v in zip(swapped, ids, vecs):
            rows.append((pid, dist6_from5(v), s.route_bucket()))
        return min(range(len(ids)), key=lambda i: equity5(vecs[i]))

    def our_move(self, board, children, d1, d2):
        if self.our_ply:
            return int(np.argmax(self.net.scores(board, d1, d2)))
        term = [c.winner_points() for c in children]
        eqs = [float(t) if t is not None else -self.net.equity(c.swap_perspective())
               for c, t in zip(children, term)]
        return max(range(len(children)), key=lambda i: eqs[i])

    def play(self, seed, ours_first):
        """One game. Returns (traj, child_rows, points_to_us).

        `traj` is [(pos_id, bucket, plies_from_start)] for every state reached;
        the caller signs the outcome per state once the result is known.
        """
        rng = random.Random(seed)
        board = bgcore.Board.starting()
        ours = ours_first
        traj, child_rows = [], []
        for ply in range(200):
            pid, bk = board.position_id(), board.route_bucket()
            # Label every trajectory state explicitly, whoever is on roll.
            #
            # The free child evaluations only cover states where WE are next to
            # move (they are the positions gnubg scored while choosing). But the
            # states our net actually EVALUATES are the children of its own
            # decisions — i.e. positions with gnubg on roll — and those are never
            # among them. Covering only one side would drop exactly the half the
            # student spends its capacity on. One eval per ply is ~10% on top of
            # the ~600 child evals per game, and it makes the trajectory complete
            # whether or not --children is set.
            traj.append((pid, bk, ply, dist6_from5(self.lab.label([pid])[0])))
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            children = bgcore.legal_moves(board, d1, d2)
            i = (self.our_move(board, children, d1, d2) if ours
                 else self.gnubg_move(children, child_rows))
            chosen = children[i]
            pts = chosen.winner_points()
            if pts is not None and pts > 0:
                return traj, child_rows, (pts if ours else -pts), ply
            board = chosen.swap_perspective()
            ours = not ours
        # Ply cap (rare crawling race): decide on pips so the API stays total.
        our_pip = board.pip_count(0 if ours else 1)
        opp_pip = board.pip_count(1 if ours else 0)
        return traj, child_rows, (1 if our_pip < opp_pip else -1), 200


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
    ap.add_argument("--net", default="net_17p5M_512x256.onnx",
                    help="OUR net — harvest the distribution of the net you intend to improve")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--our-ply", type=int, default=0,
                    help="our search depth while harvesting (0 = static, ~6 games/sec)")
    ap.add_argument("--our-candidates", type=int, default=4)
    ap.add_argument("--plies", type=int, default=2, help="gnubg's depth (its labels)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--children", action="store_true",
                    help="also emit the ~600 positions/game gnubg evaluated (kind=0). "
                         "Far more data, different distribution, counterfactual hard labels.")
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--save-every", type=int, default=100000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    our_cand = args.our_candidates if args.our_ply >= 2 else 0
    net = bgcore.Neural(str(MODELS / args.net), args.our_ply, our_cand)
    print(f"harvesting {args.games} games | our {args.net} at {args.our_ply}-ply vs gnubg "
          f"{args.plies}-ply | {args.workers} workers | children={args.children}", flush=True)

    jobs = queue.Queue()
    for g in range(args.games):
        jobs.put(g)
    pos_ids, probs, outcomes, buckets, kinds = [], [], [], [], []
    stats = {"games": 0, "traj": 0, "child": 0}
    lock = threading.Lock()
    t0 = time.time()
    out = MODELS / args.out

    def worker():
        h = Harvester(net, args.our_ply, args.plies, args.timeout)
        try:
            while True:
                try:
                    g = jobs.get_nowait()
                except queue.Empty:
                    break
                try:
                    traj, child_rows, pts, n_ply = h.play(args.seed + g, g % 2 == 0)
                except Exception:
                    h.close()
                    h = Harvester(net, args.our_ply, args.plies, args.timeout)
                    continue
                # A trajectory state's hard label is the game result from ITS
                # mover's view: the mover alternates every ply, so states an even
                # number of plies from the end share the final mover's result.
                tp, tb, tk, to, tv = [], [], [], [], []
                for pid, bk, ply, v6 in traj:
                    sign = 1 if (n_ply - ply) % 2 == 0 else -1
                    tp.append(pid); tb.append(bk); tk.append(1)
                    to.append(sign * pts if pts else 1); tv.append(v6)
                with lock:
                    # Trajectory rows carry no gnubg vector (they were not all
                    # evaluated); they exist to mark the distribution. Child rows
                    # carry the labels. Both are stored so the mix is explicit.
                    for pid, bk, k, o, v6 in zip(tp, tb, tk, to, tv):
                        pos_ids.append(pid); probs.append(v6)
                        buckets.append(bk); kinds.append(k); outcomes.append(o)
                    if args.children:
                        for pid, v6, bk in child_rows:
                            pos_ids.append(pid); probs.append(v6)
                            buckets.append(bk); kinds.append(0); outcomes.append(1)
                    stats["games"] += 1
                    stats["traj"] += len(tp)
                    stats["child"] += len(child_rows) if args.children else 0
        finally:
            h.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        with lock:
            g, tr, ch = stats["games"], stats["traj"], stats["child"]
        if g:
            dt = max(time.time() - t0, 1e-9)
            print(f"  {g:6d}/{args.games} games ({g/dt:.1f}/s) | traj {tr:,} | "
                  f"child {ch:,} | {(tr+ch)/dt:.0f} rows/sec", flush=True)
    for t in threads:
        t.join()

    # Trajectory rows have no label of their own; keep them only when they also
    # appear as a child (gnubg evaluated them), otherwise they are distribution
    # markers with no target. Emitting them unlabelled would poison training.
    labelled = {pid: v for pid, v in zip(pos_ids, probs) if v is not None}
    keep = [i for i, (pid, v) in enumerate(zip(pos_ids, probs))
            if v is not None or pid in labelled]
    fp = np.array([probs[i] if probs[i] is not None else labelled[pos_ids[i]]
                   for i in keep], dtype=np.float32)
    _atomic_savez(out,
                  pos_ids=np.array([pos_ids[i] for i in keep]),
                  probs=fp,
                  outcomes=np.array([outcomes[i] for i in keep], dtype=np.int8),
                  buckets=np.array([buckets[i] for i in keep], dtype=np.int8),
                  kind=np.array([kinds[i] for i in keep], dtype=np.int8),
                  teacher="gnubg", plies=args.plies, source=f"games-vs-gnubg:{args.net}")
    n_traj = sum(1 for i in keep if kinds[i] == 1)
    print(f"\nsaved {out} | {len(keep):,} labelled rows "
          f"({n_traj:,} trajectory, {len(keep)-n_traj:,} child) from {stats['games']} games")
    print(f"unlabelled trajectory states dropped: {stats['traj'] - n_traj:,}")


if __name__ == "__main__":
    main()
