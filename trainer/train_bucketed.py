"""Self-play trainer for the pip-count output-bucketed net (Stockfish-NNUE style).

One shared body, 8 output heads (each the six-outcome softmax), selected by total
pip count. Every position trains the shared body; only the selected head learns
that position's outcome. Unlike the phase split (separate nets, each starved of
data), the expensive body sees everything.

Python-only, like train_phase.py: never touches the live net or the app. A net
that earns promotion gets its 6-output-per-bucket ONNX and a Rust bucket selector
(already wired: NnEval reads a 48-output net and slices by pip bucket).

Run:
  .venv/Scripts/python trainer/train_bucketed.py --hidden 256,128 --iters 200 --games 200
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bgcore
from model import ValueNetBucketed, equity6, pip_bucket, N_BUCKETS
from train import roll, random_policy, hce_policy, play_match_game
from train_phase import outcome_classes, champion_policy

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def total_pips(board) -> int:
    return board.pip_count(0) + board.pip_count(1)


@torch.no_grad()
def children_mover_equity(net, board, d1, d2):
    """Equity to the side to move for each legal move, each resulting position
    scored by *its* pip-count bucket. Returns (children, mover_eq, terminal_pts).

    Total pips are perspective-invariant, so a child and its turn-passed form
    share a bucket — compute it from the child board, score the swapped features.
    """
    children = bgcore.legal_moves(board, d1, d2)
    term = [c.winner_points() for c in children]
    feats = bgcore.next_state_features(board, d1, d2)          # opponent-to-move
    buckets = torch.tensor([pip_bucket(total_pips(c)) for c in children], dtype=torch.long)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    opp_eq = equity6(net.probs_for(x, buckets))               # opponent's equity
    mover_eq = -opp_eq
    for i, p in enumerate(term):
        if p is not None:
            mover_eq[i] = float(p)
    return children, mover_eq, term


def choose_next_bucketed(net, board, d1, d2, epsilon, rng):
    children, mover_eq, term = children_mover_equity(net, board, d1, d2)
    i = rng.randrange(len(children)) if rng.random() < epsilon else int(mover_eq.argmax())
    if term[i] is not None:
        return None, term[i]
    return children[i].swap_perspective(), None


def bucketed_policy(net):
    def f(board, d1, d2, rng):
        children, mover_eq, term = children_mover_equity(net, board, d1, d2)
        i = int(mover_eq.argmax())
        return (None, term[i]) if term[i] is not None else (children[i].swap_perspective(), None)
    return f


def play_game_bucketed(net, epsilon, rng, max_plies=4000):
    board = bgcore.Board.starting()
    feats, buckets = [], []
    for _ in range(max_plies):
        feats.append(board.features())
        buckets.append(pip_bucket(total_pips(board)))
        d1, d2 = roll(rng)
        nb, pts = choose_next_bucketed(net, board, d1, d2, epsilon, rng)
        if pts is not None:
            return feats, buckets, pts
        board = nb
    return feats, buckets, 0


def train_iter_bucketed(net, opt, games, epsilon, rng, batch=1024):
    feats, buckets, cls = [], [], []
    plies = 0
    for _ in range(games):
        f_list, b_list, pts = play_game_bucketed(net, epsilon, rng)
        if pts == 0:
            continue
        feats += f_list
        buckets += b_list
        cls += outcome_classes(len(f_list), pts)
        plies += len(f_list)

    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    bk = torch.tensor(buckets, dtype=torch.long)
    y = torch.tensor(cls, dtype=torch.long)
    total, count = 0.0, 0
    perm = torch.randperm(x.size(0))
    for i in range(0, x.size(0), batch):
        idx = perm[i : i + batch]
        opt.zero_grad()
        loss = F.cross_entropy(net.logits_for(x[idx], bk[idx]), y[idx])
        loss.backward()
        opt.step()
        total += loss.item() * idx.numel()
        count += idx.numel()
    pop = torch.bincount(bk, minlength=N_BUCKETS).tolist()
    return plies, total / max(count, 1), pop


def benchmark_bucketed(net, opponent, games, rng):
    me = bucketed_policy(net)
    wins, pts = 0, 0
    for g in range(games):
        seat = g % 2
        p0, p1 = (me, opponent) if seat == 0 else (opponent, me)
        w, p = play_match_game(p0, p1, rng)
        if w == seat:
            wins += 1
            pts += p
        else:
            pts -= p
    return wins / games, pts / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--act", choices=["relu", "sqrelu"], default="relu")
    ap.add_argument("--bench-every", type=int, default=5)
    ap.add_argument("--bench-games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="td_bucket.pt")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--champion", type=str, default=None,
                    help="5-output net to head-to-head against each bench")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.hidden.split(",") if x.strip()]
    hidden = sizes[0] if len(sizes) == 1 else sizes
    MODELS_DIR.mkdir(exist_ok=True)

    net = ValueNetBucketed(hidden, args.act)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume:
        ck = torch.load(MODELS_DIR / args.resume, map_location="cpu")
        if ck.get("hidden") != hidden or ck.get("act") != args.act or not ck.get("bucketed"):
            raise SystemExit(
                f"--resume {args.resume} is hidden={ck.get('hidden')} act={ck.get('act')} "
                f"bucketed={ck.get('bucketed')}, but you asked for hidden={hidden} "
                f"act={args.act} bucketed=True")
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_iter = int(ck.get("iter", 0))
        print(f"Resumed {args.resume} at iter {start_iter}")

    champ = champion_policy(args.champion) if args.champion else None
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Bucketed TD(lambda=1) | hidden={hidden} act={args.act} | {N_BUCKETS} pip buckets "
          f"| {n_params:,} params | lr={args.lr}"
          + (f" | vs champion {args.champion}" if champ else ""))

    for it in range(start_iter + 1, start_iter + args.iters + 1):
        eps = max(0.02, 0.20 * (0.96 ** it))
        t0 = time.time()
        plies, loss, pop = train_iter_bucketed(net, opt, args.games, eps, rng)
        line = (f"iter {it:3d} | eps {eps:.3f} | plies {plies:5d} | loss {loss:.3f} "
                f"| buckets {pop} | {time.time()-t0:4.1f}s")

        if it % args.bench_every == 0 or it == start_iter + args.iters:
            wr_r, ppg_r = benchmark_bucketed(net, random_policy(), args.bench_games, rng)
            wr_h, ppg_h = benchmark_bucketed(net, hce_policy(), args.bench_games, rng)
            line += (f" || vs Random {100*wr_r:.1f}% | vs HCE {100*wr_h:.1f}% ({ppg_h:+.2f})")
            if champ is not None:
                wr_c, ppg_c = benchmark_bucketed(net, champ, args.bench_games, rng)
                line += f" | vs champ {100*wr_c:.1f}% ({ppg_c:+.2f})"
            torch.save(
                {"model": net.state_dict(), "opt": opt.state_dict(), "hidden": hidden,
                 "act": args.act, "bucketed": True, "n_buckets": N_BUCKETS, "iter": it},
                MODELS_DIR / args.out,
            )
        print(line, flush=True)

    print(f"\nSaved bucketed checkpoint to {MODELS_DIR / args.out}")


if __name__ == "__main__":
    main()
