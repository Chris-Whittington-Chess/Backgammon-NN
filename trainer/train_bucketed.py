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
from model import ValueNetBucketed, equity6, pip_bucket, N_BUCKETS, N_HEADS, net_bucketed_from_state
from train import roll, random_policy, hce_policy, play_match_game
from train_phase import outcome_classes, champion_policy

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def total_pips(board) -> int:
    return board.pip_count(0) + board.pip_count(1)


def _pip_route(board) -> int:
    """Default routing: the total-pip bucket (0..N_BUCKETS-1)."""
    return pip_bucket(total_pips(board))


def _class_route(board) -> int:
    """Class-aware routing (race/crashed/contact x pip sub-bucket, 0..N_HEADS-1),
    computed by the single-source-of-truth Rust selector."""
    return board.route_bucket()


@torch.no_grad()
def children_mover_equity(net, board, d1, d2, route=_pip_route):
    """Equity to the side to move for each legal move, each resulting position
    scored by *its* routed head. Returns (children, mover_eq, terminal_pts).

    Routing keys are perspective-invariant, so a child and its turn-passed form
    share a head — compute it from the child board, score the swapped features.
    """
    children = bgcore.legal_moves(board, d1, d2)
    term = [c.winner_points() for c in children]
    feats = bgcore.next_state_features(board, d1, d2)          # opponent-to-move
    buckets = torch.tensor([route(c) for c in children], dtype=torch.long)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    opp_eq = equity6(net.probs_for(x, buckets))               # opponent's equity
    mover_eq = -opp_eq
    for i, p in enumerate(term):
        if p is not None:
            mover_eq[i] = float(p)
    return children, mover_eq, term


def choose_next_bucketed(net, board, d1, d2, epsilon, rng, route=_pip_route):
    children, mover_eq, term = children_mover_equity(net, board, d1, d2, route)
    i = rng.randrange(len(children)) if rng.random() < epsilon else int(mover_eq.argmax())
    if term[i] is not None:
        return None, term[i]
    return children[i].swap_perspective(), None


def bucketed_policy(net, route=_pip_route):
    def f(board, d1, d2, rng):
        children, mover_eq, term = children_mover_equity(net, board, d1, d2, route)
        i = int(mover_eq.argmax())
        return (None, term[i]) if term[i] is not None else (children[i].swap_perspective(), None)
    return f


def play_game_bucketed(net, epsilon, rng, max_plies=4000, route=_pip_route):
    board = bgcore.Board.starting()
    feats, buckets = [], []
    for _ in range(max_plies):
        feats.append(board.features())
        buckets.append(route(board))
        d1, d2 = roll(rng)
        nb, pts = choose_next_bucketed(net, board, d1, d2, epsilon, rng, route)
        if pts is not None:
            return feats, buckets, pts
        board = nb
    return feats, buckets, 0


def train_iter_bucketed(net, opt, games, epsilon, rng, batch=1024, route=_pip_route, n_heads=N_BUCKETS):
    feats, buckets, cls = [], [], []
    plies = 0
    for _ in range(games):
        f_list, b_list, pts = play_game_bucketed(net, epsilon, rng, route=route)
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
    pop = torch.bincount(bk, minlength=n_heads).tolist()
    return plies, total / max(count, 1), pop


def bucketed_champion_policy(champion_file):
    """Greedy policy for a bucketed champion checkpoint (routes by its own head
    layout — inferred from the saved head width)."""
    ck = torch.load(MODELS_DIR / champion_file, map_location="cpu")
    if not ck.get("bucketed"):
        return champion_policy(champion_file)  # fall back to a 5-output champion
    net = net_bucketed_from_state(ck["model"], ck["hidden"], ck.get("act", "relu"))
    route = _class_route if net.n_heads == N_HEADS else _pip_route
    return bucketed_policy(net, route)


def benchmark_bucketed(net, opponent, games, rng, route=_pip_route):
    me = bucketed_policy(net, route)
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
    ap.add_argument("--save-every", type=int, default=25,
                    help="checkpoint the .pt every N iters (decoupled from benching)")
    ap.add_argument("--lr-decay", choices=["none", "linear"], default="none",
                    help="anneal the LR linearly from --lr to --lr-end by --lr-end-iter")
    ap.add_argument("--lr-end", type=float, default=1e-4, help="final LR for --lr-decay")
    ap.add_argument("--lr-end-iter", type=int, default=0,
                    help="absolute iter at which LR reaches --lr-end (0 = end of this run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="td_bucket.pt")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--champion", type=str, default=None,
                    help="net to head-to-head against each bench (5-output or bucketed)")
    ap.add_argument("--class-aware", action="store_true", dest="class_aware",
                    help="route by gnubg class (race/crashed/contact) x pip, 12 heads")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.hidden.split(",") if x.strip()]
    hidden = sizes[0] if len(sizes) == 1 else sizes
    MODELS_DIR.mkdir(exist_ok=True)

    route = _class_route if args.class_aware else _pip_route
    n_heads = N_HEADS if args.class_aware else N_BUCKETS
    kind = "class-aware (race/crashed/contact)" if args.class_aware else "total-pip"

    net = ValueNetBucketed(hidden, args.act, n_heads)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume:
        ck = torch.load(MODELS_DIR / args.resume, map_location="cpu")
        if (ck.get("hidden") != hidden or ck.get("act") != args.act
                or not ck.get("bucketed") or ck.get("n_buckets") != n_heads):
            raise SystemExit(
                f"--resume {args.resume} is hidden={ck.get('hidden')} act={ck.get('act')} "
                f"bucketed={ck.get('bucketed')} n_buckets={ck.get('n_buckets')}, but you asked "
                f"for hidden={hidden} act={args.act} bucketed=True n_buckets={n_heads}")
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_iter = int(ck.get("iter", 0))
        print(f"Resumed {args.resume} at iter {start_iter}")

    # Linear LR schedule. Anchored at this launch's resume point unless the
    # checkpoint already carries a schedule (so a re-resume keeps the same ramp
    # instead of snapping the LR back up to the start value).
    sched = None
    if args.lr_decay == "linear":
        inherit = args.resume and ck.get("lr_decay") == "linear"
        lr0 = ck.get("lr_start", args.lr) if inherit else args.lr
        lr1 = ck.get("lr_end", args.lr_end) if inherit else args.lr_end
        it0 = ck.get("lr_start_iter", start_iter) if inherit else start_iter
        it1 = (ck.get("lr_end_iter") if inherit
               else (args.lr_end_iter or start_iter + args.iters))
        sched = {"lr_start": lr0, "lr_end": lr1, "lr_start_iter": it0, "lr_end_iter": it1}
        print(f"Linear LR: {lr0:.1e} -> {lr1:.1e} over iters {it0}..{it1}")

    def lr_at(it):
        if sched is None:
            return args.lr
        frac = (it - sched["lr_start_iter"]) / max(1, sched["lr_end_iter"] - sched["lr_start_iter"])
        frac = min(max(frac, 0.0), 1.0)
        return sched["lr_start"] + frac * (sched["lr_end"] - sched["lr_start"])

    champ = bucketed_champion_policy(args.champion) if args.champion else None
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Bucketed TD(lambda=1) | hidden={hidden} act={args.act} | {n_heads} {kind} heads "
          f"| {n_params:,} params | lr={args.lr}"
          + (f" | vs champion {args.champion}" if champ else ""))

    for it in range(start_iter + 1, start_iter + args.iters + 1):
        cur_lr = lr_at(it)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        eps = max(0.02, 0.20 * (0.96 ** it))
        t0 = time.time()
        plies, loss, pop = train_iter_bucketed(net, opt, args.games, eps, rng, route=route, n_heads=n_heads)
        line = (f"iter {it:3d} | lr {cur_lr:.1e} | eps {eps:.3f} | plies {plies:5d} | loss {loss:.3f} "
                f"| heads {pop} | {time.time()-t0:4.1f}s")

        last = it == start_iter + args.iters
        if it % args.bench_every == 0 or last:
            wr_r, ppg_r = benchmark_bucketed(net, random_policy(), args.bench_games, rng, route)
            wr_h, ppg_h = benchmark_bucketed(net, hce_policy(), args.bench_games, rng, route)
            line += (f" || vs Random {100*wr_r:.1f}% | vs HCE {100*wr_h:.1f}% ({ppg_h:+.2f})")
            if champ is not None:
                wr_c, ppg_c = benchmark_bucketed(net, champ, args.bench_games, rng, route)
                line += f" | vs champ {100*wr_c:.1f}% ({ppg_c:+.2f})"
        if it % args.save_every == 0 or last:
            ckpt = {"model": net.state_dict(), "opt": opt.state_dict(), "hidden": hidden,
                    "act": args.act, "bucketed": True, "n_buckets": n_heads,
                    "class_aware": args.class_aware, "iter": it}
            if sched is not None:
                ckpt.update({"lr_decay": "linear", **sched})
            torch.save(ckpt, MODELS_DIR / args.out)
        print(line, flush=True)

    print(f"\nSaved bucketed checkpoint to {MODELS_DIR / args.out}")


if __name__ == "__main__":
    main()
