"""TD(lambda) self-play training loop (SPEC §8, milestone M4).

A single value network learns purely from playing itself. Move selection is
0-ply negamax on the net's equity, with epsilon-greedy exploration. After each
game we compute forward-view lambda-returns along the trajectory (handling the
two-player perspective flip) and regress the net toward them.

Usage:
    .venv/Scripts/python trainer/train.py [--iters N] [--games G] [--lam L]
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
from model import ValueNet, equity, flip, outcome_vector

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def roll(rng: random.Random) -> tuple[int, int]:
    return rng.randint(1, 6), rng.randint(1, 6)


@torch.no_grad()
def eval_batch(net: ValueNet, feats: list[list[float]]) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    return net(x)


def choose_next(net, board, d1, d2, epsilon, rng, race_aware=True):
    """Pick a move by 0-ply negamax on net equity (epsilon-greedy).

    Returns (next_board_or_None, terminal_points_or_None). The next board is the
    resulting position with the turn passed (opponent to move); a non-None points
    value means the mover just won that many points and the game is over.

    Once the sides have passed (`race_aware` and no contact), the net is a weak,
    gammon-inflating guide, so play the correct race move (minimise remaining
    pips) instead. This keeps the late-game outcomes — and hence the lambda
    targets that flow back through the whole game — honest.
    """
    children = bgcore.legal_moves(board, d1, d2)  # opponent NOT yet to move
    term_pts = [c.winner_points() for c in children]

    if race_aware and board.no_contact():
        i = min(range(len(children)), key=lambda j: children[j].pip_count(0))
    else:
        # Value of each child from the opponent's perspective; mover wants to
        # minimise it, i.e. maximise (-opp_equity). Terminal wins score points.
        vals = eval_batch(net, bgcore.next_state_features(board, d1, d2))
        mover_eq = (-equity(vals)).clone()
        for i, p in enumerate(term_pts):
            if p is not None:
                mover_eq[i] = float(p)
        if rng.random() < epsilon:
            i = rng.randrange(len(children))
        else:
            i = int(torch.argmax(mover_eq).item())

    if term_pts[i] is not None:
        return None, term_pts[i]
    return children[i].swap_perspective(), None


def play_game(net, epsilon, rng, max_plies=4000):
    """Self-play one game; return (feature_list, terminal_points).

    `feature_list[t]` is the 198-vector of decision state s_t (mover's
    perspective). `terminal_points` is the result from the perspective of the
    mover of the last state (the side that made the winning move).
    """
    board = bgcore.Board.starting()
    feats = []
    for _ in range(max_plies):
        feats.append(board.features())
        d1, d2 = roll(rng)
        next_board, pts = choose_next(net, board, d1, d2, epsilon, rng)
        if pts is not None:
            return feats, pts
        board = next_board
    return feats, 0  # safety; discarded by caller


def lambda_targets(net, feats, pts, lam):
    """Forward-view lambda-returns as regression targets, one per state.

    G[n-1] = terminal outcome (mover of s_{n-1} won `pts`).
    G[t]   = flip( (1-lam) V(s_{t+1}) + lam G[t+1] ),  t < n-1.
    The flip carries the return from s_{t+1}'s mover back to s_t's mover.
    """
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    with torch.no_grad():
        v = net(x)  # [n, 5], each in its own state's mover perspective
    n = len(feats)
    g = [None] * n
    g[n - 1] = torch.tensor(outcome_vector(pts), dtype=torch.float32)
    for t in range(n - 2, -1, -1):
        g[t] = flip((1.0 - lam) * v[t + 1] + lam * g[t + 1])
    return x, torch.stack(g)


def train_iter(net, opt, games, epsilon, lam, rng, epochs=1, batch=1024):
    xs, ts, plies = [], [], 0
    for _ in range(games):
        feats, pts = play_game(net, epsilon, rng)
        if pts == 0:
            continue
        x, t = lambda_targets(net, feats, pts, lam)
        xs.append(x)
        ts.append(t)
        plies += len(feats)
    x_all = torch.cat(xs)
    t_all = torch.cat(ts)

    total, count = 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(x_all.size(0))
        for i in range(0, x_all.size(0), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = F.mse_loss(net(x_all[idx]), t_all[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
            count += idx.numel()
    return plies, total / max(count, 1)


# --- Benchmarking (SPEC §13): net vs a fixed opponent, both seats. ---


def net_policy(net):
    def f(board, d1, d2, rng):
        return choose_next(net, board, d1, d2, 0.0, rng)  # greedy, race-aware

    return f


def random_policy():
    def f(board, d1, d2, rng):
        children = bgcore.legal_moves(board, d1, d2)
        c = children[rng.randrange(len(children))]
        p = c.winner_points()
        return (None, p) if p is not None else (c.swap_perspective(), None)

    return f


def hce_policy():
    def f(board, d1, d2, rng):
        nb = bgcore.hce_move(board, d1, d2)
        p = nb.winner_points()
        return (None, p) if p is not None else (nb.swap_perspective(), None)

    return f


def play_match_game(p0, p1, rng, max_plies=4000):
    board = bgcore.Board.starting()
    on_roll = 0
    for _ in range(max_plies):
        d1, d2 = roll(rng)
        nb, pts = (p0 if on_roll == 0 else p1)(board, d1, d2, rng)
        if pts is not None:
            return on_roll, pts
        board = nb
        on_roll ^= 1
    return 0, 1


def benchmark(net, opponent, games, rng):
    """Net vs opponent over `games` games, split evenly between seats.
    Returns (win_rate, points_per_game) for the net."""
    net_p = net_policy(net)
    wins, net_points = 0, 0
    for g in range(games):
        net_seat = g % 2
        p0, p1 = (net_p, opponent) if net_seat == 0 else (opponent, net_p)
        winner, pts = play_match_game(p0, p1, rng)
        if winner == net_seat:
            wins += 1
            net_points += pts
        else:
            net_points -= pts
    return wins / games, net_points / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=20, help="self-play games per iter")
    ap.add_argument("--lam", type=float, default=1.0, help="1.0 = Monte-Carlo; <1 can collapse")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="128",
                    help="hidden layer sizes after the pool, e.g. 128 or 256,128; "
                         "empty for none")
    ap.add_argument("--act", choices=["relu", "sqrelu"], default="relu")
    ap.add_argument("--proj", type=int, default=0,
                    help="product-pool projection width (even, e.g. 512); 0 = plain MLP")
    ap.add_argument("--bench-every", type=int, default=5)
    ap.add_argument("--bench-games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="td_latest.pt",
                    help="checkpoint filename under models/ (keep the live net safe)")
    ap.add_argument("--resume", type=str, default=None,
                    help="continue from a checkpoint under models/ (keeps the "
                         "optimizer state and the exploration schedule)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.hidden.split(",") if x.strip()]
    hidden = sizes[0] if len(sizes) == 1 else sizes   # int / list / [] for none
    proj = args.proj or None
    MODELS_DIR.mkdir(exist_ok=True)

    start_iter = 0
    if args.resume:
        ck = torch.load(MODELS_DIR / args.resume, map_location="cpu")
        # Trust the checkpoint's shape over the flags: resuming into a different
        # architecture silently trains a different net.
        if (ck.get("hidden") != hidden or ck.get("act") != args.act
                or ck.get("proj") != proj):
            raise SystemExit(
                f"--resume {args.resume} is hidden={ck.get('hidden')} act={ck.get('act')} "
                f"proj={ck.get('proj')}, but you asked for hidden={hidden} act={args.act} "
                f"proj={proj}")
        net = ValueNet(hidden, args.act, proj)
        net.load_state_dict(ck["model"])
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])   # Adam's moments matter across a long run
        start_iter = int(ck.get("iter", 0))
        print(f"Resumed {args.resume} at iter {start_iter} "
              f"(optimizer state: {'restored' if 'opt' in ck else 'MISSING — fresh Adam'})")
    else:
        net = ValueNet(hidden, args.act, proj)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"TD(lambda={args.lam}) self-play | hidden={hidden} act={args.act} "
          f"proj={proj} | {n_params:,} params | lr={args.lr}")
    if not args.resume:
        print("Baseline (untrained net):")
        wr, ppg = benchmark(net, random_policy(), args.bench_games, rng)
        print(f"  vs Random: win {100*wr:.1f}%  PPG {ppg:+.3f}")

    for it in range(start_iter + 1, start_iter + args.iters + 1):
        # Decay on the *global* iteration: restarting the schedule every chunk
        # would keep re-injecting 20% random moves into a converged net.
        eps = max(0.02, 0.20 * (0.96 ** it))
        t0 = time.time()
        plies, loss = train_iter(net, opt, args.games, eps, args.lam, rng)
        dt = time.time() - t0
        line = f"iter {it:3d} | eps {eps:.3f} | plies {plies:5d} | loss {loss:.4f} | {dt:4.1f}s"

        if it % args.bench_every == 0 or it == start_iter + args.iters:
            wr_r, ppg_r = benchmark(net, random_policy(), args.bench_games, rng)
            wr_h, ppg_h = benchmark(net, hce_policy(), args.bench_games, rng)
            line += (
                f" || vs Random win {100*wr_r:.1f}% PPG {ppg_r:+.2f}"
                f" | vs HCE win {100*wr_h:.1f}% PPG {ppg_h:+.2f}"
            )
            torch.save(
                {"model": net.state_dict(), "opt": opt.state_dict(),
                 "hidden": hidden, "act": args.act, "proj": proj, "iter": it},
                MODELS_DIR / args.out,
            )
        print(line, flush=True)

    print(f"\nSaved checkpoint to {MODELS_DIR / args.out}")


if __name__ == "__main__":
    main()
