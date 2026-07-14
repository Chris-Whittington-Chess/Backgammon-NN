"""Rollout-labeled fine-tuning (SPEC §8, hybrid) — the path to a stronger net.

Rollouts look several plies deeper than the raw 0-ply net and average over many
dice futures, so their outcome frequencies are far closer to truth. We use the
**full five-output distribution** they return — ``[win, win_g, win_bg, lose_g,
lose_bg]`` — as a supervised target: sample positions from self-play, label each
with a native parallel Rust rollout (``bgcore.Rollouts.dist``), and fine-tune the
net's probability head toward those frequencies with binary cross-entropy. This
teaches the net not just a better equity but a better *gammon/backgammon* sense,
which sharpens both cube decisions and play.

Then it proves the point: the fine-tuned net plays a head-to-head match against
the net it started from (and both against the HCE), 0-ply vs 0-ply, so any win
edge is pure evaluation quality.

Run: .venv/Scripts/python trainer/label_rollouts.py [--positions N] [--epochs E]
     add --promote to copy a stronger net to td_latest.pt and re-export td.onnx
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bgcore
from engine_api import NeuralEngine
from model import ValueNet, equity as net_equity
from train import benchmark, hce_policy, net_policy, play_match_game

MODELS = Path(__file__).resolve().parent.parent / "models"


def sample_boards(engine, n, rng, eps=0.15, race_frac=0.0):
    """Collect `n` non-terminal boards (side to move) from net self-play, with
    `eps` random moves mixed in for coverage. `race_frac` of the sample is drawn
    from pure-race positions (``no_contact``) — that is where gammon-saving lives
    and where the (now race-aware) rollout labels are trustworthy."""
    n_race = int(n * race_frac)
    races, others = [], []
    while len(races) < n_race or len(others) < n - n_race:
        board = bgcore.Board.starting()
        for _ in range(400):
            (races if board.no_contact() else others).append(board)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            if rng.random() < eps:
                children = bgcore.legal_moves(board, d1, d2)
                nb = children[rng.randrange(len(children))]
            else:
                nb = engine.analyze(board, d1, d2)[0][1]  # best result (pre-swap)
            if nb.winner_points() is not None:
                break
            board = nb.swap_perspective()
    return races[:n_race] + others[:n - n_race]


def self_play_gammon_rate(net, games, rng):
    """Fraction of greedy 0-ply self-play games that end in a gammon or
    backgammon — the direct measure of gammon-leaking (competent play ~0.25-0.35
    cubeless)."""
    play = net_policy(net)
    g_or_bg = 0
    for _ in range(games):
        board = bgcore.Board.starting()
        for _ in range(4000):
            nb, pts = play(board, rng.randint(1, 6), rng.randint(1, 6), rng)
            if pts is not None:
                g_or_bg += pts >= 2
                break
            board = nb
    return g_or_bg / games


def head_to_head(net_a, net_b, games, rng):
    """Fraction of `games` won by net_a, playing net_b 0-ply vs 0-ply, seats
    swapped every game."""
    pa, pb = net_policy(net_a), net_policy(net_b)
    a_wins = 0
    for g in range(games):
        seat = g % 2
        p0, p1 = (pa, pb) if seat == 0 else (pb, pa)
        winner, _pts = play_match_game(p0, p1, rng)
        a_wins += winner == seat
    return a_wins / games


def load_net(path):
    ck = torch.load(path, map_location="cpu")
    net = ValueNet(ck.get("hidden", 128))
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=1500)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--truncate", type=int, default=0, help="0 = roll to the end")
    ap.add_argument("--candidates", type=int, default=0)
    ap.add_argument("--race-frac", type=float, default=0.5,
                    help="fraction of training positions drawn from pure races")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bench-games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--promote", action="store_true",
                    help="if the new net gains points vs HCE without losing the "
                         "head-to-head, make it td_latest + td.onnx")
    args = ap.parse_args()

    if not hasattr(bgcore, "Rollouts"):
        raise SystemExit("Rollouts unavailable — rebuild bindings with the onnx feature.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    ckpt = MODELS / "td_latest.pt"
    net, ck = load_net(ckpt)
    base_net, _ = load_net(ckpt)  # frozen copy of the starting net for comparison

    policy = NeuralEngine(ckpt, lookahead=0)
    ro = bgcore.Rollouts(str(MODELS / "td.onnx"), args.trials, args.truncate,
                         args.candidates, args.seed, 0, 0)

    print(f"Sampling {args.positions} positions ({args.race_frac:.0%} pure races)…")
    boards = sample_boards(policy, args.positions, rng, race_frac=args.race_frac)

    kind = "full games" if args.truncate == 0 else f"{args.truncate}-ply trunc"
    print(f"Labelling with rollouts ({args.trials} trials, {kind})…")
    t0 = time.time()
    x = torch.from_numpy(np.array([b.features() for b in boards], dtype=np.float32))
    y = torch.from_numpy(np.array([ro.dist(b) for b in boards], dtype=np.float32))
    dt = time.time() - t0
    print(f"  labelled {len(boards)} positions in {dt:.1f}s ({1000 * dt / len(boards):.1f} ms/pos)")

    # Train / validation split.
    n_val = max(64, len(boards) // 10)
    perm = torch.randperm(len(boards))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    xt, yt, xv, yv = x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]

    def val_report():
        with torch.no_grad():
            p = net(xv)
            bce = F.binary_cross_entropy(p, yv).item()
            eqm = F.mse_loss(net_equity(p), net_equity(yv)).item()
        return bce, eqm

    b0, e0 = val_report()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_bce, best_state = b0, {k: v.clone() for k, v in net.state_dict().items()}
    for ep in range(1, args.epochs + 1):
        net.train()
        order = torch.randperm(xt.size(0))
        for i in range(0, xt.size(0), args.batch):
            idx = order[i:i + args.batch]
            opt.zero_grad()
            loss = F.binary_cross_entropy(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        bce, _ = val_report()
        if bce < best_bce:
            best_bce = bce
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 20 == 0 or ep == args.epochs:
            b, e = val_report()
            print(f"  epoch {ep:3d} | val BCE {b:.4f} | val equity-MSE {e:.4f}")
    net.load_state_dict(best_state)  # keep the best-validation net
    b1, e1 = val_report()
    print(f"validation:  BCE {b0:.4f} -> {b1:.4f}   equity-MSE {e0:.4f} -> {e1:.4f}")

    out = MODELS / "td_rollout.pt"
    torch.save({"model": net.state_dict(), "hidden": ck.get("hidden", 128),
                "iter": ck.get("iter")}, out)
    print(f"saved fine-tuned net to {out}")

    # --- The real test: does it leak fewer gammons and score more points? ---
    net.eval()
    print(f"\nSelf-play gammon+bg rate (lower = fewer gammons leaked):")
    gr_old = self_play_gammon_rate(base_net, args.bench_games, random.Random(args.seed + 3))
    gr_new = self_play_gammon_rate(net, args.bench_games, random.Random(args.seed + 3))
    print(f"  starting net:  {100 * gr_old:.1f}%     fine-tuned:  {100 * gr_new:.1f}%")

    print(f"\nHead-to-head & points ({args.bench_games} games, seats swapped):")
    wr = head_to_head(net, base_net, args.bench_games, random.Random(args.seed + 1))
    hce = hce_policy()
    wr_new, ppg_new = benchmark(net, hce, args.bench_games, random.Random(args.seed + 2))
    wr_old, ppg_old = benchmark(base_net, hce, args.bench_games, random.Random(args.seed + 2))
    print(f"  fine-tuned vs starting net:  {100 * wr:.1f}% wins")
    print(f"  fine-tuned vs HCE:  {100 * wr_new:.1f}%  PPG {ppg_new:+.3f}")
    print(f"  starting   vs HCE:  {100 * wr_old:.1f}%  PPG {ppg_old:+.3f}")

    # Stronger = more points per game vs the fixed HCE (gammons count), while not
    # losing the head-to-head. Points is the gammon-sensitive metric.
    stronger = ppg_new > ppg_old + 0.05 and wr >= 0.48
    verdict = "STRONGER" if stronger else "not clearly stronger"
    print(f"\nVerdict: fine-tuned net is {verdict} "
          f"(PPG {ppg_old:+.3f} -> {ppg_new:+.3f}, gammon rate "
          f"{100 * gr_old:.0f}% -> {100 * gr_new:.0f}%).")

    if args.promote and stronger:
        shutil.copyfile(out, MODELS / "td_latest.pt")
        print("promoted td_rollout.pt -> td_latest.pt; re-exporting td.onnx…")
        subprocess.run([sys.executable, str(Path(__file__).parent / "export_onnx.py"),
                        str(MODELS / "td_latest.pt")], check=True)
    elif args.promote:
        print("not promoting — no clear points gain.")


if __name__ == "__main__":
    main()
