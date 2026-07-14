"""Rollout-labeled fine-tuning (SPEC §8, hybrid) — the path to a stronger net.

Rollout equities are far closer to truth than the raw 0-ply net, so we use them
as targets: sample positions from self-play, label each with a native Rust
rollout (fast, parallel), and fine-tune the net so its equity matches. This
regresses the net's *equity* (a scalar derived from the 5 outputs) toward the
rollout equity — a simple, effective first cut. (Labelling the full
win/gammon/backgammon distribution would need the rollout to return outcome
frequencies; a natural next step.)

Run: .venv/Scripts/python trainer/label_rollouts.py [--positions N] [--epochs E]
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
from engine_api import NeuralEngine
from model import ValueNet, equity as net_equity

MODELS = Path(__file__).resolve().parent.parent / "models"


def sample_boards(policy, n, rng):
    """Collect `n` non-terminal boards (side to move) from net self-play."""
    out = []
    while len(out) < n:
        board = bgcore.Board.starting()
        for _ in range(300):
            out.append(board)
            if len(out) >= n:
                break
            nb, pts, _steps, _eq = policy.choose(board, rng.randint(1, 6), rng.randint(1, 6))
            if pts is not None:
                break
            board = nb
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--truncate", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not hasattr(bgcore, "Rollouts"):
        raise SystemExit("Rollouts unavailable — rebuild bindings with --features onnx.")

    rng = random.Random(args.seed)
    ckpt = MODELS / "td_latest.pt"
    ck = torch.load(ckpt, map_location="cpu")
    net = ValueNet(ck.get("hidden", 128))
    net.load_state_dict(ck["model"])

    policy = NeuralEngine(ckpt, lookahead=0)
    ro = bgcore.Rollouts(str(MODELS / "td.onnx"), args.trials, args.truncate, 0, args.seed)

    print(f"Sampling {args.positions} positions from self-play…")
    boards = sample_boards(policy, args.positions, rng)

    print(f"Labelling with rollouts ({args.trials}×{args.truncate})…")
    t0 = time.time()
    x = torch.from_numpy(np.array([b.features() for b in boards], dtype=np.float32))
    y = torch.tensor([ro.equity(b) for b in boards], dtype=torch.float32)
    print(f"  labelled {len(boards)} positions in {time.time() - t0:.1f}s")

    # Baseline agreement (net equity vs rollout equity) before fine-tuning.
    with torch.no_grad():
        before = F.mse_loss(net_equity(net(x)), y).item()

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = F.mse_loss(net_equity(net(x)), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = F.mse_loss(net_equity(net(x)), y).item()
    print(f"equity MSE vs rollouts:  {before:.4f} -> {after:.4f}")

    out = MODELS / "td_rollout.pt"
    torch.save({"model": net.state_dict(), "hidden": ck.get("hidden", 128), "iter": ck.get("iter")}, out)
    print(f"saved fine-tuned net to {out}")
    print("Export it with:  trainer/export_onnx.py models/td_rollout.pt")


if __name__ == "__main__":
    main()
