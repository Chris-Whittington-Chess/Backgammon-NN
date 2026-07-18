"""Supervised training on rollout-labeled data (task #4).

Trains the class-aware 12-head net to the soft/hard BLEND target:

    target = alpha * rollout_distribution  +  (1 - alpha) * onehot(game_outcome)

- rollout_distribution (soft): distillation-from-search, low variance but biased
  by our own truncated-rollout leaf eval.
- onehot(game_outcome) (hard): unbiased ground truth, high variance, anchors
  against baking in the net's own errors.
- alpha ~0.75 to start; push toward 1 as label quality rises / on relabel iters.

Loss is soft-label cross-entropy on the routed head. Unlike self-play training,
the target is a fixed dataset, so this is fast and repeatable.

Run:
  .venv/Scripts/python trainer/train_rollout.py --data rollout_data.npz \
      --alpha 0.75 --epochs 40 --champion td_latest.pt --out td_rollout.pt
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bgcore
from model import ValueNetBucketed, N_HEADS
from train import random_policy, hce_policy
from train_bucketed import benchmark_bucketed, bucketed_champion_policy, _class_route

MODELS = Path(__file__).resolve().parent.parent / "models"

# signed game points -> outcome class index [ws, wg, wbg, ls, lg, lbg]
CLASS_OF = {1: 0, 2: 1, 3: 2, -1: 3, -2: 4, -3: 5}


def load_dataset(path):
    d = np.load(MODELS / path)
    pos_ids = d["pos_ids"]
    soft = d["probs"].astype(np.float32)
    outcomes = d["outcomes"].astype(int)
    buckets = d["buckets"].astype(np.int64)
    feats = np.stack([bgcore.Board.from_id(str(p)).features() for p in pos_ids]).astype(np.float32)
    hard = np.array([CLASS_OF[int(o)] for o in outcomes], dtype=np.int64)
    return feats, soft, hard, buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rollout_data.npz")
    ap.add_argument("--alpha", type=float, default=0.75, help="soft-label weight")
    ap.add_argument("--hidden", default="256,128")
    ap.add_argument("--act", choices=["relu", "sqrelu"], default="relu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.08, help="held-out fraction for val loss")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--bench-every", type=int, default=5)
    ap.add_argument("--bench-games", type=int, default=400)
    ap.add_argument("--champion", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="td_rollout.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.hidden.split(",") if x.strip()]
    hidden = sizes[0] if len(sizes) == 1 else sizes

    feats, soft, hard, buckets = load_dataset(args.data)
    n = len(feats)
    onehot = np.zeros((n, 6), dtype=np.float32)
    onehot[np.arange(n), hard] = 1.0
    target = args.alpha * soft + (1.0 - args.alpha) * onehot

    x = torch.from_numpy(feats)
    tgt = torch.from_numpy(target)
    bk = torch.from_numpy(buckets)
    print(f"{n} positions | alpha {args.alpha} | per-bucket "
          f"{np.bincount(buckets, minlength=N_HEADS).tolist()}")

    net = ValueNetBucketed(hidden, args.act, N_HEADS)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    champ = bucketed_champion_policy(args.champion) if args.champion else None

    # Held-out validation split to watch generalisation (early-stopping signal).
    perm0 = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = perm0[:n_val], perm0[n_val:]
    n_tr = len(tr_idx)

    def lr_at(ep):
        if args.lr_schedule == "none":
            return args.lr
        frac = (ep - 1) / max(1, args.epochs - 1)
        return args.lr_end + (args.lr - args.lr_end) * (1 + math.cos(math.pi * frac)) / 2

    @torch.no_grad()
    def val_loss():
        if not n_val:
            return float("nan")
        logp = F.log_softmax(net.logits_for(x[val_idx], bk[val_idx]), dim=-1)
        return float(-(tgt[val_idx] * logp).sum(1).mean())

    best_metric, best_ep = -1e9, 0
    for ep in range(1, args.epochs + 1):
        cur_lr = lr_at(ep)
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        t0 = time.time()
        net.train()
        perm = tr_idx[torch.randperm(n_tr)]
        total = 0.0
        for i in range(0, n_tr, args.batch):
            idx = perm[i : i + args.batch]
            logp = F.log_softmax(net.logits_for(x[idx], bk[idx]), dim=-1)
            loss = -(tgt[idx] * logp).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        net.eval()
        line = (f"epoch {ep:3d} | lr {cur_lr:.1e} | train {total/n_tr:.4f} "
                f"| val {val_loss():.4f} | {time.time()-t0:4.1f}s")

        if ep % args.bench_every == 0 or ep == args.epochs:
            wr_h, ppg_h = benchmark_bucketed(net, hce_policy(), args.bench_games, rng, _class_route)
            line += f" || vs HCE {100*wr_h:.1f}% ({ppg_h:+.2f})"
            metric = wr_h
            if champ is not None:
                wr_c, ppg_c = benchmark_bucketed(net, champ, args.bench_games, rng, _class_route)
                line += f" | vs champ {100*wr_c:.1f}% ({ppg_c:+.2f})"
                metric = wr_c
            # Keep the checkpoint with the BEST head-to-head, not the last epoch.
            if metric > best_metric:
                best_metric, best_ep = metric, ep
                torch.save({"model": net.state_dict(), "hidden": hidden, "act": args.act,
                            "bucketed": True, "n_buckets": N_HEADS, "class_aware": True,
                            "alpha": args.alpha, "epoch": ep, "iter": ep,
                            "metric": metric}, MODELS / args.out)
                line += "  <- best (saved)"
        print(line, flush=True)

    print(f"\nbest at epoch {best_ep} (metric {best_metric:.3f}) -> {MODELS / args.out}")


if __name__ == "__main__":
    main()
