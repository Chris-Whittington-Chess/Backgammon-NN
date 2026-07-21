"""Task #7: strategic features on CLEAN labels.

The 14 strategic add-on features (richer-features branch) hurt under noisy TD
self-play. This retests them where they get a fair shot: a 212-input net trained
on the low-variance rollout/distillation labels (same targets as train_rollout).

The base encoding stays 198, so the 198-input champion still runs; the candidate
concatenates Board.strategic() -> 212. Benchmarks the 212 candidate against the
198 champion in one process (each encodes independently).

Run:
  .venv/Scripts/python trainer/train_features.py --data rollout_1ply.npz \
      --alpha 0.9 --epochs 40 --champion td_loop1_final.pt --out td_1ply_feat.pt
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
from model import ValueNetBucketed, ValueNetSplitBucketed, equity6, N_HEADS
from train import hce_policy, play_match_game
from train_bucketed import bucketed_champion_policy, _class_route

MODELS = Path(__file__).resolve().parent.parent / "models"
CLASS_OF = {1: 0, 2: 1, 3: 2, -1: 3, -2: 4, -3: 5}
NUM_INPUTS_212 = 212


def load_212(path, limit, limit_seed):
    """Labels from the rollout .npz, positions re-encoded as 212 features."""
    d = np.load(MODELS / path)
    pos_ids = d["pos_ids"]
    soft = d["probs"].astype(np.float32)
    outcomes = d["outcomes"].astype(int)
    buckets = d["buckets"].astype(np.int64)
    if limit and limit < len(pos_ids):
        sel = np.random.default_rng(limit_seed).choice(len(pos_ids), size=limit, replace=False)
        pos_ids, soft, outcomes, buckets = pos_ids[sel], soft[sel], outcomes[sel], buckets[sel]
    n = len(pos_ids)
    feats = np.zeros((n, NUM_INPUTS_212), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(pos_ids):
        b = bgcore.Board.from_id(str(p))
        feats[i, :198] = b.features()
        feats[i, 198:] = b.strategic()
        if (i + 1) % 200000 == 0:
            print(f"  encoded {i+1}/{n} ({(i+1)/(time.time()-t0):.0f}/s)", flush=True)
    hard = np.array([CLASS_OF[int(o)] for o in outcomes], dtype=np.int64)
    return feats, soft, hard, buckets


@torch.no_grad()
def children_mover_equity_212(net, board, d1, d2, route):
    """Mover equity of each legal move, children scored with 212 features."""
    children = bgcore.legal_moves(board, d1, d2)
    term = [c.winner_points() for c in children]
    feats = np.zeros((len(children), NUM_INPUTS_212), dtype=np.float32)
    for i, c in enumerate(children):
        s = c.swap_perspective()  # opponent to move
        feats[i, :198] = s.features()
        feats[i, 198:] = s.strategic()
    buckets = torch.tensor([route(c) for c in children], dtype=torch.long)
    x = torch.from_numpy(feats)
    mover_eq = -equity6(net.probs_for(x, buckets))
    for i, p in enumerate(term):
        if p is not None:
            mover_eq[i] = float(p)
    return children, mover_eq, term


def policy_212(net, route):
    def f(board, d1, d2, rng):
        children, mover_eq, term = children_mover_equity_212(net, board, d1, d2, route)
        i = int(mover_eq.argmax())
        return (None, term[i]) if term[i] is not None else (children[i].swap_perspective(), None)
    return f


def bench_212(net, opponent, games, rng, route=_class_route):
    """212-feature candidate vs an opponent policy, seats alternated."""
    me = policy_212(net, route)
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
    ap.add_argument("--data", default="rollout_1ply.npz")
    ap.add_argument("--arch", choices=["raw", "split"], default="split",
                    help="raw: 212 into layer 1; split: strategic injected after the "
                         "first ReLU (NNUE accumulator)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--hidden", default="256,128")
    ap.add_argument("--act", choices=["relu", "sqrelu"], default="relu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--bench-every", type=int, default=5)
    ap.add_argument("--bench-games", type=int, default=400)
    ap.add_argument("--champion", default="td_loop1_final.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="td_1ply_feat.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.hidden.split(",") if x.strip()]
    hidden = sizes[0] if len(sizes) == 1 else sizes

    print(f"loading + 212-encoding {args.data} ...", flush=True)
    feats, soft, hard, buckets = load_212(args.data, args.limit, args.limit_seed)
    n = len(feats)
    onehot = np.zeros((n, 6), dtype=np.float32)
    onehot[np.arange(n), hard] = 1.0
    target = args.alpha * soft + (1.0 - args.alpha) * onehot

    x = torch.from_numpy(feats)
    tgt = torch.from_numpy(target)
    bk = torch.from_numpy(buckets)
    print(f"{n} positions | 212 inputs | alpha {args.alpha} | per-bucket "
          f"{np.bincount(buckets, minlength=N_HEADS).tolist()}", flush=True)

    net = (ValueNetSplitBucketed(hidden, args.act, N_HEADS) if args.arch == "split"
           else ValueNetBucketed(hidden, args.act, N_HEADS, num_inputs=NUM_INPUTS_212))
    print(f"arch: {args.arch}-injection", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    champ = bucketed_champion_policy(args.champion)

    perm0 = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = perm0[:n_val], perm0[n_val:]
    n_tr = len(tr_idx)

    def lr_at(ep):
        frac = (ep - 1) / max(1, args.epochs - 1)
        return args.lr_end + (args.lr - args.lr_end) * (1 + math.cos(math.pi * frac)) / 2

    @torch.no_grad()
    def val_loss():
        logp = F.log_softmax(net.logits_for(x[val_idx], bk[val_idx]), dim=-1)
        return float(-(tgt[val_idx] * logp).sum(1).mean())

    best_metric, best_ep = -1e9, 0
    for ep in range(1, args.epochs + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(ep)
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
        line = (f"epoch {ep:3d} | lr {lr_at(ep):.1e} | train {total/n_tr:.4f} "
                f"| val {val_loss():.4f} | {time.time()-t0:4.1f}s")

        if ep % args.bench_every == 0 or ep == args.epochs:
            wr_h, ppg_h = bench_212(net, hce_policy(), args.bench_games, rng)
            wr_c, ppg_c = bench_212(net, champ, args.bench_games, rng)
            line += f" || vs HCE {100*wr_h:.1f}% | vs champ {100*wr_c:.1f}% ({ppg_c:+.2f})"
            if wr_c > best_metric:
                best_metric, best_ep = wr_c, ep
                torch.save({"model": net.state_dict(), "hidden": hidden, "act": args.act,
                            "bucketed": True, "n_buckets": N_HEADS, "class_aware": True,
                            "num_inputs": NUM_INPUTS_212, "features": "strategic",
                            "arch": args.arch, "alpha": args.alpha, "epoch": ep,
                            "metric": wr_c}, MODELS / args.out)
                line += "  <- best (saved)"
        print(line, flush=True)

    print(f"\nbest at epoch {best_ep} (vs champ {best_metric:.3f}) -> {MODELS / args.out}")


if __name__ == "__main__":
    main()
