"""Concatenate distillation/position .npz files (same schema: pos_ids, probs,
outcomes, buckets). Used to (a) stitch parallel position-gen shards into one set,
and (b) merge Machine-A + Machine-B label files into the final training set.

Run:
  .venv/Scripts/python trainer/merge_npz.py --out labels_1M_c2.npz \
      --inputs labels_A_c2.npz labels_B_c2.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="npz files under models/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup", action="store_true",
                    help="drop duplicate pos_ids (keep first occurrence)")
    args = ap.parse_args()

    pid, pr, oc, bk = [], [], [], []
    net = None
    for f in args.inputs:
        d = np.load(MODELS / f)
        pid.append(d["pos_ids"]); pr.append(d["probs"])
        oc.append(d["outcomes"]); bk.append(d["buckets"])
        if "net" in d:
            net = str(d["net"])
        print(f"  {f}: {len(d['pos_ids'])} positions")
    pid = np.concatenate(pid); pr = np.concatenate(pr)
    oc = np.concatenate(oc); bk = np.concatenate(bk)

    if args.dedup:
        _, idx = np.unique(pid, return_index=True)
        idx.sort()
        before = len(pid)
        pid, pr, oc, bk = pid[idx], pr[idx], oc[idx], bk[idx]
        print(f"  dedup: {before} -> {len(pid)} unique positions")

    out = MODELS / args.out
    np.savez_compressed(out, pos_ids=pid, probs=pr.astype(np.float32),
                        outcomes=oc.astype(np.int8), buckets=bk.astype(np.int8),
                        trials=0, truncate=1, net=net)
    pop = np.bincount(bk.astype(int), minlength=12)
    print(f"merged {len(args.inputs)} files -> {out} | {len(pid)} positions")
    print(f"per-bucket population: {pop.tolist()}")


if __name__ == "__main__":
    main()
