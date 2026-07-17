"""Export a pip-count-bucketed net to ONNX (N_BUCKETS x 6 softmax outputs).

The net emits `[N, N_BUCKETS, 6]` logits; we softmax each bucket's 6 outcomes and
flatten to `[N, N_BUCKETS*6]`. The Rust engine reads that, computes the position's
total-pip bucket, and slices its 6 probabilities — no bucket input to the graph.

Writes models/td_bucket.onnx (+ models/parity_bucket.json for the Rust test).

Run: .venv/Scripts/python trainer/export_bucketed.py [ckpt] [out.onnx]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import NUM_INPUTS, N_BUCKETS, OUTCOME_POINTS, net_bucketed_from_state, pip_bucket

MODELS = Path(__file__).resolve().parent.parent / "models"


class BucketedSoftmax(nn.Module):
    """[N,198] -> [N, N_BUCKETS*6]: softmax over each bucket's six outcomes."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        probs = torch.softmax(self.net(x), dim=-1)          # [N, B, 6]
        return probs.reshape(x.shape[0], -1)                # [N, B*6]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "td_bucket.pt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else MODELS / "td_bucket.onnx"

    ck = torch.load(MODELS / ckpt, map_location="cpu")
    net = net_bucketed_from_state(ck["model"], ck["hidden"], ck.get("act", "relu"))
    model = BucketedSoftmax(net).eval()

    dummy = torch.zeros(1, NUM_INPUTS)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13,
    )

    import bgcore
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out))
    start = bgcore.Board.starting()
    feats = np.asarray([start.features()], dtype=np.float32)
    with torch.no_grad():
        py = model(torch.from_numpy(feats)).numpy()[0]
    rt = sess.run(None, {"input": feats})[0][0]
    diff = float(np.abs(py - rt).max())

    bucket = pip_bucket(start.pip_count(0) + start.pip_count(1))
    probs = rt[bucket * 6:bucket * 6 + 6]
    pts = np.array(OUTCOME_POINTS, dtype=np.float32)
    print(f"exported {out} (bucketed, iter {ck.get('iter')}, {N_BUCKETS} buckets)")
    print(f"PyTorch vs ONNXRuntime max abs diff: {diff:.2e}")
    print(f"start bucket {bucket}  probs {np.round(probs, 4)}  sum {probs.sum():.4f}"
          f"  equity {float((probs*pts).sum()):+.4f}")

    (MODELS / "parity_bucket.json").write_text(json.dumps({
        "features": [float(x) for x in feats[0]],
        "bucket": int(bucket),
        "expected_bucket_output": [float(x) for x in probs],
    }, indent=2))
    print("wrote parity_bucket.json")

    # Also refresh the live-net fixture (parity.json) with the *folded* Value, so
    # the architecture-agnostic live-net Rust test verifies whatever td.onnx is:
    # 6-outcome probs -> nested [win, win_g, win_bg, lose_g, lose_bg].
    ws, wg, wbg, ls, lg, lbg = (float(x) for x in probs)
    folded = [ws + wg + wbg, wg + wbg, wbg, lg + lbg, lbg]
    (MODELS / "parity.json").write_text(json.dumps({
        "position_id": "4HPwATDgc/ABMA",
        "features": [float(x) for x in feats[0]],
        "expected_output": folded,
    }, indent=2))
    print("wrote parity.json (folded Value for the live-net test)")


if __name__ == "__main__":
    main()
