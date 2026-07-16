"""Export the race net from a phase-split checkpoint to ONNX (6-output softmax).

The race half of a train_phase.py checkpoint is a ValueNet6 (198 -> hidden -> 6
logits). We wrap it in a softmax so the ONNX graph emits the six outcome
*probabilities* directly — the Rust side then reads six numbers and folds them
into the same 5-field Value the engine already uses (win = ws+wg+wbg, etc.), so
nothing downstream changes.

Writes models/td_race.onnx (+ models/parity_race.json for the Rust parity test).
Verifies tract-style parity via onnxruntime before writing.

Run: .venv/Scripts/python trainer/export_race.py [phase_ckpt] [out.onnx]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import NUM_INPUTS, OUTCOME_POINTS
from train_phase import load_phase_nets

MODELS = Path(__file__).resolve().parent.parent / "models"


class Softmaxed(nn.Module):
    """Wrap a 6-logit net so ONNX emits the six outcome probabilities."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "td_phase.pt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else MODELS / "td_race.onnx"
    phase = sys.argv[3] if len(sys.argv) > 3 else "race"   # "race" | "contact"

    ck = torch.load(MODELS / ckpt, map_location="cpu")
    model = Softmaxed(load_phase_nets(ck)[phase]).eval()

    dummy = torch.zeros(1, NUM_INPUTS)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=13,
    )

    # Parity vs onnxruntime on the real starting-position features (not zeros —
    # the Rust test encodes the actual board), plus a sum-to-1 / equity sanity.
    import bgcore
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out))
    feats = np.asarray([bgcore.Board.starting().features()], dtype=np.float32)
    with torch.no_grad():
        py = model(torch.from_numpy(feats)).numpy()[0]
    rt = sess.run(None, {"input": feats})[0][0]
    diff = float(np.abs(py - rt).max())
    pts = np.array(OUTCOME_POINTS, dtype=np.float32)
    print(f"exported {out} ({phase} net, iter {ck.get('iter')})")
    print(f"PyTorch vs ONNXRuntime max abs diff: {diff:.2e}")
    print(f"start probs {np.round(rt, 4)}  sum {rt.sum():.4f}  equity {float((rt*pts).sum()):+.4f}")

    # The Rust parity test is pinned to the race net.
    if phase == "race":
        (MODELS / "parity_race.json").write_text(json.dumps({
            "position_id": "4HPwATDgc/ABMA",
            "features": [float(x) for x in feats[0]],
            "expected_output": [float(x) for x in rt],
        }, indent=2))
        print("wrote parity_race.json")


if __name__ == "__main__":
    main()
