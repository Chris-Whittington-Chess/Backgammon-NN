"""Export a trained ValueNet to ONNX and verify parity (SPEC §9, milestone M5).

Produces:
  models/td.onnx      - the network, with a dynamic batch axis
  models/parity.json  - a fixture (starting-position id + features + expected
                        output) so the Rust `tract`-based NnEval can be checked
                        against PyTorch to a tight tolerance.

Run: .venv/Scripts/python trainer/export_onnx.py [checkpoint.pt]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import bgcore
from model import ValueNet

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def main():
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODELS / "td_latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = ValueNet(ckpt.get("hidden", 128))
    net.load_state_dict(ckpt["model"])
    net.eval()

    onnx_path = MODELS / "td.onnx"
    dummy = torch.zeros(1, 198, dtype=torch.float32)
    torch.onnx.export(
        net,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
        opset_version=17,
    )
    print(f"exported {onnx_path} (from {ckpt_path.name}, iter {ckpt.get('iter', '?')})")

    # --- Parity: PyTorch vs ONNX Runtime on a batch of random inputs. ---
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    x = np.random.rand(64, 198).astype(np.float32)
    with torch.no_grad():
        torch_out = net(torch.from_numpy(x)).numpy()
    onnx_out = sess.run(["output"], {"input": x})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"PyTorch vs ONNXRuntime max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX export does not match PyTorch!"

    # --- Fixture for the Rust tract parity test. ---
    start = bgcore.Board.starting()
    feats = start.features()
    with torch.no_grad():
        start_out = net(torch.tensor([feats], dtype=torch.float32))[0].tolist()
    fixture = {
        "start_id": start.position_id(),
        "features": feats,
        "expected_output": start_out,
    }
    (MODELS / "parity.json").write_text(json.dumps(fixture, indent=2))
    print(f"wrote parity fixture; start output = {[round(v, 4) for v in start_out]}")
    print("OK")


if __name__ == "__main__":
    main()
