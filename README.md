# backgammon-2026

A backgammon engine with a self-taught neural-network evaluator and a desktop
GUI. The engine core is Rust (fast, validated move generation); training is
PyTorch (TD/Monte-Carlo self-play); inference runs natively in Rust via ONNX;
and the GUI is PySide6.

The trained net **beats the hand-crafted evaluator ~84%** and learned entirely
from self-play, starting from random weights.

## Architecture

```
PySide6 GUI  ──►  Python (trainer + engine adapters)  ──PyO3──►  Rust core (bgcore)
                        │                                            board · dice · genmoves
                   PyTorch net                                      evaluators · match runner
                   TD self-play                                     ONNX inference (tract)
                        │                                           n-ply search
                   ONNX export ──────────────────────────────────►
```

| Component | Where |
|---|---|
| Engine core (board, dice, move gen, evaluators, search, ONNX) | `crates/bgcore` |
| Python bindings (PyO3) | `crates/bgpy` → import as `bgcore` |
| Trainer, model, engine adapters | `trainer/` |
| Desktop GUI | `gui/` |
| Move-gen differential test vs [wildbg](https://github.com/carsten-wenderdel/wildbg) | `tools/movegen-difftest` |
| Full spec | `SPEC.md` |

## Highlights

- **Move generation validated** against the wildbg reference engine across
  3.15M (position, dice) pairs — zero mismatches.
- **198-input Tesauro encoding** and a 198→128→5 value net (win / gammon /
  backgammon probabilities), trained by self-play.
- **Cross-language parity**: PyTorch → ONNX → Rust `tract` inference match to
  <1e-4.
- **1-ply search** beats the same net at 0-ply **62.5%** head-to-head.
- **GnuBG-compatible Position IDs** for interop.

## Setup

Requires Rust (stable) and Python 3.9+.

```bash
# Python env + build the Rust extension
python -m venv .venv
.venv/Scripts/pip install maturin numpy torch onnx onnxruntime PySide6
cd crates/bgpy && ../../.venv/Scripts/maturin develop --release && cd ../..
```

## Run

```bash
# Play against the trained net (GUI)
.venv/Scripts/python gui/app.py

# Train from self-play
.venv/Scripts/python trainer/train.py --iters 200 --games 40 --lam 1.0

# Export the net to ONNX (for native Rust inference)
.venv/Scripts/python trainer/export_onnx.py models/td_latest.pt

# Benchmark the net natively (needs the onnx feature)
cargo run --release --features onnx --example nn_bench

# Engine-vs-engine match runner (HCE vs Random, mirrored dice)
cargo run --release --example match

# Rust tests
cargo test
```

The trained checkpoint (`models/td_latest.pt`) and its ONNX export
(`models/td.onnx`) are included, so the GUI is playable immediately after build.

## Status

The original spec (M0–M6) is complete: engine, validated move generation,
TD-trained neural evaluator, ONNX/native inference, n-ply search, and a GUI.
Possible next steps: deeper search, the doubling cube, and richer features.

## License

MIT.

## Acknowledgements

The move generator is differentially tested against
[wildbg](https://github.com/carsten-wenderdel/wildbg) (MIT/Apache-2.0). Position
ID format follows [GNU Backgammon](https://www.gnu.org/software/gnubg/).
