# Backgammon-NN

**[whittingtonchess.com/backgammon](https://whittingtonchess.com/backgammon)** —
project page, with the story of how it was built · **[Development
report](https://whittingtonchess.com/backgammon-report)** · **[Download the
app](../../releases/latest)**

A self-learning backgammon engine with a neural-network evaluator. The engine
core is Rust (fast, validated move generation); training is PyTorch
(TD/Monte-Carlo self-play); inference runs natively in Rust via ONNX; and there
are two ways to play — a PySide6 **desktop app** and a **text-only console app**.

The net learned to play entirely from self-play, starting from random weights.
It beats the hand-crafted evaluator decisively, and at equal search it holds its
own against the [wildbg](https://github.com/carsten-wenderdel/wildbg) reference
engine. It's a full toolkit: extend the training to grow stronger nets, build new
networks, run automatic engine-vs-engine matches (against other engines or
itself), and play the result on your PC.

## Download

**[Download Backgammon.exe](../../releases/latest)** (Windows 64-bit, ~62 MB) —
no install, no Python, no PyTorch. Double-click and play.

Everything is in the one file: the GUI, the Rust engine, and the trained net.
The engine runs natively via the embedded ONNX runtime, so the packaged app
plays exactly as the source build does.

> Windows SmartScreen will warn about an unrecognized publisher — the exe isn't
> code-signed. Choose *More info* → *Run anyway*.

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
| Standalone app build (PyInstaller) | `packaging/` |
| Development log (what changed, and why) | [`CHANGELOG.md`](CHANGELOG.md) |
| Move-gen differential test vs [wildbg](https://github.com/carsten-wenderdel/wildbg) | `tools/movegen-difftest` |
| Full spec | `SPEC.md` |

## Highlights

- **Move generation validated** against the wildbg reference engine across
  3.15M (position, dice) pairs — zero mismatches.
- **198-input Tesauro encoding** and a **Stockfish-NNUE-style output-bucketed**
  value net: a shared 198→256→128 body feeding **8 output heads** selected by
  total pip count, each a six-outcome softmax (win/lose × single/gammon/backgammon).
  Trained by self-play over **1M games**, it beats the previous 256-128-128 net
  ~53% head-to-head — the shared body specialises per game-stage without the data
  starvation of separate per-phase nets.
- **Cross-language parity**: PyTorch → ONNX → Rust `tract` inference match to
  <1e-4.
- **Expectiminimax search** to 2 ply with GNUbg-style candidate pruning, plus
  parallel Monte-Carlo **rollouts**. 1-ply beats 0-ply **62.5%** head-to-head.
- **Doubling cube** (money play) and a **GnuBG-compatible Position ID** for
  interop.

## Setup

Requires Rust (stable) and Python 3.9+.

```bash
# Python env + build the Rust extension
python -m venv .venv
.venv/Scripts/pip install maturin numpy torch onnx onnxruntime PySide6
# --features onnx builds the native net + rollout engine into the extension
cd crates/bgpy && ../../.venv/Scripts/maturin develop --release --features onnx && cd ../..
```

## Run

```bash
# Play against the trained net — desktop app
.venv/Scripts/python gui/app.py

# Play against the trained net — text-only console app
.venv/Scripts/python trainer/console_play.py

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

# Build the standalone app -> dist/Backgammon.exe (verifies the exe it produces)
.venv/Scripts/python packaging/build.py
```

The trained checkpoint (`models/td_latest.pt`) and its ONNX export
(`models/td.onnx`) are included, so the GUI is playable immediately after build.

## Status

The original spec (M0–M6) is complete: engine, validated move generation,
TD-trained neural evaluator, ONNX/native inference, n-ply search, and a GUI —
plus the doubling cube, Monte-Carlo rollouts, and a packaged standalone app.
Possible next steps: cubeful (Janowski) equity, match play, and richer features
such as wildbg-style split contact/race nets.

## License

Copyright © Chris Whittington 2026. All Rights Reserved.

## Acknowledgements

The move generator is differentially tested against
[wildbg](https://github.com/carsten-wenderdel/wildbg) (MIT/Apache-2.0). Position
ID format follows [GNU Backgammon](https://www.gnu.org/software/gnubg/).
