# Backgammon-NN

**[whittingtonchess.com/backgammon](https://whittingtonchess.com/backgammon)** —
project page, with the story of how it was built · **[Development
report](https://whittingtonchess.com/backgammon-report)** · **[Download the
app](../../releases/latest)**

A backgammon engine with a neural-network evaluator. The engine core is Rust
(fast, validated move generation); training is PyTorch; inference runs natively
in Rust via ONNX; and there are two ways to play — a PySide6 **desktop app** and
a **text-only console app**.

The lineage began with pure self-play: the first nets learned from random
weights, TD-Gammon style, and reached a strength ceiling that no amount of extra
architecture, features or self-generated labels could break — a student
distilled from its own engine cannot exceed it. The **current** net breaks that
ceiling by learning from an outside teacher, GNU Backgammon, and beats the last
self-play champion decisively at the same inference cost. Both nets ship, so you
can play either.

It's a full toolkit: train new nets from self-play or by distillation, run
automatic engine-vs-engine matches (against other engines or itself), benchmark
against gnubg, and play the result on your PC.

## Download

**[Download Backgammon.exe](../../releases/latest)** (Windows 64-bit, ~59 MB) —
no install, no Python, no PyTorch. Double-click and play.

Opponents range from *Random* and the hand-crafted evaluator up through
**Neural classic** (the previous champion, an easier rung), the current
**Neural** net at 0/1/2-ply, and **Rollout** — Monte-Carlo search, the strongest
and the default.

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
  value net: a shared 198→256→128 body feeding **12 output heads** selected by
  **gnubg-style position class** (race / crashed / contact) with total-pip
  sub-buckets inside each, every head a six-outcome softmax (win/lose ×
  single/gammon/backgammon). The shared body specialises per game-stage without
  the data starvation of separate per-phase nets.
- **Taught by GNU Backgammon.** The current net is trained on **22.5M positions
  labelled with gnubg's 2-ply evaluation**, not on self-play. It beats the
  previous self-play champion (3M games) **53.4% at 1-ply over 40,000 games**
  — at identical inference cost — and closes ~38% of the gap to gnubg itself.
  Earlier nets, distilled from the engine's own rollouts and searches, could
  only reach parity with themselves; an external teacher was the way past that.
- **Cross-language parity**: PyTorch → ONNX → Rust `tract` inference match to
  <1e-4.
- **Expectiminimax search** to 2 ply with GNUbg-style candidate pruning, plus
  parallel Monte-Carlo **rollouts**. Note that search buys progressively less as
  the evaluator improves: 1-ply over 0-ply was 62.5% for an early net and is
  **51.5% ±6.9** for the current one.
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
neural evaluator, ONNX/native inference, n-ply search, and a GUI — plus the
doubling cube, Monte-Carlo rollouts, and a packaged standalone app.

The current net is trained by distillation from GNU Backgammon rather than by
self-play; the [development report](DEV_REPORT.md) records that work, including
what failed. Known limits: pure distillation cannot exceed its teacher (we are
at ~46% against gnubg 2-ply, where parity is 50%), added labels stopped paying
past ~17.5M, and extra search depth no longer measurably helps. Possible next
steps: cubeful (Janowski) equity, match play, and optimising playing strength
directly rather than fitting a teacher's labels.

## License

Copyright © Chris Whittington 2026. All Rights Reserved.

## Acknowledgements

The move generator is differentially tested against
[wildbg](https://github.com/carsten-wenderdel/wildbg) (MIT/Apache-2.0). Position
ID format follows [GNU Backgammon](https://www.gnu.org/software/gnubg/).

Match play uses the **Kazaross XG2** match equity table, Copyright © 2011 Neil
Kazaross, transcribed for GNU Backgammon by Michael Petch and distributed as part
of that program. It is offered under the GNU all-permissive licence: copying and
distribution, with or without modification, are permitted in any medium without
royalty provided the copyright notice and the permission notice are preserved,
and it is offered as-is without warranty. Both notices are reproduced in full in
[`trainer/met_kazaross.py`](trainer/met_kazaross.py) and ship inside the app; the
packaged build refuses to complete if they are missing.
