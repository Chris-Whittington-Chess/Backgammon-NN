# Backgammon Engine — Technical Specification

**Status:** Draft v1 · **Date:** 2026-07-13

A backgammon-playing engine with a neural-network evaluator trained by TD(λ)
self-play, plus a desktop GUI to play against it. The engine core is written in
Rust for fast self-play and exposed to Python (PyTorch) for training.

---

## 1. Goals & scope

### In scope (v1)
- Fast, correct Rust game core: board, dice, legal-move generation, make/unmake.
- Three pluggable evaluators behind one interface: **random**, **HCE**
  (hand-crafted), **NN**.
- Play modes: **engine vs engine**, **human vs engine**, **self-play** (data /
  training).
- TD(λ) self-play training loop in PyTorch.
- ONNX export + inference so the trained net can run inside the Rust core (fast
  self-play) and in the GUI.
- Desktop GUI (play vs engine, show dice, enter moves, request hints).

### Out of scope for v1 (explicit future work)
- **Doubling cube.** v1 is *cubeless*. Cube decisions (double/take/pass, match
  equity) are a large sub-project — see §11.
- **Match play** (n-point matches, Crawford, match equity table). v1 plays
  single "money" games, cubeless.
- Mobile GUI. Architecture keeps it cheap (Rust core + ONNX Runtime Mobile), but
  no mobile front-end ships in v1.

---

## 2. High-level architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Front ends                              │
│   Desktop GUI (PySide6)      [future: Web / Mobile]            │
└───────────────┬───────────────────────────┬──────────────────┘
                │ Python API                 │  (future: Rust FFI / ORT-Mobile)
┌───────────────▼───────────────────────────▼──────────────────┐
│                    Python layer  (trainer/, app/)             │
│   PyTorch model · TD(λ) loop · ONNX export · game driver      │
└───────────────┬──────────────────────────────────────────────┘
                │ PyO3 bindings
┌───────────────▼──────────────────────────────────────────────┐
│                     Rust core  (crate: bgcore)                │
│  Board · dice · genmoves · make/unmake · game loop            │
│  Evaluators: random, HCE, NN(via ort) · self-play driver      │
│  Feature encoder (single source of truth)                     │
└──────────────────────────────────────────────────────────────┘
```

**Key principle — one source of truth for rules and encoding.** Move generation
and the NN feature encoder live *only* in Rust and are called from Python via
PyO3. This prevents the classic bug where the Python training encoder and the
engine encoder silently diverge.

### Workspace layout
```
Backgammon-2026/
├─ Cargo.toml                 # Rust workspace
├─ crates/
│  ├─ bgcore/                 # pure-Rust engine (no Python deps)
│  │  ├─ src/board.rs
│  │  ├─ src/dice.rs
│  │  ├─ src/moves.rs         # genmoves, make, unmake
│  │  ├─ src/game.rs          # game loop / result
│  │  ├─ src/features.rs      # NN input encoding (198-dim, §6)
│  │  ├─ src/eval/random.rs
│  │  ├─ src/eval/hce.rs
│  │  ├─ src/eval/nn.rs       # ONNX inference via `ort`
│  │  └─ src/selfplay.rs
│  └─ bgpy/                   # PyO3 crate -> `bgcore` Python module
├─ trainer/                   # Python (PyTorch)
│  ├─ model.py                # net definition
│  ├─ td_train.py             # TD(λ) self-play loop
│  ├─ export_onnx.py
│  ├─ benchmark.py            # strength ladder vs random/HCE/old nets
│  └─ pyproject.toml
├─ app/                       # PySide6 desktop GUI
├─ models/                    # checkpoints (.pt) + exported .onnx
└─ SPEC.md
```

---

## 3. Board representation

Standard board: 24 points, a bar, and a bear-off tray, per player.

**Canonical internal form (Rust):** always store the position *from the side to
move's perspective* so the encoder and evaluator never branch on colour.

```rust
/// Points 1..=24 are from the mover's view: 24 = mover's furthest-back point,
/// 1 = mover's ace point. Positive = mover's checkers, negative = opponent's.
pub struct Board {
    points: [i8; 25], // index 1..=24 used; index 0 unused/padding
    bar:  [u8; 2],    // [mover_on_bar, opp_on_bar]
    off:  [u8; 2],    // [mover_borne_off, opp_borne_off]
    // side_to_move is implicit: board is always mover-relative
}
```
- **Invariants** (checked in debug builds): each side has 15 checkers total
  (points + bar + off); no point holds both colours.
- `swap_perspective()` flips the board so the opponent becomes the mover
  (used after each ply). Must be an involution; unit-tested.
- Provide a compact, canonical **Position ID** (GNUbg-compatible base64 is a
  nice-to-have; a simple stable byte serialization is enough for v1) for
  logging, transposition, and dedup.

---

## 4. Rules & move generation

Move generation is the trickiest correct-code in the project. Nail it with tests.

### Dice
- Two d6. Doubles ⇒ **four** moves of that value.
- RNG: seedable (`rand` with a fixed seed in tests / self-play reproducibility).

### `genmoves(board, dice) -> Vec<Move>`
A *Move* is a full turn (a sequence of 1–4 checker steps) plus the resulting
`Board`. Rules to enforce:
1. **Bar first:** if the mover has checkers on the bar, they must all re-enter
   before any other checker moves.
2. **Blots & hits:** landing on a point with exactly one opposing checker sends
   it to the bar.
3. **Blocked points:** cannot land on a point with ≥2 opposing checkers.
4. **Bear-off:** legal only when all 15 checkers are in the home board (points
   1–6). Exact and overflow bear-off rules (a higher die bears off from the
   highest occupied point when no exact/greater point is occupied).
5. **Maximal use:** a player must play as many dice as legally possible. If
   either die alone is playable but not both, and only one order works, that
   order is forced. If only one die can be played, it must be the **larger** one
   when possible.

**Implementation:** recursive enumeration of dice applications, collecting
distinct *resulting positions* (dedup by Position ID). Then filter to the set
using the maximum number of dice (per rule 5). Returns the legal distinct moves;
empty (dance) is a valid outcome.

### `make(board, move) -> Board` / `unmake`
For search and self-play, expose apply/undo. Since a *Move* already carries the
resulting board, `make` can just return it; `unmake` restores the prior board
(store on a stack for tree search).

### Game termination & result
- A game ends when a side bears off all 15 checkers.
- Result magnitude (cubeless): **1** (single), **2** (gammon — loser bore off 0),
  **3** (backgammon — loser bore off 0 *and* has a checker in the winner's home
  or on the bar). Sign = winner. This maps directly to the NN output (§6).

### Validation ("perft" analog)
- For a fixed board + fixed dice, assert the exact set/count of legal resulting
  positions against hand-verified fixtures (doubles, bar re-entry, forced
  larger-die, bear-off exact/overflow, no-legal-move dance).
- Property tests (`proptest`): random legal games never violate invariants and
  always terminate.

---

## 5. Evaluator interface

Everything that "thinks" implements one trait so play/self-play/GUI are agnostic:

```rust
pub trait Evaluator {
    /// Cubeless value of `board` from the mover's perspective.
    /// Returns win-probability-style outputs (see §6). Higher = better for mover.
    fn evaluate(&self, board: &Board) -> Value;
}
```
`Value` = the 5-output probability vector (§6) plus a scalar **equity** derived
from it. Move choice ranks legal moves by the equity of the resulting position
*from the opponent's perspective* (negamax: minimize opponent equity).

- **`RandomEval`** — returns a random `Value`. Purposes: plumbing tests, a
  trivial baseline opponent, and injecting exploration noise early in training.
- **`HceEval`** — hand-crafted (§7). Bootstraps self-play before the net is any
  good and serves as a fixed benchmark opponent.
- **`NnEval`** — runs the exported ONNX net via the `ort` crate (fast, no
  gradients). Used for strong self-play generations and in the GUI.

During *training*, move selection instead calls back into PyTorch (gradients
needed) — see §8. The Rust `NnEval` is for inference-only play.

### Search depth (move selection)
- **0-ply:** rank legal moves by evaluating each resulting position directly.
  Fast; the training default.
- **1-ply:** for each candidate move, average the best-response value over all
  21 distinct dice rolls (weighted 1/36 non-doubles ×2, 1/36 doubles). Stronger,
  ~21× cost. Used for GUI play and stronger benchmarks.
- **2-ply:** apply 1-ply to the top-k candidates only (GNUbg-style). Future.

---

## 6. NN input & output encoding

### Inputs — 198 features (Tesauro / TD-Gammon encoding)
Per point (24 points × 2 players = 48 slots), 4 units each = 192:
- unit 1 = 1 if ≥1 checker on the point
- unit 2 = 1 if ≥2
- unit 3 = 1 if ≥3
- unit 4 = (n − 3) / 2 if n > 3, else 0
Plus:
- bar: mover / opponent count ÷ 2  → 2
- off: mover / opponent borne-off ÷ 15 → 2
- side to move: 2 units (one-hot)  → 2

Total **198**. Encoder implemented in `bgcore::features`, exposed to Python so
training and inference use identical inputs. (Richer feature sets — pip count,
race flag — are a later experiment behind a version flag.)

### Outputs — 5 probabilities (cubeless)
`[P(win), P(win_gammon), P(win_backgammon), P(lose_gammon), P(lose_backgammon)]`
with `P(lose) = 1 − P(win)`. Cubeless **equity**:
```
equity = (P(win) − P(lose))
       + (P(win_gammon) − P(lose_gammon))
       + (P(win_bg)    − P(lose_bg))
```
This mirrors the game-result magnitudes in §4 (single/gammon/backgammon).

---

## 7. Hand-crafted evaluator (HCE)

A transparent, tunable heuristic. Weighted sum of features, squashed to a
value vector. Features:
- **Pip count** differential (race).
- **Checkers on the bar** (mover penalized, opponent rewarded).
- **Home-board points made** (1–6 points closed; prime strength / consecutive
  blocks).
- **Blots** exposed, weighted by direct/indirect shot count against them.
- **Anchors** (own points in opponent's home board) — defensive value.
- **Back checkers** (on the 24/23 points) — mobility / trap risk.
- **Borne-off** count.
- **Race vs contact** switch: once positions disengage, weight pip count
  heavily and ignore contact terms.

HCE need not be world-class — it must be *better than random* and *stable* so it
can (a) seed early self-play move choice and (b) act as a fixed Elo rung.

---

## 8. Training — TD(λ) self-play

Classic TD-Gammon. The net learns purely from playing itself; no external data.

### Model (`trainer/model.py`)
- Input 198 → hidden (start 80–128 units) → output 5.
- v1: one or two hidden layers, `tanh`/`ReLU` hidden, `sigmoid` outputs
  (probabilities). Keep it small — small nets train fast and TD-Gammon reached
  strong play with ~1 hidden layer of 40–80 units.

### Self-play move selection during training
For the position to move:
1. Rust `genmoves` returns all legal resulting boards.
2. Rust `features` encodes each into a 198-vector; batch to PyTorch.
3. PyTorch evaluates the batch; pick the move minimizing the opponent's equity
   (0-ply). Add ε-greedy / softmax exploration early (or mix in HCE moves) so
   self-play doesn't collapse into a narrow style.

### TD(λ) update
Along each self-play trajectory `s₀ … s_T`:
```
δ_t = V(s_{t+1}) − V(s_t)                    # V(s_T) is the terminal result vector
w ← w + α · δ_t · e_t                         # per output component
e_t = γλ · e_{t−1} + ∇_w V(s_t)               # eligibility trace
```
- γ = 1 (undiscounted, episodic), λ tunable (~0.7 start).
- Terminal target = the actual game result vector (win/gammon/backgammon
  one-hot-ish from §4/§6).
- Implement traces with per-parameter accumulators, or use the equivalent
  forward-view / n-step target if simpler with autograd. Both are acceptable;
  document which.

### Bootstrapping schedule
1. **Gen 0:** movers = HCE (or ε-heavy net) so early games are non-random.
2. Train net on these trajectories; as it strengthens, hand move selection to
   the net (anneal ε).
3. Periodically **freeze** a checkpoint as a benchmark opponent (Elo ladder).

### Throughput
- Move selection dominated by batched child evaluation. Rust generates children
  fast; PyTorch batches. Later generations run inference via ONNX inside Rust
  self-play (no Python in the hot loop) to generate trajectories, then update in
  PyTorch — the reason for the ONNX round-trip in §9.

### Checkpointing & config
- Save `.pt` every N games with optimizer + trace state, RNG seeds, and a config
  hash (net size, λ, α, features version). Reproducibility is a first-class
  requirement.

---

## 9. Inference & ONNX

Two inference paths, one training path:
- **Training:** PyTorch with gradients (§8).
- **Play & fast self-play:** export the frozen net to **ONNX**
  (`trainer/export_onnx.py`), run via **ONNX Runtime** — `onnxruntime` in the
  Python GUI, and the `ort` crate inside `bgcore::eval::nn` so Rust self-play can
  use the strong net at native speed with no Python in the loop.
- **Future mobile:** same ONNX file via ONNX Runtime Mobile behind the shared
  engine API. No retraining, no re-encoding.

Verify parity: exported ONNX output must match PyTorch output within tolerance
on a fixture set of positions (regression test).

---

## 10. GUI (desktop, v1)

**Framework: PySide6 (Qt).** Rationale: Python is already in the stack, so the
GUI calls the PyO3 engine and `onnxruntime` directly with no extra service
layer; native desktop feel on Windows; fast to build. (A web/React front end is
the alternative if browser reach matters sooner — it needs a thin API server and
is the more natural base for mobile, but it's more moving parts for v1.)

### Features
- Render board, checkers, bar, off-tray, pip counts.
- Roll dice (animated) / accept a given roll.
- Legal-move entry: click-drag a checker; illegal drops rejected using the same
  `genmoves` set the engine uses. Highlight legal destinations.
- Modes: **Human vs Engine**, **Engine vs Engine** (watch), plus a hidden
  self-play/debug view.
- **Hint / analysis:** show the engine's top moves with equities (1-ply).
- Engine strength selector: random / HCE / NN-checkpoint, and search depth.

Engine access is behind a small `EngineApi` Python facade so a future front end
swaps the view without touching engine logic.

---

## 11. Future: doubling cube & match play
Deferred but designed-for. The 5-output cubeless probabilities are exactly what
cube decisions need. Later work: cubeful equity from cubeless probabilities
(Janowski's formula), take/drop/double thresholds, match equity table, Crawford.
Keeping v1 cubeless avoids blocking on this while leaving the door open.

---

## 12. Milestones

| # | Deliverable | Done when |
|---|-------------|-----------|
| **M0** | Rust scaffold: `Board`, dice, invariants | builds; invariant tests pass |
| **M1** | `genmoves` / make / unmake | perft-style fixtures + proptests pass |
| **M2** | Game loop + `RandomEval` + `HceEval`; engine-vs-engine in terminal | full games run to a result; HCE beats random ≫50% |
| **M3** | PyO3 bindings + Rust `features` encoder exposed to Python | Python drives a full game; encoder parity test |
| **M4** | PyTorch net + TD(λ) self-play loop + checkpoints + benchmark | net beats random ~100%, beats HCE over training |
| **M5** | ONNX export + `ort` inference in Rust; 1-ply search | ONNX/PyTorch parity test; NN self-play runs in Rust |
| **M6** | PySide6 GUI: human vs engine, dice, hints | a human can play a full game vs the NN with hints |
| **M7** | Strength tuning: 2-ply, rollouts, richer features; mobile packaging spike | Elo ladder improves; ONNX runs under ORT-Mobile |

---

## 13. Testing & quality

- **Rules:** fixture + property tests (§4). This is the highest-value test
  surface — a move-gen bug silently corrupts all training.
- **Encoder parity:** Rust `features` vs a reference decode; identical for
  random positions.
- **ONNX parity:** exported net vs PyTorch within tolerance.
- **Strength ladder** (`trainer/benchmark.py`): each checkpoint plays N games vs
  random, HCE, and prior checkpoints; track win rate / Elo over time. This is
  the real measure of progress.
- **Determinism:** seeded RNG reproduces self-play games and training runs.

---

## 14. Open decisions (pick as we go)
- Hidden layer size / count (start 1×80, tune).
- λ, α schedules; exploration schedule (ε-greedy vs softmax vs HCE-mix).
- Position ID format (custom vs GNUbg-compatible).
- Whether to add pip/race engineered features to the 198 baseline.
- GUI: stay PySide6, or invest early in a web front end for mobile reach.
