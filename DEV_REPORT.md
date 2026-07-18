# Development Report — network & architecture experiments

A running, honest log of the **net-building experiments**: what we tried, whether it
worked, and *why*. This is the "what did we learn" companion to
[`CHANGELOG.md`](CHANGELOG.md) (which tracks shipped app/engine changes newest-first).

The engine is a Rust core (validated move-gen, ONNX/tract inference, n-ply search,
Monte-Carlo rollouts) with a PyTorch-trained value net. The net learns from self-play.
Everything below is about making that net *stronger*.

## Scorecard

| # | Experiment | Verdict | Shipped |
|---|---|---|---|
| 1 | Depth over width (2 hidden layers + squared-ReLU) | ✅ **Win** — broke the single-layer ceiling | yes |
| 2 | More depth (3rd layer, 2M games) | ➖ Marginal — clear diminishing returns | yes |
| 3 | Product-pool / pairwise-multiply body | ❌ **Failed** — plateaued, washed out | reverted |
| 4 | Phase split (separate contact / race nets) | ◐ Partial — race net helps 0-ply, not rollouts | as optional opponents |
| 5 | Pip-count output buckets (SF-NNUE style, 8 heads) | ✅ **Win** — beat champion 52.6% @0-ply | **v1.7.0** |
| 6 | Class-aware routing (race/crashed/contact, 12 heads) | ◐ Routing neutral; **LR-decay recipe** drove the gain | **v1.8.0** |
| 7 | Richer input features (14 strategic, 198→212) | ❌ **Failed** — features aren't the lever | no |
| 8 | Rollout-labeled supervised training | ⏳ **In progress** | — |

---

## 1. Depth over width — ✅ win
A wider *single* hidden layer merely tied the old net. Going to **198→256→128** with
**squared-ReLU** beat the previous 128-net **58.8%** head-to-head and cut the self-play
gammon rate 70.7%→60.2%. **Lesson:** depth, not width, broke the strength ceiling.

## 2. More depth — ➖ diminishing returns
Adding a third hidden layer (198→256→128→128) over **2M games** beat the 256-128 net
~53.6% @0-ply but held only ~52% at 1-ply, and tied it vs HCE. The edge appeared in the
first ~150k games; the remaining 1.85M added nothing measurable. **Lesson:** piling on
depth had run its course — the next lever had to come from elsewhere.

## 3. Product-pool / pairwise-multiply — ❌ failed
Tried a body with multiplicative feature interactions. Plateaued ~54% vs champion and
washed out with more training. Reverted. **Lesson:** fancy interaction layers didn't buy
anything a plain MLP couldn't already learn.

## 4. Phase split (contact / race nets) — ◐ partial
Two separate nets routed by `Board::no_contact()`. The **race net was a real 0-ply win**
(+0.25 PPG, z 4.9) and shipped as selectable "Neural phase" opponents. But it did **not**
help the rollout engine (+0.075 PPG, z 0.62), so it wasn't made the default. **Lesson:**
separate per-phase nets starve on data; a static-eval win doesn't imply a *searched* win.

## 5. Pip-count output buckets — ✅ win (v1.7.0)
Stockfish-NNUE-style: **one shared body, 8 output heads** selected by total pip count
(perspective-invariant, calibrated to even octiles). The shared body sees every position,
so no data starvation. Beat the 256-128-128 champion **52.6% @0-ply** (z 3.65) and held at
1-ply. **Shipped v1.7.0.** **Lesson:** specialize the *head*, share the *body*.

## 6. Class-aware routing (race / crashed / contact) — ◐ recipe win, not routing win (v1.8.0)
Generalized the buckets to gnubg's classification: route by **race / crashed / contact**
(gnubg's exact "crashed" definition — ≤6 checkers not buried on the 1/2 points) then pip
sub-buckets = **12 heads**. Built with a single source of truth in Rust
(`Board::route_bucket`).

The honest result:
- **At equal training (1M games, constant LR): a wash** — 50.6% vs the champion (z 0.69).
  The routing *itself* added nothing.
- The gain came from a **linear LR-decay tail** (1e-3→1e-4 over 3M games): 0-ply rose to
  **55.8% (z 6.4)**.
- **But it barely survived search: 1-ply only ~52.2% (z 1.27)** — the static-eval gain
  largely washes out once both sides look ahead.

Shipped as **v1.8.0** (a genuinely stronger net, and no worse under search), but labelled
honestly: a **training-recipe win, not a routing win**. **Lessons:** (a) separate the
*lever* from the *recipe* — the LR schedule, not the crashed/race split, did the work;
(b) **verify at the ply the app actually plays** (1-ply), not just 0-ply.

## 7. Richer input features — ❌ failed
Added **14 computed strategic features** to the raw 198 (198→212): blot-exposure/shot-count
(full combinatorial), blot count, home-board points, made points, rearmost checker,
back-checkers-trapped, pip count — the gnubg-style hand-crafted inputs.

- Fed at the **raw input**: actively **hurt** (31% vs champion at iter 300 — the dense
  scalars diluted the board transform).
- Fed **after the first ReLU** (NNUE-correct: global features can't live in an
  incrementally-updated accumulator): recovered to neutral, then tracked **~5 points
  *behind* the featureless baseline** through the constant-LR phase. Cut early.

**Lesson:** for this net, richer features aren't the lever. A small net *could* benefit
from hand-crafted features (that's why shallow gnubg does), but ours doesn't — pointing
the finger squarely at **training-signal quality**, not what the net sees.

## 8. Rollout-labeled supervised training — ⏳ in progress
The through-line of experiments 6–7 is that **architecture and features have hit a
ceiling**; gnubg's real edge is **rollout-quality training labels** (+ exact bearoff
databases), not its net. We currently train on *game outcomes* — an extremely noisy label
(one game's result for a mid-game position). gnubg trains on *rollouts* — a low-variance
estimate of the true value.

Now building: label a large position set (400k, across all 12 buckets) with the engine's
own truncated rollouts, then supervised-train to a **soft/hard blend** target
`α·rollout_distribution + (1−α)·onehot(game_outcome)` — distillation-from-search anchored
by the unbiased outcome. Rollout labeling measured at ~26 pos/sec, so a large set is an
overnight job. **Verdict pending** (judged at 1-ply).

---

## Cross-cutting lessons

- **Verify at the ply the app plays.** 0-ply (static-eval) gains repeatedly shrank or
  vanished under 1-ply search. The head-to-head that matters is at the search depth used in
  real play.
- **Separate the lever from the recipe.** The "class-aware" gain was really the LR-decay
  schedule. Always A/B one variable at a time.
- **Share the body, specialize the head.** Output bucketing beat separate per-phase nets by
  avoiding data starvation.
- **The ceiling is training signal, not the net.** Routing (neutral) and features (negative)
  both flopped; both point to label quality — hence experiment 8.
- **Bucket population must be calibrated** (even octiles), or heads starve.

## Shipped milestones
- **v1.7.0** — pip-count output-bucketed net (experiment 5).
- **v1.8.0** — class-aware routing net (experiment 6), current live net.
