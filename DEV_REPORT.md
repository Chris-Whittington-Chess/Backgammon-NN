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
| 8 | Rollout-labeled supervised training | ◐ **Parity** — matches champion with ~375× less data | no |
| 9 | Absolute benchmark vs gnubg (0-ply) | 📊 Champion **~43%** — first world-class placement | tool |
| 10 | Rollout-label **bootstrapping loop** | ◐ Round 1 gained: 43% → **45.5%** vs gnubg, then converged | no |
| 11 | Loop past round 1 — data / quantity / α | ◐ **Fixed point** — round 2, 3.9M, α=1.0 all ~parity | no |
| 12 | Untruncated (λ=1) rollout labels | ❌ **Worse** — the ceiling is champion *play*, not truncation | no |
| 13 | Fast rollout engine (wave + step-free move-gen) | ✅ **Engine win** — ~5× labeling; per-position beats wave | infra |
| 14 | n-ply **search distillation** (1-ply → 2-ply) | ✅ **Best net** — 2-ply beats 1-ply; ~52% vs champ at 1-ply, search-robust | **v1.9.0** |
| 15 | Strategic features on **clean** labels (198→212) | ❌ **Failed again** — raw & split both ~parity; features aren't the lever | no |
| 16 | Exact **bear-off EGTB** + wire into eval | ◐ Exact endgame, but **0-ply-neutral** (never flips a 0-ply move) | infra |

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

## 8. Rollout-labeled supervised training — ◐ parity, not a win
The through-line of experiments 6–7 is that **architecture and features have hit a
ceiling**; gnubg's real edge is **rollout-quality training labels** (+ exact bearoff
databases), not its net. We train on *game outcomes* — an extremely noisy label (one
game's result for a mid-game position); gnubg trains on *rollouts* — a low-variance
estimate of the true value.

We labeled **400k positions** (across all 12 buckets, ~26 pos/sec = ~5h) with the engine's
own truncated rollouts, then supervised-trained the class-aware net to a **soft/hard blend**
`α·rollout_distribution + (1−α)·onehot(game_outcome)`.

**Result: parity, not a win.** The best net (α=0.95) scores **49.4% vs the champion over
3,000 games (z −0.66, dead 50/50)** and is *stronger* vs HCE (~94% vs ~89%). Two findings:
- **Trust the rollouts.** Strength rose monotonically with α (0.5/0.75/0.9/0.95 →
  37/43/47/~50%): the low-variance rollout label *is* the signal; the noisy game outcome
  mostly re-adds the variance we're trying to escape, so α≈0.95 (barely anchored) wins.
- **Label quality substitutes for compute.** It reached parity with the 3M-game self-play
  champion from **400k supervised examples (~375× fewer positions)** — minutes of training
  vs ~16h of self-play. The thesis holds; the *ceiling* of this dataset is champion-level,
  though, because the labels use the champion as their rollout leaf. Exceeding it needs
  **stronger labels** (untruncated / 2-ply-leaf rollouts) — the honest next lever.

**Single-round verdict across 6–8:** routing (neutral), features (negative), rollout labels
(parity) — none *singly* beats v1.8.0. But two things followed that changed the picture: an
absolute yardstick (§9), and *iterating* the rollout labels (§10), which does break past the
champion.

## 9. Absolute benchmark — vs gnubg at 0-ply — 📊 the real yardstick
Everything above was measured against our *own* champion or wildbg. Installing **gnubg**
(world-class) and bridging via the GNU Position ID gave the first *absolute* placement. A
direct 0-ply head-to-head — our engine generates the moves, gnubg picks by evaluating each
resulting position, parallelized across 32 gnubg processes — puts **our champion at ~43%
(PPG −0.20)**: gnubg's 0-ply genuinely out-plays ours, but we're *competitive, not
outclassed*. A phase breakdown locates the gap in **contact** (race/bear-off are near-even,
and gnubg's exact bear-off database owns the endgame anyway). **This sets the target: +7
points / +0.2 PPG at 0-ply.** (We first tried a millipoint *error-rate* metric but it was
self-scoring-biased — gnubg can't lose measured against its own eval — so the head-to-head is
the honest measure.)

## 10. Rollout-label bootstrapping loop — ◐ gaining
Experiment 8 capped at champion-level because its labels used the champion as their rollout
leaf. The escape — **expert iteration**, exactly how gnubg itself was trained: *relabel each
round with the improved net*, so the leaf strengthens and the label ceiling rises every round.
- **Round 1:** labeled **2.4M** positions (2M fresh @180 trials + the earlier 400k, all
  11-ply truncated), trained α=0.9. The extra data alone broke experiment-8's parity —
  **beats the champion ~55% @0-ply** (vs 49.4% from 400k), capturing the full
  rollout-over-0-ply gain the smaller set left on the table. Converged and epoch-independent
  (epoch-10 and epoch-60 both ~45.5% vs gnubg).
- **It moved the *absolute* needle:** **43% → 45.5% vs gnubg**, PPG gap roughly halved
  (−0.20 → −0.12). Beating our own lineage *did* translate to gnubg progress — at a
  diminishing per-round rate.
- **Round 1 was the peak.** The loop gained *once* then hit a fixed point (§11). The front
  line is now label *quality* (§12–14), not more rounds.

## 11. The loop's fixed point — data, quantity, and α all ruled out — ◐ parity
Round 1's gain did not repeat. **Round 2** (relabel 1.54M with the round-1 net, α=0.9) landed
at **45.6% vs gnubg (z −1.56)** — dead level with round 1. Suspecting a data-quantity confound
(1.54M vs round-1's 2.4M), we retrained on the **combined 3.9M** pool: still ~parity vs
champion. And **α=1.0** (drop the noisy game-outcome anchor entirely) on that same 3.9M: also
~parity (49.4%). The mechanism is a genuine fixed point — the loop converges when *net static
== its own 11-ply-rollout-bootstrapped-on-static target*; once round 1's net predicts that,
using it as round 2's leaf reproduces the same target and the trainee has caught the teacher.
**More data estimates the same fixed point more precisely; α only trims noise around it.
Neither raises the ceiling — the truncation does.**

## 12. Untruncated (λ=1) rollout labels — ❌ worse, not better
If the truncation-leaf is the ceiling, roll to the *end*: unbiased Monte-Carlo, no leaf. We
generated **1.37M** untruncated labels (the fast engine, §13, made this affordable) and
trained — **~46.8% vs champion, *below* the truncated runs.** The theory was wrong, and
informatively so: **the ceiling was never the truncation — it's the champion's *play*.**
Rolling the weak 0-ply greedy policy ~50 plies to the game's end accumulates its blunders;
truncating at ply 11 and trusting the champion's *static eval* gives a **better** value
estimate (a trained value function beats 40 more plies of its own weak play — bias–variance
favours truncation). This closes the entire "rollout under champion 0-ply play" family:
data, α, truncation depth — every axis lands at champion parity (~45.5% vs gnubg).

## 13. Fast rollout engine — ✅ engine win
To afford the label experiments we rebuilt the rollout hot path. A **batched "wave" engine**
(all playouts in lockstep, one big matmul per ply) gave 2.1× — but revealed the real
bottleneck: rollouts are **move-generation-bound, not inference-bound** (throughput was flat
across batch size, so a GPU is moot). The decisive win was a **step-free playout move
generator** — no per-node `Vec<Step>`, a `Copy` 30-byte board, dice as a count-multiset
instead of per-level heap allocation. Per-position labeling jumped to **~57 pos/sec**, enough
that it now *beats* the wave engine: move-gen is no longer the bottleneck, so the simpler
per-position path (trials parallel across cores) wins. Both paths are proven bit-identical to
the reference by property tests.

## 14. n-ply search distillation — ✅ the win (v1.9.0)
With rollouts exhausted, distil a *stronger* teacher: the champion's own **n-ply search
value**. Cost first — stronger *play inside* rollouts is infeasible (~0.04 pos/sec: a rollout
label is ~1,000 move-decisions, each n-ply decision 250×+ costlier), but distilling the search
*value* is **one search per label**. It needed a new **distribution-returning expectiminimax**
that propagates the principal variation's win/gammon/backgammon split (not just equity, which
the trainer can't use), proven to fold to the exact same equity as the scalar search.
- **1-ply distillation = parity** (~48% vs champion): a TD net's static eval already
  approximates its own one-ply backup, so there's nothing to learn.
- **2-ply distillation is the lever.** At *equal* data the 2-ply pilot beat 1-ply (~45% vs
  ~42.7% mean vs champion); the full **1.37M-label** net (`td_2ply_full`) is the strongest of
  the session: ~50% vs champion at 0-ply, and — crucially — **~52% vs the champion at 1-ply
  (1000 games, PPG +0.058)**. Small (~2%, not significant) but **search-robust**: the first
  improvement all project that does *not* wash out under search (§ cross-cutting). Promoted as
  **v1.9.0**.

*(PUCT/MCTS is the wrong tool here: chance nodes dilute simulations across 21 rolls per ply,
dice reset the tree each turn, and the value net is too accurate for selective deep search to
pay — which is why no strong backgammon engine uses MCTS. Expectiminimax + rollouts is the
paradigm.)*

## 15. Strategic features on clean labels — ❌ failed again
Experiment 7 (features under noisy TD self-play) failed, but that's the *worst* regime for
extra inputs. Retested the 14 strategic features (198→212) on the clean 1-ply distillation
labels — kept as a separate `strategic()` block so the 198 champion still runs and a 212-input
candidate coexists. **Both raw-input injection (~48.9%) and NNUE-style after-first-ReLU
injection (~47.8%) landed at parity with the 198 baseline (~48.3%).** Clean labels rescued raw
from its old 31% collapse but produced no win. **Features are not the lever, in *any* regime —
the raw 198 encoding already captures what they provide given a clean signal.**

## 16. Exact bear-off EGTB — ◐ exact, but 0-ply-neutral
Built the exact one-sided bear-off database by backward DP over all 54,264 home-board configs
(no rollouts — the graph is a DAG in pip order): rolls-to-finish + rolls-to-first-checker
distributions under expected-roll-minimising play, convolved into exact win/gammon race
equities. Wired into the eval (`is_home_race` → table). **But attribution showed it's
0-ply-neutral**: verified active (home-race eval `0.99990` table vs `0.99939` net), yet its
ultra-precise values *never flip a 0-ply move choice* — a strong net already plays the same
bear-off moves. So it adds nothing to 0-ply play; its value is for deeper search, cube
decisions, and exact labels. A within-N-points hybrid extension (exact ≤9, mean+var 9–12) is
queued.

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
  both flopped; both point to label quality — hence experiments 8 and 10.
- **The teacher sets the ceiling.** A distilled net can't exceed its teacher. Our own rollouts
  — however deep — cap at our-net level (proven in §8); to go higher, *iterate* the loop
  (raise the leaf each round, §10) or learn from a stronger teacher (gnubg).
- **Measure absolutely, not just internally.** Beating our own champion ≠ closing the external
  gap. The loop gains 55% vs our lineage but only +2.5 pts vs gnubg — the world-class
  head-to-head (§9) is the honest yardstick.
- **Bucket population must be calibrated** (even octiles), or heads starve.
- **The teacher's *policy* is the ceiling, not the label depth.** Deeper rollouts (λ=1)
  *hurt* when the playout policy is weak — a good truncation-leaf eval beats extra plies of
  weak play. Raise the ceiling by strengthening the *teacher* (search value), not the rollout.
- **Price the search before believing the intuition.** Stronger-*play* rollouts sounded ideal
  but cost ~1000× (rollouts are decision-dense); one-shot search-*value* distillation is the
  affordable form of the same idea. Measure the cost of a label before committing a run.
- **Optimise the actual bottleneck.** The wave engine chased the matmul; the real cost was
  move generation. A micro-benchmark of the wrong stage (36× matmul) predicted a win that
  didn't materialise — profile the whole path, not a slice.
- **A search-robust small gain beats a big 0-ply one.** v1.9.0's edge is tiny at 0-ply but
  *holds* at 1-ply — the opposite of v1.7.0/v1.8.0, whose larger 0-ply gains evaporated under
  search. Distilling *search value* (not static self-play) is what makes the gain survive
  search. Prefer the improvement that lives at the ply the app actually plays.
- **Compare on the *same* benchmark code — a tooling change can masquerade as strength.** A
  gnubg-h2h fix (resolving crawling races by pip count, which the old code scored as non-wins)
  lifted *every* net ~+3 points. It briefly looked like a parity breakthrough; on the corrected
  metric the *champion itself* was already ~46% vs gnubg (not 42.7%), and 2-ply distillation
  added only ~+1. Re-baseline every historical number when the harness changes.

## Shipped milestones
- **v1.7.0** — pip-count output-bucketed net (experiment 5).
- **v1.8.0** — class-aware routing net (experiment 6).
- **v1.9.0** — 2-ply search-distillation net (experiment 14), current live net. Near-parity
  with gnubg at 0-ply; the exact bear-off EGTB (§16) ships alongside for search/cube/labels.
