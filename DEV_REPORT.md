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
| 17 | **gnubg as external teacher** — is it strong enough? | 📊 0-ply **parity** (useless); 2-ply **clearly stronger** (43.9%, z −3.86) → valid teacher | tool |
| 18 | **Distilling gnubg 2-ply labels** | ✅ **The win** — breaks the self-distillation fixed point; **+0.082 PPG at 1-ply**, z +13.70 | **v1.10.0** |
| 19 | Warm-start vs scratch, at scale | 🔄 **Reversal** — `--init` helps at 60K, *hurts* at 2.5M+ (53.1% vs 52.4%) | recipe |
| 20 | Capacity revisited (512×256, 2.7× params) | 🔄 **Reversal** — depth/width pays again *with a stronger teacher*; but same-speed net catches it on data | no |
| 21 | Label volume scaling (500K → 2.5M → 17.5M) | ➖ **Flattening** — 5× gave a clear win, a further 7× was unresolved (z +1.91) | yes |
| 22 | **DAgger harvest** (positions from games vs gnubg) | ❌ **No gain** — 5M harvested rows on top of 17.5M changed nothing (z −0.64) | infra |
| 23 | Search depth against gnubg (0 / 1 / 2 ply) | ⚠️ **RETRACTED** — the harness never varied depth; see §23 and §25 | no |
| 24 | gnubg **rollouts** as a teacher | ❌ **Rejected on cost** — ~0.3 s/trial/position and no machine-readable output | no |
| 25 | Equal-depth rematch vs gnubg (harness fixed) | ✅ **Parity** — 49.1% money (3,000 games), 48.5% of 400 7-point matches | tool |
| 26 | Cube graded decision-by-decision vs gnubg | ✅ **Win** — match cube 45.1 → 5.8 mEMG; money 2.1 mEMG | **unreleased** |
| 27 | Teacher depth: gnubg 3-ply and 4-ply labels | ❌ **Closed** — mean \|Δequity\| 0.009 / 0.007 vs 2-ply | no |
| 28 | Search *width* (12 candidates vs 4) | ❌ **Null** — 50.5%, z +0.47, 2,600 games | no |
| 29 | App default on a big box (rollout vs 2-ply) | ➖ **Equal strength, ~50× slower** — 51.0%, z +0.38 | recipe |

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

## 17. gnubg as an external teacher — 📊 the pre-check that unlocked §18
Experiments 8–15 all landed on the same fixed point: **a student distilled from its own
engine cannot exceed it.** Rollouts, untruncated rollouts, 1-ply and 2-ply search values,
richer features — every axis converged to champion parity. The only escape is a teacher
from *outside* the loop, and gnubg is the obvious candidate. But "world-class" is not the
same as "stronger than us at the setting we can afford", so it had to be measured first.

Two head-to-heads (mirrored dice, cubeless, our net at 0-ply):

| teacher candidate | our champion scores | verdict |
|---|---|---|
| gnubg **0-ply** | **49.0%** (z −0.28, 200 games) | **parity — useless as a teacher** |
| gnubg **2-ply** | **43.9%** (z −3.86, 1000 games) | **clearly stronger — valid teacher** |

That distinction is the whole experiment. Labelling 60K positions with a parity teacher
would have reproduced our own ceiling exactly, and the result would have looked like yet
another "◐ parity" row in this table.

**Two gnubg behaviours cost hours and are worth recording:**
- **`set evaluation chequerplay eval plies N` does not affect the `eval` command.** Verified
  by byte-comparing output at N = 0, 2 and 3: identical. gnubg *always* computes the full
  static / 1-ply / 2-ply table and always ends with a `N-ply cubeless equity` summary. Its
  strength in a bridge like ours is therefore decided **entirely by which row you parse** —
  and the corollary is that the deep evaluation was being computed and thrown away all along,
  so reading it costs nothing (~0.04 s per position either way).
- **Positions gnubg answers from its bearoff databases print `static:` and no deeper rows.**
  A parser that counts ` 2 ply:` lines silently comes up short, blocks, and times out. In the
  head-to-head harness that *discarded whole games* — and the survivors were the fast ones,
  i.e. a biased sample presented as a result. The fix is to delimit on the always-present
  summary line and take the deepest row available; those static rows are database-exact, so
  nothing is lost.

## 18. Distilling gnubg 2-ply — ✅ the win (v1.10.0)
Label positions with gnubg's 2-ply evaluation (the five nested probabilities, not just
equity) and train on them directly. **~2,600 positions/sec** across 60 gnubg processes, so
2.5M labels cost ~15 minutes — against **6h15m** for the 30K 1-ply-rollout labels this
replaces, which had come back at parity.

Progress against the v1.9.0 champion, all native, mirrored dice, verified at 1-ply:

| labels | recipe | 0-ply | 1-ply |
|---|---|---|---|
| 500K | warm-start, 256×128 | 51.5% (z +4.16) PPG +0.026 | 50.9% (z +3.73) PPG +0.008 |
| 2.5M | warm-start, 256×128 | 52.4% (z +6.80) PPG +0.061 | 52.3% (z +9.26) PPG +0.047 |
| 2.5M | **scratch**, 512×256 | 54.3% (z +12.18) PPG +0.114 | 53.4% (z +13.52) PPG +0.077 |
| 17.5M | scratch, 512×256 | 54.8% (z +13.51) PPG +0.131 | 53.6% (z +14.44) PPG +0.086 |
| **22.5M** | **scratch, 256×128** | **53.9% (z +11.17) PPG +0.105** | **53.4% (z +13.70) PPG +0.082** |

**The shipped net is the last row**, and the reason is the third column of §20: it is a
256×128 — *the same architecture and inference cost as the champion it replaces* — yet it is
statistically indistinguishable from the 2.7×-larger net (49.5%, z −1.50, 20,000 games).

**It also moved the absolute needle**, which is the number that matters (4,000 games each):

| | vs gnubg 2-ply |
|---|---|
| v1.9.0 champion | 43.4%, PPG −0.173 |
| v1.10.0 net | **46.1%**, PPG −0.113 |

~38% of the deficit to the teacher closed.

> **Correction (2026-08-05).** This section originally continued: *"gnubg 2-ply beats the
> champion by ~6 points, so a perfect distillation would score ~56% against it. We capture
> roughly 40% of that headroom."* Both figures come from the retracted §23 harness and
> compare **our 0-ply against gnubg's 2-ply** — they measure a depth handicap, not the
> quality of the distillation. Measured like for like (§25) the distilled net is at
> **parity** with its teacher: 49.1% over 3,000 money games, 48.5% of 400 seven-point
> matches. There is no 60% of remaining headroom. Distillation from gnubg 2-ply is
> **finished**, not partially exploited, and any plan premised on "we still have most of the
> gap to close" is planning against an artefact.

## 19. Warm-start vs scratch — 🔄 the advice inverts with scale
`train_rollout.py` gained `--init` to warm-start from the champion, because training from
scratch on 60K labels reached only **41%** against it — a number that measures *data volume*,
not teacher quality. With the warm start the same data reached parity.

At 2.5M the comparison reverses: **scratch 53.1% vs warm-started 52.4%**, identical data and
architecture. The warm start anchors the net to the champion's weaker evaluation, and once
there is enough teacher signal that anchor is a handicap rather than a head start. **Lesson:
"initialise from the incumbent" is not a fixed truth — it is a statement about the ratio of
teacher data to incumbent knowledge, and it flips as the data grows.**

## 20. Capacity revisited — 🔄 it pays again, but data buys it back
The CHANGELOG's standing verdict was that extra depth gave "clearly diminishing returns".
That was measured **against self-play labels — a teacher no stronger than the student**.
With gnubg as teacher, at fixed data (2.5M), 2.7× the parameters bought **+1.2 points**
(53.1% → 54.3% at 0-ply): the bigger net has something it cannot already produce to learn
*from*.

But capacity is not free, and the app is not a benchmark:
- **1.58×** the cost per static evaluation (measured on an idle box; a contended measurement
  had said 1.96× — re-measure timings when the machine is busy).
- **73 → 37 games/sec** in `nn_bench`, i.e. ~2× on whole games.
- At **equal movetime** in the rollout engine — the app's default opponent, where the
  evaluator *is* the playout policy — the advantage could not be demonstrated at all:
  **52.0%, z +0.80** over 400 games.

And then more data on the *small* net caught it (§18, last row): same speed, indistinguishable
strength. **Lesson: price capacity in the currency the application spends — time per move —
not in evaluations. A gain that needs 2× the compute must beat what the same compute buys
elsewhere.**

## 21. Label volume — ➖ flattening
500K → 2.5M was a clear win. 2.5M → 17.5M (7×, using 15M labels generated on a second
machine) moved PPG against the champion only **+0.114 → +0.131**, and head-to-head between
those two nets was **z +1.91 — unresolved**. Validation loss was still falling, so the nets
were still fitting the teacher better; it simply stopped translating into games.

**Lesson: more of the same distribution has a knee, and validation loss will not tell you
where it is** — only head-to-head games will. (A caution on reading those games early: this
run's 1-ply gate read 54.8% at 7,000 games and finished at 53.6% over 40,000. Partial
head-to-heads wander by more than the effects being measured.)

## 22. DAgger harvest — ❌ no measurable gain
Every label to this point came from **champion self-play positions**, so all the scaling above
varied *volume* while holding *distribution* fixed. The harvester plays our net against gnubg
and keeps the evaluations gnubg computes anyway — ~500 labelled positions per game, in the
distribution where our net's errors are actually punished, at ~2,600 rows/sec. The DAgger
argument is textbook: label the states the *student* visits, using the *expert*.

It first looked promising. The 22.5M net (17.5M self-play + 5M harvested) beat the same
architecture trained on 2.5M by **+0.050 PPG (z +5.37)**, where 7× more *self-play* data had
given the 512×256 only an unresolved +0.019 — which reads as the distribution doing work.

**The control says otherwise.** Training the identical architecture and recipe on the 17.5M
*without* the harvest and playing the two directly: **49.8%, z −0.64, 20,000 games — no
difference.** The harvest arm had 28% *more* data and still did not win, so the earlier
+0.050 was volume, not distribution.

The honest reading is that this is §21 again: past ~17.5M, additional labels stop paying
**whatever distribution they come from**. That does not refute DAgger — it says we are on the
flat part of the curve, where 5M more rows of anything is lost in the noise. A real test would
need the harvest to *replace* rather than *supplement* self-play data at a volume where the
curve still has slope.

**Lesson: when a promising result arrives confounded, the control is not optional.** The
confound here (28% more rows) favoured the hypothesis, and the hypothesis still lost.

**A subtlety worth keeping** for whenever this is retried: the free child evaluations only
cover states where *we* are next to move. The states our net actually *evaluates* are the
children of its own decisions — positions with the opponent on roll — and those never appear
among them. Harvesting only the free ones would systematically miss the half the student
spends its capacity on.

**A subtlety worth recording:** the free child evaluations only cover states where *we* are
next to move. The states our net actually *evaluates* are the children of its own decisions —
positions with the opponent on roll — and those never appear among them. Harvesting only the
free ones would systematically miss the half the student spends its capacity on.

## 23. Search depth against gnubg — ⚠️ RETRACTED (2026-08-05)

**This experiment measured our 0-ply evaluation three times and called the agreement a
null result.** `gnubg_h2h.py` built its net as `bgcore.Neural(path, 0, 0)` with the plies
hardcoded. Search depth lives in that constructor, not in which scoring function is called:
`scores()` on a net built at 0 ply returns static values however it is invoked. So `--our-ply`
only chose between `our_best` and `our_best_searched`, and both then evaluated statically.
`--our-candidates` was parsed and never reached the engine at all.

The original text is preserved below, because the way it argued is the lesson. What it
reported:

> our net scores the same against gnubg 2-ply at every depth: **46.1% / 46.3% / 46.3%** at
> 0 / 1 / 2 ply, 3,000 games each, identical dice.
>
> This is a real null, not a broken harness: over 315 sampled positions, 0-ply and 1-ply
> choose **different moves 29.5%** of the time. Nor is it the candidate filter — 1-ply is
> full width, with no pruning and no mixed static/searched comparison, and scores the same
> as pruned 2-ply. An independent check agrees: `nn_bench` puts 1-ply over 0-ply at
> **51.5% ±6.9** for the champion.
>
> **Lesson: as the evaluator improves, search buys less** ... Evaluation is the binding
> constraint; search tuning was parked on this evidence.

Every corroboration there is about the *engines* — that 0-ply and 1-ply pick different moves
is true and irrelevant, because the harness never asked them to. The three numbers agreeing
to within 0.2 points across 9,000 games was not evidence of a null; **it was the bug's
signature**, and it was read as the finding.

Re-measured with the constructor fixed (§25), our true 2-ply scores **49.1%** against gnubg
2-ply where 0-ply scores 46.1%. Search is worth roughly **+3 points, ~21 Elo** — not nothing.
The conclusion this experiment was cited for ("evaluation is the binding constraint, park
search tuning") does not follow from it. Search *width* is separately null (§28); search
*depth* is not.

## 24. gnubg rollouts as a teacher — ❌ rejected on cost
If gnubg's 2-ply sets the ceiling, its rollouts would raise it. Measured before building
anything: **14s for 36 trials, 51s for 144 trials — ~0.3 s per trial per position**, barely
changed by dropping to 0-ply chequerplay with variance reduction and quasi-random off (11s).
That is ~10⁵× the cost of a 2-ply label. Worse, `gnubg-cli -t -q` prints "Rollout done.
Printing final results." and nothing parseable follows.

**gnubg's 2-ply `eval` is the strongest teacher it will practically give us**, which fixes the
distillation ceiling at ~parity with gnubg 2-ply. **Lesson: price the teacher before designing
around it** — the same lesson as §14, learned again on the other side of the fence.

> **Confirmed (2026-08-05).** The predicted ceiling is where we landed: §25 measures parity
> with gnubg 2-ply. This section's *cost* verdict on rollouts still stands, but note that its
> second objection — "no machine-readable output" — was specific to the CLI. gnubg's embedded
> **Python API** returns evaluations as data (that is how §27 and the Kazaross MET were
> obtained), so the parsing barrier has since fallen for everything it was invoked against.
> Rollouts remain the one teacher axis §27 does **not** close: 3-ply and 4-ply agree with
> 2-ply because all three bottom out in gnubg's same net, whereas a rollout bottoms out in
> game outcomes. If any teacher work restarts, it starts there — and it starts by re-pricing
> `gnubg.rolloutcontext()`, not by trusting this section's CLI-era estimate.

## 25. Equal-depth rematch vs gnubg — ✅ parity
With the §23 harness bug fixed, the comparison the project thought it had been running for
months was run for the first time. Two harnesses, two formats:

| | our net | gnubg | result |
|---|---|---|---|
| Money games, cubeless, mirrored dice | 2-ply | 2-ply | **49.1%**, PPG −0.022, z −0.95, 3,000 games |
| 7-point matches, cube + Crawford | 2-ply | 2-ply | **48.5%** of matches, z −0.60, 400 matches |

**We are level with gnubg at equal depth**, not 4 points behind. Both figures sit just under
50% and neither is significant, so "level, possibly a hair behind" is the defensible reading —
two independent harnesses agreeing at 48.5% and 49.1% makes a real gap of more than ~2 points
unlikely.

This is the number that closes §18: a net distilled from gnubg 2-ply has reached gnubg 2-ply.

## 26. Grading the cube decision-by-decision — ✅ win
`cube.py` and `match.py` had never faced an opponent: `gnubg_h2h.py` is cubeless money play,
so the doubling logic was the largest untested surface in the project.

Playing matches is the obvious test and the wrong one — a match is ~15 games of large variance
holding a handful of cube decisions, and the result confounds cube errors with checker errors.
`grade_cube.py` instead has gnubg analyse each cube decision directly, at any score, pricing
every alternative, so a disagreement costs a measurable number of millipoints. **2,250
decisions/sec across 60 processes**; 30,000 graded in 13 seconds.

| | agreement | mean error | blunders >80 mEMG |
|---|---|---|---|
| Money | 87.8% | 2.13 mEMG | 0.67% |
| Match, before | 73.1% | 45.05 mEMG | 12.21% |
| Match, after | **80.6%** | **5.77 mEMG** | **2.71%** |

The match model had no cube model at all: `_mwc_from_game` played the game out at the current
cube value, so *keeping* the cube was worth nothing and doubling almost always looked better
(289 wrong doubles against 2 wrong holds at 2-away/4-away). Replaced with Janowski's cubeful
equity in MWC space, with score-dependent take and cash barriers. Held out on an independent
24,424-decision sample: 5.59 mEMG.

Efficiency `x` is now **fitted, not assumed**, and the two modes differ: **0.55 for match play
against 0.68 for money**, because the score truncates the cube's value. Money's own optimum
measures 0.66 — 0.01 mEMG from what we ship, so it was deliberately left alone rather than
chased into the noise.

**The diagnostic that found it:** sweeping `x` produced *bit-identical* results from 0.40 to
0.95. A parameter that cannot change the answer is not a parameter.

## 27. Teacher depth: gnubg 3-ply and 4-ply — ❌ closed
Before committing to a relabelling run, measure whether the deeper teacher says anything
different. 2,400 real positions from games against gnubg:

| | mean \|Δwin\| | mean \|Δequity\| | share >0.02 equity |
|---|---|---|---|
| 2-ply vs 3-ply | 0.0036 | 0.0093 | 13.6% |
| 2-ply vs 4-ply | 0.0025 | **0.0071** | **7.8%** |

**4-ply is closer to 2-ply than 3-ply is** — the even/odd ply parity effect, since 2 and 4
share the same side to move at the leaf. So most of the already-small 3-ply difference is a
parity artefact rather than deeper insight. Cost was never the obstacle (3-ply is ~10× a
2-ply label, not the ~10⁵× that killed rollouts); the obstacle is that it has nothing to say.

Note what this does **not** close: all of these bottom out in gnubg's same network. A rollout
bottoms out in game outcomes, which is a different signal (see §24's addendum).

*Incidental:* the run appeared to cost ~6 s/position until `set display off` — the ASCII board
echoed on every `set board` was the bottleneck, not the search. 2,400 positions then took
under 3 minutes.

## 28. Search width — ❌ null
The ladder's search rungs are nearly flat (0-ply 1458, 1-ply 1465, 2-ply 1469), and
`our_best_searched` documents a mechanism that would explain it: a pruned move keeps its
static value in `scores` and can outrank a searched move's deep value. Measured directly over
3,153 positions: mean width actually searched **3.81**, and the chosen move **was never
searched 3.3% of the time**. The mechanism is real.

It is also worthless. Widening the window three-fold — 12 candidates against the shipped 4,
both at 2-ply, 2,600 games — scores **50.5%, z +0.47, PPG +0.018**: about +3.5 Elo with a
confidence interval spanning −10 to +17. The static-best move is usually fine.

**Search depth pays (§23), search width does not.**

## 29. The app's default opponent on a big box — ➖ equal, and much slower
`ladder.py` builds its rollout with `rollout_threads=8` to approximate a laptop, but the app
hands it the whole machine — so the ladder's Rollout **1444** against Neural 2-ply's **1469**
never described what the app actually runs on a 128-core box. Re-measured with the rollout
given every core, and games played **serially**, because a fixed movetime means contention
would starve the very engine under test:

**Rollout wins 51.0%** (z +0.38, +7 Elo, 341 games) — level, not 25 Elo weaker.

The finding is the cost, not the strength: **53.1 s/game**, i.e. 800 ms/move against 2-ply's
near-instant reply, *for the same strength*. Users on big machines wait ~50× longer per move
to face an equally strong opponent. Raising `ROLLOUT_MIN_CORES` is a one-line change that
costs nothing. Caveat: at the current 32-core threshold the rollout gets a quarter of these
trials, so this argues for raising the bar, not for abandoning rollouts.

---

## Infrastructure that unblocked the above
- **A memory bug was silently capping dataset size.** `train_rollout.py` built its feature
  matrix with `np.stack` over a list comprehension, where every row is 198 *boxed* Python
  floats: measured **1.61 GB per 200K positions — 182 GB at 22.5M**, against 128 GB of RAM.
  The 22.5M run died during loading with an empty log; the 17.5M run had only survived by
  paging. Filling a preallocated array holds peak at ~19 GB, so 50M+ is now reachable. Every
  "how much data can we use" question before this was answering the wrong question.
- **Free labels.** gnubg comparison runs discard ~500 labelled positions per game; roughly
  **7.6M** were binned in a single day of benchmarking before `--dump-labels` was added to the
  h2h and bench harnesses.
- **`merge_npz.py` could not merge a merged file** — it wrote `net=None`, which numpy stores
  as an object array that will not load without `allow_pickle`.

---

## Cross-cutting lessons

- **Verify at the ply the app plays.** 0-ply (static-eval) gains repeatedly shrank or
  vanished under 1-ply search. The head-to-head that matters is at the search depth used in
  real play.
- **Suspiciously clean agreement is a bug signature, not a finding.** §23 got 46.1 / 46.3 /
  46.3% across three depths and 9,000 games and published it as a null; the three runs were
  the same measurement, because the depth never reached the engine. §26 found the same shape
  in a parameter sweep that returned *bit-identical* numbers from x=0.40 to 0.95. **When a
  knob you turned changes nothing, first prove the knob is connected** — the null hypothesis
  for a flat response is a disconnected wire, not a real invariance.
- **A control that varies the wrong thing corroborates nothing.** §23 defended its null with
  "0-ply and 1-ply choose different moves 29.5% of the time". True of the engines, irrelevant
  to the harness — which never asked them to. Check that a control exercises the same code
  path as the result it is defending.
- **Silence is not success.** A run producing no output for ten minutes was read as "slow but
  working" and used to clear a diagnosis; it was crashing on startup. Verify a harness by an
  output it *must* produce, not by absence of an error.
- **Price the teacher, then check it has something to say.** §24 priced rollouts and rejected
  them on cost. §27 found the cheaper upgrade — 3-ply, 4-ply — affordable but *empty*: they
  bottom out in the same network, so they repeat it. Affordable and informative are separate
  questions and both need measuring.
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
- **v1.9.0** — 2-ply search-distillation net (experiment 14). Kept in the app as the
  **"Neural classic"** opponent — an easier rung between HCE and the current net.
- **v1.10.0** — **gnubg-distilled net** (experiments 17–18), current live net. A 198→256→128
  body with 12 routed heads, trained from scratch on **22.5M gnubg-2-ply-labelled positions**
  in under an hour. Beats the v1.9.0 champion **53.9% at 0-ply** and **53.4% at 1-ply**
  (z +13.70, 40,000 games) at **identical inference cost**, and closes ~38% of the gap to
  gnubg 2-ply itself (43.4% → 46.1%).
