# Development log

Changes to **Backgammon-NN**, newest first. Engine internals and the app are both
covered; the story of how the *network* was trained is in the
[development report](https://whittingtonchess.com/backgammon-report).

Downloads: [Releases](../../releases). The app is a single self-contained
`Backgammon.exe` — no installer, no Python, no PyTorch.

---

## v1.12.0 — the cube, measured against GNU Backgammon

- **The app now opens on the neural 2-ply engine on every machine.** It used to
  switch to rollouts on anything with 32+ cores, on the untested assumption that
  a big machine would let a rollout "take the lead". Measured with all 128
  logical cores of a Threadripper, the rollout scores **51.0%** against 2-ply —
  level, not ahead — while taking **800 ms per move against a near-instant
  reply**. Every user on a large machine was waiting roughly 50× longer for an
  equally strong opponent. Rollouts remain available in the Opponent box.
- **Match equity now comes from Kazaross XG2**, the rollout-derived published
  table (transcribed from gnubg, which uses it as its own default), replacing the
  cubeless recursion v1.11.0 shipped with.
  The recursion was not merely imprecise, it was **biased**: being cubeless it
  never priced the trailer's ability to double their way back into the match, so
  it flattered whoever was ahead — by **7 to 9 points of match equity** at exactly
  the lopsided scores where cube decisions turn. At 1-away/5-away it read 93.6%
  where the real figure is 84.2%. It passed every test it had, because the tests
  checked the properties it *did* get right (50% at double match point,
  antisymmetry, monotonicity) and none that a wrong-but-consistent table would
  fail.
- **Crawford and post-Crawford are now separate tables.** They were being read as
  one, which understated the trailer badly: at 2-away/1-away the Crawford game is
  worth 32.3% but the game *after* it is worth 48.8%, because the cube comes back
  and a trailer who doubles immediately needs to win only one game. Every equity
  lookup now carries which phase it is in, including the point that the children
  of a Crawford game are post-Crawford.
- **This changes how the app plays a match**: across a sample of scores and
  positions, about a third of cube decisions differ, and facing a double at
  2-away it now takes much more freely — which grading against gnubg confirms is
  right (97.5% of take/pass calls agree with it).
- **Match play now has a cube model at all.** It turned out not to have one:
  `_mwc_from_game` played the game out at the current cube value, so *keeping* the
  cube was worth exactly nothing and doubling almost always looked better. Match
  play now uses Janowski's cubeful equity in match-winning-chance space — the same
  model money play already used, with the take and cash barriers moving with the
  score, which is the point of it. At 2-away a doubled game wins the match, so the
  doubling window collapses; the old model could not represent that.
- **Measured against gnubg**, over 24,424 graded cube decisions:

  | | before | after |
  |---|---|---|
  | agreement on double/no-double | 73.1% | **80.6%** |
  | mean error | 45.05 mEMG | **5.77 mEMG** |
  | blunders (>80 mEMG) | 12.21% | **2.71%** |
  | agreement on take/pass | 97.5% | **99.4%** |

  Confirmed on an independent 24,424-decision sample (5.59 mEMG, 2.67% blunders).
  The worst score improved most: 2-away/4-away went from 138.9 to 5.7 mEMG, and
  wrong doubles there from 289 to 143 against 2 wrong holds.
- **Cube efficiency is now fitted rather than assumed, and the two modes differ.**
  Match play uses **x = 0.55** against money's 0.68 — the cube is genuinely less
  efficient in a match because the score truncates it. Money's own optimum
  measures at 0.66 against the shipped 0.68, a 0.01 mEMG difference, so money is
  deliberately left alone rather than chased into the noise.
- **New tool: `trainer/grade_cube.py`.** gnubg will analyse a cube decision at any
  score and price every alternative, so a disagreement costs a measurable number
  of millipoints. That is a far better instrument than playing matches, which
  would confound cube errors with checker errors and need thousands of matches to
  see anything. It grades 2,250 decisions/sec across 60 processes — 30,000 in 13
  seconds — and caches gnubg's answers so sweeping our own parameters is free.
- The packaged app's selftest now asserts the equity table actually made it into
  the bundle, by checking that its Crawford and post-Crawford answers differ.

## v1.11.0 — a real doubling cube, and match play

- **The cube model is no longer three hard-coded thresholds.** Cube decisions now
  come from **Janowski's cubeful equity** computed from the net's own five
  probabilities — take and cash points *derive* from the position instead of
  being tuned constants, and they reduce to the canonical 25%/20% and 75%/80% at
  a dead/live cube.
- **Gammons finally count.** The app used to hand the cube model a single equity
  number, which discarded exactly what a cube decision turns on. At a fixed 78%
  win probability, a dry race is now a *take* while the same win rate with heavy
  gammon threats is a *pass*. Cube **ownership** is likewise passed through
  rather than assumed centred.
- **"Too good" works properly** — the engine plays on for the gammon instead of
  cashing, which the old equity window could only approximate with a ceiling.
- **Match play.** A *Match* selector offers money play or a match to 1/3/5/7/11.
  In a match the decisions trade in **match-winning chance**, not points, via a
  gammon-aware match equity table; a take that is routine for money can be wrong
  at some scores. The **Crawford rule** is implemented: the single game after
  either side reaches match-point is played with no cube, and the cube returns
  afterwards.
- Note for anyone comparing with the old app: the taker now takes down to a
  mover equity of ~0.573 where the old fixed threshold was 0.50.

## v1.10.0 — taught by GNU Backgammon

- **The net no longer learns from itself.** Every label the project had tried —
  self-play outcomes, its own rollouts, its own 1-ply and 2-ply search values —
  converged on the same fixed point, because a student distilled from its own
  engine cannot exceed it. The new net is trained on **22.5M positions labelled
  with gnubg's 2-ply evaluation**, from scratch, in under an hour.
- **It beats the v1.9.0 champion 53.9% at 0-ply and 53.4% at 1-ply** (z +13.70,
  40,000 mirrored-dice games, PPG +0.082) — and it is the *same* 198→256→128
  architecture, so it costs exactly what its predecessor cost per move. A
  2.7×-larger net scored no better than it (49.5%, z −1.50).
- **Against gnubg itself**, the deficit narrows from 43.4% to **46.1%** (PPG
  −0.173 → −0.113), closing ~38% of the gap.
- **New opponent: "Neural classic"** — the v1.9.0 net, kept as an easier rung.
  The jump from *HCE* to the current net was the biggest step on the ladder.
- Honest limits, all measured: pure distillation cannot pass its teacher (~50%
  would be parity with gnubg 2-ply); labels stopped paying past ~17.5M; the
  DAgger harvest added nothing on top of that; and extra search depth is worth
  nothing against gnubg (0/1/2-ply all score ~46.2%). See
  [`DEV_REPORT.md`](DEV_REPORT.md) §17-24.

## v1.8.0 — class-aware value net (race / crashed / contact)

- The live net now routes its output heads by **gnubg-style position class** —
  race, crashed, contact — with total-pip sub-buckets inside each (**12 heads**),
  instead of pip count alone (8). "Crashed" is gnubg's exact definition: a side
  with at most 6 checkers not buried on its own 1- and 2-points. Routing
  (`Board::route_bucket`) is one source of truth shared by Rust inference and
  Python training; the engine reads the 12-head net and slices the routed head.
- Trained **3M self-play games** with a linear learning-rate decay (1e-3 → 1e-4).
- **Honest strength picture.** It beats the previous 8-bucket net **decisively at
  0-ply (55.8%, z 6.4)** but only **~52% at 1-ply (within noise)** — the
  static-evaluation gain largely washes out once both sides search. So it's a
  clear upgrade as the rollout *leaf* evaluator and no worse under search, but not
  the breakthrough the class split might suggest: at equal training the routing
  was a wash, and the measured gain came from the LR-decay recipe, not the
  race/crashed/contact split itself. Previous champion kept as
  `models/td_bucket_champion.pt`.

## Bucketed net promoted (SF-style pip-count output buckets)

- The live net is now a **Stockfish-NNUE-style output-bucketed** net: one shared
  198→256→128 body, **8 output heads** selected by total pip count, each a
  six-outcome softmax. The body trains on every position; only the selected head
  specialises — no data starvation, unlike separate per-phase nets.
- Trained 1M self-play games; **beats the 256-128-128 champion 52.6%** head-to-head
  at 0-ply (z 3.65 over 4,800 games), 56.8% on the final net with mirrored dice,
  and **holds at 1-ply** (53%, PPG +0.14). vs HCE ~90%, matching the old champion.
- Bucket edges calibrated to the octiles of champion self-play (even population).
  The Rust engine reads the 48-output net and slices the position's pip bucket;
  the live-net parity test is now architecture-agnostic (folded Value). Outgoing
  champion kept as `models/td_256-128-128.pt`.
- Also fixes an engine-move **animation glitch** where the moving checker was drawn
  twice (static at its source and flying), so the source copy vanished on landing.

## Deeper net promoted (256-128-128)

- The live net gains a third hidden layer: **198→256→128→128→5**, squared-ReLU,
  trained over **2M self-play games** (resuming through the run so tests never
  cost training). It beats the previous 256-128 champion **53.6%** head-to-head
  at 0-ply (z=8.7 over 14,800 games) and holds ~52% at 1-ply; the two tie against
  HCE.
- **The gains from extra depth are clearly diminishing.** The edge appeared in
  the first ~150k games and the remaining 1.85M added nothing measurable — a
  smaller step than the 58.8% the 256-128 net won over *its* predecessor, for far
  more training. The next lever is wildbg-style split contact/race nets, not more
  layers. The outgoing champion is kept as `models/td_256-128_champion.pt`.

## v1.5.0 — no more freezing while the engine thinks

- The engine chose its move on the UI thread, freezing the window for the best
  part of a second every turn. Its move choice and cube decision now run on a
  worker thread: worst UI stall during a full engine turn is **164ms, down from
  ~800ms**. The Rust rollout engine had to release the GIL first, or a worker
  thread would only have moved the stall rather than removed it.

## v1.4.3 — the dice you see are the dice it plays

- **The engine appeared to roll one pair and move on another.** The tumble's last
  frame set the dice and called `update()`, which only *queues* a repaint — then
  handed straight to the engine, which blocks the UI thread for the best part of
  a second choosing via rollouts. The queued paint couldn't run until after that,
  so the last random tumble frame sat on screen looking like the roll, and the
  real dice only appeared as the engine moved. `repaint()` paints synchronously,
  so the roll is on screen before the block.
- Found by watching the app play. The selftest and headless test passed
  throughout: they check the engine rolls and moves correctly, not what's drawn
  mid-animation, so a rendering-order artifact is invisible to them.

## v1.4.2 — engine dice fixes

Both introduced by v1.4.1's roll animation:

- **The engine re-rolled the opening throw it had just won.** The throw is handed
  to `engine_play`, which now tumbles before moving — so winning the throw 2-6
  meant visibly rolling again before playing 2-6. A roll already on the table is
  played as-is.
- **The roll was shown before it landed.** The status line named the roll before
  tumbling, and `refresh()` writes the final dice into the view — so the pair
  appeared, the tumble rolled other pairs over the top, and it came back.
  `refresh()` now leaves the dice alone while a tumble owns them.

## v1.4.1 — the engine rolls where you can see it

- **The engine's dice never rolled.** It set its roll and moved in one step —
  no tumble, no sound. Both sides now roll through one shared `_tumble()`: sound,
  a brief tumble, the dice land, and only then does the mover act. The engine
  tumbles *before* it thinks, since choosing can block for ~0.8s with rollouts.
- **Less time rolling** — it sits between every move: the tumble is 360ms (was
  880ms) and the roll sound 0.39s (was 1.00s), so the two now line up instead of
  a second-long sound running past a stale board.

## v1.4.0 — both numberings on the board

- **Dual point numbers.** Every point is labelled with both numberings: yours in
  ivory, the engine's in red. Backgammon has no single numbering — each player
  counts 1–24 from their own home and moves are always written from the mover's
  own view, so the two always total 25 (your 8 is its 17). Your moves and hints
  read off the ivory numbers; the engine's log lines read off the red ones.
  Mirroring the engine's moves into your numbers was considered and rejected: it
  would disagree with every backgammon book, with GNUbg, and with the app's own
  (mover-relative) GnuBG Position IDs.

## v1.3.0 — board numbers, hint panel, dice that sound like dice

- **Point numbers on the board**, so notation like `8/5 6/5` can be found at a
  glance.
- **Hint is a panel.** Hovering *Hint* lists the best five moves with their
  equities, best first, instead of squeezing three into the status line. Cached
  by position and roll — at 2-ply the search is ~0.6s and hover would re-run it.
- **Two-dice targets.** Selecting a checker also lights where it can land using
  *both* dice; clicking there plays both legs, each snapshotted separately so
  Undo still steps back one checker at a time. Routes whose intermediate landing
  hits a blot are excluded — a hit is a move to choose deliberately, not a
  waypoint. Verified over 4,000 positions: 3,217 hitting first-legs seen, none
  ever offered as a combined target.
- **The roll had a tune in it.** Each knock was a stack of harmonic modes (giving
  it a pitch) and the landings stepped 880→760→690→620 Hz — a descending melody.
  Knocks are now noise through a wide band-pass with jittered centres:
  pitchiness 0.83 → 0.27, zero-crossings 4.8k (between the old white-noise click
  at 21k and the ring at 1.5k).
- **Checkers are audible.** That sound was 90 ms at peak 0.40 — inaudible in
  practice. Now a woody knock at peak 0.82 with a silent tail, since the audio
  sink is created per play and a buffer that short could expire before the device
  finished opening.
- Default volume 50%.

## v1.2.1 — sound actually works

- **The app could be completely silent on a machine whose audio was fine.** Qt's
  `QSoundEffect` reported `Status.Ready` and `isPlaying() == true` while emitting
  nothing. Playing the same WAV through `winsound` (audible) and then
  `QAudioSink` (audible) isolated the fault to QSoundEffect's decode path. The
  app synthesises its own samples, so there is nothing to decode: it now hands
  raw PCM straight to `QAudioSink`.
- `is_playing` compares sink state **by name**: Qt 6.7 renamed the `QAudio`
  namespace to `QtAudio`, so the imported enum isn't the one `state()` returns
  and `==` quietly returned False for an actively playing sink.
- **The build now fails if playing a sound doesn't make the device go active.** A
  silent build looked perfectly healthy from outside — which is how this shipped
  twice.
- **The opening throw is the winner's first roll.** Win the throw 6-3 and you
  play 6-3; no rolling again, and no double can be offered before it, because the
  roll has already been made.
- Help panel covers the eval bar, pip counts and cube.

## v1.2.0 — in-app help

- **Help panel.** Hovering **?** overlays what pulses, what hovering gives you,
  how to move and how to take back. The text lives next to the code it describes
  so it can't quietly drift.
- Opening dice count 1-2-3-4-5-6 in step, a little slower, until clicked.
- Qt silently drops `play()` while a WAV is still loading, so the first sound
  after launch could vanish; it's now played once loaded.
- The selftest reported sound as "the object isn't None", which proved nothing.

## v1.1.1 — the opening roll throws for real

- **A tied opening throw is shown and thrown again**, as at the board. It used to
  loop internally until the dice happened to differ, so a tie never appeared.

## v1.1.0 — hover controls, takeback

- **Anything that wants a click pulses, and hovering it says what the click
  does** — Roll dice on the dice, Double on the cube, Accept/Fold while the cube
  pulses at you. The hover region deliberately includes the boxes themselves, or
  reaching for a box would dismiss it.
- **Opening roll winds through 1-6 until you click**; the click is what rolls.
- **Takeback.** *Undo* / Ctrl+Z steps back the checkers you've moved this turn,
  restoring each die, to the start of your turn. Playing your last die commits
  the turn and the engine replies, so the final checker can't be recalled without
  a full game-history rewind.
- Volume slider, remembered between sessions.
- Wider window so move equities don't clip.

## v1.0.1 — plays its best by default

- The opponent selector opens on the **strongest engine available** (Monte-Carlo
  rollouts) instead of 1-ply. Hints stay on the deepest *neural* search, because
  hints rank every move and the rollout engine only reports the one it picked.

## v1.0.0 — the standalone app

- **One self-contained `Backgammon.exe`** (~62 MB): the PySide6 GUI, the Rust
  engine (`bgcore.pyd`, embedding the ONNX runtime) and the trained net. No
  Python, no PyTorch.
- **A torch-free play path.** `bgcore.Neural` exposes the Rust net + n-ply search
  to Python, so the app runs the net through the engine instead of PyTorch —
  which would have added 200 MB+ to the download.
- **The search was never batched.** Every chance node scored its moves one
  position at a time, so it never used `Evaluator::evaluate_batch` — only the
  rollouts did. Scoring a move list in one `[n, 198]` matmul made native 1-ply
  **7.5× faster** (195 ms → 26 ms per move) and 2-ply **8×** (4.8 s → 0.6 s),
  which also speeds up `SearchEngine` and the benchmarks. Verified against the
  torch engine over 40 self-play positions: equities agree to 2.5e-7 and the
  chosen move matches everywhere.
- **`build.py` verifies the exe it produced** rather than trusting a clean build.
  A windowed app has nowhere to print, and a missing `td.onnx` wouldn't crash it
  — the neural opponents would silently vanish — so `app.py --selftest` reports
  what actually loaded and the build asserts on it.

---

## Before the app

The engine itself (Rust core, validated move generation, TD self-play trainer,
ONNX export, n-ply search, rollouts, the doubling cube and the GUI) predates this
log. Two results worth carrying forward:

- **Move generation is validated** against the independent
  [wildbg](https://github.com/carsten-wenderdel/wildbg) engine across **3.15M**
  (position, dice) pairs — zero mismatches.
- **Depth broke the net's strength ceiling.** A wider single-layer net merely
  tied the old one; `198→256→128→5` with squared-ReLU beat it **58.8%**
  head-to-head and cut the self-play gammon rate from 70.7% to 60.2%. At equal
  search the engine now holds its own against wildbg.

See [`SPEC.md`](SPEC.md) for the design and milestones.
