# Development log

Changes to **Backgammon-NN**, newest first. Engine internals and the app are both
covered; the story of how the *network* was trained is in the
[development report](https://whittingtonchess.com/backgammon-report).

Downloads: [Releases](../../releases). The app is a single self-contained
`Backgammon.exe` — no installer, no Python, no PyTorch.

---

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
