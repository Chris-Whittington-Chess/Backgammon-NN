"""Drive the strength ladder, giving each rung the settings it actually needs.

`ladder.py --games-slow N` treats every expensive rung alike, which is wrong in
two ways:

  * The ROLLOUT engine's strength scales with the cores it gets, because it has a
    fixed movetime and cores buy trials. Measuring it with one thread (so that 60
    games can run in parallel) understates it by roughly the core count. It needs
    few parallel games and many threads each.
  * The rollout rung's cost is dominated by its OPPONENT. A rollout-vs-2-ply game
    is ~24s of rollout and ~160s of 2-ply search. Since 2-ply and 1-ply are
    statistically tied for this net (50.8%, z +0.76 over 2000 games), chaining the
    rollout off 1-ply loses nothing and is ~10x cheaper.

Run:  .venv/Scripts/python trainer/ladder_run.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ladder import elo_gap, match, plot

ROOT = Path(__file__).resolve().parent.parent

# (upper, lower, label_upper, games, workers, rollout_threads)
RUNGS = [
    ("hce",      "random",   "HCE (heuristic)",        6000, 60, 1),
    ("classic0", "hce",      "Neural classic — 0-ply", 6000, 60, 1),
    ("classic1", "classic0", "Neural classic — 1-ply", 3000, 60, 1),
    ("nn0",      "classic1", "Neural — 0-ply",         3000, 60, 1),
    ("nn1",      "nn0",      "Neural — 1-ply",         3000, 60, 1),
    ("nn2",      "nn1",      "Neural — 2-ply",          400, 60, 1),
    # Rollout: 8 cores per move (a typical laptop), 8 games at a time, and
    # measured against 1-ply rather than 2-ply for the reason in the docstring.
    ("rollout",  "nn1",      "Rollout (800ms, 8 cores)", 300,  8, 8),
]


def main():
    labels = ["Random"]
    elo = [0.0]
    rows = []
    # nn2 hangs off nn1, and so does rollout — so track Elo by opponent key
    # rather than assuming a single chain.
    by_key = {"random": 0.0}
    t_all = time.time()
    for upper, lower, label, games, workers, rthreads in RUNGS:
        t0 = time.time()
        wr, ppg, n = match(upper, lower, games, workers, 800, rthreads)
        gap = elo_gap(wr)
        by_key[upper] = by_key[lower] + gap
        labels.append(label)
        elo.append(by_key[upper])
        rows.append({"upper": label, "lower": lower, "win_rate": wr, "ppg": ppg,
                     "games": n, "elo_gap": gap, "elo": by_key[upper]})
        print(f"  {label:28s} vs {lower:10s} {100*wr:5.1f}%  ppg {ppg:+.3f}  "
              f"({n} games, {time.time()-t0:.0f}s)  {gap:+.0f} Elo  ->  {by_key[upper]:.0f}",
              flush=True)

    print(f"\ntotal {time.time()-t_all:.0f}s\n\nchained Elo (Random = 0):")
    for lab, e in zip(labels, elo):
        print(f"  {lab:28s} {e:6.0f}")

    (ROOT / "models" / "ladder.json").write_text(
        json.dumps({"rungs": rows, "labels": labels, "elo": elo}, indent=2), encoding="utf-8")
    plot(labels, elo, str(Path(r"C:/Users/chris/source/repos/whittingtonchess/images/backgammon-elo.png")))


if __name__ == "__main__":
    main()
