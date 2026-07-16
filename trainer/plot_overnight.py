"""Plot a training run's progress from the overnight log.

Three panels, all against games played:
  * head-to-head vs the champion — the number that decides whether the candidate
    replaces it. Drawn with a 95% band, because at a few hundred games per point
    the scatter is mostly dice.
  * vs HCE — an independent yardstick that doesn't move, with the champion's own
    score for reference.
  * Elo relative to the champion, derived from the head-to-head. Backgammon win
    rates are compressed by luck, so treat the scale as indicative.

Run: .venv/Scripts/python trainer/plot_overnight.py [models/overnight_log.md]
Out: models/overnight_progress.png
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "overnight_log.md"
OUT = ROOT / "models" / "overnight_progress.png"

INK, GRID, GOOD, WARN, DIM = "#1b2432", "#dfe3ea", "#2f7d4f", "#9e2b25", "#8892a4"


def elo(p):
    """Elo difference implied by a win rate."""
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -400.0 * math.log10(1.0 / p - 1.0)


def main():
    txt = LOG.read_text(encoding="utf-8")
    games = [int(g.replace(",", "")) for g in
             re.findall(r"## cycle \d+ — \S+ \(([\d,]+) games", txt)]
    h2h_a = [float(x) for x in
             re.findall(r"Head-to-head \(0-ply, race-aware\):\s+A wins ([\d.]+)%", txt)]
    # One bench line survives per cycle (the log keeps the chunk's last lines).
    hce = [float(x) for x in re.findall(r"vs HCE win ([\d.]+)%", txt)]
    n_games = int(re.search(r"\*\*head-to-head vs champion\*\* \((\d+) games", txt).group(1))
    champ_hce = float(re.search(r"vs HCE:\s+A ([\d.]+)%", txt).group(1))

    n = min(len(games), len(h2h_a), len(hce))   # a cycle in flight has no result yet
    games, h2h_a, hce = games[:n], h2h_a[:n], hce[:n]
    new = [100 - a for a in h2h_a]                  # the candidate's score
    band = 1.96 * math.sqrt(0.25 / n_games) * 100   # 95% on a single point

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle("Backgammon-NN — 198→256→128→128→5 (sqrelu) overnight run",
                 fontsize=13, fontweight="bold", color=INK, y=0.965)
    x = [g / 1000 for g in games]

    ax = axes[0]
    ax.axhspan(50 - band, 50 + band, color=DIM, alpha=0.15,
               label=f"95% noise band ({n_games} games/point)")
    ax.axhline(50, color=DIM, lw=1.2, ls="--")
    ax.plot(x, new, "o-", color=GOOD, lw=2, ms=5, label="candidate")
    ax.set_ylabel("head-to-head vs champion (%)")
    ax.set_title("Does it beat the champion?  (above 50% = yes)", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.axhline(champ_hce, color=WARN, lw=1.4, ls="--", label=f"champion ({champ_hce:.1f}%)")
    ax.plot(x, hce, "o-", color=GOOD, lw=2, ms=5, label="candidate")
    ax.set_ylabel("win rate vs HCE (%)")
    ax.set_title("Independent yardstick: vs the hand-crafted evaluator",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    ax.axhline(0, color=DIM, lw=1.2, ls="--")
    ax.plot(x, [elo(p / 100) for p in new], "o-", color=GOOD, lw=2, ms=5)
    ax.set_ylabel("Elo vs champion")
    ax.set_xlabel("self-play games (thousands)")
    ax.set_title("Implied Elo (backgammon win rates are luck-compressed — indicative)",
                 fontsize=10, color=INK)

    for ax in axes:
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    pooled = sum(new[3:]) / max(1, len(new[3:]))
    tot = len(new[3:]) * n_games
    se = math.sqrt(0.25 / max(1, tot)) * 100
    fig.text(0.5, 0.005,
             f"pooled since crossover: {pooled:.1f}% over {tot:,} games "
             f"(z = {(pooled-50)/se:.2f})   |   {games[-1]:,} games trained",
             ha="center", fontsize=9, color=DIM)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}  ({len(games)} cycles, {games[-1]:,} games)")


if __name__ == "__main__":
    main()
