"""Plot an estimated Elo ladder for the engines (SPEC §13).

Elo is derived by chaining measured head-to-head win rates, anchoring Random at
0. Each rung uses one real match result for the current live net (SF-style
pip-count output-bucketed net, 198-256-128 body + 8 heads, 1M self-play games):
    Random -> HCE   : HCE wins 99.5%
    HCE    -> NN     : NN wins 88.0% (0-ply, 1000 games)
    NN0    -> NN1    : 1-ply wins 56.5% over 0-ply (200 games)

The ladder stops at 1-ply on purpose. 2-ply no longer measurably beats 1-ply for
this net: nn_bench put it at 46.7% over 60 games (+/-12.6), i.e. tied with 50% --
the stronger the static evaluator, the less an extra ply of lookahead buys, and
resolving so small a delta would take many hours of 2-ply search. The search
rungs shrink as the net improves (1-ply over 0-ply was 62.5% for an early net).
Backgammon win rates compress toward 50% because of dice luck, so these Elo gaps
are approximate; points-per-game gaps are larger (NN averages ~+1.6 ppg vs HCE).
Run: .venv/Scripts/python trainer/plot_elo.py [out.png]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("elo.png")


def elo_gap(win_rate: float) -> float:
    """Elo difference implied by a head-to-head win rate."""
    win_rate = min(max(win_rate, 1e-4), 1 - 1e-4)
    return 400.0 * math.log10(win_rate / (1.0 - win_rate))


# Measured head-to-head win rates (winner listed first), current live net.
HCE_VS_RANDOM = 0.995        # unchanged — HCE and Random are fixed baselines
NN0_VS_HCE = 0.880           # nn_bench, 1000 games, ±2.0
NN1_VS_NN0 = 0.565           # nn_bench, 200 games, ±6.9
# 2-ply vs 1-ply omitted: 46.7% over 60 games (±12.6) is a statistical tie —
# deeper search no longer measurably helps this net, so the ladder stops at 1-ply.

random_elo = 0.0
hce_elo = random_elo + elo_gap(HCE_VS_RANDOM)
nn0_elo = hce_elo + elo_gap(NN0_VS_HCE)
nn1_elo = nn0_elo + elo_gap(NN1_VS_NN0)

labels = ["Random", "HCE", "Neural net\n(0-ply)", "Neural net\n(1-ply)"]
elos = [random_elo, hce_elo, nn0_elo, nn1_elo]
colors = ["#8a8f98", "#d08a34", "#2f6f8f", "#2f8f77"]

for name, e in zip(("Random", "HCE", "NN0", "NN1"), elos):
    print(f"{name:8s} {e:7.0f}")

fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=130)
bars = ax.bar(labels, elos, color=colors, width=0.62, edgecolor="white", linewidth=0.6)
for b, e in zip(bars, elos):
    ax.text(b.get_x() + b.get_width() / 2, e + 18, f"{e:.0f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222")

ax.set_ylabel("Estimated Elo  (Random = 0)", fontsize=11)
ax.set_title("Backgammon engine strength ladder", fontsize=14, fontweight="bold", pad=14)
ax.text(0.5, 1.005,
        "Chained from head-to-head win rates · dice luck compresses win% so gaps are approximate",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#666")
ax.set_ylim(0, max(elos) * 1.16)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.margins(x=0.02)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
