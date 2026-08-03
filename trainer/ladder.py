"""Measure the strength ladder of the opponents the app actually offers.

`nply_h2h.py` compares two NETS; this compares two OPPONENTS as the GUI defines
them — which is a different thing, because an opponent is a net *plus* a search
setting (0/1/2-ply, phase routing, or Monte-Carlo rollouts), and those are what a
player picks between in the Opponent box.

Adjacent rungs are played head-to-head with mirrored dice, then Elo is chained
from the measured win rates with Random anchored at 0. Backgammon win rates are
compressed by dice luck, so treat the Elo gaps as indicative; points-per-game is
reported alongside because it is the more sensitive statistic.

Games run across PROCESSES, not threads: engine_api is a GUI-shaped API whose
`analyze()` does substantial Python work per move under the GIL, so a threaded
version of this managed about four of sixty-four cores. Game budgets are tiered
by rung cost — a 0-ply game is ~20ms, a 2-ply or rollout game tens of seconds.

Run:
  .venv/Scripts/python trainer/ladder.py --out models/ladder.json
  .venv/Scripts/python trainer/ladder.py --plot images/backgammon-elo.png
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bgcore
from engine_api import HceEngine, NativeNeuralEngine, RandomEngine, RolloutEngine

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ROLLOUT_THREADS = 8


# Opponent keys -> labels, weakest first. Mirrors gui/app.py's Opponent box.
def opponents(rollout_ms: int):
    td, classic = MODELS / "td.onnx", MODELS / "td_classic.onnx"
    out = [("random", "Random"), ("hce", "HCE (heuristic)")]
    if classic.exists():
        out += [("classic0", "Neural classic — 0-ply"), ("classic1", "Neural classic — 1-ply")]
    out += [("nn0", "Neural — 0-ply"), ("nn1", "Neural — 1-ply"), ("nn2", "Neural — 2-ply"),
            ("rollout", f"Rollout ({rollout_ms}ms)")]
    return out


def make(key: str, rollout_ms: int):
    """Build one opponent. Called inside each worker PROCESS, never pickled."""
    td, classic = MODELS / "td.onnx", MODELS / "td_classic.onnx"
    if key == "random":
        return RandomEngine(0)
    if key == "hce":
        return HceEngine()
    if key.startswith("classic"):
        return NativeNeuralEngine(classic, int(key[-1]), label="Neural classic")
    if key.startswith("nn"):
        return NativeNeuralEngine(td, int(key[-1]))
    if key == "rollout":
        # Threads matter enormously here: the rollout gets a fixed MOVETIME, so
        # cores translate directly into trials, and trials are what make the
        # estimate accurate. The app uses threads=0 (the whole machine), so a
        # 1-thread measurement understates it by roughly the core count. Default
        # 8 approximates a typical user's laptop; games still parallelise.
        return RolloutEngine(td, movetime_ms=rollout_ms, truncate_plies=9,
                             candidates=5, threads=ROLLOUT_THREADS)
    raise ValueError(key)


def play(ea, eb, seed, a_first):
    """One game between two engine_api engines. Returns points to A."""
    rng = random.Random(seed)
    board = bgcore.Board.starting()
    a_turn = a_first
    for _ in range(300):
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        nxt, pts, _steps, _eq = (ea if a_turn else eb).choose(board, d1, d2)
        if pts is not None:
            return pts if a_turn else -pts
        board = nxt
        a_turn = not a_turn
    a_pip = board.pip_count(0 if a_turn else 1)
    b_pip = board.pip_count(1 if a_turn else 0)
    return 1 if a_pip < b_pip else -1


_EA = _EB = None


def _init(key_a, key_b, rollout_ms, rollout_threads=8):
    global _EA, _EB, ROLLOUT_THREADS
    ROLLOUT_THREADS = rollout_threads
    _EA, _EB = make(key_a, rollout_ms), make(key_b, rollout_ms)


def _play(job):
    g = job
    return play(_EA, _EB, 4000 + g // 2, g % 2 == 0)   # mirrored: same dice, seats swapped


def match(key_a, key_b, games, workers, rollout_ms, rollout_threads=8):
    """A vs B over `games` mirrored-dice games. Returns (win_rate, ppg, n).

    PROCESSES, not threads. engine_api is a GUI-shaped API: `analyze()` rebuilds
    the move list with step history and sorts it in Python, all under the GIL,
    and only the search itself releases it. A threaded version of this measured
    38,000 CPU-seconds at 6% box utilisation — about four cores of sixty-four.
    Windows caps mp.Pool near 60 workers (WaitForMultipleObjects, 63 handles).
    """
    with mp.Pool(min(workers, 60), initializer=_init,
                 initargs=(key_a, key_b, rollout_ms, rollout_threads)) as pool:
        res = list(pool.imap_unordered(_play, range(games), chunksize=4))
    n = len(res)
    return sum(1 for x in res if x > 0) / n, sum(res) / n, n


def elo_gap(win_rate: float) -> float:
    """Elo difference implied by a head-to-head win rate."""
    win_rate = min(max(win_rate, 1e-4), 1 - 1e-4)
    return 400.0 * math.log10(win_rate / (1.0 - win_rate))


def main():
    ap = argparse.ArgumentParser()
    # Rung cost varies by ~100x, so budget games per tier rather than uniformly:
    # a 0-ply game is ~20ms, a 2-ply or rollout game tens of seconds. Chaining
    # noisy rungs accumulates error, so the cheap rungs should be measured hard.
    ap.add_argument("--games", type=int, default=6000, help="games per 0-ply rung")
    ap.add_argument("--games-mid", type=int, default=3000, help="games per 1-ply rung")
    ap.add_argument("--games-slow", type=int, default=800, help="games per 2-ply / rollout rung")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--rollout-ms", type=int, default=800, help="matches the app default")
    ap.add_argument("--rollout-threads", type=int, default=8,
                    help="cores the rollout engine gets per move. The app gives it the whole "
                         "machine; 8 approximates a typical laptop. Cores buy trials, and "
                         "trials are what make a rollout accurate — this materially changes "
                         "its measured strength.")
    ap.add_argument("--rollout-workers", type=int, default=8,
                    help="parallel games on rollout rungs (workers x threads should fit the box)")
    ap.add_argument("--out", default=None, help="write results json here")
    ap.add_argument("--plot", default=None, help="write the Elo chart png here")
    args = ap.parse_args()

    engines = opponents(args.rollout_ms)
    print(f"ladder: {len(engines)} opponents, {len(engines)-1} rungs, mirrored dice\n", flush=True)

    # Cost tier per opponent: 0 = static eval, 1 = one ply of search, 2 = two
    # plies or rollouts. A rung costs whatever its more expensive side costs.
    TIER = {"random": 0, "hce": 0, "classic0": 0, "nn0": 0,
            "classic1": 1, "nn1": 1, "nn2": 2, "rollout": 2}
    budget = [args.games, args.games_mid, args.games_slow]

    rungs, elo, labels = [], [0.0], [engines[0][1]]
    for (ka, la), (kb, lb) in zip(engines, engines[1:]):
        games = budget[max(TIER.get(ka, 0), TIER.get(kb, 0))]
        t0 = time.time()
        slow_rollout = "rollout" in (ka, kb)
        w = args.rollout_workers if slow_rollout else args.workers
        wr, ppg, n = match(kb, ka, games, w, args.rollout_ms, args.rollout_threads)
        gap = elo_gap(wr)
        elo.append(elo[-1] + gap)
        labels.append(lb)
        rungs.append({"upper": lb, "lower": la, "win_rate": wr, "ppg": ppg,
                      "games": n, "elo_gap": gap})
        print(f"  {lb:26s} vs {la:26s} {100*wr:5.1f}%  ppg {ppg:+.3f}  "
              f"({n} games, {time.time()-t0:.0f}s)  +{gap:.0f} Elo", flush=True)

    print("\nchained Elo (Random = 0):")
    for lab, e in zip(labels, elo):
        print(f"  {lab:26s} {e:6.0f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"rungs": rungs, "labels": labels, "elo": elo}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.plot:
        plot(labels, elo, args.plot)


def elo_se(win_rate: float, n: int) -> float:
    """1 s.e. of a rung's Elo gap, propagated from the binomial s.e. of its win
    rate. d(Elo)/dp = 400 / (ln10 * p(1-p))."""
    p = min(max(win_rate, 1e-3), 1 - 1e-3)
    return (400.0 / (math.log(10) * p * (1 - p))) * math.sqrt(p * (1 - p) / n)


def plot(labels, elos, out, rungs=None):
    """Two panels: the whole ladder, and a zoom on the neural cluster.

    One panel cannot show both. Anchored at Random = 0 the interesting
    differences between the neural options — tens of Elo — are invisible next to
    a 990-point first step, which would imply the choices matter more than they
    do. The zoom carries error bars because those differences are individually
    at the edge of significance and should not be read as a settled order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = [l.replace(" — ", "\n").replace(" (heuristic)", "").replace(" (", "\n(")
             for l in labels]
    colors = ["#8a8f98", "#d08a34", "#7796a8", "#6d8fa8", "#2f6f8f", "#2f7f85",
              "#2f8f77", "#2f9f5f"][:len(labels)]

    # Cumulative 1-s.e. band: chained rungs add variance.
    err = [0.0]
    if rungs:
        var = 0.0
        for r in rungs:
            var += elo_se(r["win_rate"], r["games"]) ** 2
            err.append(math.sqrt(var))
    else:
        err = [0.0] * len(elos)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=130,
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    bars = ax.bar(short, elos, color=colors, width=0.66, edgecolor="white", linewidth=0.6)
    for b, e in zip(bars, elos):
        ax.text(b.get_x() + b.get_width() / 2, e + max(elos) * 0.012, f"{e:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")
    ax.set_ylabel("Estimated Elo  (Random = 0)", fontsize=11)
    ax.set_title("The whole ladder", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(0, max(elos) * 1.16)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)

    # Zoom: everything above HCE.
    k = 2
    z_lab, z_elo, z_err, z_col = short[k:], elos[k:], err[k:], colors[k:]
    ax2.bar(z_lab, z_elo, color=z_col, width=0.6, edgecolor="white", linewidth=0.6,
            yerr=z_err, capsize=4, error_kw={"ecolor": "#555", "elinewidth": 1.2})
    lo, hi = min(z_elo) - max(z_err) - 18, max(z_elo) + max(z_err) + 22
    ax2.set_ylim(lo, hi)
    ax2.set_title("The neural options, magnified (±1 s.e.)", fontsize=12, fontweight="bold", pad=8)
    ax2.grid(axis="y", alpha=0.25)
    ax2.tick_params(axis="x", labelsize=8)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.margins(x=0.03)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("Backgammon-NN — strength of each opponent in the app",
                 fontsize=14.5, fontweight="bold", y=1.005)
    fig.text(0.5, 0.945, "Chained from measured head-to-head win rates · dice luck compresses "
             "win% so gaps are approximate · the neural options are all within ~35 Elo",
             ha="center", va="bottom", fontsize=9, color="#666")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
