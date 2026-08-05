"""Three unattended experiments, in order, each asking "where is Elo left?".

Run after the parity result (2026-08-05): our net at a genuine 2-ply is level with
gnubg 2-ply, which closes the distillation axis — we are AT the ceiling experiment
24 predicted, not 40% of the way to it as DEV_REPORT §18 says. So the remaining
Elo has to come from somewhere other than more gnubg 2-ply labels.

  1. Is the app's default opponent wrong on a big box?
     The ladder put Rollout at 1444 against Neural 2-ply at 1469, and the app
     defaults to rollouts at >=32 cores. But the ladder built its rollout with
     `rollout_threads=8` to approximate a laptop, while the app hands it the whole
     machine — so 1444 never described what the app actually runs here. Rollouts
     get a fixed MOVETIME, so cores buy trials and trials buy accuracy. This is
     the one experiment that must have the box to itself, and it plays its games
     SERIALLY for the same reason.

  2. Is candidate pruning costing search its value?
     The ladder's search rungs are nearly flat: 0-ply 1458, 1-ply 1465, 2-ply
     1469 — 11 Elo for two plies. `our_best_searched` documents a mechanism that
     would explain it: "a pruned move keeps its static value in `scores`, which
     can top a searched move's deep value". An unsearched move can outrank a
     searched one on static score alone. Widening the window tests it directly.

  3. Is gnubg 3-ply a better teacher than gnubg 2-ply?
     A cheap divergence check before committing to any relabelling run — see
     `teacher_depth.py`.

Everything is time-budgeted and logs as it goes, because a run killed at its
limit having printed nothing has wasted the night.

Run: .venv/Scripts/python trainer/elo_axes.py
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
GNUBG = os.environ.get("GNUBG_CLI", r"C:/Users/chris/AppData/Local/gnubg/gnubg-cli.exe")

# Wall-clock budgets. Generous, but bounded so experiment 3 still gets to run.
BUDGET_1 = float(os.environ.get("BGNN_BUDGET_1", 5 * 3600))
BUDGET_2 = float(os.environ.get("BGNN_BUDGET_2", 3 * 3600))
BUDGET_3 = float(os.environ.get("BGNN_BUDGET_3", 1 * 3600))

WAIT_FOR = [ROOT / "h2h_true_2ply.log", ROOT / "match_7pt_400.log"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def elo(win_rate):
    w = min(max(win_rate, 1e-4), 1 - 1e-4)
    return 400.0 * math.log10(w / (1 - w))


def zscore(wins, n):
    return (wins / n - 0.5) / math.sqrt(0.25 / n) if n else 0.0


def wait_for_box(max_wait=4 * 3600):
    """Block until the in-flight runs finish. Experiment 1 is movetime-based, so
    a busy box would understate the rollout engine — the very thing under test."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        done = 0
        for p in WAIT_FOR:
            if not p.exists():
                done += 1
                continue
            tail = p.read_text(errors="ignore").strip().splitlines()[-3:]
            if any(ln.startswith("=>") for ln in tail):
                done += 1
        if done == len(WAIT_FOR):
            log("box is free")
            return True
        log(f"waiting for the box ({done}/{len(WAIT_FOR)} runs finished)")
        time.sleep(300)
    log("WARNING: gave up waiting; experiment 1 will be contended, which is "
        "pessimistic for the rollout engine")
    return False


# --------------------------------------------------------------------------- 1

def exp1_app_default(budget):
    from engine_api import NativeNeuralEngine, RolloutEngine
    from ladder import play

    log("EXP 1: Rollout (800ms, ALL cores — as the app runs it) vs Neural 2-ply, "
        "serial games")
    ro = RolloutEngine(MODELS / "td.onnx", movetime_ms=800, truncate_plies=9,
                       candidates=5, threads=0)      # 0 = whole machine, like the app
    nn = NativeNeuralEngine(MODELS / "td.onnx", 2)
    t0 = time.time()
    res = []
    g = 0
    while time.time() - t0 < budget:
        # Mirrored dice: consecutive games share a seed with the seats swapped.
        res.append(play(ro, nn, 9000 + g // 2, g % 2 == 0))
        g += 1
        if g % 10 == 0:
            w = sum(1 for x in res if x > 0)
            log(f"  exp1 {g} games: rollout wins {100*w/g:.1f}%  "
                f"ppg {sum(res)/g:+.3f}  ({(time.time()-t0)/g:.1f}s/game)")
    n = len(res)
    if not n:
        return "EXP 1: no games completed"
    w = sum(1 for x in res if x > 0)
    wr, z = w / n, zscore(w, n)
    verdict = ("ROLLOUT STRONGER — the app default is right" if z > 1.96 else
               "2-PLY STRONGER — the app default is COSTING users strength"
               if z < -1.96 else "too close to call at this sample")
    return (f"EXP 1 app default: rollout(all cores) vs Neural 2-ply — rollout wins "
            f"{100*wr:.1f}% (z={z:+.2f}), PPG {sum(res)/n:+.3f}, implied "
            f"{elo(wr):+.0f} Elo, {n} games\n         => {verdict}")


# --------------------------------------------------------------------------- 2

def exp2_candidate_width(budget):
    log("EXP 2: candidate width at 2-ply — A=12 candidates vs B=4 (what we ship)")
    out = ROOT / "exp2_width.log"
    cmd = [str(ROOT / ".venv/Scripts/python"), str(ROOT / "trainer/nply_h2h.py"),
           "--a", "td.onnx", "--b", "td.onnx", "--ply", "2",
           "--candidates-a", "12", "--candidates-b", "4",
           "--games", "4000", "--workers", "56"]
    with open(out, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        try:
            p.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            p.kill()
            log("  exp2 hit its budget — reading the last progress line instead")
    lines = [l for l in out.read_text(errors="ignore").splitlines() if l.strip()]
    final = next((l for l in reversed(lines) if l.startswith("A (")), None)
    tail = final or (lines[-1] if lines else "no output")
    return ("EXP 2 candidate width: " + tail +
            "\n         (A winning => pruning to 4 is costing search its value)")


# --------------------------------------------------------------------------- 3

def exp3_teacher_depth(budget):
    """Run the divergence check at 3-ply and again at 4-ply.

    4-ply is far too slow to ever label millions of positions with, but that is
    not what it is for here: if even 4-ply says what 2-ply says, the teacher-depth
    axis is closed outright rather than merely unattractive.
    """
    out = []
    for deep in (3, 4):
        out.append(_divergence(deep, budget / 2))
    return "\n         ".join(out)


def _divergence(deep_ply, budget):
    log(f"EXP 3: gnubg {deep_ply}-ply vs 2-ply label divergence")
    d = np.load(MODELS / "harvest_gnubg_10k.npz")
    pos = d["pos_ids"][d["kind"] == 1] if "kind" in d.files else d["pos_ids"]
    rng = np.random.default_rng(7)
    sample = pos[rng.choice(len(pos), size=2400, replace=False)]

    # 3-ply costs seconds per real position, so shard across processes. Each
    # writes its own csv incrementally; whatever exists when the budget expires
    # is still a valid (and still random) sample.
    shards = 24
    procs, csvs = [], []
    for k in range(shards):
        pin = ROOT / f"exp3_positions_{deep_ply}_{k}.txt"
        csv = ROOT / f"exp3_teacher_{deep_ply}_{k}.csv"
        pin.write_text("\n".join(str(p) for p in sample[k::shards]))
        csvs.append(csv)
        env = dict(os.environ, BGNN_POS_IN=str(pin), BGNN_CSV_OUT=str(csv),
                   BGNN_BASE_PLY="2", BGNN_DEEP_PLY=str(deep_ply))
        fh = open(ROOT / f"exp3_teacher_{deep_ply}_{k}.log", "w")
        procs.append((subprocess.Popen(
            [GNUBG, "-q", "-t", "-p", str(ROOT / "trainer/teacher_depth.py")],
            stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)), fh))

    deadline = time.time() + budget
    for p, fh in procs:
        try:
            p.wait(timeout=max(5, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
        fh.close()

    rows = []
    for csv in csvs:
        if csv.exists():
            rows += [l.split(",") for l in csv.read_text().splitlines()[1:] if l.strip()]
    if not rows:
        return f"EXP 3 ({deep_ply}-ply): gnubg wrote no rows"
    if not rows:
        return "EXP 3: csv empty"
    wb = np.array([float(r[1]) for r in rows])
    eb = np.array([float(r[2]) for r in rows])
    wd = np.array([float(r[3]) for r in rows])
    ed = np.array([float(r[4]) for r in rows])
    dw, de = np.abs(wd - wb), np.abs(ed - eb)
    # A label difference below what training noise washes out is not a teacher
    # upgrade, whatever its p-value.
    material = float((de > 0.02).mean())
    call = ("worth relabelling at 3-ply" if material > 0.15 else
            "NOT worth relabelling — 3-ply says what 2-ply says")
    return (f"EXP 3 teacher depth 2-ply vs {deep_ply}-ply: {len(rows)} positions | mean |dwin| "
            f"{dw.mean():.4f} (p95 {np.percentile(dw,95):.4f}) | mean |dequity| "
            f"{de.mean():.4f} (p95 {np.percentile(de,95):.4f}) | "
            f"{100*material:.1f}% differ by >0.02 equity\n         => {call}")


def main():
    log("elo_axes starting")
    wait_for_box()
    results = []
    for name, fn, budget in (("1", exp1_app_default, BUDGET_1),
                             ("2", exp2_candidate_width, BUDGET_2),
                             ("3", exp3_teacher_depth, BUDGET_3)):
        try:
            r = fn(budget)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = f"EXP {name}: FAILED — {type(e).__name__}: {e}"
        log(r)
        results.append(r)
    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78, flush=True)
    for r in results:
        print("  " + r.replace("\n", "\n  "), flush=True)


if __name__ == "__main__":
    main()
