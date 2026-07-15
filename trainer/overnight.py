"""Train a net overnight, testing it at intervals, resuming after each test.

Self-play runs in chunks. After every chunk the checkpoint is exported to ONNX
and measured; then the next chunk *resumes* from that same checkpoint, so testing
never costs training progress. Everything is appended to a log with timestamps,
so the run reads back in the morning.

What gets measured, every chunk:

  vs Random / vs HCE at 0-ply   (train.py's own bench, in-process)
  head-to-head vs the champion  (compare_nets.py — the honest test: identical
                                 race play, so the result turns on contact
                                 evaluation)

Deliberately *not* measured here: 1-ply and rollout strength (nn_bench, and
wildbg-bench against the reference engine). Both are dominated by slow search —
238s and 400s+ per run — and while a net is still being built all we want is a
rough progress-o-meter. Run them on the finished candidate instead:

  target/release/examples/nn_bench.exe models/td_deep3.onnx
  tools/wildbg-bench/target/release/wildbg-bench.exe 40 150 200 td_deep3.onnx
  (wildbg-bench takes a bare filename — it prepends models/ itself)

Run:
  .venv/Scripts/python trainer/overnight.py --hours 10

Resuming a run that was interrupted is the same command: it picks up from the
checkpoint automatically.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
MODELS = ROOT / "models"


def run(cmd, timeout=None):
    """Run a command, returning its combined output (never raising)."""
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return "(timed out)"
    except Exception as exc:                       # a broken tool must not end the night
        return f"(failed: {exc})"


def tail(text, n=6):
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join("    " + l for l in lines[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0, help="wall-clock budget")
    ap.add_argument("--hidden", default="256,128,128")
    ap.add_argument("--act", default="sqrelu")
    ap.add_argument("--games", type=int, default=200, help="self-play games per iter")
    ap.add_argument("--chunk", type=int, default=120, help="iters between tests")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="td_deep3.pt")
    ap.add_argument("--champion", default="td_latest.pt", help="net to beat")
    ap.add_argument("--h2h-games", type=int, default=400)
    ap.add_argument("--log", default="models/overnight_log.md")
    args = ap.parse_args()

    ckpt = MODELS / args.out
    onnx = MODELS / (Path(args.out).stem + ".onnx")
    log_path = ROOT / args.log
    deadline = time.time() + args.hours * 3600

    def log(msg=""):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    log(f"\n# Overnight run — {datetime.now():%Y-%m-%d %H:%M}")
    log(f"\nnet `{args.hidden}` act={args.act} | {args.games} games/iter | "
        f"chunk {args.chunk} iters ({args.chunk * args.games:,} games) | "
        f"budget {args.hours}h (until {datetime.now() + timedelta(hours=args.hours):%H:%M})")
    log(f"champion to beat: `{args.champion}`\n")

    cycle, total_games = 0, 0
    while time.time() < deadline:
        cycle += 1
        left = (deadline - time.time()) / 3600
        # Train. Resume once a checkpoint exists, so an interrupted night can be
        # restarted with the same command.
        cmd = [str(VENV_PY), "trainer/train.py",
               "--hidden", args.hidden, "--act", args.act,
               "--iters", str(args.chunk), "--games", str(args.games),
               "--lam", "1.0", "--lr", str(args.lr),
               "--bench-every", str(max(1, args.chunk // 2)), "--bench-games", "300",
               "--out", args.out]
        if ckpt.exists():
            cmd += ["--resume", args.out]
        t0 = time.time()
        out = run(cmd, timeout=6 * 3600)
        mins = (time.time() - t0) / 60
        total_games += args.chunk * args.games

        log(f"\n## cycle {cycle} — {datetime.now():%H:%M} "
            f"({total_games:,} games so far, {mins:.0f} min, {left:.1f}h left)")
        log("\n**self-play** (vs Random / vs HCE, 0-ply)\n```")
        log(tail(out, 3))
        log("```")

        if not ckpt.exists():
            log("\n**checkpoint missing — training failed, stopping**")
            log(tail(out, 12))
            return

        # Export so the native tools can read it. The second argument is not
        # optional here: export_onnx.py defaults to models/td.onnx, which is the
        # *live* net — exporting a candidate without it overwrites the champion
        # the app ships with.
        run([str(VENV_PY), "trainer/export_onnx.py", f"models/{args.out}",
             f"models/{onnx.name}"], timeout=600)
        if not onnx.exists():
            log(f"\n**export failed — {onnx.name} not written; skipping native tests**")

        # The key test: is the new net actually better than the live one?
        h2h = run([str(VENV_PY), "trainer/compare_nets.py",
                   f"models/{args.champion}", f"models/{args.out}", str(args.h2h_games)],
                  timeout=3600)
        log(f"\n**head-to-head vs champion** ({args.h2h_games} games, 0-ply)\n```")
        log(tail(h2h, 8))
        log("```")

    log(f"\n# Done — {datetime.now():%H:%M} | {total_games:,} games over {cycle} cycles")
    log(f"Checkpoint: `models/{args.out}` (the live net was never touched)")


if __name__ == "__main__":
    main()
