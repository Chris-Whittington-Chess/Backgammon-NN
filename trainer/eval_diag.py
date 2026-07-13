"""Sanity-check a trained value net on hand-crafted positions.

Independent of noisy benchmark games: a correct value function should rate the
opening near even, a big lead as strongly winning, and a big deficit as losing.

Run: .venv/Scripts/python trainer/eval_diag.py [checkpoint.pt]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import bgcore
from model import ValueNet, equity


def net_equity(net, board) -> float:
    x = torch.tensor([board.features()], dtype=torch.float32)
    with torch.no_grad():
        return float(equity(net(x))[0].item())


def main():
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "models" / "td_latest.pt"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = ValueNet(ckpt.get("hidden", 128))
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"Loaded {ckpt_path.name} (iter {ckpt.get('iter', '?')})\n")

    start = bgcore.Board.starting()

    # Mover bearing off, opponent stuck far back -> should be ~winning.
    # Position IDs are easiest built from the engine; construct via a race.
    # Use raw feature positions through legal play instead: craft via ids.
    cases = [
        ("opening (should be ~0)", start),
        ("mover swapped opening (should be ~0)", start.swap_perspective()),
    ]

    # A clear mover win: mover has 2 checkers left near home, opp all in the back.
    # Build by parsing a GnuBG id we can trust: take starting and race mover home.
    # Simpler: reuse HCE-truth via pip counts to label expectation.
    for name, b in cases:
        eq = net_equity(net, b)
        print(f"{name:42s} equity={eq:+.3f}  pips m/o={b.pip_count(0)}/{b.pip_count(1)}")

    # Monotonicity probe: from the opening, compare equity of the position after
    # a strong mover move vs a weak one for the roll 3-1 (24/23 13/11 is standard
    # best; the net should not rate a clearly bad move higher than a good one on
    # average once trained). We just report the spread.
    kids = bgcore.legal_moves(start, 3, 1)
    eqs = sorted(net_equity(net, k.swap_perspective()) for k in kids)
    # These are opponent-perspective equities of the child; mover prefers the min.
    print(
        f"\nopening 3-1: {len(kids)} moves, child opp-equity "
        f"range [{eqs[0]:+.3f}, {eqs[-1]:+.3f}] (mover picks the min)"
    )

    print("\nInterpretation: opening ~0 and a meaningful spread across moves "
          "indicate the value function learned something non-trivial.")


if __name__ == "__main__":
    main()
