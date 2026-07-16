"""Isolate the value of a trained race net: champion+racenet vs champion+minpip.

The phase split's likely payoff is the race net — it replaces min-pip race play,
which has no network incumbent, whereas a fresh contact net fights the mature
champion at contact. So the decisive test isn't the phase pair against the
champion; it's whether *adding* the race net to the existing champion beats the
status quo (champion for contact, min-pip for race), with contact play identical
on both sides.

Both engines route by the current board's phase:
  contact position -> the 5-output champion net
  race position    -> race net (hybrid)  vs  min-pip move (baseline)

Dice are mirrored (each pair of games shares a seed, seats swapped), so a few
hundred games mean something. Also reports how often games reach a race, which
bounds how much the race net can matter.

Run: .venv/Scripts/python trainer/eval_hybrid.py [phase_ckpt] [champion] [games]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

import bgcore
from model import net_from_ckpt, equity as equity5, equity6
from train_phase import load_phase_nets, MODELS_DIR


def _opp_equity(kind, net, feats):
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    with torch.no_grad():
        if kind == "net6":
            return equity6(torch.softmax(net(x), dim=-1))
        return equity5(net(x))


def composite_policy(contact, race):
    """A move policy that evaluates by the current board's phase. `contact` and
    `race` are ('net5'|'net6', net) or ('minpip', None)."""
    def f(board, d1, d2, rng):
        children = bgcore.legal_moves(board, d1, d2)
        term = [c.winner_points() for c in children]
        kind, net = race if board.no_contact() else contact
        if kind == "minpip":
            i = min(range(len(children)), key=lambda j: children[j].pip_count(0))
        else:
            mover_eq = -_opp_equity(kind, net, bgcore.next_state_features(board, d1, d2))
            for k, p in enumerate(term):
                if p is not None:
                    mover_eq[k] = float(p)
            i = int(mover_eq.argmax())
        return (None, term[i]) if term[i] is not None else (children[i].swap_perspective(), None)
    return f


def play(p_a, p_b, seed, a_seat, track):
    """One game with dice fixed by `seed`; A in seat `a_seat`. Returns points to A."""
    rng = random.Random(seed)
    board = bgcore.Board.starting()
    seat = 0
    saw_race = False
    for _ in range(4000):
        if board.no_contact():
            saw_race = True
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        pol = p_a if seat == a_seat else p_b
        nb, pts = pol(board, d1, d2, rng)
        if pts is not None:
            track[0] += saw_race
            return pts if seat == a_seat else -pts
        board = nb
        seat ^= 1
    return 0


def main():
    phase_file = sys.argv[1] if len(sys.argv) > 1 else "td_phase.pt"
    champ_file = sys.argv[2] if len(sys.argv) > 2 else "td_latest.pt"
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 400

    champ = net_from_ckpt(torch.load(MODELS_DIR / champ_file, map_location="cpu"))
    ck = torch.load(MODELS_DIR / phase_file, map_location="cpu")
    race = load_phase_nets(ck)["race"]
    print(f"champion {champ_file}  |  race net from {phase_file} (iter {ck.get('iter')})")
    print(f"{games} games, mirrored dice.\n")

    hybrid = composite_policy(("net5", champ), ("net6", race))     # champion + race net
    baseline = composite_policy(("net5", champ), ("minpip", None))  # champion + min-pip

    track = [0]           # games that reached a race
    wins_h = pts_h = 0
    for g in range(games):
        seed = 2000 + g // 2
        a_seat = g % 2            # alternate which seat the hybrid takes; dice mirrored
        p = play(hybrid, baseline, seed, a_seat, track)
        pts_h += p
        if p > 0:
            wins_h += 1

    import math
    wr = wins_h / games
    se = math.sqrt(0.25 / games)
    z = (wr - 0.5) / se
    print(f"champion+racenet vs champion+minpip:  {100*wr:.1f}%  (z {z:+.2f})  PPG {pts_h/games:+.3f}")
    print(f"games that reached a race: {100*track[0]/games:.0f}%")
    verdict = ("race net HELPS" if z > 1.96 else
               "race net HURTS" if z < -1.96 else "too close to call")
    print(f"=> {verdict}")


if __name__ == "__main__":
    main()
