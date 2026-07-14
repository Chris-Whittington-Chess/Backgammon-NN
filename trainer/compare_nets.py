"""Compare two value nets head-to-head (0-ply, race-aware) plus their points and
gammon rates vs the fixed HCE. The head-to-head is the honest strength test:
identical race play, so the result turns purely on contact evaluation.

Run: .venv/Scripts/python trainer/compare_nets.py models/td256.pt models/td_latest.pt
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_rollouts import head_to_head, load_net, self_play_gammon_rate
from train import benchmark, hce_policy

GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 600


def main():
    a_path = sys.argv[1] if len(sys.argv) > 1 else "models/td256.pt"
    b_path = sys.argv[2] if len(sys.argv) > 2 else "models/td_latest.pt"
    a_net, a_ck = load_net(a_path)
    b_net, b_ck = load_net(b_path)
    a = f"{Path(a_path).name} (hidden {a_ck.get('hidden')}, iter {a_ck.get('iter')})"
    b = f"{Path(b_path).name} (hidden {b_ck.get('hidden')}, iter {b_ck.get('iter')})"
    print(f"A = {a}\nB = {b}\n{GAMES} games each, seats swapped.\n")

    wr = head_to_head(a_net, b_net, GAMES, random.Random(11))
    print(f"Head-to-head (0-ply, race-aware):  A wins {100 * wr:.1f}%")

    hce = hce_policy()
    wa, pa = benchmark(a_net, hce, GAMES, random.Random(22))
    wb, pb = benchmark(b_net, hce, GAMES, random.Random(22))
    print(f"vs HCE:   A {100 * wa:.1f}% PPG {pa:+.3f}    B {100 * wb:.1f}% PPG {pb:+.3f}")

    ga = self_play_gammon_rate(a_net, GAMES, random.Random(33))
    gb = self_play_gammon_rate(b_net, GAMES, random.Random(33))
    print(f"self-play gammon+bg rate:   A {100 * ga:.1f}%    B {100 * gb:.1f}%")

    print()
    if wr >= 0.53:
        print(f"=> A is STRONGER ({100 * wr:.1f}% head-to-head).")
    elif wr <= 0.47:
        print(f"=> B is STRONGER ({100 * (1 - wr):.1f}% head-to-head).")
    else:
        print(f"=> too close to call ({100 * wr:.1f}% head-to-head).")


if __name__ == "__main__":
    main()
