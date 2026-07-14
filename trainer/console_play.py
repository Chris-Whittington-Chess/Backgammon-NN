"""Play Backgammon-NN in the terminal — a console app to play on your PC.

You are 'O' (moving 24 -> 1, bearing off at the right); the engine is 'X'. Roll,
then choose a move from the numbered list of legal plays. The engine replies.

Run:
    .venv/Scripts/python trainer/console_play.py            # vs the neural net
    .venv/Scripts/python trainer/console_play.py --ply 1    # net with 1-ply search
    .venv/Scripts/python trainer/console_play.py --opponent hce
    .venv/Scripts/python trainer/console_play.py --demo     # self-running demo
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bgcore
from engine_api import HceEngine, NeuralEngine, RandomEngine, RolloutEngine, format_steps

MODEL = Path(__file__).resolve().parent.parent / "models" / "td_latest.pt"


def render(board) -> None:
    """Print the board from the given perspective ('O' = positive side)."""
    def cell(p):
        n = board.point(p)
        if n == 0:
            return "  · "
        return f"{abs(n):>2}{'O' if n > 0 else 'X'} "

    top = " ".join(cell(p) for p in range(13, 25))
    bot = " ".join(cell(p) for p in range(12, 0, -1))
    print()
    print("   13                                              24")
    print("  " + top)
    print("  " + bot)
    print("   12                                               1")
    print(f"   bar O={board.bar(0)} X={board.bar(1)}   off O={board.off(0)} X={board.off(1)}"
          f"   pips O={board.pip_count(0)} X={board.pip_count(1)}")


def human_move(board, d1, d2):
    moves = bgcore.legal_moves_with_steps(board, d1, d2)
    if len(moves) == 1 and not moves[0][0]:
        print("   (no legal move — you dance)")
        return moves[0][1]
    for i, (steps, _res) in enumerate(moves):
        print(f"   [{i}] {format_steps(steps)}")
    while True:
        choice = input(f"   Your move 0-{len(moves) - 1}: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(moves):
            return moves[int(choice)][1]
        print("   Please enter a number from the list.")


def make_engine(name, ply):
    if name == "rollout":
        return RolloutEngine(MODEL.parent / "td.onnx", movetime_ms=600, candidates=4)
    if name == "neural" and MODEL.exists():
        return NeuralEngine(MODEL, lookahead=ply)
    if name == "random":
        return RandomEngine()
    return HceEngine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", choices=["neural", "rollout", "hce", "random"], default="neural")
    ap.add_argument("--ply", type=int, default=0, help="engine search depth (neural)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--demo", action="store_true", help="auto-play both sides")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    engine = make_engine(args.opponent, args.ply)
    print(f"Backgammon-NN — you are O, opponent: {engine.name}")

    board = bgcore.Board.starting()
    human = True
    plies = 0
    while plies < 4000:
        plies += 1
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        if human:
            render(board)
            print(f"\n You rolled {d1}-{d2}")
            if args.demo:
                moves = bgcore.legal_moves_with_steps(board, d1, d2)
                board = moves[rng.randrange(len(moves))][1]
            else:
                board = human_move(board, d1, d2)
            pts = board.winner_points()
            if pts:
                render(board)
                print(f"\n You win {pts} point(s)! 🎉")
                return
            board = board.swap_perspective()
            human = False
        else:
            nxt, pts, steps, eq = engine.choose(board, d1, d2)
            print(f"\n Engine rolled {d1}-{d2}: {format_steps(steps)}  (eq {eq:+.2f})")
            if pts:
                print(f"\n Engine wins {pts} point(s).")
                return
            board = nxt
            human = True
            if args.demo and plies > 60:  # keep the demo short
                print("\n (demo cut short)")
                return


if __name__ == "__main__":
    main()
