"""Headless smoke test + screenshot generator for the GUI (offscreen Qt).

Drives the game by pumping the animation timers manually (there is no event
loop offscreen), exercising roll -> human submoves -> animated engine reply.
"""
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gui"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT

from PySide6.QtWidgets import QApplication  # noqa: E402
import app as gui  # noqa: E402


def grab(win, name):
    QApplication.processEvents()
    win.view.grab().save(str(OUT / name))
    print("saved", name)


def pump_roll(win):
    win.on_dice()
    win._roll_timer.stop()
    win._roll_frames = 1
    win._roll_frame()


def pump_engine(win):
    win.view.grab()          # force a paint so board geometry exists
    win.engine_move()
    guard = 0
    while win._anim_timer.isActive():
        win._anim_frame()
        guard += 1
        assert guard < 2000, "engine animation runaway"


def main():
    _ = QApplication(sys.argv)
    win = gui.MainWindow()
    win.resize(1080, 700)
    win.show()
    grab(win, "board_start.png")

    win.opp_box.setCurrentText("Random")   # fast opponent for the smoke test
    win.rng.seed(11)

    for turn in range(4):
        if win.game_over:
            break
        pump_roll(win)
        guard = 0
        while win.human_turn and win.remaining and win.subs and not win.game_over:
            win.apply_submove(win.subs[0])
            guard += 1
            assert guard < 12
        if win.human_turn and not win.remaining is None and not win.subs and not win.game_over:
            # dance (no legal move): the singleShot won't fire headless
            if win.human_turn and not win.game_over and not win.remaining == []:
                pass
        if not win.human_turn and not win.game_over:
            pump_engine(win)
        print(f"turn {turn}: moves logged = {win.moves.count()}, status = {win.status.text()!r}")

    grab(win, "board_features.png")
    print("OK: no crashes;", "game over" if win.game_over else "in progress",
          "| move panel entries:", win.moves.count())


if __name__ == "__main__":
    main()
