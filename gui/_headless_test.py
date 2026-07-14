"""Headless smoke test + screenshot generator for the GUI (offscreen Qt)."""
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


def resolve_opening(win):
    """Force the opening roll to resolve with the human starting (deterministic)."""
    if getattr(win, "opening", False):
        win._open_winner = 0
        win._open_finish()


def play_turns(win, n):
    resolve_opening(win)
    for _ in range(n):
        if win.game_over:
            break
        win.on_dice()
        win._roll_timer.stop()
        win._roll_frames = 1
        win._roll_frame()
        g = 0
        while win.human_turn and win.remaining and win.subs and not win.game_over:
            win.apply_submove(win.subs[0])
            g += 1
            assert g < 12
        if not win.human_turn and not win.game_over:
            win.view.grab()
            win.engine_play()   # skip the double check for a fast deterministic smoke
            gg = 0
            while win._anim_timer.isActive():
                win._anim_frame()
                gg += 1
                assert gg < 2000


def main():
    _ = QApplication(sys.argv)
    win = gui.MainWindow()
    win.resize(1080, 700)
    win.show()
    grab(win, "board_start.png")           # centered cube shows "1"

    win.opp_box.setCurrentText("Random")
    win.rng.seed(11)
    play_turns(win, 4)
    grab(win, "board_features.png")
    print("after play: moves", win.moves.count(), "| status", repr(win.status.text()))

    # --- doubling cube mechanics ---
    win.new_game()
    resolve_opening(win)
    win.on_double()                        # you offer; engine takes or drops
    assert win.cube_value == 2 or win.game_over, "double had no effect"
    print("human double ->", "cube", win.cube_value, "owner", win.cube_owner,
          "over", win.game_over, "score", win.score)

    win.new_game()
    resolve_opening(win)
    win.pending_double = True
    win.busy = True
    win.on_take()
    assert win.cube_value == 2 and win.cube_owner == 0 and not win.pending_double
    print("take -> cube", win.cube_value, "owner", win.cube_owner)

    win.new_game()
    resolve_opening(win)
    win.pending_double = True
    before = list(win.score)
    win.on_drop()
    assert win.game_over and win.score[1] == before[1] + 1
    print("drop -> engine +1, score", win.score)

    grab(win, "board_cube.png")
    print("OK: no crashes")


if __name__ == "__main__":
    main()
