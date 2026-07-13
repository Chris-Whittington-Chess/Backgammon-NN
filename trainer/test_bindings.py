"""Smoke test for the bgcore PyO3 bindings (M3).

Run: .venv/Scripts/python trainer/test_bindings.py
"""

import random

import bgcore


def test_constants_and_starting():
    assert bgcore.NUM_INPUTS == 198
    b = bgcore.Board.starting()
    assert b.position_id() == "4HPwATDgc/ABMA"
    assert b.pip_count(0) == 167 and b.pip_count(1) == 167
    assert not b.is_terminal()
    assert b.winner_points() is None


def test_features():
    b = bgcore.Board.starting()
    f = b.features()
    assert len(f) == bgcore.NUM_INPUTS
    # Mover point 6 (5 checkers) -> (1,1,1,1.0) at offset 20.
    assert f[20:24] == [1.0, 1.0, 1.0, 1.0]
    # Turn one-hot at the tail.
    assert f[196:198] == [1.0, 0.0]


def test_move_generation_alignment():
    b = bgcore.Board.starting()
    moves = bgcore.legal_moves(b, 3, 1)
    feats = bgcore.children_features(b, 3, 1)
    assert len(moves) == len(feats) > 0
    # children_features must equal each move's own features, index for index.
    for mv, fv in zip(moves, feats):
        assert mv.features() == fv
        assert len(fv) == 198


def test_position_id_roundtrip():
    b = bgcore.Board.starting()
    again = bgcore.Board.from_id(b.position_id())
    assert again == b
    try:
        bgcore.Board.from_id("not-valid!!")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_selfplay_game_via_hce():
    """Drive a full game with HCE moves; it must terminate with a valid result."""
    rng = random.Random(12345)
    board = bgcore.Board.starting()
    on_roll = 0
    for ply in range(4000):
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        board = bgcore.hce_move(board, d1, d2)
        wp = board.winner_points()
        if wp is not None:
            assert 1 <= wp <= 3  # the mover (side that just moved) won
            print(f"  game ended after {ply + 1} plies, on_roll={on_roll} won {wp} point(s)")
            return
        board = board.swap_perspective()
        on_roll ^= 1
    raise AssertionError("game did not terminate")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nAll {len(tests)} binding tests passed.")
