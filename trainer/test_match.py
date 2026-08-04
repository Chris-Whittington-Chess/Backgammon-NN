"""Tests for match play (trainer/match.py).

Run: .venv/Scripts/python trainer/test_match.py
"""
from __future__ import annotations

import match
from cube import CENTER, MOVER, OPP


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def gammonless(win):
    return [win, 0.0, 0.0, 0.0, 0.0]


def test_met_boundaries():
    assert match.mwc(0, 5) == 1.0
    assert match.mwc(5, 0) == 0.0
    for a in range(1, 8):
        for b in range(1, 8):
            m = match.mwc(a, b)
            assert 0.0 <= m <= 1.0, (a, b, m)


def test_double_match_point_is_50():
    assert approx(match.mwc(1, 1), 0.5)


def test_met_symmetry():
    for a in range(1, 10):
        for b in range(1, 10):
            assert approx(match.mwc(a, b), 1.0 - match.mwc(b, a)), (a, b)


def test_met_monotone():
    # Needing fewer points is better; opponent needing fewer is worse for me.
    for b in range(1, 9):
        for a in range(1, 9):
            assert match.mwc(a, b) <= match.mwc(a - 1, b) + 1e-12, (a, b)
            assert match.mwc(a, b) <= match.mwc(a, b + 1) + 1e-12, (a, b)


def test_leader_has_more_than_half():
    # Ahead in the match (needing fewer) => >50% match equity.
    assert match.mwc(2, 5) > 0.5
    assert match.mwc(5, 2) < 0.5


def test_take_point_in_range():
    for (a, b) in [(2, 2), (3, 5), (5, 3), (7, 7), (2, 6), (4, 2)]:
        tp = match.take_point_match(a, b, cube=1)
        assert 0.0 <= tp <= 1.0, (a, b, tp)


def test_crawford_no_cube():
    d = match.match_cube_action(gammonless(0.8), a=1, b=3, cube=1,
                                owner=CENTER, crawford=True)
    assert d.action == "no cube (crawford)"


def test_even_position_even_scores_is_half():
    # Same away scores, dead-even game -> 50% MWC from playing it out.
    m = match._mwc_from_game(gammonless(0.5), a=5, b=5, cube=1, g=match.DEFAULT_GAMMON_RATE)
    assert approx(m, 0.5, 1e-9), m


def test_clear_double_pass_when_far_ahead():
    # Near-certain win, money would cash; in a match with room it should double.
    d = match.match_cube_action(gammonless(0.92), a=7, b=7, cube=1, owner=CENTER)
    assert d.action in ("double/pass", "double/take"), d
    assert not d.take  # 92% winner -> opponent drops


def test_no_double_when_behind_in_game():
    d = match.match_cube_action(gammonless(0.40), a=7, b=7, cube=1, owner=CENTER)
    assert d.action == "no double", d


def test_opponent_owns_cannot_double():
    d = match.match_cube_action(gammonless(0.75), a=7, b=7, cube=1, owner=OPP)
    assert d.action == "no double", d


def test_trailer_takes_lighter_than_money():
    # 2-away vs 4-away: the trailer (4-away) faces a double at 2-away/... has strong
    # recube leverage, so the take point should sit below the money 25% dead point.
    tp = match.take_point_match(a=4, b=2, cube=1)   # taker is 4-away
    assert tp < 0.25, tp


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
