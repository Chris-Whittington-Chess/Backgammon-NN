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


def test_two_away_taker_takes_lighter_than_money():
    # The taker is 2-away, so a doubled game *wins the match* — the gain from
    # taking is enormous and the risk small. Take points here sit well under the
    # money 25%.
    tp = match.take_point_match(a=2, b=5, cube=1)   # taker 2-away, doubler 5-away
    assert tp < 0.20, tp


def test_two_away_doubler_raises_the_take_point():
    # The mirror image, and the one that is easy to get backwards: against a
    # 2-away DOUBLER the cube is dead at 2 — losing ends the match — so the take
    # is expensive and the take point rises above the money 25%.
    for taker in (3, 4, 5, 6, 7):
        tp = match.take_point_match(a=taker, b=2, cube=1)
        assert tp > 0.25, (taker, tp)


def test_never_drop_when_a_drop_loses_the_match():
    # Doubler is 1-away: passing hands them the match, so there is no price at
    # which dropping is right.
    for taker in range(1, 8):
        assert match.take_point_match(a=taker, b=1, cube=1) == 0.0, taker


def test_post_crawford_beats_crawford_for_the_trailer():
    # The Crawford game is played cubeless; the moment it is over the cube comes
    # back and the trailer's equity jumps. Reading one table for both was the bug
    # a single mwc() lookup used to have.
    assert match.mwc(2, 1) < 0.35                                # Crawford game
    assert match.mwc(2, 1, post_crawford=True) > 0.45            # cube is back
    for n in range(1, 10):
        assert approx(match.mwc(1, n, post_crawford=True),
                      1.0 - match.mwc(n, 1, post_crawford=True)), n


def test_published_table_disagrees_with_the_cubeless_recursion():
    # Guards the swap itself: the recursion is still reachable past the table's
    # edge, and it is biased towards the leader at lopsided scores — which is the
    # whole reason for using a published table.
    assert approx(match.mwc(1, 5), 0.84179)                      # Kazaross XG2
    recursion = match._mwc_recursive(1, 5, 260)
    assert recursion > match.mwc(1, 5) + 0.05, recursion


def test_cubeful_reduces_to_dead_at_zero_efficiency():
    # x = 0 means the cube can never be used, which is exactly playing the game
    # out at the current value. Anchors the blend at one end.
    d = [0.62, 0.18, 0.01, 0.12, 0.01]
    for owner in (CENTER, MOVER, OPP):
        assert approx(match.cubeful_mwc(d, 4, 4, 1, owner, x=0.0),
                      match._mwc_from_game(d, 4, 4, 1, match.DEFAULT_GAMMON_RATE)), owner


def test_holding_the_cube_is_worth_something():
    # The bug this model replaced: the value of keeping the cube was zero, so
    # doubling nearly always looked better. Owning it must beat not owning it,
    # and both must beat having the opponent own it.
    d = [0.62, 0.18, 0.01, 0.12, 0.01]
    mine = match.cubeful_mwc(d, 4, 4, 1, MOVER)
    centred = match.cubeful_mwc(d, 4, 4, 1, CENTER)
    theirs = match.cubeful_mwc(d, 4, 4, 1, OPP)
    assert mine > centred > theirs, (mine, centred, theirs)


def test_efficiency_actually_reaches_match_decisions():
    # A sweep that came out bit-identical across x was how we found the cube model
    # was missing entirely; guard against it silently going missing again.
    d = [0.62, 0.18, 0.01, 0.12, 0.01]
    lo = match.cubeful_mwc(d, 4, 4, 1, CENTER, x=0.2)
    hi = match.cubeful_mwc(d, 4, 4, 1, CENTER, x=0.9)
    assert abs(hi - lo) > 1e-4, (lo, hi)


def test_does_not_double_the_whole_window_away_at_two_away():
    # Leading 2-away, a doubled game wins the match, so the model used to double
    # on almost anything: 289 wrong doubles against 2 wrong holds at this score.
    # A modest edge is not enough.
    d = match.match_cube_action(gammonless(0.55), a=2, b=4, cube=1, owner=CENTER)
    assert d.action == "no double", d


def test_beyond_the_table_falls_back_and_stays_sane():
    big = match.MAX_AWAY + 5
    m = match.mwc(big, big)
    assert approx(m, 0.5, 1e-9), m
    assert 0.0 <= match.mwc(big, 3) <= match.mwc(3, big) <= 1.0


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
