"""Canonical-point tests for the money-game cube model (trainer/cube.py).

Run: .venv/Scripts/python trainer/test_cube.py
"""
from __future__ import annotations

import cube
from cube import CENTER, MOVER, OPP


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def gammonless(win):
    return [win, 0.0, 0.0, 0.0, 0.0]


def test_cubeless_equity_matches_convention():
    # even → 0; sure win → +1; a gammon-laden edge
    assert approx(cube.cubeless_equity(gammonless(0.5)), 0.0)
    assert approx(cube.cubeless_equity(gammonless(1.0)), 1.0)
    # win 60%, of wins half are gammons: win_g=0.30; opp no gammons
    assert approx(cube.cubeless_equity([0.6, 0.3, 0.0, 0.0, 0.0]), (2*0.6-1) + 0.3)


def test_take_point_gammonless():
    d = gammonless(0.5)  # w=l=1
    assert approx(cube.take_point(d, x=0.0), 0.25), cube.take_point(d, 0.0)   # dead
    assert approx(cube.take_point(d, x=1.0), 0.20), cube.take_point(d, 1.0)   # live


def test_cash_point_gammonless():
    d = gammonless(0.5)
    assert approx(cube.cash_point(d, x=0.0), 0.75), cube.cash_point(d, 0.0)   # dead
    assert approx(cube.cash_point(d, x=1.0), 0.80), cube.cash_point(d, 1.0)   # live


def test_cubeful_reduces_to_cubeless_at_x0():
    for w in (0.2, 0.4, 0.5, 0.7, 0.9):
        d = gammonless(w)
        for owner in (CENTER, MOVER, OPP):
            assert approx(cube.cubeful_equity(d, owner, x=0.0),
                          cube.cubeless_equity(d)), (w, owner)


def test_even_position_zero_equity():
    d = gammonless(0.5)
    assert approx(cube.cubeful_equity(d, CENTER, x=1.0), 0.0)


def test_owning_cube_is_worth_more_than_giving_it_up():
    # At the same position, owning the cube >= centered >= opponent owns it.
    for w in (0.35, 0.5, 0.65):
        d = gammonless(w)
        e_mine = cube.cubeful_equity(d, MOVER, x=0.7)
        e_center = cube.cubeful_equity(d, CENTER, x=0.7)
        e_opp = cube.cubeful_equity(d, OPP, x=0.7)
        assert e_mine >= e_center - 1e-9 >= e_opp - 2e-9, (w, e_mine, e_center, e_opp)


def test_take_drop_boundary():
    x = cube.DEFAULT_EFFICIENCY
    tp = cube.take_point(gammonless(0.5), x)  # ~0.216
    # just above the take point -> take; just below -> drop
    assert cube.should_take(-cube.cubeless_equity(gammonless(tp + 0.02)))
    assert not cube.should_take(-cube.cubeless_equity(gammonless(tp - 0.02)))


def test_cube_action_progression():
    # very behind -> no double; clear but takeable lead -> double/take;
    # near-certain -> double/pass; huge gammon threat -> too good.
    assert cube.cube_action(gammonless(0.45), CENTER).action == "no double"
    d_take = gammonless(0.70)
    assert cube.cube_action(d_take, CENTER).action in ("double/take", "no double")
    assert cube.cube_action(gammonless(0.90), CENTER).action == "double/pass"
    # 85% win, almost all wins are gammons -> playing on beats cashing +1
    too_good = [0.85, 0.80, 0.10, 0.0, 0.0]
    assert cube.cube_action(too_good, CENTER).action == "too good", \
        cube.cube_action(too_good, CENTER)


def test_gammon_threat_lowers_takers_take_point():
    # If the taker can win gammons, their take point drops (they risk less net).
    plain = cube.take_point(gammonless(0.30))
    with_gammons = cube.take_point([0.30, 0.15, 0.0, 0.0, 0.0])
    assert with_gammons < plain, (plain, with_gammons)


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
