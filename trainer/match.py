"""Match play (SPEC §11, Phase 2): match equity table, Crawford, cube decisions.

Money-game cube decisions (``cube.py``) trade in *points*; match-play decisions
trade in **match-winning chance (MWC)**, because points only matter through their
effect on the odds of reaching the match target. A take that is trivial for money
can be wrong at some scores and vice versa — the whole game changes near the end
of a match.

Match equity table
------------------
``mwc(a, b)`` = probability the player who needs ``a`` points wins the match,
opponent needs ``b``. Here it is built by a **self-consistent recursion** over game
outcomes rather than transcribed from a published table: a fresh game is symmetric
(either side wins ~50%), a decided game is a single (prob ``1-g``) or a gammon
(prob ``g``), and the recursion bottoms out when a side reaches 0-away. This is a
*cubeless* MET — it omits the extra value the cube gives the leader — so it is an
approximation, but a self-consistent, gammon-aware one that gets the anchors right
(DMP = 50%, ``mwc(a,b) = 1 - mwc(b,a)``, monotonic). ``MET_TABLE`` can be replaced
with published values (e.g. Kazaross-XG2) without touching the decision logic.

Cube decisions consume the MET through the standard risk/gain-in-MWC formulas.
Crawford: the Crawford game is played with no cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

# Fraction of *decided* games that end in a gammon (backgammons folded in as
# gammons for the table). ~0.26 is a common empirical figure.
DEFAULT_GAMMON_RATE = 0.26

# Optional override: a published table as {(a, b): mwc_for_a}. Empty -> use the
# recursion. Filling this (e.g. Kazaross-XG2) upgrades accuracy with no code change.
MET_TABLE: dict[tuple[int, int], float] = {}


@lru_cache(maxsize=None)
def _mwc_recursive(a: int, b: int, g_milli: int) -> float:
    if a <= 0:
        return 1.0
    if b <= 0:
        return 0.0
    g = g_milli / 1000.0
    single = 1.0 - g
    # I win this game (prob 1/2): opponent still needs b, I need a - (1 or 2).
    win = single * _mwc_recursive(a - 1, b, g_milli) + g * _mwc_recursive(a - 2, b, g_milli)
    # Opponent wins (prob 1/2): I still need a, opponent needs b - (1 or 2).
    loss = single * _mwc_recursive(a, b - 1, g_milli) + g * _mwc_recursive(a, b - 2, g_milli)
    return 0.5 * win + 0.5 * loss


def mwc(a: int, b: int, g: float = DEFAULT_GAMMON_RATE) -> float:
    """Match-winning chance for the side needing ``a`` points (opponent ``b``)."""
    if a <= 0:
        return 1.0
    if b <= 0:
        return 0.0
    if (a, b) in MET_TABLE:
        return MET_TABLE[(a, b)]
    return _mwc_recursive(a, b, int(round(g * 1000)))


def _clamp_away(a: int) -> int:
    return a if a > 0 else 0


def take_point_match(a: int, b: int, cube: int, g: float = DEFAULT_GAMMON_RATE,
                     gammon_win: float = 0.0, gammon_lose: float = 0.0) -> float:
    """Game-winning probability at which taking a double (cube -> 2*cube) equals
    dropping, in MWC terms. ``a``/``b`` are the taker's / doubler's away scores
    *before* the double. Gammon rates (of the taker) shift the take point.

    risk  = MWC(drop) - MWC(lose the doubled game)
    gain  = MWC(win the doubled game) - MWC(drop)
    take point = risk / (risk + gain).
    """
    d = 2 * cube
    # Drop: doubler banks `cube` points.
    mwc_drop = mwc(a, _clamp_away(b - cube), g)
    # Win the doubled game: taker scores d (or 2d on a gammon).
    mwc_win_single = mwc(_clamp_away(a - d), b, g)
    mwc_win_gammon = mwc(_clamp_away(a - 2 * d), b, g)
    mwc_win = (1 - gammon_win) * mwc_win_single + gammon_win * mwc_win_gammon
    # Lose the doubled game: doubler scores d (or 2d on a gammon).
    mwc_lose_single = mwc(a, _clamp_away(b - d), g)
    mwc_lose_gammon = mwc(a, _clamp_away(b - 2 * d), g)
    mwc_lose = (1 - gammon_lose) * mwc_lose_single + gammon_lose * mwc_lose_gammon

    risk = mwc_drop - mwc_lose
    gain = mwc_win - mwc_drop
    if risk + gain <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, risk / (risk + gain)))


@dataclass
class MatchCubeDecision:
    action: str          # "no double", "double/take", "double/pass", "no cube (crawford)"
    take: bool
    take_point: float    # taker's game-win threshold (for reference)
    mwc_nodouble: float
    mwc_double: float


def match_cube_action(dist: Sequence[float], a: int, b: int, cube: int,
                      owner: int, crawford: bool = False,
                      g: float = DEFAULT_GAMMON_RATE) -> MatchCubeDecision:
    """Match-play cube decision for the side on roll.

    ``dist`` is the mover's cubeless 5-vector; ``a``/``b`` are the mover's / the
    opponent's away scores; ``owner`` follows ``cube.py`` (CENTER/MOVER/OPP).
    """
    from cube import CENTER, MOVER  # local import to avoid a cycle at import time

    win, win_g, win_bg, lose_g, lose_bg = dist
    W = min(max(float(win), 0.0), 1.0)

    if crawford:
        # No cube this game. MWC of just playing it out.
        m = _mwc_from_game(dist, a, b, cube, g)
        return MatchCubeDecision("no cube (crawford)", take=False,
                                 take_point=0.0, mwc_nodouble=m, mwc_double=m)

    may_double = owner in (CENTER, MOVER) and cube < (1 << 10)
    # Taker (opponent) take point at these scores, given the mover's gammon rates
    # become the taker's *loss* gammons and vice versa.
    g_taker_win = (lose_g) / (1 - W) if W < 1 else 0.0     # opp wins gammon | opp wins
    g_taker_lose = (win_g) / W if W > 0 else 0.0           # opp loses gammon | opp loses
    tp = take_point_match(b, a, cube, g, gammon_win=g_taker_win, gammon_lose=g_taker_lose)
    opp_winprob = 1.0 - W
    take = opp_winprob > tp

    mwc_nodouble = _mwc_from_game(dist, a, b, cube, g)
    mwc_double_take = _mwc_from_game(dist, a, b, 2 * cube, g)
    # If they pass, mover banks `cube` points.
    mwc_double_pass = mwc(_clamp_away(a - cube), b, g)
    mwc_double = mwc_double_take if take else mwc_double_pass

    if not may_double:
        return MatchCubeDecision("no double", take=take, take_point=tp,
                                 mwc_nodouble=mwc_nodouble, mwc_double=mwc_nodouble)

    if mwc_double > mwc_nodouble + 1e-9:
        action = "double/take" if take else "double/pass"
    else:
        action = "no double"
    return MatchCubeDecision(action, take=take, take_point=tp,
                             mwc_nodouble=mwc_nodouble, mwc_double=mwc_double)


def _mwc_from_game(dist: Sequence[float], a: int, b: int, cube: int,
                   g: float) -> float:
    """Mover's MWC from playing the current game out at the given cube value,
    integrating over win/gammon/backgammon outcomes (no further cube action)."""
    win, win_g, win_bg, lose_g, lose_bg = dist
    p_win_single = win - win_g
    p_win_gammon = win_g - win_bg
    p_win_bg = win_bg
    p_lose_single = (1 - win) - lose_g
    p_lose_gammon = lose_g - lose_bg
    p_lose_bg = lose_bg
    out = 0.0
    for p, pts in ((p_win_single, 1), (p_win_gammon, 2), (p_win_bg, 3)):
        out += p * mwc(_clamp_away(a - pts * cube), b, g)
    for p, pts in ((p_lose_single, 1), (p_lose_gammon, 2), (p_lose_bg, 3)):
        out += p * mwc(a, _clamp_away(b - pts * cube), g)
    return out
