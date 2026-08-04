"""Match play (SPEC §11, Phase 2): match equity table, Crawford, cube decisions.

Money-game cube decisions (``cube.py``) trade in *points*; match-play decisions
trade in **match-winning chance (MWC)**, because points only matter through their
effect on the odds of reaching the match target. A take that is trivial for money
can be wrong at some scores and vice versa — the whole game changes near the end
of a match.

Match equity table
------------------
``mwc(a, b)`` = probability the player who needs ``a`` points wins the match,
opponent needs ``b``. The numbers come from **Kazaross XG2** (``met_kazaross.py``,
transcribed from gnubg), a rollout-derived published table.

It replaced a self-consistent cubeless recursion, which is kept below as a fallback
past the table's 25-away edge. That recursion was not merely imprecise, it was
biased: being cubeless it priced away the *trailer's* ability to double their way
back, so it flattered the leader by 7-9 points of match equity at exactly the
lopsided scores where cube decisions turn (1-away/5-away: 93.6% by the recursion,
84.2% by Kazaross). It got the anchors right — DMP = 50%, ``mwc(a,b) = 1-mwc(b,a)``,
monotonic — which is why the error was invisible from the tests.

Crawford and post-Crawford are *different tables*, not the same score read twice.
The Crawford game is played with no cube; the moment it is over the cube returns
and the trailer's equity jumps (2-away/1-away: 32.3% during Crawford, 48.8% after),
so every lookup carries a ``post_crawford`` flag. Note that the children of a
Crawford game are post-Crawford, which is why the two are threaded separately.

Cube decisions consume the MET through the standard risk/gain-in-MWC formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

from met_kazaross import MAX_AWAY, POST, PRE

# Fraction of *decided* games that end in a gammon (backgammons folded in as
# gammons for the table). ~0.26 is a common empirical figure. Only the fallback
# recursion uses this; the published table has gammons already rolled in.
DEFAULT_GAMMON_RATE = 0.26

# Cube efficiency for match play — the same parameter as ``cube.DEFAULT_EFFICIENCY``
# but deliberately a different value. Fitted against gnubg over 24,424 graded cube
# decisions (`trainer/grade_cube.py`), the minimum sits at 0.56 with an
# independent 24,424-decision validation sample putting it at 0.54; the curve is
# flat from 0.52 to 0.60, so 0.55 is the middle of a broad optimum rather than a
# point estimate. Money play fits 0.70 — the cube is genuinely *less* efficient in
# a match, because the score truncates it: at 2-away a doubled game already wins
# the match, so the recube that money play still has to fear is dead.
DEFAULT_EFFICIENCY_MATCH = 0.55

# Optional override, consulted before the published table: {(a, b): mwc_for_a}.
# Left empty in the repo — it exists so a caller can swap in a different MET
# (or pin a score for a test) without touching the decision logic.
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


def mwc(a: int, b: int, g: float = DEFAULT_GAMMON_RATE,
        post_crawford: bool = False) -> float:
    """Match-winning chance for the side needing ``a`` points (opponent ``b``).

    ``post_crawford`` selects the post-Crawford table, where the cube is back and
    one side is 1-away. It changes the answer a lot — see the module docstring —
    so callers must thread it through rather than let it default.
    """
    if a <= 0:
        return 1.0
    if b <= 0:
        return 0.0
    if (a, b) in MET_TABLE:
        return MET_TABLE[(a, b)]
    if post_crawford:
        # Post-Crawford one side is always 1-away, so a single row suffices; the
        # 1-away player's equity is the complement of the trailer's.
        if b == 1 and a <= MAX_AWAY:
            return POST[a - 1]
        if a == 1 and b <= MAX_AWAY:
            return 1.0 - POST[b - 1]
    elif a <= MAX_AWAY and b <= MAX_AWAY:
        return PRE[a - 1][b - 1]
    return _mwc_recursive(a, b, int(round(g * 1000)))


def _clamp_away(a: int) -> int:
    return a if a > 0 else 0


def take_point_match(a: int, b: int, cube: int, g: float = DEFAULT_GAMMON_RATE,
                     gammon_win: float = 0.0, gammon_lose: float = 0.0,
                     post_crawford: bool = False) -> float:
    """Game-winning probability at which taking a double (cube -> 2*cube) equals
    dropping, in MWC terms. ``a``/``b`` are the taker's / doubler's away scores
    *before* the double. Gammon rates (of the taker) shift the take point.

    risk  = MWC(drop) - MWC(lose the doubled game)
    gain  = MWC(win the doubled game) - MWC(drop)
    take point = risk / (risk + gain).
    """
    d = 2 * cube
    pc = post_crawford
    # Drop: doubler banks `cube` points.
    mwc_drop = mwc(a, _clamp_away(b - cube), g, pc)
    # Win the doubled game: taker scores d (or 2d on a gammon).
    mwc_win_single = mwc(_clamp_away(a - d), b, g, pc)
    mwc_win_gammon = mwc(_clamp_away(a - 2 * d), b, g, pc)
    mwc_win = (1 - gammon_win) * mwc_win_single + gammon_win * mwc_win_gammon
    # Lose the doubled game: doubler scores d (or 2d on a gammon).
    mwc_lose_single = mwc(a, _clamp_away(b - d), g, pc)
    mwc_lose_gammon = mwc(a, _clamp_away(b - 2 * d), g, pc)
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
                      g: float = DEFAULT_GAMMON_RATE,
                      post_crawford: bool = False,
                      x: float = DEFAULT_EFFICIENCY_MATCH) -> MatchCubeDecision:
    """Match-play cube decision for the side on roll.

    ``dist`` is the mover's cubeless 5-vector; ``a``/``b`` are the mover's / the
    opponent's away scores; ``owner`` follows ``cube.py`` (CENTER/MOVER/OPP).
    ``crawford`` means *this* game is the Crawford game; ``post_crawford`` means it
    has already been played, which is what puts the cube back in play at 1-away.
    """
    from cube import CENTER, MOVER, OPP  # local import to avoid a cycle at import time

    win, win_g, win_bg, lose_g, lose_bg = dist
    W = min(max(float(win), 0.0), 1.0)

    if crawford:
        # No cube this game. MWC of just playing it out — and whatever score it
        # ends at is post-Crawford, so the continuation is priced from that table.
        m = _mwc_from_game(dist, a, b, cube, g, post_crawford=True)
        return MatchCubeDecision("no cube (crawford)", take=False,
                                 take_point=0.0, mwc_nodouble=m, mwc_double=m)

    may_double = owner in (CENTER, MOVER) and cube < (1 << 10)
    # Taker (opponent) take point at these scores, given the mover's gammon rates
    # become the taker's *loss* gammons and vice versa. Reported for reference;
    # the take itself is decided on equity below, as in ``cube.cube_action``.
    g_taker_win = (lose_g) / (1 - W) if W < 1 else 0.0     # opp wins gammon | opp wins
    g_taker_lose = (win_g) / W if W > 0 else 0.0           # opp loses gammon | opp loses
    tp = take_point_match(b, a, cube, g, gammon_win=g_taker_win, gammon_lose=g_taker_lose,
                          post_crawford=post_crawford)

    # Holding the cube is worth more than playing the game out, and that value is
    # what the old dead-cube model threw away.
    mwc_nodouble = cubeful_mwc(dist, a, b, cube, owner, x, g, post_crawford)
    # After doubling, the opponent owns the cube at twice its value.
    mwc_double_take = cubeful_mwc(dist, a, b, 2 * cube, OPP, x, g, post_crawford)
    # If they pass, mover banks `cube` points.
    mwc_double_pass = mwc(_clamp_away(a - cube), b, g, post_crawford)
    take = mwc_double_take < mwc_double_pass
    mwc_double = mwc_double_take if take else mwc_double_pass

    if not may_double:
        return MatchCubeDecision("no double", take=take, take_point=tp,
                                 mwc_nodouble=mwc_nodouble, mwc_double=mwc_nodouble)

    if not take:
        # They would pass. Doubling is still wrong if playing on is worth more
        # than the points a pass banks — the match-play form of "too good".
        if mwc_nodouble > mwc_double_pass + 1e-12:
            return MatchCubeDecision("too good", take=False, take_point=tp,
                                     mwc_nodouble=mwc_nodouble,
                                     mwc_double=mwc_double_pass)
        return MatchCubeDecision("double/pass", take=False, take_point=tp,
                                 mwc_nodouble=mwc_nodouble, mwc_double=mwc_double_pass)

    action = "double/take" if mwc_double > mwc_nodouble + 1e-12 else "no double"
    return MatchCubeDecision(action, take=True, take_point=tp,
                             mwc_nodouble=mwc_nodouble, mwc_double=mwc_double)


def _outcome_mwcs(dist: Sequence[float], a: int, b: int, cube: int,
                  g: float, post_crawford: bool) -> tuple[float, float, float]:
    """``(W, mwc_given_win, mwc_given_loss)`` at the given cube value.

    Splitting the two conditionals out is what lets the cubeful model below
    anchor its live segments without re-integrating: the dead value is just
    ``W * m_win + (1 - W) * m_lose``.
    """
    win, win_g, win_bg, lose_g, lose_bg = dist
    W = min(max(float(win), 0.0), 1.0)
    L = 1.0 - W
    wins = ((win - win_g, 1), (win_g - win_bg, 2), (win_bg, 3))
    losses = ((L - lose_g, 1), (lose_g - lose_bg, 2), (lose_bg, 3))
    m_win = sum(p * mwc(_clamp_away(a - pts * cube), b, g, post_crawford)
                for p, pts in wins) / W if W > 1e-9 else 1.0
    m_lose = sum(p * mwc(a, _clamp_away(b - pts * cube), g, post_crawford)
                 for p, pts in losses) / L if L > 1e-9 else 0.0
    return W, m_win, m_lose


def _mwc_from_game(dist: Sequence[float], a: int, b: int, cube: int,
                   g: float, post_crawford: bool = False) -> float:
    """Mover's MWC from playing the current game out at the given cube value,
    integrating over win/gammon/backgammon outcomes (**no further cube action**).

    This is the *dead cube* value. Using it as the value of holding the cube was
    the flaw that graded at 45 mEMG against gnubg: it prices keeping the cube at
    zero, so doubling almost always looked better and the model over-doubled
    massively — 289 wrong doubles to 2 wrong holds at 2-away/4-away. It is still
    exactly right for a Crawford game, where the cube really is dead by rule.

    ``post_crawford`` describes the scores this game *leads to*, not the game being
    played — so a Crawford game passes ``True``, because once it ends the cube is
    back.
    """
    W, m_win, m_lose = _outcome_mwcs(dist, a, b, cube, g, post_crawford)
    return W * m_win + (1.0 - W) * m_lose


def _live_mwc(W: float, tp: float, cp: float, m_win: float, m_lose: float,
              m_cash: float, m_drop: float, owner: int) -> float:
    """Fully-live cubeful MWC, mover's view — the MWC analogue of
    ``cube._live_equity``.

    Same shape as the money model, with match-winning chances in place of its
    ±1 saturation points: above the cash point the mover doubles the opponent out
    and banks ``m_cash``; below its own take point it gets doubled out at
    ``m_drop``; in between the value is linear in the win probability. Which
    barriers apply depends on who holds the cube.
    """
    from cube import CENTER, MOVER

    def seg(lo_W, lo_E, hi_W, hi_E):
        if hi_W <= lo_W:
            return lo_E
        t = (W - lo_W) / (hi_W - lo_W)
        return lo_E + t * (hi_E - lo_E)

    if owner == CENTER:
        if W <= tp:
            return m_drop
        if W >= cp:
            return m_cash
        return seg(tp, m_drop, cp, m_cash)
    if owner == MOVER:
        # Mover holds the cube, so it cannot be doubled out: no lower barrier,
        # and the bottom end runs to the raw value of losing the game.
        if W >= cp:
            return m_cash
        return seg(0.0, m_lose, cp, m_cash)
    # Opponent owns it: mirror — the mover can be doubled out but cannot cash.
    if W <= tp:
        return m_drop
    return seg(tp, m_drop, 1.0, m_win)


def cubeful_mwc(dist: Sequence[float], a: int, b: int, cube: int, owner: int,
                x: float = DEFAULT_EFFICIENCY_MATCH,
                g: float = DEFAULT_GAMMON_RATE,
                post_crawford: bool = False) -> float:
    """Janowski cubeful MWC: the dead value blended with the fully-live one by
    the cube efficiency ``x``, exactly as ``cube.cubeful_equity`` does for money.

    The take and cash barriers are *score-dependent* here, which is the whole
    point — at 2-away a doubled game wins the match, so the window collapses.
    """
    win, win_g, _win_bg, lose_g, _lose_bg = dist
    W, m_win, m_lose = _outcome_mwcs(dist, a, b, cube, g, post_crawford)
    dead = W * m_win + (1.0 - W) * m_lose
    if x <= 0.0:
        return dead

    # Mover's own take point, facing a double from `cube` to 2 * cube.
    tp = take_point_match(a, b, cube, g,
                          gammon_win=(win_g / W if W > 1e-9 else 0.0),
                          gammon_lose=(lose_g / (1 - W) if W < 1 else 0.0),
                          post_crawford=post_crawford)
    # Cash point: the mirror of the opponent's take point.
    tp_opp = take_point_match(b, a, cube, g,
                              gammon_win=(lose_g / (1 - W) if W < 1 else 0.0),
                              gammon_lose=(win_g / W if W > 1e-9 else 0.0),
                              post_crawford=post_crawford)
    cp = 1.0 - tp_opp

    m_cash = mwc(_clamp_away(a - cube), b, g, post_crawford)   # opponent passes
    m_drop = mwc(a, _clamp_away(b - cube), g, post_crawford)   # mover passes
    live = _live_mwc(W, tp, cp, m_win, m_lose, m_cash, m_drop, owner)
    return (1.0 - x) * dead + x * live
