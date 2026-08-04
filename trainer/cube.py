"""Doubling-cube decisions (SPEC §11) from the net's cubeless probabilities.

The evaluator emits the 5-vector ``[win, win_g, win_bg, lose_g, lose_bg]`` (nested:
``win`` = P(win any), ``win_g`` = P(win 2+), ``win_bg`` = P(win 3); likewise for
losses). That is exactly the input a cube decision needs.

Money game (this module, Phase 1): cubeful equity via Rick Janowski's model
(*Take Points in Money Games*, 1993), which interpolates between the **dead cube**
(cubeless equity — the cube can never be used) and the **fully live cube** (used
with perfect efficiency) by a single **cube-efficiency** parameter ``x`` in [0, 1].
Real cubes sit in between; ``x ≈ 0.65-0.70`` is the money-game standard, races
higher (the cube turns faster). Take/drop/double fall out of comparing cubeful
equities — no hand-tuned thresholds.

Canonical checks the tests pin (gammonless): take point 25% dead / 20% live, cash
point 75% dead / 80% live.

Match play (MET, Crawford) is Phase 2 — separate module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Cube efficiency x in Janowski's model: 0 = dead (cubeless), 1 = perfectly live.
DEFAULT_EFFICIENCY = 0.68

# Cube ownership, from the perspective of the side to move ("mover").
CENTER = 0     # nobody owns it (either may double)
MOVER = 1      # mover owns it (only mover may double)
OPP = -1       # opponent owns it (only opponent may double)


def _win_loss_values(dist: Sequence[float]) -> tuple[float, float, float]:
    """(W, w, l): win probability, and the average magnitude of a win / of a loss
    in points, folding gammons and backgammons in. Nested inputs, so
    P(win exactly single) = win - win_g, P(win gammon) = win_g - win_bg, etc.,
    giving expected win magnitude (win + win_g + win_bg) / win."""
    win, win_g, win_bg, lose_g, lose_bg = dist
    W = min(max(float(win), 0.0), 1.0)
    L = 1.0 - W
    w = (W + win_g + win_bg) / W if W > 1e-9 else 1.0
    l = (L + lose_g + lose_bg) / L if L > 1e-9 else 1.0
    return W, w, l


def cubeless_equity(dist: Sequence[float]) -> float:
    """Standard cubeless equity in points, matching ``model.equity`` / Rust."""
    win, win_g, win_bg, lose_g, lose_bg = dist
    return (2.0 * win - 1.0) + (win_g + win_bg) - (lose_g + lose_bg)


def take_point(dist: Sequence[float], x: float = DEFAULT_EFFICIENCY) -> float:
    """Mover's take point: the win probability below which taking a double is worse
    than dropping. Derived from equating take vs drop equity; the ``0.5 x`` term is
    the recube vig the live cube grants the taker. Gammonless: 0.25 (dead) → 0.20 (live)."""
    _, w, l = _win_loss_values(dist)
    return (l - 0.5) / (w + l + 0.5 * x)


def cash_point(dist: Sequence[float], x: float = DEFAULT_EFFICIENCY) -> float:
    """Mover's cash point: the win probability at/above which the mover can double
    the opponent out (opponent is at their take point). It is the opponent's take
    point mirrored. Gammonless: 0.75 (dead) → 0.80 (live)."""
    _, w, l = _win_loss_values(dist)
    return 1.0 - (w - 0.5) / (w + l + 0.5 * x)


def _live_equity(W: float, w: float, l: float, owner: int, x: float) -> float:
    """Fully-live (x = 1 reference) cubeful equity per unit cube, mover's view.

    Continuous model: in the doubling window the cubeful equity is linear in W
    between the two barriers, and it saturates at ±1 (single cube turn) once a
    side can cash. Which barriers apply depends on who holds the cube:
      - CENTER: both barriers live — bounded in [-1, +1].
      - MOVER owns: mover can double (upper barrier at the cash point) but cannot
        be doubled out, so the lower end runs to the raw loss value -l.
      - OPP owns: mirror — capped at +w above, -1 barrier below.
    """
    # Barriers as win-probabilities, using the same denominator that gave the
    # canonical take/cash points (x folded in so this is the x-live reference).
    tp = (l - 0.5) / (w + l + 0.5)   # mover gets doubled out below this
    cp = 1.0 - (w - 0.5) / (w + l + 0.5)  # mover cashes at/above this

    def seg(lo_W, lo_E, hi_W, hi_E):
        if hi_W <= lo_W:
            return lo_E
        t = (W - lo_W) / (hi_W - lo_W)
        return lo_E + t * (hi_E - lo_E)

    if owner == CENTER:
        if W <= tp:
            return -1.0
        if W >= cp:
            return 1.0
        return seg(tp, -1.0, cp, 1.0)
    if owner == MOVER:
        # No lower cash for the opponent; mover holds. Below cp, equity runs from
        # the pure-loss value at W=0 up to +1 at the cash point.
        if W >= cp:
            return 1.0
        return seg(0.0, -l, cp, 1.0)
    # OPP owns: mirror of MOVER.
    if W <= tp:
        return -1.0
    return seg(tp, -1.0, 1.0, w)


def cubeful_equity(dist: Sequence[float], owner: int = CENTER,
                   x: float = DEFAULT_EFFICIENCY) -> float:
    """Janowski cubeful equity per unit cube, mover's perspective: a blend of the
    dead cube (cubeless) and the fully-live cube by efficiency ``x``."""
    W, w, l = _win_loss_values(dist)
    e_dead = W * w - (1.0 - W) * l          # == cubeless_equity(dist)
    e_live = _live_equity(W, w, l, owner, x)
    return (1.0 - x) * e_dead + x * e_live


@dataclass
class CubeDecision:
    action: str          # "no double", "double/take", "double/pass", "too good"
    equity_nodouble: float
    equity_double: float  # mover's cubeful equity if it doubles (opp takes optimally)
    take: bool            # would the opponent take?


def cube_action(dist: Sequence[float], owner: int = CENTER,
                x: float = DEFAULT_EFFICIENCY) -> CubeDecision:
    """Money-game cube decision for the side on roll.

    Compares holding the cube against doubling. The opponent takes iff their
    resulting equity beats dropping (−1 to them, i.e. +1 to us). "Too good" is a
    double the opponent would drop but where playing on (holding) is worth *more*
    than the +1 a pass banks — i.e. we cash gammons instead.
    """
    W, w, l = _win_loss_values(dist)
    may_double = owner in (CENTER, MOVER)

    e_hold = cubeful_equity(dist, owner, x)
    if not may_double:
        return CubeDecision("no double", e_hold, e_hold, take=W < cash_point(dist, x))

    # After doubling, we own nothing (cube goes to opp) at value 2. Opponent's
    # take/drop: they take iff our post-double equity < +1 (their drop loss).
    owner_after = OPP  # opponent now owns the (turned) cube
    e_double_if_take = 2.0 * cubeful_equity(dist, owner_after, x)
    take = e_double_if_take < 1.0
    e_double = e_double_if_take if take else 1.0  # if they pass, we bank +1

    if not take:
        # They pass. Double only if banking +1 beats playing on for gammons.
        if e_hold > 1.0:
            return CubeDecision("too good", e_hold, 1.0, take=False)
        return CubeDecision("double/pass", e_hold, 1.0, take=False)

    action = "double/take" if e_double >= e_hold else "no double"
    return CubeDecision(action, e_hold, e_double, take=True)


# --- Backwards-compatible API used by gui/app.py --------------------------------

def should_double(mover_equity: float, may_double: bool) -> bool:
    """Deprecated shim: the app passes only a scalar cubeless equity. Kept working
    so the GUI import doesn't break; prefer ``cube_action`` with the full 5-vector.
    Maps the scalar to a gammonless distribution and asks the real model."""
    if not may_double:
        return False
    dist = _dist_from_equity(mover_equity)
    return cube_action(dist, CENTER).action in ("double/take", "double/pass")


def should_take(mover_equity: float) -> bool:
    """Deprecated shim (see ``should_double``): the taker holds ``-mover_equity``."""
    dist = _dist_from_equity(-mover_equity)
    W, _, _ = _win_loss_values(dist)
    return W >= take_point(dist)


def _dist_from_equity(equity: float) -> list[float]:
    """A gammonless distribution whose cubeless equity is ``equity`` (in −1..+1
    after clamping): win = (E+1)/2, no gammons. Only for the scalar shims."""
    e = min(max(equity, -1.0), 1.0)
    win = (e + 1.0) / 2.0
    return [win, 0.0, 0.0, 0.0, 0.0]
