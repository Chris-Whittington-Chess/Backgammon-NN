"""Doubling-cube decisions (SPEC §11), money-play heuristics.

Decisions are driven by the *mover's* cubeless equity from the evaluator (the
net or HCE). These are simplified but sound money-game rules:

- **Take point ~25%**: the opponent takes a double while their cubeless equity is
  at least -0.5 (i.e. the mover's equity is at most +0.5). Below that they drop
  and concede the current stake.
- **Doubling window**: the mover offers a double when clearly ahead but not so
  far ahead that playing on for a gammon is better ("too good").

Equity here is the standard cubeless value in points (range about -3..+3), so
gammon/backgammon chances shift the thresholds naturally.
"""

from __future__ import annotations

# Mover offers a double inside this equity window.
DOUBLE_MIN_E = 0.40
DOUBLE_MAX_E = 1.60      # beyond this, usually "too good" — play on for a gammon
# Taker accepts while their equity is at least this (≈ 25% cubeless win chance).
TAKE_MIN_TAKER_E = -0.50


def should_double(mover_equity: float, may_double: bool) -> bool:
    """Whether the side on roll should offer a double."""
    return may_double and DOUBLE_MIN_E <= mover_equity <= DOUBLE_MAX_E


def should_take(mover_equity: float) -> bool:
    """Whether the doubled player should take (True) or drop (False).
    `mover_equity` is the doubler's cubeless equity; the taker's is its negative.
    """
    return -mover_equity >= TAKE_MIN_TAKER_E
