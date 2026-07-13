"""Neural-network value model for the backgammon engine (SPEC §6, §8).

The net maps the 198 Tesauro inputs to five cubeless probabilities
``[win, win_gammon, win_backgammon, lose_gammon, lose_backgammon]``. Equity and
the perspective-flip below mirror ``bgcore::eval::Value`` exactly so Rust and
Python agree on what the outputs mean.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_INPUTS = 198
NUM_OUTPUTS = 5


class ValueNet(nn.Module):
    """198 -> hidden -> 5 (sigmoid) probability head."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_INPUTS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_OUTPUTS),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def equity(v: torch.Tensor) -> torch.Tensor:
    """Cubeless equity from the 5-vector(s), matching ``Value::equity`` in Rust:
    ``(win - lose) + (win_g - lose_g) + (win_bg - lose_bg)`` with ``lose = 1-win``.
    Accepts shape ``[..., 5]`` and returns shape ``[...]``.
    """
    win, win_g, win_bg, lose_g, lose_bg = v.unbind(dim=-1)
    lose = 1.0 - win
    return (win - lose) + (win_g - lose_g) + (win_bg - lose_bg)


def flip(v: torch.Tensor) -> torch.Tensor:
    """Convert a value vector from the opponent's perspective to the mover's.

    ``[win, win_g, win_bg, lose_g, lose_bg]`` ->
    ``[1-win, lose_g, lose_bg, win_g, win_bg]``. This is the two-player identity
    that lets one network score both sides: my win chance is the opponent's loss
    chance, and gammon/backgammon win/loss terms swap.
    """
    win, win_g, win_bg, lose_g, lose_bg = v.unbind(dim=-1)
    return torch.stack([1.0 - win, lose_g, lose_bg, win_g, win_bg], dim=-1)


def outcome_vector(points: int) -> list[float]:
    """The terminal target from the winner's perspective: +points -> a win
    vector; the sign convention is handled by the caller. `points` in 1..=3."""
    return [1.0, float(points >= 2), float(points >= 3), 0.0, 0.0]
