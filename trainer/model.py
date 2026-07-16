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


class SquaredReLU(nn.Module):
    """ReLU squared (Primer): ``max(0, x)**2``, written as a Mul so it exports to
    ONNX ops (`Relu`, `Mul`) that tract supports."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.relu(x)
        return r * r


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "sqrelu": SquaredReLU}[name]()


class ProductPool(nn.Module):
    """Split the input in two halves, ReLU each, multiply elementwise.

    A bilinear / multiplicative unit — the pairwise-product trick from modern
    chess NNUE. A projection to ``2d`` becomes ``d`` features carrying degree-2
    interactions of the inputs, which a plain additive (Linear+ReLU) net can only
    approximate with many neurons. Backgammon's value function is full of such
    products (hit chance ≈ "I have a blot" × "opponent bears on it").

    ReLU on each half *before* the product bounds it away from the runaway
    magnitudes an unbounded activation (e.g. squared-ReLU) on the product would
    give. Exports to ONNX as ``Split`` + ``Relu`` + ``Mul`` — all tract ops.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return torch.relu(a) * torch.relu(b)


class ValueNet(nn.Module):
    """198 -> [optional product pool] -> hidden layer(s) -> 5 (sigmoid) head.

    `hidden` is an ``int`` for one hidden layer (e.g. ``128``) or a list for a
    deeper net (e.g. ``[256, 128]``); ``[]`` means straight from the pool/inputs
    to the output. Layer indices are unchanged for the single-layer ReLU case, so
    older 198->128->5 checkpoints still load. Widths that are multiples of
    8/16/32 tile cleanly for SIMD. `act` is ``"relu"`` or ``"sqrelu"``.

    `proj` (even, e.g. 512) inserts a `Linear(198, proj)` -> [`ProductPool`] first,
    so the hidden tail starts from `proj // 2` multiplicative features. `None`
    (default) is the plain MLP, unchanged.
    """

    def __init__(self, hidden=128, act: str = "relu", proj=None):
        super().__init__()
        sizes = [hidden] if isinstance(hidden, int) else list(hidden)
        layers = []
        prev = NUM_INPUTS
        if proj is not None:
            if proj % 2 != 0:
                raise ValueError(f"proj must be even (split in two), got {proj}")
            layers += [nn.Linear(NUM_INPUTS, proj), ProductPool()]
            prev = proj // 2
        for h in sizes:
            layers += [nn.Linear(prev, h), _activation(act)]
            prev = h
        layers += [nn.Linear(prev, NUM_OUTPUTS), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def net_from_ckpt(ck) -> "ValueNet":
    """Rebuild the exact architecture recorded in a checkpoint dict."""
    net = ValueNet(ck.get("hidden", 128), ck.get("act", "relu"), ck.get("proj"))
    net.load_state_dict(ck["model"])
    net.eval()
    return net


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
