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


class ValueNet(nn.Module):
    """198 -> hidden layer(s) -> 5 (sigmoid) probability head.

    `hidden` is an ``int`` for one hidden layer (e.g. ``128``) or a list for a
    deeper net (e.g. ``[256, 128]``). Layer indices are unchanged for the
    single-layer ReLU case, so older 198->128->5 checkpoints still load. Widths
    that are multiples of 8/16/32 (128, 256) also tile cleanly for SIMD.
    `act` is ``"relu"`` or ``"sqrelu"`` (ReLU squared).
    """

    def __init__(self, hidden=128, act: str = "relu"):
        super().__init__()
        sizes = [hidden] if isinstance(hidden, int) else list(hidden)
        layers = []
        prev = NUM_INPUTS
        for h in sizes:
            layers += [nn.Linear(prev, h), _activation(act)]
            prev = h
        layers += [nn.Linear(prev, NUM_OUTPUTS), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def net_from_ckpt(ck) -> "ValueNet":
    """Rebuild the exact architecture recorded in a checkpoint dict."""
    net = ValueNet(ck.get("hidden", 128), ck.get("act", "relu"))
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


# ---------------------------------------------------------------------------
# Six-outcome (softmax) head — the phase-split experiment.
#
# Instead of the 5 nested sigmoids above, model the six *mutually exclusive*
# game outcomes directly and train with cross-entropy. Same equity as the
# 5-output net (verified by construction), but a properly normalised
# distribution and a loss that fits the one-hot Monte-Carlo target cleanly.
# ---------------------------------------------------------------------------

NUM_OUTCOMES = 6

# Points won (+) or lost (-) for each outcome class, in output order:
# [win single, win gammon, win bg, lose single, lose gammon, lose bg].
OUTCOME_POINTS = (1.0, 2.0, 3.0, -1.0, -2.0, -3.0)


class ValueNet6(nn.Module):
    """198 -> hidden layer(s) -> 6 logits over the mutually-exclusive outcomes.

    Same body as :class:`ValueNet`; the head is 6 raw logits (no sigmoid).
    Apply softmax for probabilities (`probs`), or train from the logits with
    cross-entropy against the outcome class.
    """

    def __init__(self, hidden=128, act: str = "relu"):
        super().__init__()
        sizes = [hidden] if isinstance(hidden, int) else list(hidden)
        layers = []
        prev = NUM_INPUTS
        for h in sizes:
            layers += [nn.Linear(prev, h), _activation(act)]
            prev = h
        layers += [nn.Linear(prev, NUM_OUTCOMES)]   # logits, no activation
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                          # logits [..., 6]

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def net6_from_state(sd, hidden, act) -> "ValueNet6":
    net = ValueNet6(hidden, act)
    net.load_state_dict(sd)
    net.eval()
    return net


def equity6(p: torch.Tensor) -> torch.Tensor:
    """Cubeless equity from a 6-outcome probability distribution `p` (`[..., 6]`):
    ``1*ws + 2*wg + 3*wbg - 1*ls - 2*lg - 3*lbg``. Equal to the 5-output
    ``equity`` for the same underlying position."""
    pts = torch.tensor(OUTCOME_POINTS, dtype=p.dtype, device=p.device)
    return (p * pts).sum(dim=-1)


def flip6(p: torch.Tensor) -> torch.Tensor:
    """Opponent-perspective distribution -> mover's: swap the win and lose halves.
    ``[ws, wg, wbg, ls, lg, lbg]`` -> ``[ls, lg, lbg, ws, wg, wbg]``."""
    ws, wg, wbg, ls, lg, lbg = p.unbind(dim=-1)
    return torch.stack([ls, lg, lbg, ws, wg, wbg], dim=-1)


def outcome_class(points: int) -> int:
    """The outcome class index (0..2) for a WIN of `points` (1..=3), from the
    winner's perspective: 1->win single, 2->win gammon, 3->win backgammon."""
    return min(points, 3) - 1


# ---------------------------------------------------------------------------
# Pip-count output buckets (Stockfish-NNUE style).
#
# One shared body, N_BUCKETS output heads (each the 6-outcome softmax), selected
# by *total* pip count (both sides). Total pips is the faithful analog of SF's
# total piece count: it decreases monotonically over the game and is
# perspective-invariant, so a position and its swap share a bucket and equity
# stays antisymmetric. The body trains on every position (no data starvation);
# only the selected head specialises.
# ---------------------------------------------------------------------------

# Bucket edges on total pip count (both sides), calibrated by
# trainer/calibrate_buckets.py to the octiles of champion self-play so each of the
# 8 buckets holds ~1/8 of positions (uniform 42-pip slices were ~90x imbalanced).
# MUST match the array in crates/bgcore/src/eval/nn.rs.
PIP_BUCKET_EDGES = (85, 131, 169, 205, 238, 271, 305)
N_BUCKETS = len(PIP_BUCKET_EDGES) + 1  # 8


def pip_bucket(total_pips: int) -> int:
    """Bucket index (0..N_BUCKETS-1) from the total pip count of both sides — the
    number of edges it meets or exceeds. Must match the Rust selector exactly."""
    return sum(1 for e in PIP_BUCKET_EDGES if total_pips >= e)


# Class-aware routing: race / crashed / contact, each split into total-pip
# sub-buckets (3 + 2 + 7 = 12 heads). The routing itself is the single-source-of-
# truth `Board.route_bucket()` in Rust (crates/bgcore/src/board.rs) — Python just
# calls it — so there are no edge constants to keep in sync here.
N_HEADS = 12


class ValueNetBucketed(nn.Module):
    """Shared body -> ``n_heads`` x 6-outcome heads, one selected per position.

    `forward` returns all heads' logits, shape ``[..., n_heads, 6]``. Training
    gathers the per-sample head; inference (Rust) runs all heads and slices the one
    the position routes to. ``n_heads`` defaults to the total-pip layout
    (`N_BUCKETS` = 8); the class-aware net uses `N_HEADS` = 12. Same body as ValueNet6.
    """

    def __init__(self, hidden=(256, 128), act: str = "relu", n_heads: int = N_BUCKETS,
                 num_inputs: int = NUM_INPUTS):
        super().__init__()
        sizes = [hidden] if isinstance(hidden, int) else list(hidden)
        layers = []
        prev = num_inputs
        for h in sizes:
            layers += [nn.Linear(prev, h), _activation(act)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.n_heads = n_heads
        self.heads = nn.Linear(prev, n_heads * NUM_OUTCOMES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.heads(h).view(*h.shape[:-1], self.n_heads, NUM_OUTCOMES)

    def logits_for(self, x: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        """Per-sample selected head logits, ``[N, 6]``. `buckets` is a LongTensor
        of head indices."""
        out = self.forward(x)                                   # [N, n_heads, 6]
        return out[torch.arange(out.size(0)), buckets]          # [N, 6]

    def probs_for(self, x: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits_for(x, buckets), dim=-1)


def net_bucketed_from_state(sd, hidden, act, n_heads: int = None) -> "ValueNetBucketed":
    if n_heads is None:  # infer from the head layer's width
        n_heads = sd["heads.weight"].shape[0] // NUM_OUTCOMES
    net = ValueNetBucketed(hidden, act, n_heads)
    net.load_state_dict(sd)
    net.eval()
    return net


# Raw Tesauro inputs and the appended strategic add-on (see features.rs strategic()).
RAW_INPUTS = 198
STRAT_INPUTS = 14


class ValueNetSplitBucketed(nn.Module):
    """Like ValueNetBucketed, but the 14 strategic features are injected **after
    the first ReLU** (the NNUE 'accumulator'), not at the raw input.

    The first layer transforms only the 198 raw board inputs; the strategic
    scalars are concatenated onto that activation and fed to the second layer, so
    the head reads them directly instead of diluting them through the board
    transform. Requires exactly two hidden layers. Input `x` is the full 212
    (raw ++ strategic); the net splits it internally.
    """

    def __init__(self, hidden=(256, 128), act: str = "relu", n_heads: int = N_HEADS,
                 raw_inputs: int = RAW_INPUTS, strat_inputs: int = STRAT_INPUTS):
        super().__init__()
        sizes = [hidden] if isinstance(hidden, int) else list(hidden)
        assert len(sizes) == 2, "split-inject net needs exactly two hidden layers"
        self.raw_inputs = raw_inputs
        self.strat_inputs = strat_inputs
        self.num_inputs = raw_inputs + strat_inputs
        self.n_heads = n_heads
        self.l1 = nn.Linear(raw_inputs, sizes[0])
        self.a1 = _activation(act)
        self.l2 = nn.Linear(sizes[0] + strat_inputs, sizes[1])
        self.a2 = _activation(act)
        self.heads = nn.Linear(sizes[1], n_heads * NUM_OUTCOMES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x[..., : self.raw_inputs]
        strat = x[..., self.raw_inputs : self.raw_inputs + self.strat_inputs]
        h = self.a1(self.l1(raw))
        h = torch.cat([h, strat], dim=-1)          # inject after the first ReLU
        h = self.a2(self.l2(h))
        return self.heads(h).view(*h.shape[:-1], self.n_heads, NUM_OUTCOMES)

    def logits_for(self, x: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return out[torch.arange(out.size(0)), buckets]

    def probs_for(self, x: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits_for(x, buckets), dim=-1)


def net_split_from_state(sd, hidden, act, n_heads: int = None) -> "ValueNetSplitBucketed":
    if n_heads is None:
        n_heads = sd["heads.weight"].shape[0] // NUM_OUTCOMES
    raw_inputs = sd["l1.weight"].shape[1]
    strat_inputs = sd["l2.weight"].shape[1] - sd["l1.weight"].shape[0]
    net = ValueNetSplitBucketed(hidden, act, n_heads, raw_inputs, strat_inputs)
    net.load_state_dict(sd)
    net.eval()
    return net
