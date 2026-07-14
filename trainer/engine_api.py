"""Engine adapters for the GUI (SPEC §10): neural / HCE / random behind one API.

Each engine can `analyze` a position (rank every legal full turn by equity, best
first) and `choose` the move it would play. Move steps come from the Rust engine
so hints can be shown in standard notation.
"""

from __future__ import annotations

import random
from pathlib import Path

import bgcore

BAR, OFF = bgcore.BAR, bgcore.OFF

# The 21 distinct dice rolls with their probabilities (doubles 1/36, else 2/36).
ALL21 = [
    (a, c, (1.0 if a == c else 2.0) / 36.0) for a in range(1, 7) for c in range(a, 7)
]


def format_steps(steps) -> str:
    """Render a list of (from, to, die) steps as e.g. '24/23 13/11', 'bar/20',
    '6/off'."""
    def one(frm, to):
        s = "bar" if frm == BAR else str(frm)
        d = "off" if to == OFF else str(to)
        return f"{s}/{d}"

    return " ".join(one(f, t) for f, t, _ in steps) or "(dance)"


class BaseEngine:
    name = "Base"

    def analyze(self, board, d1, d2):
        """Return [(steps, result_board, mover_equity), ...], best first."""
        raise NotImplementedError

    def choose(self, board, d1, d2):
        """Return (next_board_or_None, points_or_None, steps, equity)."""
        steps, result, eq = self.analyze(board, d1, d2)[0]
        pts = result.winner_points()
        if pts is not None and pts > 0:
            return None, pts, steps, eq  # this move wins the game
        return result.swap_perspective(), None, steps, eq


class RandomEngine(BaseEngine):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def analyze(self, board, d1, d2):
        moves = [(s, r, 0.0) for s, r in bgcore.legal_moves_with_steps(board, d1, d2)]
        self.rng.shuffle(moves)
        return moves


class HceEngine(BaseEngine):
    name = "HCE (heuristic)"

    def analyze(self, board, d1, d2):
        scored = []
        for steps, result in bgcore.legal_moves_with_steps(board, d1, d2):
            pts = result.winner_points()
            eq = float(pts) if (pts is not None and pts > 0) else -bgcore.hce_equity(
                result.swap_perspective()
            )
            scored.append((steps, result, eq))
        scored.sort(key=lambda t: -t[2])
        return scored


class RolloutEngine(BaseEngine):
    """Native parallel Monte-Carlo rollouts (via the Rust `bgcore.Rollouts`,
    which needs the onnx-feature bindings). Strong but heavy — a second or so per
    move at these settings."""

    def __init__(self, onnx_path, trials=0, truncate_plies=9, candidates=5, seed=1,
                 movetime_ms=800, threads=0):
        self._ro = bgcore.Rollouts(
            str(onnx_path), trials, truncate_plies, candidates, seed, movetime_ms, threads)
        self.name = (f"Rollout ({movetime_ms}ms)" if movetime_ms
                     else f"Rollout ({trials}×{truncate_plies})")

    def analyze(self, board, d1, d2):
        chosen, eq = self._ro.best_move(board, d1, d2)  # eq: mover's perspective
        steps = next(
            (s for s, r in bgcore.legal_moves_with_steps(board, d1, d2) if r == chosen), []
        )
        return [(steps, chosen, eq)]


class NeuralEngine(BaseEngine):
    """Neural evaluator with optional 1-ply search.

    `lookahead=0` ranks moves by the net's static equity. `lookahead=1` scores
    each move by averaging over all 21 opponent rolls of the opponent's best
    static reply — stronger, but ~21x slower.
    """

    def __init__(self, ckpt_path: str | Path, lookahead: int = 0, share=None):
        import numpy as np
        import torch

        if share is not None:
            self.net, self._torch, self._np, self._equity = (
                share.net, share._torch, share._np, share._equity
            )
            base = share.name.split(" —")[0]
        else:
            from model import ValueNet, equity

            ck = torch.load(ckpt_path, map_location="cpu")
            self.net = ValueNet(ck.get("hidden", 128))
            self.net.load_state_dict(ck["model"])
            self.net.eval()
            self._torch, self._np, self._equity = torch, np, equity
            base = f"Neural (iter {ck.get('iter', '?')})"

        self.lookahead = lookahead
        self.name = f"{base} — {lookahead}-ply"

    def static_equity(self, board):
        """Net equity for the side to move at a single position (0-ply)."""
        return float(self._static_equity_batch([board.features()])[0])

    def win_prob(self, board):
        """Net win probability P(win) for the side to move at `board`."""
        x = self._torch.from_numpy(self._np.asarray([board.features()], dtype=self._np.float32))
        with self._torch.no_grad():
            return float(self.net(x)[0, 0])

    def _static_equity_batch(self, feats):
        """Equity (from each position's mover perspective) for a batch of
        feature vectors."""
        x = self._torch.from_numpy(self._np.asarray(feats, dtype=self._np.float32))
        with self._torch.no_grad():
            return self._equity(self.net(x))

    def _one_ply_value(self, board):
        """Expected equity for the side to move at `board`, assuming it rolls and
        plays its best static reply (matches Rust `position_value(board, 1)`)."""
        total = 0.0
        for a, c, w in ALL21:
            children = bgcore.legal_moves(board, a, c)
            # Grandchild features are opponent-of-opponent = this side; negate to
            # get this side's value of each reply.
            gc_eq = self._static_equity_batch(bgcore.next_state_features(board, a, c))
            best = None
            for i, ch in enumerate(children):
                pts = ch.winner_points()
                v = float(pts) if (pts is not None and pts > 0) else -float(gc_eq[i])
                if best is None or v > best:
                    best = v
            total += w * best
        return total

    # Candidate limits for 2-ply search (root / opponent node) — keep it usable.
    ROOT_CAND = 4
    OPP_CAND = 3

    def _two_ply_value(self, board):
        """Expected equity for the side to move at `board`, searched 2-ply with
        candidate pruning at the opponent node (matches the Rust pruned search)."""
        total = 0.0
        for a, c, w in ALL21:
            children = bgcore.legal_moves(board, a, c)
            eq0 = self._static_equity_batch(bgcore.next_state_features(board, a, c))
            vals0 = []
            for i, ch in enumerate(children):
                pts = ch.winner_points()
                vals0.append(float(pts) if (pts is not None and pts > 0) else -float(eq0[i]))
            order = sorted(range(len(children)), key=lambda i: -vals0[i])[: self.OPP_CAND]
            best = None
            for i in order:
                ch = children[i]
                pts = ch.winner_points()
                v = float(pts) if (pts is not None and pts > 0) else -self._one_ply_value(
                    ch.swap_perspective())
                if best is None or v > best:
                    best = v
            total += w * best
        return total

    def analyze(self, board, d1, d2):
        moves = bgcore.legal_moves_with_steps(board, d1, d2)
        scored = []
        if self.lookahead == 0:
            opp_eq = self._static_equity_batch(bgcore.next_state_features(board, d1, d2))
            for i, (steps, result) in enumerate(moves):
                pts = result.winner_points()
                eq = float(pts) if (pts is not None and pts > 0) else -float(opp_eq[i])
                scored.append((steps, result, eq))
        elif self.lookahead == 1:
            for steps, result in moves:
                pts = result.winner_points()
                eq = (float(pts) if (pts is not None and pts > 0)
                      else -self._one_ply_value(result.swap_perspective()))
                scored.append((steps, result, eq))
        else:  # 2-ply with root candidate pruning
            opp_eq = self._static_equity_batch(bgcore.next_state_features(board, d1, d2))
            base = []
            for i, (steps, result) in enumerate(moves):
                pts = result.winner_points()
                v0 = float(pts) if (pts is not None and pts > 0) else -float(opp_eq[i])
                base.append(v0)
            top = set(sorted(range(len(moves)), key=lambda i: -base[i])[: self.ROOT_CAND])
            for i, (steps, result) in enumerate(moves):
                pts = result.winner_points()
                if pts is not None and pts > 0:
                    eq = float(pts)
                elif i in top:
                    eq = -self._two_ply_value(result.swap_perspective())
                else:
                    eq = base[i]  # shallow value; won't be the best anyway
                scored.append((steps, result, eq))
        scored.sort(key=lambda t: -t[2])
        return scored
