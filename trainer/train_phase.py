"""Phase-split self-play trainer: two six-output nets, contact and race.

A single evaluator has to be good at two quite different games at once — the
tactical contact game (hits, primes, blocks) and the pure race/bear-off. This
trains **two** nets, picked per position by ``Board.no_contact()``: the contact
net plays while checkers can still hit, the race net takes over once the armies
have passed. Same idea as wildbg's split.

Both nets are the six-outcome softmax head (`model.ValueNet6`): the six mutually
exclusive results [win s/g/bg, lose s/g/bg], trained with cross-entropy toward
the actual Monte-Carlo outcome (lambda=1). The two nets are architecturally
identical for now — one ``--hidden`` shape, one ``--act``.

This is a Python-only experiment: it trains and evaluates two nets and never
touches the live 5-output net, its ONNX, or the app. Porting the six-output
`Value`/`NnEval` and phase routing into Rust comes only if a phase net earns it.

Run:
  .venv/Scripts/python trainer/train_phase.py --hidden 256,128 --iters 200 --games 200
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bgcore
from model import ValueNet6, equity6, outcome_class
# Net-agnostic helpers reused verbatim.
from train import roll, random_policy, hce_policy, play_match_game

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PHASES = ("contact", "race")


def phase_of(board) -> str:
    return "race" if board.no_contact() else "contact"


@torch.no_grad()
def children_mover_equity(nets, board, d1, d2):
    """For each legal move, the equity to the side to move, with every resulting
    position scored by the net for *its* phase.

    Returns ``(children, mover_eq, terminal_points)``: `children` are the mover's
    resulting boards (turn not yet passed), `mover_eq` a tensor of equities to the
    mover, `terminal_points` the win points where a move ends the game else None.
    """
    children = bgcore.legal_moves(board, d1, d2)
    term = [c.winner_points() for c in children]
    feats = bgcore.next_state_features(board, d1, d2)   # opponent-to-move, aligned
    opp_eq = torch.zeros(len(children))
    # Route each child to its phase net (no_contact is symmetric, so the child's
    # own phase is well-defined regardless of whose turn it is).
    for ph in PHASES:
        idx = [i for i, c in enumerate(children) if phase_of(c) == ph]
        if not idx:
            continue
        x = torch.from_numpy(np.asarray([feats[i] for i in idx], dtype=np.float32))
        eq = equity6(torch.softmax(nets[ph](x), dim=-1))   # opponent's equity
        for k, i in enumerate(idx):
            opp_eq[i] = eq[k]
    mover_eq = -opp_eq
    for i, p in enumerate(term):
        if p is not None:
            mover_eq[i] = float(p)                          # a win is worth its points
    return children, mover_eq, term


def choose_next_phase(nets, board, d1, d2, epsilon, rng):
    """Pick a move by phase-routed equity (epsilon-greedy). Returns
    ``(next_board_or_None, terminal_points_or_None)``, next board with the turn
    passed."""
    children, mover_eq, term = children_mover_equity(nets, board, d1, d2)
    i = rng.randrange(len(children)) if rng.random() < epsilon else int(mover_eq.argmax())
    if term[i] is not None:
        return None, term[i]
    return children[i].swap_perspective(), None


def phase_policy(nets):
    """A greedy match policy over the two nets, for benchmarks / head-to-head."""
    def f(board, d1, d2, rng):
        children, mover_eq, term = children_mover_equity(nets, board, d1, d2)
        i = int(mover_eq.argmax())
        return (None, term[i]) if term[i] is not None else (children[i].swap_perspective(), None)
    return f


def play_game_phase(nets, epsilon, rng, max_plies=4000):
    """Self-play one game. Returns ``(feature_list, phase_list, terminal_points)``
    — each decision state's 198 features and its phase."""
    board = bgcore.Board.starting()
    feats, phases = [], []
    for _ in range(max_plies):
        feats.append(board.features())
        phases.append(phase_of(board))
        d1, d2 = roll(rng)
        nb, pts = choose_next_phase(nets, board, d1, d2, epsilon, rng)
        if pts is not None:
            return feats, phases, pts
        board = nb
    return feats, phases, 0


def outcome_classes(n_states: int, points: int) -> list[int]:
    """Per-state target class for a game the last mover won by `points`.

    lambda=1, so each state's target is the real result from that state's mover's
    perspective. The mover alternates every ply, so states an even number of
    plies from the end share the winner's result (win class 0..2); odd states get
    the mirror loss class (3..5)."""
    base = outcome_class(points)
    return [base if (n_states - 1 - t) % 2 == 0 else base + 3 for t in range(n_states)]


def train_iter_phase(nets, opts, games, epsilon, rng, batch=1024):
    """One self-play + learn iteration. Each net trains only on its own phase's
    states, with cross-entropy toward the outcome class."""
    feats = {ph: [] for ph in PHASES}
    cls = {ph: [] for ph in PHASES}
    plies = 0
    for _ in range(games):
        f_list, ph_list, pts = play_game_phase(nets, epsilon, rng)
        if pts == 0:
            continue
        for f, ph, c in zip(f_list, ph_list, outcome_classes(len(f_list), pts)):
            feats[ph].append(f)
            cls[ph].append(c)
        plies += len(f_list)

    losses, counts = {}, {}
    for ph in PHASES:
        counts[ph] = len(feats[ph])
        if not feats[ph]:
            losses[ph] = 0.0
            continue
        x = torch.from_numpy(np.asarray(feats[ph], dtype=np.float32))
        y = torch.tensor(cls[ph], dtype=torch.long)
        total, count = 0.0, 0
        perm = torch.randperm(x.size(0))
        for i in range(0, x.size(0), batch):
            idx = perm[i : i + batch]
            opts[ph].zero_grad()
            loss = F.cross_entropy(nets[ph](x[idx]), y[idx])
            loss.backward()
            opts[ph].step()
            total += loss.item() * idx.numel()
            count += idx.numel()
        losses[ph] = total / max(count, 1)
    return plies, losses, counts


def benchmark_phase(nets, opponent, games, rng):
    """Phase engine vs a fixed opponent policy, both seats. Returns
    (win_rate, points_per_game)."""
    me = phase_policy(nets)
    wins, pts = 0, 0
    for g in range(games):
        seat = g % 2
        p0, p1 = (me, opponent) if seat == 0 else (opponent, me)
        w, p = play_match_game(p0, p1, rng)
        if w == seat:
            wins += 1
            pts += p
        else:
            pts -= p
    return wins / games, pts / games


def champion_policy(champion_file):
    """A greedy policy for the current 5-output champion, to head-to-head against.
    Plays as it was benchmarked during its own training (race-aware)."""
    from model import net_from_ckpt
    from train import net_policy
    ck = torch.load(MODELS_DIR / champion_file, map_location="cpu")
    return net_policy(net_from_ckpt(ck))


def build_nets(hidden, act):
    return {ph: ValueNet6(hidden, act) for ph in PHASES}


def save_ckpt(path, nets, opts, hidden, act, it):
    torch.save(
        {"contact": nets["contact"].state_dict(), "race": nets["race"].state_dict(),
         "opt_contact": opts["contact"].state_dict(), "opt_race": opts["race"].state_dict(),
         "hidden": hidden, "act": act, "outputs": 6, "phase_split": True, "iter": it},
        path,
    )


def load_phase_nets(ck):
    """Rebuild the two eval-mode nets from a phase-split checkpoint."""
    nets = build_nets(ck["hidden"], ck.get("act", "relu"))
    nets["contact"].load_state_dict(ck["contact"])
    nets["race"].load_state_dict(ck["race"])
    for n in nets.values():
        n.eval()
    return nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=200, help="self-play games per iter")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="256,128",
                    help="hidden sizes for BOTH nets, e.g. 128 or 256,128")
    ap.add_argument("--act", choices=["relu", "sqrelu"], default="relu")
    ap.add_argument("--bench-every", type=int, default=5)
    ap.add_argument("--bench-games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="td_phase.pt")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--champion", type=str, default=None,
                    help="5-output net to head-to-head against at each bench (e.g. td_latest.pt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.hidden.split(",") if x.strip()]
    hidden = sizes[0] if len(sizes) == 1 else sizes
    MODELS_DIR.mkdir(exist_ok=True)

    nets = build_nets(hidden, args.act)
    opts = {ph: torch.optim.Adam(nets[ph].parameters(), lr=args.lr) for ph in PHASES}

    start_iter = 0
    if args.resume:
        ck = torch.load(MODELS_DIR / args.resume, map_location="cpu")
        if ck.get("hidden") != hidden or ck.get("act") != args.act or not ck.get("phase_split"):
            raise SystemExit(
                f"--resume {args.resume} is hidden={ck.get('hidden')} act={ck.get('act')} "
                f"phase_split={ck.get('phase_split')}, but you asked for hidden={hidden} "
                f"act={args.act} phase_split=True")
        nets["contact"].load_state_dict(ck["contact"])
        nets["race"].load_state_dict(ck["race"])
        if "opt_contact" in ck:
            opts["contact"].load_state_dict(ck["opt_contact"])
            opts["race"].load_state_dict(ck["opt_race"])
        start_iter = int(ck.get("iter", 0))
        print(f"Resumed {args.resume} at iter {start_iter}")

    champ = champion_policy(args.champion) if args.champion else None
    n_params = sum(p.numel() for p in nets["contact"].parameters())
    print(f"Phase-split TD(lambda=1) | hidden={hidden} act={args.act} | "
          f"{n_params:,} params/net x2 | lr={args.lr}"
          + (f" | vs champion {args.champion}" if champ else ""))

    for it in range(start_iter + 1, start_iter + args.iters + 1):
        eps = max(0.02, 0.20 * (0.96 ** it))
        t0 = time.time()
        plies, losses, counts = train_iter_phase(nets, opts, args.games, eps, rng)
        dt = time.time() - t0
        line = (f"iter {it:3d} | eps {eps:.3f} | plies {plies:5d} | "
                f"loss c {losses['contact']:.3f} r {losses['race']:.3f} | "
                f"states c {counts['contact']} r {counts['race']} | {dt:4.1f}s")

        if it % args.bench_every == 0 or it == start_iter + args.iters:
            wr_r, ppg_r = benchmark_phase(nets, random_policy(), args.bench_games, rng)
            wr_h, ppg_h = benchmark_phase(nets, hce_policy(), args.bench_games, rng)
            line += (f" || vs Random {100*wr_r:.1f}% ({ppg_r:+.2f})"
                     f" | vs HCE {100*wr_h:.1f}% ({ppg_h:+.2f})")
            if champ is not None:
                wr_c, ppg_c = benchmark_phase(nets, champ, args.bench_games, rng)
                line += f" | vs champ {100*wr_c:.1f}% ({ppg_c:+.2f})"
            save_ckpt(MODELS_DIR / args.out, nets, opts, hidden, args.act, it)
        print(line, flush=True)

    print(f"\nSaved phase-split checkpoint to {MODELS_DIR / args.out}")


if __name__ == "__main__":
    main()
