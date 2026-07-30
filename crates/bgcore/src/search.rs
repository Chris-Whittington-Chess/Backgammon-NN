//! Depth-limited expectiminimax search with candidate pruning (SPEC §5).
//!
//! 0-ply ranks moves by static evaluation. n-ply looks n half-moves deeper,
//! averaging over all 21 dice rolls at each chance node. Full-width 2-ply is
//! ~170x the cost of a 1-ply decision, so — like GNU Backgammon — at deep nodes
//! we shallow-rank the legal moves (0-ply) and only search the best few
//! (`candidates`). This keeps 2-ply to a fraction of a second per move with
//! little strength loss.

use std::cell::Cell;
use std::hash::{Hash, Hasher};

use crate::board::Board;
use crate::dice::{Dice, Rng};
use crate::eval::Evaluator;
use crate::game::{result, Engine, GameResult};
use crate::moves::{genmoves, Move};

/// Deterministic per-position seed so the sampled PV extension is reproducible.
fn board_seed(b: &Board) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    b.hash(&mut h);
    h.finish() ^ 0x9E37_79B9_7F4A_7C15
}

/// How many moves to keep at a pruned node, as a function of its *remaining*
/// search depth. `Uniform(n)` keeps `n` everywhere (`0` = full width — the old
/// single-`candidates` behaviour). `PerDepth(s)` keeps `s[depth]` at a node with
/// that remaining depth (out of range ⇒ full width), so a search can prune wider
/// near the root than deep — for a D-ply search `s[D]` is the root, `s[D-1]` the
/// next ply, and `s[1]`/`s[0]` are unused (the last ply and leaves aren't pruned).
#[derive(Clone, Copy)]
pub enum Cands<'a> {
    Uniform(usize),
    PerDepth(&'a [usize]),
}

impl Cands<'_> {
    /// Candidate limit at a node with `depth` half-moves remaining (`0` = full width).
    #[inline]
    fn at(&self, depth: u8) -> usize {
        match self {
            Cands::Uniform(n) => *n,
            Cands::PerDepth(s) => s.get(depth as usize).copied().unwrap_or(0),
        }
    }
}

/// Static (0-ply) value of each move's result to the side that just moved: its
/// points if the move wins outright, else the negated opponent equity.
///
/// The non-terminal results are scored in a single batched forward pass. This is
/// the search's hot path — every chance node scores all its legal moves — and one
/// `[n, 198]` matmul beats `n` `[1, 198]` ones by enough to dominate the search's
/// runtime.
fn shallow_all<E: Evaluator>(moves: &[Move], eval: &E) -> Vec<f32> {
    let mut out = vec![0.0f32; moves.len()];
    let mut pending = Vec::with_capacity(moves.len());
    let mut at = Vec::with_capacity(moves.len());
    for (i, m) in moves.iter().enumerate() {
        match result(&m.result) {
            // A won position needs no net evaluation.
            GameResult::MoverWins(p) => out[i] = p as f32,
            _ => {
                at.push(i);
                pending.push(m.result.swap_perspective());
            }
        }
    }
    for (k, v) in eval.evaluate_batch(&pending).into_iter().enumerate() {
        out[at[k]] = -v.equity();
    }
    out
}

/// Expected equity for the side to move at `board`, searching `depth` half-moves
/// deep with `eval` at the leaves. At nodes deeper than one ply, only the top
/// `candidates` moves (by static value) are explored; `candidates == 0` searches
/// all moves (full width).
pub fn position_value<E: Evaluator>(board: &Board, depth: u8, eval: &E) -> f32 {
    pv(board, depth, eval, 0)
}

fn pv<E: Evaluator>(board: &Board, depth: u8, eval: &E, candidates: usize) -> f32 {
    match result(board) {
        GameResult::MoverWins(p) => return p as f32,
        GameResult::OppWins(p) => return -(p as f32),
        GameResult::InProgress => {}
    }
    if depth == 0 {
        return eval.evaluate(board).equity();
    }

    let mut total = 0.0f32;
    for a in 1..=6u8 {
        for c in a..=6u8 {
            let weight = if a == c { 1.0 / 36.0 } else { 2.0 / 36.0 };
            let mut moves = genmoves(board, &Dice::new(a, c));
            let vals = shallow_all(&moves, eval);

            // At the last ply a move's static value *is* its searched value, so
            // the batched pass above already answered this chance node.
            if depth == 1 {
                total += weight * vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                continue;
            }

            // Prune only where it pays: below the last ply the deep search per
            // move is expensive, so keep just the best `candidates`.
            if candidates > 0 && moves.len() > candidates {
                let mut idx: Vec<usize> = (0..moves.len()).collect();
                idx.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());
                idx.truncate(candidates);
                idx.sort_unstable();
                let mut keep = idx.iter().map(|&i| moves[i].clone()).collect();
                std::mem::swap(&mut moves, &mut keep);
            }

            let mut best = f32::NEG_INFINITY;
            for m in &moves {
                let v = match result(&m.result) {
                    GameResult::MoverWins(p) => p as f32,
                    _ => -pv(&m.result.swap_perspective(), depth - 1, eval, candidates),
                };
                if v > best {
                    best = v;
                }
            }
            total += weight * best;
        }
    }
    total
}

// --- Pass extension -----------------------------------------------------------
//
// A forced pass (dance: no legal move for a roll) is a null, non-branching ply.
// Here it does NOT consume a depth ply, so the opponent's reply is searched one
// ply deeper — cheap, and it fires only in forced (closed-out / blocked) lines
// where the static eval is least reliable. Explosion guard: an absolute-ply
// counter that always advances; when it reaches `abs_cap` the node is statically
// evaluated no matter what, so pass-heavy lines can't recurse without bound.

/// `pv` with the pass extension. Equal to `pv` on any line with no forced passes.
#[allow(clippy::too_many_arguments)]
fn pv_pass<E: Evaluator>(
    board: &Board,
    depth: u8,
    abs_ply: u16,
    abs_cap: u16,
    eval: &E,
    candidates: usize,
) -> f32 {
    match result(board) {
        GameResult::MoverWins(p) => return p as f32,
        GameResult::OppWins(p) => return -(p as f32),
        GameResult::InProgress => {}
    }
    if depth == 0 || abs_ply >= abs_cap {
        return eval.evaluate(board).equity();
    }

    let mut total = 0.0f32;
    // Every pass-die at this node recurses to the SAME opponent board (the board is
    // unchanged by a dance), so compute that value once and reuse it — otherwise a
    // fully-closed-out node recurses 21x per level and nests 21^depth. This dedup,
    // together with `abs_cap`, is what makes the extension safe.
    let mut pass_value: Option<f32> = None;
    for a in 1..=6u8 {
        for c in a..=6u8 {
            let weight = if a == c { 1.0 / 36.0 } else { 2.0 / 36.0 };
            let mut moves = genmoves(board, &Dice::new(a, c));

            // Dance for this roll: a single pass move whose result is the unchanged
            // board. Recurse the opponent WITHOUT decrementing depth (the guard
            // still advances via abs_ply).
            if moves.len() == 1 && moves[0].result == *board {
                let v = *pass_value.get_or_insert_with(|| {
                    -pv_pass(&board.swap_perspective(), depth, abs_ply + 1, abs_cap, eval, candidates)
                });
                total += weight * v;
                continue;
            }

            let vals = shallow_all(&moves, eval);
            if depth == 1 {
                total += weight * vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                continue;
            }
            if candidates > 0 && moves.len() > candidates {
                let mut idx: Vec<usize> = (0..moves.len()).collect();
                idx.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());
                idx.truncate(candidates);
                idx.sort_unstable();
                let mut keep = idx.iter().map(|&i| moves[i].clone()).collect();
                std::mem::swap(&mut moves, &mut keep);
            }
            let mut best = f32::NEG_INFINITY;
            for m in &moves {
                let v = match result(&m.result) {
                    GameResult::MoverWins(p) => p as f32,
                    _ => -pv_pass(
                        &m.result.swap_perspective(),
                        depth - 1,
                        abs_ply + 1,
                        abs_cap,
                        eval,
                        candidates,
                    ),
                };
                if v > best {
                    best = v;
                }
            }
            total += weight * best;
        }
    }
    total
}

/// Per-move equities like [`score_moves`] but with the pass extension in the deep
/// search. `max_pass_ext` bounds how many extra plies passes may add per line.
pub fn score_moves_pass<E: Evaluator>(
    board: &Board,
    dice: &Dice,
    depth: u8,
    candidates: usize,
    eval: &E,
    max_pass_ext: u16,
) -> Vec<f32> {
    let moves = genmoves(board, dice);
    let mut scores = shallow_all(&moves, eval);
    if depth == 0 {
        return scores;
    }
    let abs_cap = depth as u16 + max_pass_ext;
    let mut order: Vec<usize> = (0..moves.len()).collect();
    if depth >= 2 && candidates > 0 && moves.len() > candidates {
        order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
        order.truncate(candidates);
    }
    for &i in &order {
        scores[i] = match result(&moves[i].result) {
            GameResult::MoverWins(p) => p as f32,
            _ => -pv_pass(&moves[i].result.swap_perspective(), depth, 0, abs_cap, eval, candidates),
        };
    }
    scores
}

// --- Singular extension -------------------------------------------------------
//
// Generalises the pass extension to a SINGULAR move (exactly one legal, non-pass
// move — a forced play): also a non-decision, so it doesn't consume a ply. Unlike
// a pass, a singular move changes the board, so the same-board dedup does NOT
// apply and singular is far more common — hence extra guards: a per-branch cap
// (`max_sing` extensions per root-to-leaf path) AND a hard node budget. Pass
// extension is still applied throughout.

/// `pv_pass` plus the singular extension. `sing_used` counts singular extensions
/// on the current path; `nodes` counts visited nodes (both are explosion guards).
#[allow(clippy::too_many_arguments)]
fn pv_sing<E: Evaluator>(
    board: &Board,
    depth: u8,
    abs_ply: u16,
    abs_cap: u16,
    sing_used: u8,
    max_sing: u8,
    nodes: &Cell<u64>,
    node_cap: u64,
    eval: &E,
    candidates: usize,
) -> f32 {
    nodes.set(nodes.get() + 1);
    match result(board) {
        GameResult::MoverWins(p) => return p as f32,
        GameResult::OppWins(p) => return -(p as f32),
        GameResult::InProgress => {}
    }
    if depth == 0 || abs_ply >= abs_cap || nodes.get() >= node_cap {
        return eval.evaluate(board).equity();
    }

    let mut total = 0.0f32;
    let mut pass_value: Option<f32> = None;
    for a in 1..=6u8 {
        for c in a..=6u8 {
            let weight = if a == c { 1.0 / 36.0 } else { 2.0 / 36.0 };
            let mut moves = genmoves(board, &Dice::new(a, c));

            // Pass (dance): non-decrement, deduped across dice.
            if moves.len() == 1 && moves[0].result == *board {
                let v = *pass_value.get_or_insert_with(|| {
                    -pv_sing(&board.swap_perspective(), depth, abs_ply + 1, abs_cap,
                             sing_used, max_sing, nodes, node_cap, eval, candidates)
                });
                total += weight * v;
                continue;
            }

            // Singular: exactly one (real) forced move, with per-branch budget left.
            if moves.len() == 1 && sing_used < max_sing {
                let v = match result(&moves[0].result) {
                    GameResult::MoverWins(p) => p as f32,
                    _ => -pv_sing(&moves[0].result.swap_perspective(), depth, abs_ply + 1, abs_cap,
                                  sing_used + 1, max_sing, nodes, node_cap, eval, candidates),
                };
                total += weight * v;
                continue;
            }

            let vals = shallow_all(&moves, eval);
            if depth == 1 {
                total += weight * vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                continue;
            }
            if candidates > 0 && moves.len() > candidates {
                let mut idx: Vec<usize> = (0..moves.len()).collect();
                idx.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());
                idx.truncate(candidates);
                idx.sort_unstable();
                let mut keep = idx.iter().map(|&i| moves[i].clone()).collect();
                std::mem::swap(&mut moves, &mut keep);
            }
            let mut best = f32::NEG_INFINITY;
            for m in &moves {
                let v = match result(&m.result) {
                    GameResult::MoverWins(p) => p as f32,
                    _ => -pv_sing(&m.result.swap_perspective(), depth - 1, abs_ply + 1, abs_cap,
                                  sing_used, max_sing, nodes, node_cap, eval, candidates),
                };
                if v > best {
                    best = v;
                }
            }
            total += weight * best;
        }
    }
    total
}

/// Per-move equities with BOTH the pass extension and the singular extension.
/// `max_sing` = singular extensions allowed per branch; `node_cap` = hard node
/// budget (both guards). `max_sing = 0` reduces exactly to `score_moves_pass`.
pub fn score_moves_sing<E: Evaluator>(
    board: &Board,
    dice: &Dice,
    depth: u8,
    candidates: usize,
    eval: &E,
    max_pass_ext: u16,
    max_sing: u8,
    node_cap: u64,
) -> Vec<f32> {
    let moves = genmoves(board, dice);
    let mut scores = shallow_all(&moves, eval);
    if depth == 0 {
        return scores;
    }
    let abs_cap = depth as u16 + max_pass_ext + max_sing as u16;
    let mut order: Vec<usize> = (0..moves.len()).collect();
    if depth >= 2 && candidates > 0 && moves.len() > candidates {
        order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
        order.truncate(candidates);
    }
    for &i in &order {
        scores[i] = match result(&moves[i].result) {
            GameResult::MoverWins(p) => p as f32,
            _ => {
                let nodes = Cell::new(0u64);
                -pv_sing(&moves[i].result.swap_perspective(), depth, 0, abs_cap,
                         0, max_sing, &nodes, node_cap, eval, candidates)
            }
        };
    }
    scores
}

// --- gnubg-style move filter --------------------------------------------------
//
// Instead of a fixed top-N (`candidates`), keep the best `floor` moves ALWAYS,
// plus up to `extra` more that are within `threshold` equity of the best — then
// hard-cap at `floor + extra`. Adaptive width (searches more when several moves
// are close, fewer when one dominates) but bounded. `extra == 0` reproduces a
// fixed top-`floor`.

/// Indices (in original order) of the moves to search deeper under the filter.
fn filtered_moves(vals: &[f32], floor: usize, extra: usize, threshold: f32) -> Vec<usize> {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());
    let best = vals[idx[0]];
    let cap = (floor + extra).min(n);
    let mut keep = floor.clamp(1, n);
    while keep < cap && vals[idx[keep]] >= best - threshold {
        keep += 1;
    }
    idx.truncate(keep);
    idx.sort_unstable(); // restore original order (matches pv/pvd)
    idx
}

/// `pv` with the gnubg-style move filter in place of the fixed `candidates`.
fn pv_filter<E: Evaluator>(
    board: &Board,
    depth: u8,
    eval: &E,
    floor: usize,
    extra: usize,
    threshold: f32,
) -> f32 {
    match result(board) {
        GameResult::MoverWins(p) => return p as f32,
        GameResult::OppWins(p) => return -(p as f32),
        GameResult::InProgress => {}
    }
    if depth == 0 {
        return eval.evaluate(board).equity();
    }
    let mut total = 0.0f32;
    for a in 1..=6u8 {
        for c in a..=6u8 {
            let weight = if a == c { 1.0 / 36.0 } else { 2.0 / 36.0 };
            let moves = genmoves(board, &Dice::new(a, c));
            let vals = shallow_all(&moves, eval);
            if depth == 1 {
                total += weight * vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                continue;
            }
            let mut best = f32::NEG_INFINITY;
            for &i in &filtered_moves(&vals, floor, extra, threshold) {
                let v = match result(&moves[i].result) {
                    GameResult::MoverWins(p) => p as f32,
                    _ => -pv_filter(&moves[i].result.swap_perspective(), depth - 1, eval,
                                    floor, extra, threshold),
                };
                if v > best {
                    best = v;
                }
            }
            total += weight * best;
        }
    }
    total
}

/// Per-move equities using the gnubg-style filter (root + deep). `extra = 0`
/// reproduces the fixed top-`floor` behaviour of `score_moves`.
pub fn score_moves_filter<E: Evaluator>(
    board: &Board,
    dice: &Dice,
    depth: u8,
    floor: usize,
    extra: usize,
    threshold: f32,
    eval: &E,
) -> Vec<f32> {
    let moves = genmoves(board, dice);
    let mut scores = shallow_all(&moves, eval);
    if depth == 0 {
        return scores;
    }
    let order: Vec<usize> = if depth >= 2 {
        filtered_moves(&scores, floor, extra, threshold)
    } else {
        (0..moves.len()).collect()
    };
    for &i in &order {
        scores[i] = match result(&moves[i].result) {
            GameResult::MoverWins(p) => p as f32,
            _ => -pv_filter(&moves[i].result.swap_perspective(), depth, eval, floor, extra, threshold),
        };
    }
    scores
}

// --- Distribution-returning search --------------------------------------------
//
// `pv` folds each searched position to a scalar equity. For distillation labels we
// need the full 5-outcome distribution [win, win_g, win_bg, lose_g, lose_bg] (mover
// frame). `pvd` runs the SAME expectiminimax — average over the 21 rolls at chance
// nodes, take the equity-best move at choice nodes — but carries that best move's
// *distribution* instead of only its equity. By construction its folded equity
// equals `position_value` exactly (verified by `position_dist_folds_to_pv`).

fn win_vec5(points: u8) -> [f32; 5] {
    [1.0, (points >= 2) as u8 as f32, (points >= 3) as u8 as f32, 0.0, 0.0]
}

/// Opponent-frame 5-vector -> mover frame (swap win/lose), matching `rollout::flip5`.
fn flip5(v: [f32; 5]) -> [f32; 5] {
    [1.0 - v[0], v[3], v[4], v[1], v[2]]
}

/// Equity of a 5-outcome distribution — identical to `Value::equity`.
fn equity5(v: [f32; 5]) -> f32 {
    let lose = 1.0 - v[0];
    (v[0] - lose) + (v[1] - v[3]) + (v[2] - v[4])
}

fn add5(a: [f32; 5], b: [f32; 5]) -> [f32; 5] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4]]
}

fn scale5(w: f32, v: [f32; 5]) -> [f32; 5] {
    [w * v[0], w * v[1], w * v[2], w * v[3], w * v[4]]
}

/// The static leaf distribution of a single move's result, mover frame.
fn leaf_dist<E: Evaluator>(m: &Move, eval: &E) -> [f32; 5] {
    match result(&m.result) {
        GameResult::MoverWins(p) => win_vec5(p),
        _ => {
            let v = eval.evaluate(&m.result.swap_perspective());
            flip5([v.win, v.win_g, v.win_bg, v.lose_g, v.lose_bg])
        }
    }
}

/// Expectiminimax distribution for the side to move at `board`, searched `depth`
/// half-moves deep, carrying the principal variation's outcome distribution. Folds
/// to the same equity as [`position_value`]. `candidates` prunes deep nodes (as in
/// `pv`); `0` = full width.
pub fn position_dist<E: Evaluator>(
    board: &Board,
    depth: u8,
    candidates: usize,
    eval: &E,
) -> [f32; 5] {
    pvd(board, depth, eval, Cands::Uniform(candidates))
}

/// Like [`position_dist`] but with a per-ply candidate schedule (see [`Cands`]),
/// so a distillation teacher can search wider near the root than deep.
pub fn position_dist_cands<E: Evaluator>(
    board: &Board,
    depth: u8,
    cands: Cands,
    eval: &E,
) -> [f32; 5] {
    pvd(board, depth, eval, cands)
}

fn pvd<E: Evaluator>(board: &Board, depth: u8, eval: &E, cands: Cands) -> [f32; 5] {
    match result(board) {
        GameResult::MoverWins(p) => return win_vec5(p),
        GameResult::OppWins(p) => return flip5(win_vec5(p)),
        GameResult::InProgress => {}
    }
    if depth == 0 {
        let v = eval.evaluate(board);
        return [v.win, v.win_g, v.win_bg, v.lose_g, v.lose_bg];
    }

    let mut acc = [0.0f32; 5];
    for a in 1..=6u8 {
        for c in a..=6u8 {
            let weight = if a == c { 1.0 / 36.0 } else { 2.0 / 36.0 };
            let mut moves = genmoves(board, &Dice::new(a, c));
            let vals = shallow_all(&moves, eval);

            // Last ply: static values are the searched values; propagate the
            // static-best move's leaf distribution (first max, as `pv` folds).
            if depth == 1 {
                let mut bi = 0;
                for i in 1..vals.len() {
                    if vals[i] > vals[bi] {
                        bi = i;
                    }
                }
                acc = add5(acc, scale5(weight, leaf_dist(&moves[bi], eval)));
                continue;
            }

            // Prune to the best `c` for this ply before the deep search (mirrors `pv`).
            let c = cands.at(depth);
            if c > 0 && moves.len() > c {
                let mut idx: Vec<usize> = (0..moves.len()).collect();
                idx.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());
                idx.truncate(c);
                idx.sort_unstable();
                let keep: Vec<Move> = idx.iter().map(|&i| moves[i].clone()).collect();
                moves = keep;
            }

            // Choose the move with the best deep equity; propagate its distribution.
            let mut best_eq = f32::NEG_INFINITY;
            let mut best_dist = [0.0f32; 5];
            for m in &moves {
                let d = match result(&m.result) {
                    GameResult::MoverWins(p) => win_vec5(p),
                    _ => flip5(pvd(&m.result.swap_perspective(), depth - 1, eval, cands)),
                };
                let eq = equity5(d);
                if eq > best_eq {
                    best_eq = eq;
                    best_dist = d;
                }
            }
            acc = add5(acc, scale5(weight, best_dist));
        }
    }
    acc
}

// --- Selective deepening (prototype) -----------------------------------------
//
// Extends the "all-rank-1" frontier by one ply: a leaf whose every ancestor
// candidate-choice (per dice roll) was that node's rank-1 (best static) move is
// searched one ply deeper instead of statically evaluated. This spends the extra
// depth on the lines that determine the value, at a fraction of a full extra
// ply's cost — ~1/C^(D-1) of leaves for a D-ply, C-candidate search (1/9 for
// 3-ply/C=3, and higher wherever a node has fewer than C legal moves).

/// Tallies base leaves visited vs. leaves extended, for instrumentation.
struct ExtCount {
    total: Cell<u64>,
    extended: Cell<u64>,
}

/// Selective-deepening distribution search. `on_pv` stays true while every
/// ancestor candidate-choice on this roll-path was the rank-1 move; such leaves
/// are extended by one ply (a plain depth-1 [`pvd`]). Folds like [`pvd`]; with
/// `on_pv` forced false it *is* `pvd` (see `pvd_ext_offpv_matches_pvd`).
fn pvd_ext<E: Evaluator>(
    board: &Board,
    depth: u8,
    eval: &E,
    cands: Cands,
    on_pv: bool,
    ext_plies: u8,
    ext_cands: usize,
    cnt: &ExtCount,
) -> [f32; 5] {
    match result(board) {
        GameResult::MoverWins(p) => return win_vec5(p),
        GameResult::OppWins(p) => return flip5(win_vec5(p)),
        GameResult::InProgress => {}
    }
    if depth == 0 {
        let v = eval.evaluate(board);
        return [v.win, v.win_g, v.win_bg, v.lose_g, v.lose_bg];
    }

    let mut acc = [0.0f32; 5];
    for a in 1..=6u8 {
        for c in a..=6u8 {
            let weight = if a == c { 1.0 / 36.0 } else { 2.0 / 36.0 };
            let moves = genmoves(board, &Dice::new(a, c));
            let vals = shallow_all(&moves, eval);
            // rank-1 = first index holding the max static value (matches `pv`).
            let mut r1 = 0usize;
            for i in 1..vals.len() {
                if vals[i] > vals[r1] {
                    r1 = i;
                }
            }

            if depth == 1 {
                // Leaf ply: take the static-best move, extend it a ply if on-PV.
                cnt.total.set(cnt.total.get() + 1);
                let d = if on_pv && ext_plies > 0 {
                    cnt.extended.set(cnt.extended.get() + 1);
                    match result(&moves[r1].result) {
                        GameResult::MoverWins(p) => win_vec5(p),
                        _ => {
                            // (1,1)-style sampled rollout on the PV: from the leaf's
                            // (already-chosen) best-move result, play `ext_plies`
                            // greedy SAMPLED plies then truncate+eval, averaged over
                            // `ext_cands` trials. An EVEN ext_plies lands the
                            // truncation on the base leaf's parity. One move-gen +
                            // eval per ply — far cheaper than enumerating all dice.
                            let start = moves[r1].result.swap_perspective();
                            let trials = ext_cands.max(1);
                            let mut seed = board_seed(&start);
                            let mut acc = [0.0f32; 5];
                            for _ in 0..trials {
                                let mut rng = Rng::new(seed);
                                acc = add5(acc, crate::rollout::rollout_once_dist(
                                    &start, eval, ext_plies as usize, false, &mut rng));
                                seed = seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1);
                            }
                            flip5(scale5(1.0 / trials as f32, acc))
                        }
                    }
                } else {
                    leaf_dist(&moves[r1], eval)
                };
                acc = add5(acc, scale5(weight, d));
                continue;
            }

            // Deeper node: keep the best `cc` (restored to original order, as
            // `pvd` does), recurse, take the equity-best; only the rank-1 child
            // stays on-PV.
            let cc = cands.at(depth);
            let mut idx: Vec<usize> = (0..moves.len()).collect();
            if cc > 0 && idx.len() > cc {
                idx.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());
                idx.truncate(cc);
                idx.sort_unstable();
            }
            let mut best_eq = f32::NEG_INFINITY;
            let mut best = [0.0f32; 5];
            for &i in &idx {
                let child_pv = on_pv && i == r1;
                let d = match result(&moves[i].result) {
                    GameResult::MoverWins(p) => win_vec5(p),
                    _ => flip5(pvd_ext(
                        &moves[i].result.swap_perspective(),
                        depth - 1,
                        eval,
                        cands,
                        child_pv,
                        ext_plies,
                        ext_cands,
                        cnt,
                    )),
                };
                let eq = equity5(d);
                if eq > best_eq {
                    best_eq = eq;
                    best = d;
                }
            }
            acc = add5(acc, scale5(weight, best));
        }
    }
    acc
}

/// Selective-deepening [`position_dist`]: returns the searched distribution plus
/// `(total_leaves, extended_leaves)` for instrumentation.
pub fn position_dist_ext<E: Evaluator>(
    board: &Board,
    depth: u8,
    cands: Cands,
    eval: &E,
    ext_plies: u8,
    ext_cands: usize,
) -> ([f32; 5], u64, u64) {
    let cnt = ExtCount { total: Cell::new(0), extended: Cell::new(0) };
    let d = pvd_ext(board, depth, eval, cands, true, ext_plies, ext_cands, &cnt);
    (d, cnt.total.get(), cnt.extended.get())
}

/// Equity of every legal move for `dice`, from the mover's perspective, in
/// [`genmoves`] order — the ranked list a GUI needs (best move, hints, and the
/// cost of the alternatives), not just the single pick [`SearchEngine::choose`]
/// returns.
///
/// Moves are searched `depth` half-moves deep. At `depth >= 2` only the best
/// `candidates` (by static value) are searched deeply; the rest keep their static
/// value, which is enough to rank also-rans for display. Note this means the
/// argmax here can differ from `choose`, which only ever considers its candidate
/// set — a pruned move's *static* value can top the candidates' *deep* values.
/// `choose` remains the engine's move for play and benchmarks.
pub fn score_moves<E: Evaluator>(
    board: &Board,
    dice: &Dice,
    depth: u8,
    candidates: usize,
    eval: &E,
) -> Vec<f32> {
    let moves = genmoves(board, dice);
    let mut scores = shallow_all(&moves, eval);
    if depth == 0 {
        return scores;
    }

    let mut order: Vec<usize> = (0..moves.len()).collect();
    if depth >= 2 && candidates > 0 && moves.len() > candidates {
        order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
        order.truncate(candidates);
    }
    for &i in &order {
        scores[i] = match result(&moves[i].result) {
            GameResult::MoverWins(p) => p as f32,
            _ => -pv(&moves[i].result.swap_perspective(), depth, eval, candidates),
        };
    }
    scores
}

/// Like [`score_moves`] but scores each searched candidate through the
/// distribution search, optionally with selective deepening (`selective` routes
/// the deep search through [`pvd_ext`], extending the all-rank-1 frontier one
/// ply). With `selective == false` it folds to the same equities as
/// [`score_moves`] — so an A/B of the two isolates the extension at play time.
pub fn score_moves_ext<E: Evaluator>(
    board: &Board,
    dice: &Dice,
    depth: u8,
    candidates: usize,
    eval: &E,
    ext_plies: u8,
    ext_cands: usize,
) -> Vec<f32> {
    let moves = genmoves(board, dice);
    let mut scores = shallow_all(&moves, eval);
    if depth == 0 {
        return scores;
    }
    let cands = Cands::Uniform(candidates);
    let cnt = ExtCount { total: Cell::new(0), extended: Cell::new(0) };

    let mut order: Vec<usize> = (0..moves.len()).collect();
    if depth >= 2 && candidates > 0 && moves.len() > candidates {
        order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
        order.truncate(candidates);
    }
    for &i in &order {
        scores[i] = match result(&moves[i].result) {
            GameResult::MoverWins(p) => p as f32,
            _ => {
                let child = moves[i].result.swap_perspective();
                let d = if ext_plies > 0 {
                    pvd_ext(&child, depth, eval, cands, true, ext_plies, ext_cands, &cnt)
                } else {
                    pvd(&child, depth, eval, cands)
                };
                -equity5(d)
            }
        };
    }
    scores
}

/// An [`Engine`] that picks its move by `lookahead`-ply search. `candidates`
/// bounds the branching of deep (2-ply+) searches, including at the root; use
/// `0` for full width (fine for 0/1-ply).
pub struct SearchEngine<E: Evaluator> {
    eval: E,
    lookahead: u8,
    candidates: usize,
    name: String,
}

impl<E: Evaluator> SearchEngine<E> {
    /// Full-width search (no candidate pruning).
    pub fn new(eval: E, lookahead: u8, name: impl Into<String>) -> Self {
        SearchEngine { eval, lookahead, candidates: 0, name: name.into() }
    }

    /// Search keeping only the best `candidates` moves at deep nodes.
    pub fn with_candidates(eval: E, lookahead: u8, candidates: usize, name: impl Into<String>) -> Self {
        SearchEngine { eval, lookahead, candidates, name: name.into() }
    }
}

impl<E: Evaluator> Engine for SearchEngine<E> {
    fn choose(&mut self, board: &Board, dice: &Dice) -> crate::moves::Move {
        let mut moves = genmoves(board, dice);

        // At the root, prune to the best `candidates` before the (expensive)
        // deep search, when doing 2-ply or deeper.
        let order: Vec<usize> = if self.lookahead >= 2
            && self.candidates > 0
            && moves.len() > self.candidates
        {
            let vals = shallow_all(&moves, &self.eval);
            let mut idx: Vec<usize> = (0..moves.len()).collect();
            idx.sort_by(|&i, &j| vals[j].partial_cmp(&vals[i]).unwrap());
            idx.truncate(self.candidates);
            idx
        } else {
            (0..moves.len()).collect()
        };

        let mut best_i = order[0];
        let mut best = f32::NEG_INFINITY;
        for &i in &order {
            let s = match result(&moves[i].result) {
                GameResult::MoverWins(p) => p as f32,
                _ => -pv(&moves[i].result.swap_perspective(), self.lookahead, &self.eval, self.candidates),
            };
            if s > best {
                best = s;
                best_i = i;
            }
        }
        moves.swap_remove(best_i)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Evaluator, HceEval};

    #[test]
    fn zero_ply_equals_static_eval() {
        let b = Board::starting_position();
        let hce = HceEval::new();
        assert_eq!(position_value(&b, 0, &hce), hce.evaluate(&b).equity());
    }

    #[test]
    fn deeper_search_is_finite_and_bounded() {
        let b = Board::starting_position();
        for depth in [1u8, 2] {
            let v = position_value(&b, depth, &HceEval::new());
            assert!(v.is_finite() && v.abs() <= 3.0, "depth {depth} value {v}");
        }
    }

    #[test]
    fn pruned_two_ply_runs() {
        // Candidate-pruned 2-ply should produce a finite value quickly.
        let v = pv(&Board::starting_position(), 2, &HceEval::new(), 4);
        assert!(v.is_finite() && v.abs() <= 3.0);
    }

    /// With the PV flag forced off, selective deepening is exactly the base
    /// `pvd` (nothing extended) — the extension is the only behavioural change.
    #[test]
    fn pvd_ext_offpv_matches_pvd() {
        let hce = HceEval::new();
        let cnt = ExtCount { total: Cell::new(0), extended: Cell::new(0) };
        let b = Board::starting_position();
        // Full width is cheap only to 2-ply; check the deeper (3-ply) case pruned.
        for depth in [1u8, 2] {
            assert_eq!(
                pvd(&b, depth, &hce, Cands::Uniform(0)),
                pvd_ext(&b, depth, &hce, Cands::Uniform(0), false, 2, 1, &cnt),
                "off-PV pvd_ext != pvd (full width) at depth {depth}"
            );
        }
        for depth in [2u8, 3] {
            assert_eq!(
                pvd(&b, depth, &hce, Cands::Uniform(2)),
                pvd_ext(&b, depth, &hce, Cands::Uniform(2), false, 2, 1, &cnt),
                "off-PV pvd_ext != pvd (pruned) at depth {depth}"
            );
        }
        assert_eq!(cnt.extended.get(), 0, "off-PV must extend nothing");
        assert!(cnt.total.get() > 0, "leaves should have been counted");
    }

    /// First index holding the maximum — the same tie-break `choose` uses (it
    /// keeps the incumbent on `v > best`). HCE is a pip-race eval, so distinct
    /// moves using the same pips tie exactly and the tie-break decides.
    fn argmax(v: &[f32]) -> usize {
        let mut best = 0;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best
    }

    #[test]
    fn score_moves_scores_every_move() {
        let b = Board::starting_position();
        let d = Dice::new(3, 1);
        let hce = HceEval::new();
        for depth in [0u8, 1, 2] {
            let s = score_moves(&b, &d, depth, 4, &hce);
            assert_eq!(s.len(), genmoves(&b, &d).len(), "depth {depth}");
            assert!(s.iter().all(|v| v.is_finite() && v.abs() <= 3.0), "depth {depth}");
        }
    }

    /// 0-ply scores are just the negated opponent equity of each result.
    #[test]
    fn score_moves_zero_ply_is_static_value() {
        let b = Board::starting_position();
        let d = Dice::new(6, 5);
        let hce = HceEval::new();
        let s = score_moves(&b, &d, 0, 0, &hce);
        for (m, got) in genmoves(&b, &d).iter().zip(s) {
            let want = -hce.evaluate(&m.result.swap_perspective()).equity();
            assert_eq!(got, want);
        }
    }

    /// Batching must not change what the search computes: the batched
    /// `shallow_all` path and a per-position loop agree exactly.
    #[test]
    fn shallow_all_matches_per_position_eval() {
        let b = Board::starting_position();
        let hce = HceEval::new();
        let moves = genmoves(&b, &Dice::new(4, 2));
        for (m, got) in moves.iter().zip(shallow_all(&moves, &hce)) {
            let want = match result(&m.result) {
                GameResult::MoverWins(p) => p as f32,
                _ => -hce.evaluate(&m.result.swap_perspective()).equity(),
            };
            assert_eq!(got, want);
        }
    }

    /// Full width (no pruning), the ranked list's best move is the one the engine
    /// actually plays — `score_moves` and `choose` agree wherever they can.
    #[test]
    fn score_moves_best_matches_choose_full_width() {
        let hce = HceEval::new();
        for (d1, d2) in [(3u8, 1u8), (6, 5), (5, 5)] {
            let b = Board::starting_position();
            let d = Dice::new(d1, d2);
            for depth in [0u8, 1] {
                let best = argmax(&score_moves(&b, &d, depth, 0, &hce));
                let chosen = SearchEngine::new(&hce, depth, "t").choose(&b, &d);
                assert_eq!(genmoves(&b, &d)[best].result, chosen.result, "{d1}{d2} depth {depth}");
            }
        }
    }

    /// The distribution-returning search must fold to exactly the same equity as
    /// the scalar `pv`/`position_value` — full width AND candidate-pruned, across
    /// depths. HCE is deterministic so the match is exact (no float-reorder slack).
    #[test]
    fn position_dist_folds_to_pv() {
        let hce = HceEval::new();
        let mut race = Board::empty();
        race.set_off(crate::board::MOVER, 3);
        race.set_point(4, 6);
        race.set_point(5, 6);
        race.set_point(20, 6);
        race.set_point(21, 6);
        let boards = [Board::starting_position(), race];
        for b in &boards {
            for depth in [0u8, 1, 2] {
                let d = position_dist(b, depth, 0, &hce);
                let eq = equity5(d);
                let pval = position_value(b, depth, &hce);
                assert!((eq - pval).abs() < 1e-4, "full depth {depth}: {eq} vs {pval}");
                assert!(eq.is_finite() && eq.abs() <= 3.0);
            }
            // Candidate-pruned 2-ply must also agree with the pruned `pv`.
            let dp = position_dist(b, 2, 4, &hce);
            assert!((equity5(dp) - pv(b, 2, &hce, 4)).abs() < 1e-4, "pruned 2-ply mismatch");
        }
    }
}
