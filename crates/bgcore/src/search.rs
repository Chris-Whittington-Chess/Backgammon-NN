//! Depth-limited expectiminimax search with candidate pruning (SPEC §5).
//!
//! 0-ply ranks moves by static evaluation. n-ply looks n half-moves deeper,
//! averaging over all 21 dice rolls at each chance node. Full-width 2-ply is
//! ~170x the cost of a 1-ply decision, so — like GNU Backgammon — at deep nodes
//! we shallow-rank the legal moves (0-ply) and only search the best few
//! (`candidates`). This keeps 2-ply to a fraction of a second per move with
//! little strength loss.

use crate::board::Board;
use crate::dice::Dice;
use crate::eval::Evaluator;
use crate::game::{result, Engine, GameResult};
use crate::moves::{genmoves, Move};

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
}
