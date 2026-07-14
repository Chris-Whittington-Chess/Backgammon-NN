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
use crate::moves::genmoves;

/// Static (0-ply) value of a resulting position `r` to the side that just moved:
/// its points if the move wins outright, else the negated opponent equity.
fn shallow<E: Evaluator>(r: &Board, eval: &E) -> f32 {
    match result(r) {
        GameResult::MoverWins(p) => p as f32,
        _ => -eval.evaluate(&r.swap_perspective()).equity(),
    }
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

            // Prune only where it pays: below the last ply the deep search per
            // move is expensive, so keep just the best `candidates`.
            if depth > 1 && candidates > 0 && moves.len() > candidates {
                moves.sort_by(|x, y| {
                    shallow(&y.result, eval)
                        .partial_cmp(&shallow(&x.result, eval))
                        .unwrap()
                });
                moves.truncate(candidates);
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
            let mut idx: Vec<usize> = (0..moves.len()).collect();
            idx.sort_by(|&i, &j| {
                shallow(&moves[j].result, &self.eval)
                    .partial_cmp(&shallow(&moves[i].result, &self.eval))
                    .unwrap()
            });
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
}
