//! Depth-limited expectiminimax search (SPEC §5, "1-ply / 2-ply").
//!
//! 0-ply ranks moves by the static evaluation of the resulting position. n-ply
//! looks n half-moves deeper, averaging over all 21 dice rolls at each chance
//! node and assuming both sides play their best static (0-ply) reply. 1-ply is
//! the usual strength/speed sweet spot; it costs ~21x a 0-ply decision.

use crate::board::Board;
use crate::dice::Dice;
use crate::eval::Evaluator;
use crate::game::{result, Engine, GameResult};
use crate::moves::genmoves;

/// Expected equity for the side to move at `board`, searching `depth` half-moves
/// deep with `eval` as the static leaf evaluator.
///
/// At `depth == 0` this is just the static equity. Deeper, it averages over the
/// 21 distinct rolls (doubles weight 1/36, non-doubles 2/36) of the best reply.
pub fn position_value<E: Evaluator>(board: &Board, depth: u8, eval: &E) -> f32 {
    // Terminal positions are scored directly (normally not reached mid-search).
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
            let mut best = f32::NEG_INFINITY;
            for m in genmoves(board, &Dice::new(a, c)) {
                let v = match result(&m.result) {
                    GameResult::MoverWins(p) => p as f32, // mover wins immediately
                    _ => -position_value(&m.result.swap_perspective(), depth - 1, eval),
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

/// An [`Engine`] that picks its move by `lookahead`-ply search: rank each legal
/// move by the (negated) value of the position it leaves the opponent, searched
/// `lookahead` half-moves deep. `lookahead == 0` reproduces plain 0-ply play.
pub struct SearchEngine<E: Evaluator> {
    eval: E,
    lookahead: u8,
    name: String,
}

impl<E: Evaluator> SearchEngine<E> {
    pub fn new(eval: E, lookahead: u8, name: impl Into<String>) -> Self {
        SearchEngine {
            eval,
            lookahead,
            name: name.into(),
        }
    }
}

impl<E: Evaluator> Engine for SearchEngine<E> {
    fn choose(&mut self, board: &Board, dice: &Dice) -> crate::moves::Move {
        let mut moves = genmoves(board, dice);
        let mut best_i = 0;
        let mut best = f32::NEG_INFINITY;
        for (i, m) in moves.iter().enumerate() {
            let s = match result(&m.result) {
                GameResult::MoverWins(p) => p as f32,
                _ => -position_value(&m.result.swap_perspective(), self.lookahead, &self.eval),
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
    use crate::eval::HceEval;

    #[test]
    fn one_ply_value_is_finite_and_bounded() {
        let v = position_value(&Board::starting_position(), 1, &HceEval::new());
        assert!(v.is_finite() && v.abs() <= 3.0, "value {v}");
    }

    #[test]
    fn zero_ply_equals_static_eval() {
        use crate::eval::Evaluator;
        let b = Board::starting_position();
        let hce = HceEval::new();
        assert_eq!(position_value(&b, 0, &hce), hce.evaluate(&b).equity());
    }
}
