//! Monte-Carlo rollouts (SPEC §5, §13) — the backgammon gold standard.
//!
//! To estimate a position's value, play it out many times and average the
//! result. The evaluator serves as both the **0-ply playout policy** (which move
//! to make) and the **truncation-leaf evaluator** (what a partial game is worth).
//! Rollouts are embarrassingly parallel, so trials run across threads with
//! `rayon`.
//!
//! Key ingredients (all standard in GNU Backgammon / XG):
//! - **Truncation:** stop after `truncate_plies` and evaluate the leaf with the
//!   net, instead of playing to the end. Most of the accuracy, a fraction of the
//!   cost.
//! - **Candidate filtering:** only roll out the best few moves (by 0-ply), since
//!   rollouts are expensive.
//! - **Common random numbers:** every candidate faces the *same* dice streams
//!   (trial `t` uses the same seed regardless of candidate), so dice luck cancels
//!   in the comparison — the variance reduction that makes rollouts practical.

use crate::board::Board;
use crate::dice::{Dice, Rng};
use crate::eval::Evaluator;
use crate::game::{result, Engine, GameResult};
use crate::moves::{genmoves, Move};
use rayon::prelude::*;

/// Parameters for a rollout.
#[derive(Clone, Debug)]
pub struct RolloutConfig {
    /// Number of playouts averaged per position.
    pub trials: usize,
    /// Evaluate the leaf with the net after this many plies; `0` plays to the end.
    pub truncate_plies: usize,
    /// Roll out only the best this-many moves at the root; `0` rolls out all.
    pub candidates: usize,
    /// Base seed — fixed across candidates for common random numbers.
    pub seed: u64,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        RolloutConfig { trials: 180, truncate_plies: 11, candidates: 6, seed: 0x5EED }
    }
}

/// 0-ply value of a resulting position to the side that just moved.
fn shallow<E: Evaluator>(r: &Board, eval: &E) -> f32 {
    match result(r) {
        GameResult::MoverWins(p) => p as f32,
        _ => -eval.evaluate(&r.swap_perspective()).equity(),
    }
}

/// One truncated playout from `board`, returning equity **from the perspective
/// of `board`'s side to move**. Uses a 0-ply policy (greedy on `eval`).
fn rollout_once<E: Evaluator>(board: &Board, eval: &E, truncate: usize, rng: &mut Rng) -> f32 {
    let mut b = board.clone();
    let mut plies = 0usize;
    loop {
        // Every ply swaps perspective, so even plies are `board`'s mover.
        let sign = if plies.is_multiple_of(2) { 1.0 } else { -1.0 };
        match result(&b) {
            GameResult::MoverWins(p) => return sign * p as f32,
            GameResult::OppWins(p) => return -sign * p as f32,
            GameResult::InProgress => {}
        }
        if truncate > 0 && plies >= truncate {
            return sign * eval.evaluate(&b).equity();
        }

        let moves = genmoves(&b, &rng.roll());
        let mut best_i = 0;
        let mut best = f32::NEG_INFINITY;
        for (i, m) in moves.iter().enumerate() {
            let s = shallow(&m.result, eval);
            if s > best {
                best = s;
                best_i = i;
            }
        }
        let chosen = &moves[best_i];
        if let GameResult::MoverWins(p) = result(&chosen.result) {
            return sign * p as f32; // the side that just moved won
        }
        b = chosen.result.swap_perspective();
        plies += 1;
    }
}

/// Expected equity for the side to move at `board`, from `cfg.trials` truncated
/// rollouts run in parallel.
pub fn rollout_equity<E: Evaluator + Sync>(board: &Board, eval: &E, cfg: &RolloutConfig) -> f32 {
    if cfg.trials == 0 {
        return eval.evaluate(board).equity();
    }
    let sum: f32 = (0..cfg.trials)
        .into_par_iter()
        .map(|t| {
            let mut rng = Rng::new(
                cfg.seed
                    .wrapping_add(t as u64 + 1)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            rollout_once(board, eval, cfg.truncate_plies, &mut rng)
        })
        .sum();
    sum / cfg.trials as f32
}

/// An [`Engine`] that picks its move by rolling out the top candidates and
/// choosing the highest rollout equity. Far stronger than static/1-ply play,
/// and much heavier — meant for strong play and for labelling training data.
pub struct RolloutEngine<E: Evaluator + Sync> {
    eval: E,
    cfg: RolloutConfig,
    name: String,
}

impl<E: Evaluator + Sync> RolloutEngine<E> {
    pub fn new(eval: E, cfg: RolloutConfig, name: impl Into<String>) -> Self {
        RolloutEngine { eval, cfg, name: name.into() }
    }
}

impl<E: Evaluator + Sync> Engine for RolloutEngine<E> {
    fn choose(&mut self, board: &Board, dice: &Dice) -> Move {
        let mut moves = genmoves(board, dice);

        // Roll out only the most promising moves (by 0-ply).
        let mut order: Vec<usize> = (0..moves.len()).collect();
        if self.cfg.candidates > 0 && moves.len() > self.cfg.candidates {
            order.sort_by(|&i, &j| {
                shallow(&moves[j].result, &self.eval)
                    .partial_cmp(&shallow(&moves[i].result, &self.eval))
                    .unwrap()
            });
            order.truncate(self.cfg.candidates);
        }

        let mut best_i = order[0];
        let mut best = f32::NEG_INFINITY;
        for &i in &order {
            // Common random numbers: same cfg.seed for every candidate.
            let s = match result(&moves[i].result) {
                GameResult::MoverWins(p) => p as f32,
                _ => -rollout_equity(&moves[i].result.swap_perspective(), &self.eval, &self.cfg),
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
    fn rollout_equity_is_finite_and_bounded() {
        let cfg = RolloutConfig { trials: 24, truncate_plies: 6, candidates: 0, seed: 1 };
        let v = rollout_equity(&Board::starting_position(), &HceEval::new(), &cfg);
        assert!(v.is_finite() && v.abs() <= 3.0, "value {v}");
    }

    #[test]
    fn winning_position_rolls_out_positive() {
        // Mover is on the ace point (pip 15); opponent stuck deep (pip ~345).
        let mut b = Board::empty();
        b.set_point(1, 15);
        b.set_point(2, -15);
        let cfg = RolloutConfig { trials: 40, truncate_plies: 0, candidates: 0, seed: 2 };
        let v = rollout_equity(&b, &HceEval::new(), &cfg);
        assert!(v > 0.7, "a won position rolled out to {v}");
    }

    #[test]
    fn common_random_numbers_are_deterministic() {
        let cfg = RolloutConfig { trials: 32, truncate_plies: 8, candidates: 0, seed: 7 };
        let b = Board::starting_position();
        let a = rollout_equity(&b, &HceEval::new(), &cfg);
        let c = rollout_equity(&b, &HceEval::new(), &cfg);
        assert_eq!(a, c, "same seed must give the same estimate");
    }
}
