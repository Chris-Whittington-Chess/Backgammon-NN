//! Position evaluation (SPEC §5, §6).
//!
//! Every "thinking" component implements [`Evaluator`], returning a [`Value`]:
//! the five cubeless probability outputs plus a derived scalar equity. Move
//! selection ranks candidate positions by equity, so all three evaluators
//! (`random`, `hce`, and later the neural net) are interchangeable.

use crate::board::Board;

pub mod hce;
#[cfg(feature = "onnx")]
pub mod nn;
pub mod random;

pub use hce::HceEval;
#[cfg(feature = "onnx")]
pub use nn::NnEval;
pub use random::RandomEval;

/// Cubeless evaluation of a position from the mover's perspective.
///
/// The five fields are the standard nested win/loss probabilities:
/// `win` is the total probability of winning; `win_g`/`win_bg` the probabilities
/// of winning a gammon / backgammon; `lose_g`/`lose_bg` the mirror for losses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Value {
    pub win: f32,
    pub win_g: f32,
    pub win_bg: f32,
    pub lose_g: f32,
    pub lose_bg: f32,
}

impl Value {
    /// A value carrying only a single-game win probability (gammon/backgammon
    /// terms zero). Used by the racing-style evaluators in M2.
    pub fn from_win_prob(win: f32) -> Self {
        Value {
            win,
            win_g: 0.0,
            win_bg: 0.0,
            lose_g: 0.0,
            lose_bg: 0.0,
        }
    }

    /// Cubeless equity in points (range roughly `-3..=3`), matching the
    /// single/gammon/backgammon result magnitudes (SPEC §6):
    /// `(P(win) − P(lose)) + (P(win_g) − P(lose_g)) + (P(win_bg) − P(lose_bg))`.
    pub fn equity(&self) -> f32 {
        let lose = 1.0 - self.win;
        (self.win - lose) + (self.win_g - self.lose_g) + (self.win_bg - self.lose_bg)
    }
}

/// Anything that can score a position from the mover's perspective.
pub trait Evaluator {
    fn evaluate(&self, board: &Board) -> Value;

    /// Evaluate several positions at once. The default loops over
    /// [`Evaluator::evaluate`]; neural evaluators override it with a single
    /// batched forward pass (one `[N, 198]` matmul instead of N `[1, 198]`
    /// ones), which is far more SIMD-efficient — the hot path in rollouts, where
    /// every ply scores all legal moves.
    fn evaluate_batch(&self, boards: &[Board]) -> Vec<Value> {
        boards.iter().map(|b| self.evaluate(b)).collect()
    }
}

/// Let a shared reference act as an evaluator, so one (expensive) evaluator such
/// as [`NnEval`] can back several engines without cloning.
impl<T: Evaluator + ?Sized> Evaluator for &T {
    fn evaluate(&self, board: &Board) -> Value {
        (**self).evaluate(board)
    }
    fn evaluate_batch(&self, boards: &[Board]) -> Vec<Value> {
        (**self).evaluate_batch(boards)
    }
}
