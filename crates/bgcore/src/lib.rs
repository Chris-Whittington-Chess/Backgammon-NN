//! # bgcore — backgammon engine core
//!
//! Pure-Rust engine with no Python dependencies. Provides the board
//! representation, dice, (later) move generation and evaluators.
//!
//! ## Perspective convention
//! A [`board::Board`] is **always stored from the point of view of the side to
//! move** (the "mover"). Points `1..=24` run from the mover's ace point (1) to
//! their furthest-back point (24). Positive counts are the mover's checkers,
//! negative are the opponent's. After a turn, call
//! [`board::Board::swap_perspective`] so the opponent becomes the mover. This
//! keeps every evaluator and encoder colour-agnostic.
//!
//! Milestone M0: `board` + `dice` + invariants.

pub mod board;
pub mod dice;
pub mod eval;
pub mod features;
pub mod game;
pub mod moves;
pub mod posid;
pub mod rollout;
pub mod search;

pub use board::Board;
pub use dice::{Dice, Rng};
pub use eval::{Evaluator, HceEval, RandomEval, Value};
pub use features::{encode, encode_into, NUM_INPUTS};
pub use game::{
    play_game, result, run_match, Engine, EvalEngine, GameOutcome, GameResult, MatchStats,
};
pub use moves::{genmoves, next_submoves, Move, Step, SubMove, BAR, OFF};
pub use rollout::{
    build_pool, rollout_best, rollout_best_scored, rollout_equity, RolloutConfig, RolloutEngine,
};
pub use search::{position_value, SearchEngine};
