//! Random evaluator (SPEC §5): a trivial baseline and plumbing test.
//!
//! Returns a pseudo-random win probability derived by hashing the position, so
//! it is deterministic per position (reproducible games) yet effectively picks
//! arbitrary moves. Useful as the weakest rung on the strength ladder.

use super::{Evaluator, Value};
use crate::board::Board;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Evaluator that scores positions by a seeded hash.
pub struct RandomEval {
    seed: u64,
}

impl RandomEval {
    pub fn new(seed: u64) -> Self {
        RandomEval { seed }
    }
}

impl Evaluator for RandomEval {
    fn evaluate(&self, board: &Board) -> Value {
        let mut h = DefaultHasher::new();
        self.seed.hash(&mut h);
        board.hash(&mut h);
        // Map the top 53 bits into [0, 1).
        let p = (h.finish() >> 11) as f32 / (1u64 << 53) as f32;
        Value::from_win_prob(p)
    }
}
