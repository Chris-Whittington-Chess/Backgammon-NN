//! An [`Evaluator`] wrapper that memoizes leaf evaluations within a search.
//!
//! Backgammon expectiminimax re-reaches the same position through different
//! (dice, move) paths and at different depths, and the neural leaf eval is the
//! search's hot cost. Caching `board -> Value` collapses those repeats to a
//! single forward pass — GNU Backgammon's "evaluation cache". A cache hit returns
//! the exact [`Value`] a miss would have computed, so search output is *identical*
//! (bit-for-bit) — only faster; see the `cached_search_matches_uncached` test.
//!
//! Scoped to one search (the caller wraps its evaluator per `position_dist` /
//! `score_moves` call), so the map is bounded by that search's node count and
//! dropped afterwards. It is single-threaded within a search — the engine's
//! parallelism is across processes — so a [`RefCell`] suffices and `CachedEval`
//! is intentionally neither `Sync` nor `Send`.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::board::Board;
use crate::eval::{Evaluator, Value};

/// Memoizes `board -> Value` in front of any [`Evaluator`]. `Board` derives
/// `Hash + Eq`, so it is the key directly (no position-id round-trip).
pub struct CachedEval<E: Evaluator> {
    inner: E,
    cache: RefCell<HashMap<Board, Value>>,
}

impl<E: Evaluator> CachedEval<E> {
    pub fn new(inner: E) -> Self {
        CachedEval { inner, cache: RefCell::new(HashMap::new()) }
    }

    /// Distinct positions currently memoized (diagnostics / tests).
    pub fn cached(&self) -> usize {
        self.cache.borrow().len()
    }
}

impl<E: Evaluator> Evaluator for CachedEval<E> {
    fn evaluate(&self, board: &Board) -> Value {
        if let Some(v) = self.cache.borrow().get(board) {
            return *v;
        }
        let v = self.inner.evaluate(board);
        self.cache.borrow_mut().insert(*board, v);
        v
    }

    fn evaluate_batch(&self, boards: &[Board]) -> Vec<Value> {
        // Serve cached hits immediately; batch only the misses so the net still
        // does one matmul over the positions it has not seen. Input order is
        // preserved, so the result is indistinguishable from `inner`'s.
        let mut out: Vec<Option<Value>> = vec![None; boards.len()];
        let mut miss_at: Vec<usize> = Vec::new();
        let mut miss_boards: Vec<Board> = Vec::new();
        {
            let cache = self.cache.borrow();
            for (i, b) in boards.iter().enumerate() {
                match cache.get(b) {
                    Some(v) => out[i] = Some(*v),
                    None => {
                        miss_at.push(i);
                        miss_boards.push(*b);
                    }
                }
            }
        }
        if !miss_boards.is_empty() {
            let vals = self.inner.evaluate_batch(&miss_boards);
            let mut cache = self.cache.borrow_mut();
            for (k, v) in vals.into_iter().enumerate() {
                out[miss_at[k]] = Some(v);
                cache.insert(miss_boards[k], v);
            }
        }
        out.into_iter().map(|v| v.expect("every slot filled")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, MOVER};
    use crate::eval::HceEval;
    use crate::search::{position_dist, position_value};

    /// Wrapping an evaluator in the cache must not change ANY search result —
    /// scalar or distribution, full-width or candidate-pruned, at every depth.
    /// HCE is deterministic, so equality is exact.
    #[test]
    fn cached_search_matches_uncached() {
        let hce = HceEval::new();
        let cached = CachedEval::new(&hce);

        let mut race = Board::empty();
        race.set_off(MOVER, 3);
        race.set_point(4, 6);
        race.set_point(5, 6);
        race.set_point(20, 6);
        race.set_point(21, 6);

        for b in [Board::starting_position(), race] {
            // Full-width (candidates = 0) is cheap only up to 2-ply; cover the
            // scalar and distribution leaves there.
            for depth in [0u8, 1, 2] {
                assert_eq!(
                    position_value(&b, depth, &hce),
                    position_value(&b, depth, &cached),
                    "scalar mismatch at depth {depth}"
                );
                assert_eq!(
                    position_dist(&b, depth, 0, &hce),
                    position_dist(&b, depth, 0, &cached),
                    "full-width dist mismatch at depth {depth}"
                );
            }
            // Deeper (3-ply) only candidate-pruned — fast, and it still exercises
            // the recursion and the batched-eval cache path.
            for depth in [2u8, 3] {
                assert_eq!(
                    position_dist(&b, depth, 2, &hce),
                    position_dist(&b, depth, 2, &cached),
                    "pruned dist mismatch at depth {depth}"
                );
            }
        }
        assert!(cached.cached() > 0, "cache never populated");
    }
}
