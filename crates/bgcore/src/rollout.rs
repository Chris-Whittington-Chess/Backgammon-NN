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
    /// Number of playouts averaged per position (ignored when `movetime_ms > 0`).
    pub trials: usize,
    /// Evaluate the leaf with the net after this many plies; `0` plays to the end.
    pub truncate_plies: usize,
    /// Roll out only the best this-many moves at the root; `0` rolls out all.
    pub candidates: usize,
    /// Base seed — fixed across candidates for common random numbers.
    pub seed: u64,
    /// If non-zero, roll out until this many milliseconds elapse instead of a
    /// fixed `trials` count (for a move, the budget is split across candidates).
    pub movetime_ms: u64,
    /// Worker threads; `0` uses the global rayon pool (honours RAYON_NUM_THREADS).
    pub threads: usize,
    /// Play race (no-contact) positions in the playout with the evaluator rather
    /// than min-pip. Only sound with a race-competent evaluator (a phase net whose
    /// race half is trained on races); with a plain contact net, leave `false` and
    /// keep the min-pip fallback. Default `false`.
    pub net_race: bool,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        RolloutConfig {
            trials: 180,
            truncate_plies: 11,
            candidates: 6,
            seed: 0x5EED,
            movetime_ms: 0,
            threads: 0,
            net_race: false,
        }
    }
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// 0-ply score of every move's resulting position, to the side that just moved
/// (terminal wins score their points). The non-terminal children are evaluated
/// in **one batched forward pass** — the rollout hot path, since it runs at every
/// ply for all legal moves.
fn shallow_scores<E: Evaluator>(moves: &[Move], eval: &E) -> Vec<f32> {
    let mut scores = vec![0.0f32; moves.len()];
    let mut boards = Vec::with_capacity(moves.len());
    let mut slots = Vec::with_capacity(moves.len());
    for (i, m) in moves.iter().enumerate() {
        match result(&m.result) {
            GameResult::MoverWins(p) => scores[i] = p as f32,
            _ => {
                slots.push(i);
                boards.push(m.result.swap_perspective());
            }
        }
    }
    let vals = eval.evaluate_batch(&boards);
    for (k, &i) in slots.iter().enumerate() {
        scores[i] = -vals[k].equity();
    }
    scores
}

fn argmax(scores: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if s > best {
            best = s;
            best_i = i;
        }
    }
    best_i
}

/// Choose the playout move and return its index into `moves`.
///
/// Once the sides have passed (a pure race), the net is a weak and biased guide
/// — it blunders bear-offs and so wildly over-produces gammons, poisoning any
/// rollout labels. There, blots are irrelevant and the only goal is to get home
/// and off fastest, which minimising the resulting pip count does near-optimally
/// (it also races the back checkers home to save the gammon when behind).
/// Otherwise fall back to greedy 0-ply on the evaluator.
fn pick_playout<E: Evaluator>(b: &Board, moves: &[Move], eval: &E, net_race: bool) -> usize {
    if b.no_contact() && !net_race {
        let mut best_i = 0;
        let mut best = i32::MAX;
        for (i, m) in moves.iter().enumerate() {
            let pip = m.result.pip_count(crate::board::MOVER);
            if pip < best {
                best = pip;
                best_i = i;
            }
        }
        return best_i;
    }
    argmax(&shallow_scores(moves, eval))
}

/// One truncated playout from `board`, returning equity **from the perspective
/// of `board`'s side to move**. Uses a 0-ply policy (greedy on `eval`).
fn rollout_once<E: Evaluator>(
    board: &Board,
    eval: &E,
    truncate: usize,
    net_race: bool,
    rng: &mut Rng,
) -> f32 {
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
        let chosen = &moves[pick_playout(&b, &moves, eval, net_race)];
        if let GameResult::MoverWins(p) = result(&chosen.result) {
            return sign * p as f32; // the side that just moved won
        }
        b = chosen.result.swap_perspective();
        plies += 1;
    }
}

/// Expected equity for the side to move at `board`, from parallel truncated
/// rollouts — a fixed `cfg.trials` count, or (if `cfg.movetime_ms > 0`) as many
/// as fit in the time budget.
pub fn rollout_equity<E: Evaluator + Sync>(board: &Board, eval: &E, cfg: &RolloutConfig) -> f32 {
    if cfg.movetime_ms > 0 {
        return rollout_timed(board, eval, cfg);
    }
    if cfg.trials == 0 {
        return eval.evaluate(board).equity();
    }
    let sum: f32 = (0..cfg.trials)
        .into_par_iter()
        .map(|t| {
            let mut rng = Rng::new(cfg.seed.wrapping_add(t as u64 + 1).wrapping_mul(GOLDEN));
            rollout_once(board, eval, cfg.truncate_plies, cfg.net_race, &mut rng)
        })
        .sum();
    sum / cfg.trials as f32
}

/// Roll out in parallel batches until the movetime budget elapses.
fn rollout_timed<E: Evaluator + Sync>(board: &Board, eval: &E, cfg: &RolloutConfig) -> f32 {
    use std::time::{Duration, Instant};
    let deadline = Instant::now() + Duration::from_millis(cfg.movetime_ms);
    let batch: u64 = 64;
    let mut total = 0.0f32;
    let mut count = 0u64;
    loop {
        let base = count;
        let s: f32 = (0..batch)
            .into_par_iter()
            .map(|t| {
                let mut rng = Rng::new(cfg.seed.wrapping_add(base + t + 1).wrapping_mul(GOLDEN));
                rollout_once(board, eval, cfg.truncate_plies, cfg.net_race, &mut rng)
            })
            .sum();
        total += s;
        count += batch;
        if Instant::now() >= deadline {
            break;
        }
    }
    total / count as f32
}

/// Pick the best move by rolling out the top candidates (with common random
/// numbers) and choosing the highest rollout equity.
pub fn rollout_best<E: Evaluator + Sync>(
    board: &Board,
    dice: &Dice,
    eval: &E,
    cfg: &RolloutConfig,
) -> Move {
    rollout_best_scored(board, dice, eval, cfg).0
}

/// As [`rollout_best`], but also returns the chosen move's rollout equity (from
/// the mover's perspective).
pub fn rollout_best_scored<E: Evaluator + Sync>(
    board: &Board,
    dice: &Dice,
    eval: &E,
    cfg: &RolloutConfig,
) -> (Move, f32) {
    let mut moves = genmoves(board, dice);

    let mut order: Vec<usize> = (0..moves.len()).collect();
    if cfg.candidates > 0 && moves.len() > cfg.candidates {
        let scores = shallow_scores(&moves, eval);
        order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
        order.truncate(cfg.candidates);
    }

    // Under a movetime budget, share it across the candidates being rolled out.
    let mut cc = cfg.clone();
    if cfg.movetime_ms > 0 {
        cc.movetime_ms = (cfg.movetime_ms / order.len().max(1) as u64).max(1);
    }

    let mut best_i = order[0];
    let mut best = f32::NEG_INFINITY;
    for &i in &order {
        let s = match result(&moves[i].result) {
            GameResult::MoverWins(p) => p as f32,
            _ => -rollout_equity(&moves[i].result.swap_perspective(), eval, &cc),
        };
        if s > best {
            best = s;
            best_i = i;
        }
    }
    (moves.swap_remove(best_i), best)
}

// --- Outcome-distribution rollouts (for training labels & cube decisions) ---

fn win_vec(points: u8) -> [f32; 5] {
    [1.0, (points >= 2) as u8 as f32, (points >= 3) as u8 as f32, 0.0, 0.0]
}

/// Convert an opponent-perspective 5-vector to the mover's, matching the trainer:
/// `[w, wg, wbg, lg, lbg] -> [1-w, lg, lbg, wg, wbg]`.
fn flip5(v: [f32; 5]) -> [f32; 5] {
    [1.0 - v[0], v[3], v[4], v[1], v[2]]
}

fn orient(v: [f32; 5], plies: usize) -> [f32; 5] {
    if plies.is_multiple_of(2) {
        v
    } else {
        flip5(v)
    }
}

/// One playout, returning the outcome distribution `[win, win_g, win_bg,
/// lose_g, lose_bg]` from the perspective of `board`'s side to move.
fn rollout_once_dist<E: Evaluator>(
    board: &Board,
    eval: &E,
    truncate: usize,
    net_race: bool,
    rng: &mut Rng,
) -> [f32; 5] {
    let mut b = board.clone();
    let mut plies = 0usize;
    loop {
        match result(&b) {
            GameResult::MoverWins(p) => return orient(win_vec(p), plies),
            GameResult::OppWins(p) => return orient(win_vec(p), plies + 1),
            GameResult::InProgress => {}
        }
        if truncate > 0 && plies >= truncate {
            let v = eval.evaluate(&b);
            return orient([v.win, v.win_g, v.win_bg, v.lose_g, v.lose_bg], plies);
        }
        let moves = genmoves(&b, &rng.roll());
        let chosen = &moves[pick_playout(&b, &moves, eval, net_race)];
        if let GameResult::MoverWins(p) = result(&chosen.result) {
            return orient(win_vec(p), plies);
        }
        b = chosen.result.swap_perspective();
        plies += 1;
    }
}

fn add5(a: [f32; 5], b: [f32; 5]) -> [f32; 5] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4]]
}

/// Mean outcome distribution for the side to move at `board`, from parallel
/// truncated rollouts (fixed `trials`, or `movetime_ms` if set).
pub fn rollout_dist<E: Evaluator + Sync>(board: &Board, eval: &E, cfg: &RolloutConfig) -> [f32; 5] {
    let mean = |sums: [f32; 5], n: f32| [sums[0] / n, sums[1] / n, sums[2] / n, sums[3] / n, sums[4] / n];

    if cfg.movetime_ms > 0 {
        use std::time::{Duration, Instant};
        let deadline = Instant::now() + Duration::from_millis(cfg.movetime_ms);
        let batch: u64 = 64;
        let mut total = [0.0f32; 5];
        let mut count = 0u64;
        loop {
            let base = count;
            let s = (0..batch)
                .into_par_iter()
                .map(|t| {
                    let mut rng = Rng::new(cfg.seed.wrapping_add(base + t + 1).wrapping_mul(GOLDEN));
                    rollout_once_dist(board, eval, cfg.truncate_plies, cfg.net_race, &mut rng)
                })
                .reduce(|| [0.0; 5], add5);
            total = add5(total, s);
            count += batch;
            if Instant::now() >= deadline {
                break;
            }
        }
        return mean(total, count as f32);
    }

    if cfg.trials == 0 {
        let v = eval.evaluate(board);
        return [v.win, v.win_g, v.win_bg, v.lose_g, v.lose_bg];
    }
    let sums = (0..cfg.trials)
        .into_par_iter()
        .map(|t| {
            let mut rng = Rng::new(cfg.seed.wrapping_add(t as u64 + 1).wrapping_mul(GOLDEN));
            rollout_once_dist(board, eval, cfg.truncate_plies, cfg.net_race, &mut rng)
        })
        .reduce(|| [0.0; 5], add5);
    mean(sums, cfg.trials as f32)
}

/// An [`Engine`] that picks its move by rolling out the top candidates and
/// choosing the highest rollout equity. Far stronger than static/1-ply play,
/// and much heavier — meant for strong play and for labelling training data.
pub struct RolloutEngine<E: Evaluator + Sync> {
    eval: E,
    cfg: RolloutConfig,
    name: String,
    pool: Option<rayon::ThreadPool>,
}

impl<E: Evaluator + Sync> RolloutEngine<E> {
    pub fn new(eval: E, cfg: RolloutConfig, name: impl Into<String>) -> Self {
        let pool = build_pool(cfg.threads);
        RolloutEngine { eval, cfg, name: name.into(), pool }
    }
}

impl<E: Evaluator + Sync> Engine for RolloutEngine<E> {
    fn choose(&mut self, board: &Board, dice: &Dice) -> Move {
        match &self.pool {
            Some(pool) => pool.install(|| rollout_best(board, dice, &self.eval, &self.cfg)),
            None => rollout_best(board, dice, &self.eval, &self.cfg),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Build a dedicated rayon pool with `threads` workers, or `None` (global pool)
/// when `threads == 0`.
pub fn build_pool(threads: usize) -> Option<rayon::ThreadPool> {
    if threads == 0 {
        None
    } else {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::MOVER;
    use crate::eval::HceEval;

    #[test]
    fn rollout_equity_is_finite_and_bounded() {
        let cfg = RolloutConfig { trials: 24, truncate_plies: 6, candidates: 0, seed: 1, ..Default::default() };
        let v = rollout_equity(&Board::starting_position(), &HceEval::new(), &cfg);
        assert!(v.is_finite() && v.abs() <= 3.0, "value {v}");
    }

    #[test]
    fn winning_position_rolls_out_positive() {
        // Mover is on the ace point (pip 15); opponent stuck deep (pip ~345).
        let mut b = Board::empty();
        b.set_point(1, 15);
        b.set_point(2, -15);
        let cfg = RolloutConfig { trials: 40, truncate_plies: 0, candidates: 0, seed: 2, ..Default::default() };
        let v = rollout_equity(&b, &HceEval::new(), &cfg);
        assert!(v > 0.7, "a won position rolled out to {v}");
    }

    #[test]
    fn race_policy_saves_the_gammon() {
        // Pure race, no contact. Mover has 10 off and 5 on the ace point — it
        // needs ~3 rolls to finish. The opponent has 0 off but all 15 checkers
        // in its OWN home board (mover-relative points 19..=24), so it gets two
        // or three turns first. Any competent race play bears off several
        // checkers in that time, so a gammon (opponent bears off none) should
        // be rare and a backgammon (a checker still stuck deep) rarer still.
        let mut b = Board::empty();
        b.set_off(MOVER, 10);
        b.set_point(1, 5);
        b.set_point(24, -3);
        b.set_point(23, -3);
        b.set_point(22, -3);
        b.set_point(21, -3);
        b.set_point(20, -3);
        assert!(b.no_contact());
        let cfg = RolloutConfig { trials: 200, truncate_plies: 0, candidates: 0, seed: 4, ..Default::default() };
        let d = rollout_dist(&b, &HceEval::new(), &cfg);
        assert!(d[0] > 0.9, "mover should almost always win, win={}", d[0]);
        assert!(d[1] < 0.1, "gammon should be rare with good race play, win_g={}", d[1]);
        assert!(d[2] < 0.02, "backgammon should be very rare, win_bg={}", d[2]);
    }

    #[test]
    fn movetime_runs_within_its_budget() {
        let cfg = RolloutConfig {
            movetime_ms: 60,
            truncate_plies: 8,
            candidates: 0,
            seed: 3,
            ..Default::default()
        };
        let t = std::time::Instant::now();
        let v = rollout_equity(&Board::starting_position(), &HceEval::new(), &cfg);
        let ms = t.elapsed().as_millis();
        assert!(v.is_finite() && v.abs() <= 3.0);
        assert!((50..2000).contains(&ms), "movetime rollout took {ms}ms");
    }

    #[test]
    fn common_random_numbers_are_deterministic() {
        let cfg = RolloutConfig { trials: 32, truncate_plies: 8, candidates: 0, seed: 7, ..Default::default() };
        let b = Board::starting_position();
        let a = rollout_equity(&b, &HceEval::new(), &cfg);
        let c = rollout_equity(&b, &HceEval::new(), &cfg);
        assert_eq!(a, c, "same seed must give the same estimate");
    }

    #[test]
    fn dist_is_a_coherent_distribution() {
        let cfg = RolloutConfig { trials: 64, truncate_plies: 0, candidates: 0, seed: 5, ..Default::default() };
        let d = rollout_dist(&Board::starting_position(), &HceEval::new(), &cfg);
        // Every component is a frequency in [0, 1].
        assert!(d.iter().all(|&x| (0.0..=1.0).contains(&x)), "{d:?}");
        // Nested outcomes: win >= win_g >= win_bg, and lose_g >= lose_bg.
        assert!(d[0] >= d[1] - 1e-6 && d[1] >= d[2] - 1e-6, "win nesting {d:?}");
        assert!(d[3] >= d[4] - 1e-6, "lose nesting {d:?}");
        // A near-even opening: win probability should sit around a half.
        assert!((0.3..0.7).contains(&d[0]), "opening win prob {}", d[0]);
        // Equity from the distribution matches rollout_equity closely.
        let eq_from_dist = (2.0 * d[0] - 1.0) + d[1] + d[2] - d[3] - d[4];
        let eq = rollout_equity(&Board::starting_position(), &HceEval::new(), &cfg);
        assert!((eq_from_dist - eq).abs() < 0.15, "eq {eq} vs from dist {eq_from_dist}");
    }
}
