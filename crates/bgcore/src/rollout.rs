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
use crate::moves::{genmoves, genmoves_playout, Move};
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

/// `shallow_scores` over resulting boards directly (the playout hot path uses
/// `genmoves_playout`, which yields boards without step history).
fn shallow_scores_boards<E: Evaluator>(children: &[Board], eval: &E) -> Vec<f32> {
    let mut scores = vec![0.0f32; children.len()];
    let mut boards = Vec::with_capacity(children.len());
    let mut slots = Vec::with_capacity(children.len());
    for (i, c) in children.iter().enumerate() {
        match result(c) {
            GameResult::MoverWins(p) => scores[i] = p as f32,
            _ => {
                slots.push(i);
                boards.push(c.swap_perspective());
            }
        }
    }
    let vals = eval.evaluate_batch(&boards);
    for (k, &i) in slots.iter().enumerate() {
        scores[i] = -vals[k].equity();
    }
    scores
}

/// `pick_playout` over resulting boards (see [`shallow_scores_boards`]).
fn pick_playout_boards<E: Evaluator>(
    b: &Board,
    children: &[Board],
    eval: &E,
    net_race: bool,
) -> usize {
    if b.no_contact() && !net_race {
        let mut best_i = 0;
        let mut best = i32::MAX;
        for (i, c) in children.iter().enumerate() {
            let pip = c.pip_count(crate::board::MOVER);
            if pip < best {
                best = pip;
                best_i = i;
            }
        }
        return best_i;
    }
    argmax(&shallow_scores_boards(children, eval))
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
    let mut b = *board;
    let mut plies = 0usize;
    let mut children: Vec<Board> = Vec::new();
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

        genmoves_playout(&b, &rng.roll(), &mut children);
        let chosen = children[pick_playout_boards(&b, &children, eval, net_race)];
        if let GameResult::MoverWins(p) = result(&chosen) {
            return sign * p as f32; // the side that just moved won
        }
        b = chosen.swap_perspective();
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
    let mut b = *board;
    let mut plies = 0usize;
    let mut children: Vec<Board> = Vec::new();
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
        genmoves_playout(&b, &rng.roll(), &mut children);
        let chosen = children[pick_playout_boards(&b, &children, eval, net_race)];
        if let GameResult::MoverWins(p) = result(&chosen) {
            return orient(win_vec(p), plies);
        }
        b = chosen.swap_perspective();
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

// --- Batched "wave" rollouts: label MANY positions in one pass -------------
//
// `rollout_dist` above rolls out ONE position; each ply scores that position's
// ~18 legal children in a tiny `[~18, 198]` matmul. Small matmuls badly
// underutilise the hardware. The wave engine instead keeps every `(board,
// trial)` playout of MANY source positions in lockstep and, per ply-step,
// gathers *all* playouts' pending net queries (move-selection children +
// truncation leaves) into ONE huge batched forward pass. Same playouts, same
// dice, same policy — just the net evals coalesced. Per-board output is
// identical to calling `rollout_dist` on each board (within f32 tolerance:
// tract's matmul may reduce a 198-dot in a different SIMD order at a different
// batch size, which can rarely flip an ultra-close argmax).

#[derive(Clone, Copy, PartialEq)]
enum StepKind {
    /// Finished, or nothing to do this step.
    Idle,
    /// Needs a truncation-leaf eval of the current board.
    Leaf,
    /// Needs the net to score this ply's (contact) children before advancing.
    Move,
}

/// A single `(board, trial)` playout kept in flight by the wave engine.
struct WavePlayout {
    b: Board,
    plies: usize,
    rng: Rng,
    /// Index of the source board this trial belongs to (within the wave chunk).
    board_idx: usize,
    /// Set once the playout terminates or truncates: its outcome 5-vector,
    /// oriented to the source board's mover.
    done: Option<[f32; 5]>,
    /// What this playout needs from the batched eval this step.
    kind: StepKind,
    /// Transient: this ply's legal resulting boards (only for `StepKind::Move`).
    moves: Vec<Board>,
    /// Transient: offset of this playout's rows in the wave's batch this step.
    batch_start: usize,
}

/// Batched Monte-Carlo rollouts for MANY positions at once. Produces the same
/// per-board outcome distribution `[win, win_g, win_bg, lose_g, lose_bg]` as
/// calling [`rollout_dist`] on each board.
///
/// The parallelism axis matters. tract's matmul is single-threaded, so the win
/// is *per-thread batching*, not one giant matmul: coalescing a whole wave into
/// a single huge matmul would serialise the hot path onto one core and lose the
/// core-level parallelism the per-position rollout already gets for free. So we
/// parallelise **across** `wave_boards`-sized chunks (one core each) and, within
/// a chunk, run its `(board, trial)` playouts in lockstep — every ply's net
/// evals coalesce into one batch that sits right in tract's batched sweet spot.
/// With the default `trials`, `wave_boards = 1` already yields a ~2–3k-row batch
/// per ply, so keep the chunk small: bigger chunks only cost memory and coarsen
/// load balancing.
pub fn rollout_dist_wave<E: Evaluator + Sync>(
    boards: &[Board],
    eval: &E,
    cfg: &RolloutConfig,
    wave_boards: usize,
) -> Vec<[f32; 5]> {
    let chunk = if wave_boards == 0 { 1 } else { wave_boards };
    let parts: Vec<Vec<[f32; 5]>> =
        boards.par_chunks(chunk).map(|group| wave_chunk(group, eval, cfg)).collect();
    parts.concat()
}

/// One wave: roll out all trials of `boards` together to completion, on a single
/// thread (the caller parallelises across waves). Every ply-step coalesces all
/// live playouts' net queries into one batched forward pass.
fn wave_chunk<E: Evaluator>(boards: &[Board], eval: &E, cfg: &RolloutConfig) -> Vec<[f32; 5]> {
    let n = boards.len();
    if n == 0 {
        return Vec::new();
    }
    // trials == 0 degenerates to a static (batched) eval, matching rollout_dist.
    if cfg.trials == 0 {
        return eval
            .evaluate_batch(boards)
            .into_iter()
            .map(|v| [v.win, v.win_g, v.win_bg, v.lose_g, v.lose_bg])
            .collect();
    }
    let truncate = cfg.truncate_plies;
    let net_race = cfg.net_race;

    // One playout per (board, trial). Trial `t` uses the exact per-trial seed of
    // rollout_dist, so each board's trial set reproduces its standalone rollout.
    let mut playouts: Vec<WavePlayout> = Vec::with_capacity(n * cfg.trials);
    for (bi, b) in boards.iter().enumerate() {
        for t in 0..cfg.trials {
            let rng = Rng::new(cfg.seed.wrapping_add(t as u64 + 1).wrapping_mul(GOLDEN));
            playouts.push(WavePlayout {
                b: b.clone(),
                plies: 0,
                rng,
                board_idx: bi,
                done: None,
                kind: StepKind::Idle,
                moves: Vec::new(),
                batch_start: 0,
            });
        }
    }

    loop {
        // Phase A (no net): advance every alive playout up to its next net query.
        // Terminals resolve; pure-race plies play min-pip in place (no net), so a
        // playout may cross several race plies here; contact plies stop with their
        // moves stashed for the batch.
        playouts.iter_mut().for_each(|p| {
            if p.done.is_some() {
                p.kind = StepKind::Idle;
                return;
            }
            loop {
                match result(&p.b) {
                    GameResult::MoverWins(pt) => {
                        p.done = Some(orient(win_vec(pt), p.plies));
                        p.kind = StepKind::Idle;
                        return;
                    }
                    GameResult::OppWins(pt) => {
                        p.done = Some(orient(win_vec(pt), p.plies + 1));
                        p.kind = StepKind::Idle;
                        return;
                    }
                    GameResult::InProgress => {}
                }
                if truncate > 0 && p.plies >= truncate {
                    p.kind = StepKind::Leaf;
                    return;
                }
                let dice = p.rng.roll();
                genmoves_playout(&p.b, &dice, &mut p.moves);
                if p.b.no_contact() && !net_race {
                    // Pure race: min-pip, no net. Advance in place and keep going.
                    let mut best_i = 0usize;
                    let mut best = i32::MAX;
                    for (i, c) in p.moves.iter().enumerate() {
                        let pip = c.pip_count(crate::board::MOVER);
                        if pip < best {
                            best = pip;
                            best_i = i;
                        }
                    }
                    let chosen = p.moves[best_i];
                    if let GameResult::MoverWins(pt) = result(&chosen) {
                        p.done = Some(orient(win_vec(pt), p.plies));
                        p.kind = StepKind::Idle;
                        return;
                    }
                    p.b = chosen.swap_perspective();
                    p.plies += 1;
                    continue;
                }
                // Contact: the net must score the children — defer to the batch.
                // p.moves already holds this ply's resulting boards.
                p.kind = StepKind::Move;
                return;
            }
        });

        // Phase B (gather): concatenate every pending net query into one batch.
        // Leaf -> the playout's board; Move -> its non-terminal swapped children,
        // in move order (mirrors shallow_scores). Sequential to fix batch order.
        let mut batch: Vec<Board> = Vec::new();
        for p in playouts.iter_mut() {
            match p.kind {
                StepKind::Leaf => {
                    p.batch_start = batch.len();
                    batch.push(p.b.clone());
                }
                StepKind::Move => {
                    p.batch_start = batch.len();
                    for c in &p.moves {
                        if !matches!(result(c), GameResult::MoverWins(_)) {
                            batch.push(c.swap_perspective());
                        }
                    }
                }
                StepKind::Idle => {}
            }
        }
        if batch.is_empty() {
            break; // every playout has finished
        }

        // Phase C: the whole point — ONE batched forward pass for the ply.
        let vals = eval.evaluate_batch(&batch);

        // Phase D (scatter): fold leaves; for contact plies rebuild
        // shallow_scores from the batched vals, argmax, and advance (or finish).
        playouts.iter_mut().for_each(|p| match p.kind {
            StepKind::Leaf => {
                let v = vals[p.batch_start];
                p.done = Some(orient([v.win, v.win_g, v.win_bg, v.lose_g, v.lose_bg], p.plies));
            }
            StepKind::Move => {
                let mut scores = vec![0.0f32; p.moves.len()];
                let mut k = p.batch_start;
                for (i, c) in p.moves.iter().enumerate() {
                    match result(c) {
                        GameResult::MoverWins(pt) => scores[i] = pt as f32,
                        _ => {
                            scores[i] = -vals[k].equity();
                            k += 1;
                        }
                    }
                }
                let idx = argmax(&scores);
                let chosen = p.moves[idx];
                if let GameResult::MoverWins(pt) = result(&chosen) {
                    p.done = Some(orient(win_vec(pt), p.plies));
                } else {
                    p.b = chosen.swap_perspective();
                    p.plies += 1;
                }
                p.moves.clear();
            }
            StepKind::Idle => {}
        });
    }

    // Aggregate: mean over each source board's trials.
    let mut sums = vec![[0.0f32; 5]; n];
    for p in &playouts {
        let d = p.done.expect("every playout finishes");
        sums[p.board_idx] = add5(sums[p.board_idx], d);
    }
    let inv = 1.0 / cfg.trials as f32;
    sums.into_iter()
        .map(|s| [s[0] * inv, s[1] * inv, s[2] * inv, s[3] * inv, s[4] * inv])
        .collect()
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
    fn wave_matches_per_trial_rollout_dist() {
        // The batched wave engine must reproduce, per board, the same outcome
        // distribution as calling rollout_dist on each board. With HceEval the
        // batched and single-position evals are bit-identical (no matmul), so any
        // gap here is a logic bug, not float reordering. (The tiny residual is
        // just the order the 180 per-trial vectors are summed in.)
        let cfg = RolloutConfig { trials: 96, truncate_plies: 7, candidates: 0, seed: 11, ..Default::default() };
        let eval = HceEval::new();

        // A spread of positions: opening, a mid-game contact position, and a race.
        let opening = Board::starting_position();
        let mut contact = Board::empty();
        contact.set_point(6, 4);
        contact.set_point(8, 3);
        contact.set_point(13, 5);
        contact.set_point(24, 2);
        contact.set_point(1, -3);
        contact.set_point(12, -5);
        contact.set_point(19, -4);
        contact.set_point(17, -3);
        let mut race = Board::empty();
        race.set_off(MOVER, 3);
        race.set_point(1, 4);
        race.set_point(2, 4);
        race.set_point(3, 4);
        race.set_point(22, -4);
        race.set_point(23, -4);
        race.set_point(24, -4);
        assert!(race.no_contact());
        let boards = vec![opening, contact, race];

        let per_trial: Vec<[f32; 5]> =
            boards.iter().map(|b| rollout_dist(b, &eval, &cfg)).collect();
        // wave_boards = 0 (all in one wave) and a small chunk must both match.
        for wb in [0usize, 2] {
            let wave = rollout_dist_wave(&boards, &eval, &cfg, wb);
            let mut max_diff = 0.0f32;
            for (w, s) in wave.iter().zip(&per_trial) {
                for k in 0..5 {
                    max_diff = max_diff.max((w[k] - s[k]).abs());
                }
            }
            assert!(
                max_diff < 1e-4,
                "wave (wave_boards={wb}) diverged from per-trial rollout_dist by {max_diff}"
            );
        }
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
