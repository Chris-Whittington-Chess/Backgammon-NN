//! Benchmark our engine against wildbg on an equal footing (SPEC §13).
//!
//! wildbg's net is a stronger *evaluator* than ours (separate contact/race nets,
//! richer inputs). To compare fairly we run **both nets through the identical
//! bgcore search** — an adapter lets wildbg's evaluator drive our engines — so
//! the only difference is evaluation quality, at matched search effort:
//!
//!   * 1-ply:   our net vs wildbg's net, static eval of each move (isolates eval)
//!   * rollout: our net vs wildbg's net, same fixed-trial rollout (eval + search)
//!
//! Games are mirrored pairs (same dice, each engine in each seat once). We also
//! report wall-clock per move, since wildbg's eval is heavier per call.
//!
//! Run:  cargo run --release --manifest-path tools/wildbg-bench/Cargo.toml -- [pairs] [trials]

use std::time::{Duration, Instant};

use bgcore::board::{MOVER, NUM_POINTS, OPP};
use bgcore::eval::{Evaluator as BgEvaluator, NnEval, Value};
use bgcore::{result, Board, Engine, EvalEngine, GameResult, RolloutConfig, RolloutEngine, Rng};
use engine::composite::CompositeEvaluator;
use engine::evaluator::Evaluator as WildbgEvaluator;
use engine::position::{Position as WPos, O_BAR, X_BAR};

/// Convert our mover-relative board to a wildbg `Position` (same orientation).
fn to_wildbg(b: &Board) -> WPos {
    let mut pips = [0i8; 26];
    for p in 1..=NUM_POINTS {
        pips[p] = b.point(p);
    }
    pips[X_BAR] = b.bar(MOVER) as i8;
    pips[O_BAR] = -(b.bar(OPP) as i8);
    WPos::try_from(pips).expect("sampled board should be a legal wildbg position")
}

/// wildbg's neural evaluator behind our `Evaluator` trait, so it can drive any
/// bgcore engine. wildbg's `Probabilities` are the six exclusive outcomes; our
/// `Value` is the nested form, so we accumulate.
struct WildbgAdapter(CompositeEvaluator);

impl BgEvaluator for WildbgAdapter {
    fn evaluate(&self, board: &Board) -> Value {
        let p = self.0.eval(&to_wildbg(board));
        Value {
            win: p.win_normal + p.win_gammon + p.win_bg,
            win_g: p.win_gammon + p.win_bg,
            win_bg: p.win_bg,
            lose_g: p.lose_gammon + p.lose_bg,
            lose_bg: p.lose_bg,
        }
    }
}

fn wildbg_adapter() -> WildbgAdapter {
    let nets = format!("{}/../../external/wildbg/neural-nets", env!("CARGO_MANIFEST_DIR"));
    WildbgAdapter(
        CompositeEvaluator::from_file_paths(
            &format!("{nets}/contact.onnx"),
            &format!("{nets}/race.onnx"),
        )
        .expect("load wildbg nets"),
    )
}

fn our_net() -> NnEval {
    NnEval::from_path(&format!("{}/../../models/td.onnx", env!("CARGO_MANIFEST_DIR")))
        .expect("load our net")
}

/// One game between engines `a` and `b`; `a` sits in seat `a_seat`, seat 0 moves
/// first. Accumulates per-engine think time. Returns `(winning_seat, points)`.
fn play(
    a: &mut dyn Engine,
    b: &mut dyn Engine,
    a_seat: usize,
    seed: u64,
    t: &mut [(Duration, u64); 2],
) -> (usize, u32) {
    let mut board = Board::starting_position();
    let mut rng = Rng::new(seed);
    let mut on_roll = 0usize;
    for _ in 0..4000 {
        let d = rng.roll();
        let is_a = on_roll == a_seat;
        let clock = Instant::now();
        let mv = if is_a { a.choose(&board, &d) } else { b.choose(&board, &d) };
        let slot = if is_a { 0 } else { 1 };
        t[slot].0 += clock.elapsed();
        t[slot].1 += 1;
        match result(&mv.result) {
            GameResult::MoverWins(p) => return (on_roll, p as u32),
            _ => {
                board = mv.result.swap_perspective();
                on_roll ^= 1;
            }
        }
    }
    (0, 1)
}

/// Mirrored-pair duel of `a` (our engine) vs `b` (wildbg-backed). Prints A's
/// win% / points-per-game and each side's average ms per move.
fn duel(label: &str, a: &mut dyn Engine, b: &mut dyn Engine, pairs: usize) {
    let mut a_pts: i64 = 0;
    let mut a_wins = 0usize;
    let mut t = [(Duration::ZERO, 0u64); 2]; // [0]=a, [1]=b
    let total = pairs * 2;
    for g in 0..pairs {
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(g as u64 + 1);
        for &a_seat in &[0usize, 1usize] {
            let (winner, pts) = play(a, b, a_seat, seed, &mut t);
            if winner == a_seat {
                a_wins += 1;
                a_pts += pts as i64;
            } else {
                a_pts -= pts as i64;
            }
        }
    }
    let ms = |(d, n): (Duration, u64)| if n == 0 { 0.0 } else { d.as_secs_f64() * 1000.0 / n as f64 };
    println!(
        "  {label:18}  ours win {:.1}%  PPG {:+.3}   [ours {:.1} ms/mv, wildbg {:.1} ms/mv]  ({total} games)",
        100.0 * a_wins as f64 / total as f64,
        a_pts as f64 / total as f64,
        ms(t[0]),
        ms(t[1]),
    );
}

fn main() {
    let pairs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let trials: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    println!(
        "Equal-footing: our net vs wildbg's net through the SAME bgcore search.\n\
         (wildbg net = 2 specialised contact/race nets; ours = 1x 198-Tesauro)\n"
    );

    // 1-ply: static eval of each move — isolates raw evaluation quality.
    let mut ours0 = EvalEngine::new(our_net(), "ours");
    let mut wild0 = EvalEngine::new(wildbg_adapter(), "wildbg");
    duel("1-ply eval", &mut ours0, &mut wild0, pairs * 3);

    // Rollout, equal SEARCH EFFORT: identical fixed-trial count for both.
    let cfg = RolloutConfig { trials, truncate_plies: 9, candidates: 5, ..Default::default() };
    let mut oursr = RolloutEngine::new(our_net(), cfg.clone(), "ours");
    let mut wildr = RolloutEngine::new(wildbg_adapter(), cfg, "wildbg");
    duel(&format!("rollout x{trials}"), &mut oursr, &mut wildr, pairs);

    // Rollout, equal WALL-CLOCK: each side gets the same movetime per move, so
    // our cheaper eval simply does more trials. This is the fair "equal thinking
    // time" fight.
    let mt: u64 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let cfg_t = RolloutConfig { movetime_ms: mt, truncate_plies: 9, candidates: 5, ..Default::default() };
    let mut ourst = RolloutEngine::new(our_net(), cfg_t.clone(), "ours");
    let mut wildt = RolloutEngine::new(wildbg_adapter(), cfg_t, "wildbg");
    duel(&format!("rollout {mt}ms"), &mut ourst, &mut wildt, pairs);
}
