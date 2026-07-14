//! Benchmark our engines against the wildbg reference engine (SPEC §13).
//!
//! wildbg is an independent neural-net backgammon engine (~5.9 GnuBG-2-ply error
//! rate), so beating or matching it places our engine on a real external scale
//! rather than only against our own HCE. Both engines share our board
//! orientation and agree on the GnuBG Position ID, so a game alternates between
//! `bgcore` moves and wildbg moves on one shared position.
//!
//! Games are played in mirrored pairs (same dice, our engine in each seat once)
//! to cancel dice luck. Money-game equity drives both sides' move choice.
//!
//! Run:  cargo run --release --manifest-path tools/wildbg-bench/Cargo.toml -- [games]

use bgcore::board::{MOVER, NUM_POINTS, OPP};
use bgcore::eval::NnEval;
use bgcore::{result, Board, Dice, Engine, EvalEngine, GameResult, RolloutConfig, RolloutEngine, Rng};
use engine::composite::CompositeEvaluator;
use engine::dice::Dice as WDice;
use engine::evaluator::Evaluator;
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

/// wildbg's chosen move, returned as our board in the mover's perspective (turn
/// not yet passed) so it drops straight into the shared game loop. wildbg's
/// `best_position` returns the position with sides already switched, so we switch
/// back and round-trip through the Position ID both engines agree on.
fn wildbg_move(wildbg: &CompositeEvaluator, b: &Board, d: &Dice) -> Board {
    let (a, c) = d.pair();
    let best_switched =
        wildbg.best_position(&to_wildbg(b), &WDice::new(a as usize, c as usize), |p| p.equity());
    Board::from_position_id(&best_switched.sides_switched().position_id())
        .expect("wildbg position id should round-trip")
}

/// Play one game; return `(winning_seat, points)`. Seat 0 moves first.
fn play_game(ours: &mut dyn Engine, wildbg: &CompositeEvaluator, our_seat: usize, seed: u64) -> (usize, u32) {
    let mut b = Board::starting_position();
    let mut rng = Rng::new(seed);
    let mut on_roll = 0usize;
    for _ in 0..4000 {
        let d = rng.roll();
        let played = if on_roll == our_seat {
            ours.choose(&b, &d).result
        } else {
            wildbg_move(wildbg, &b, &d)
        };
        match result(&played) {
            GameResult::MoverWins(p) => return (on_roll, p as u32),
            _ => {
                b = played.swap_perspective();
                on_roll ^= 1;
            }
        }
    }
    (0, 1)
}

/// Mirrored-pair benchmark of one engine vs wildbg. Prints win% and points/game.
fn bench(name: &str, ours: &mut dyn Engine, wildbg: &CompositeEvaluator, pairs: usize) {
    let mut our_points: i64 = 0;
    let mut our_wins = 0usize;
    let total = pairs * 2;
    for g in 0..pairs {
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(g as u64 + 1);
        for &seat in &[0usize, 1usize] {
            let (winner, pts) = play_game(ours, wildbg, seat, seed);
            if winner == seat {
                our_wins += 1;
                our_points += pts as i64;
            } else {
                our_points -= pts as i64;
            }
        }
    }
    println!(
        "  {name:20} vs wildbg:  win {:.1}%   PPG {:+.3}   ({total} games)",
        100.0 * our_wins as f64 / total as f64,
        our_points as f64 / total as f64
    );
}

fn main() {
    let root = env!("CARGO_MANIFEST_DIR");
    let nets = format!("{root}/../../external/wildbg/neural-nets");
    let wildbg = CompositeEvaluator::from_file_paths(
        &format!("{nets}/contact.onnx"),
        &format!("{nets}/race.onnx"),
    )
    .expect("load wildbg nets");
    let onnx = format!("{root}/../../models/td.onnx");

    let pairs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(150);
    println!("bgcore vs wildbg (mirrored pairs; wildbg ~5.9 GnuBG-2-ply error rate)\n");

    let nn = NnEval::from_path(&onnx).expect("load our net");
    let mut zero_ply = EvalEngine::new(nn, "0-ply net");
    bench("0-ply net", &mut zero_ply, &wildbg, pairs);

    let nn2 = NnEval::from_path(&onnx).expect("load our net");
    let cfg = RolloutConfig { movetime_ms: 300, truncate_plies: 9, candidates: 5, ..Default::default() };
    let mut rollout = RolloutEngine::new(nn2, cfg, "rollout");
    bench("rollout (300ms)", &mut rollout, &wildbg, (pairs / 5).max(10));
}
