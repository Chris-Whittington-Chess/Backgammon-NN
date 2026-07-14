//! Quick demo of the parallel Monte-Carlo rollout engine with the trained net.
//! Run:  cargo run --release --features onnx --example rollout_demo

use std::time::Instant;

use bgcore::eval::NnEval;
use bgcore::{rollout_equity, run_match, Board, RolloutConfig, RolloutEngine, SearchEngine};

fn main() {
    let path = format!("{}/../../models/td.onnx", env!("CARGO_MANIFEST_DIR"));
    let nn = NnEval::from_path(&path).expect("load net (run trainer/export_onnx.py first)");
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("Monte-Carlo rollouts — {threads} threads\n");

    // Sanity: rollout equities should be sensible.
    let cfg = RolloutConfig { trials: 200, truncate_plies: 9, candidates: 0, seed: 0x5EED };
    let t0 = Instant::now();
    let eq = rollout_equity(&Board::starting_position(), &nn, &cfg);
    println!(
        "opening position:  rollout equity {eq:+.3}   ({} trials in {:.2}s)",
        cfg.trials,
        t0.elapsed().as_secs_f64()
    );

    let mut won = Board::empty();
    won.set_point(1, 15); // mover on the ace point
    won.set_point(2, -15); // opponent stuck deep
    println!("won position:      rollout equity {:+.3}", rollout_equity(&won, &nn, &cfg));

    // Rollout engine vs 1-ply search — same net. Pairs from argv[1] (default 3).
    let pairs: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let mcfg = RolloutConfig { trials: 64, truncate_plies: 6, candidates: 3, seed: 0x5EED };
    let t1 = Instant::now();
    let mut a = RolloutEngine::new(&nn, mcfg, "NN-rollout");
    let mut b = SearchEngine::new(&nn, 1, "NN-1ply");
    let stats = run_match(&mut a, &mut b, pairs, 4242);
    println!(
        "\nNN(rollout 64x6, top-3) vs NN(1-ply):\n  {}  [{:.2} games/sec]",
        stats.summary(),
        (pairs * 2) as f64 / t1.elapsed().as_secs_f64()
    );
}
