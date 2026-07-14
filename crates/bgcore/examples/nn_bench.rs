//! Benchmark the trained ONNX net (as a native Rust engine) against HCE and
//! Random using the mirrored-dice match runner (SPEC §9, §13; milestone M5).
//!
//! Run:  cargo run --release --features onnx --example nn_bench [path/to/td.onnx]

use std::time::Instant;

use bgcore::eval::{HceEval, NnEval, RandomEval};
use bgcore::{run_match, EvalEngine, RolloutConfig, RolloutEngine, SearchEngine};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/../../models/td.onnx", env!("CARGO_MANIFEST_DIR")));

    let nn = match NnEval::from_path(&path) {
        Ok(nn) => nn,
        Err(e) => {
            eprintln!("Could not load {path}: {e}\nRun trainer/export_onnx.py first.");
            std::process::exit(1);
        }
    };
    println!("Loaded net from {path}\n");

    // 0-ply baselines (fast, so run more games).
    {
        let mut a = EvalEngine::new(&nn, "NN");
        let mut b = EvalEngine::new(RandomEval::new(0xABCD), "Random");
        println!("NN(0-ply) vs Random\n  {}", run_match(&mut a, &mut b, 500, 12345).summary());
    }
    {
        let t0 = Instant::now();
        let mut a = EvalEngine::new(&nn, "NN");
        let mut b = EvalEngine::new(HceEval::new(), "HCE");
        let stats = run_match(&mut a, &mut b, 500, 6789);
        let g = 1000.0 / t0.elapsed().as_secs_f64();
        println!("NN(0-ply) vs HCE\n  {}  [{g:.0} games/sec]", stats.summary());
    }

    // 1-ply search (~21x slower, so fewer games).
    {
        let t0 = Instant::now();
        let mut a = SearchEngine::new(&nn, 1, "NN-1ply");
        let mut b = EvalEngine::new(HceEval::new(), "HCE");
        let stats = run_match(&mut a, &mut b, 100, 6789);
        let g = 200.0 / t0.elapsed().as_secs_f64();
        println!("NN(1-ply) vs HCE\n  {}  [{g:.0} games/sec]", stats.summary());
    }
    {
        // The head-to-head that proves the search helps: 1-ply vs 0-ply, same net.
        let t0 = Instant::now();
        let mut a = SearchEngine::new(&nn, 1, "NN-1ply");
        let mut b = EvalEngine::new(&nn, "NN-0ply");
        let stats = run_match(&mut a, &mut b, 100, 424242);
        let g = 200.0 / t0.elapsed().as_secs_f64();
        println!("NN(1-ply) vs NN(0-ply)\n  {}  [{g:.1} games/sec]", stats.summary());
    }
    {
        // Candidate-pruned 2-ply vs 1-ply (same net). Fewer games — 2-ply is slower.
        let t0 = Instant::now();
        let mut a = SearchEngine::with_candidates(&nn, 2, 6, "NN-2ply");
        let mut b = SearchEngine::new(&nn, 1, "NN-1ply");
        let stats = run_match(&mut a, &mut b, 30, 55555);
        let g = 60.0 / t0.elapsed().as_secs_f64();
        println!("NN(2-ply) vs NN(1-ply)\n  {}  [{g:.1} games/sec]", stats.summary());
    }
    {
        // Parallel Monte-Carlo rollouts vs 1-ply. Rollouts are heavy — few games.
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let cfg = RolloutConfig { trials: 80, truncate_plies: 7, candidates: 3, seed: 0x5EED };
        let t0 = Instant::now();
        let mut a = RolloutEngine::new(&nn, cfg, "NN-rollout");
        let mut b = SearchEngine::new(&nn, 1, "NN-1ply");
        let stats = run_match(&mut a, &mut b, 4, 999);
        let g = 8.0 / t0.elapsed().as_secs_f64();
        println!(
            "NN(rollout, 80x7) vs NN(1-ply)  [{threads} threads]\n  {}  [{g:.2} games/sec]",
            stats.summary()
        );
    }
}
