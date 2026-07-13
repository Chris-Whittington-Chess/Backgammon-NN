//! Engine-vs-engine match runner (SPEC §13). Runs mirrored-dice matches between
//! the built-in evaluators and prints the results.
//!
//! Run with:  cargo run --release --example match

use bgcore::eval::{HceEval, RandomEval};
use bgcore::{run_match, EvalEngine};

fn main() {
    let pairs = 500; // 1000 games per match

    println!("Backgammon match runner — mirrored dice, {} games each\n", pairs * 2);

    // HCE vs Random: the M2 milestone check.
    {
        let mut a = EvalEngine::new(HceEval::new(), "HCE");
        let mut b = EvalEngine::new(RandomEval::new(0xABCD), "Random");
        let stats = run_match(&mut a, &mut b, pairs, 12345);
        println!("HCE vs Random");
        println!("  {}", stats.summary());
    }

    // Random vs Random: sanity — should hover around 50% within the CI.
    {
        let mut a = EvalEngine::new(RandomEval::new(1), "RandomA");
        let mut b = EvalEngine::new(RandomEval::new(2), "RandomB");
        let stats = run_match(&mut a, &mut b, pairs, 6789);
        println!("\nRandom vs Random (sanity)");
        println!("  {}", stats.summary());
    }
}
