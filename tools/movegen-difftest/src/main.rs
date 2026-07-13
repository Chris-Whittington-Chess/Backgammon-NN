//! Differential move-generation test: our `bgcore::genmoves` vs the wildbg
//! reference engine (SPEC §13, milestone M1.5).
//!
//! For a large sample of realistic positions (drawn from random self-play) and
//! all 21 distinct dice, we generate the set of legal resulting positions with
//! both engines and compare them. Positions are compared via their **GnuBG
//! Position ID**, a canonical key both engines agree on.
//!
//! Our board and wildbg's `Position` share the same orientation (our "mover" is
//! wildbg's "x", points `1..=24` line up, and the bars/off map directly), so the
//! conversion is a straight copy.
//!
//! Run:  cargo run --release            (default 20k positions)
//!       cargo run --release -- 100000   (bigger sweep)

use bgcore::board::{MOVER, NUM_POINTS, OPP};
use bgcore::{genmoves, Board, Dice, Rng};
use engine::dice::Dice as WDice;
use engine::position::{Position as WPos, O_BAR, X_BAR};
use std::collections::HashSet;

/// Convert our mover-relative `Board` into a wildbg `Position`.
fn to_wildbg(b: &Board) -> WPos {
    let mut pips = [0i8; 26];
    for p in 1..=NUM_POINTS {
        pips[p] = b.point(p);
    }
    pips[X_BAR] = b.bar(MOVER) as i8; // index 25 = x (mover) bar, positive
    pips[O_BAR] = -(b.bar(OPP) as i8); // index 0 = o (opponent) bar, negative
    WPos::try_from(pips).expect("sampled board should be a legal wildbg position")
}

/// Canonical set of resulting positions from our engine.
///
/// wildbg's `all_positions_after_moving` returns positions with the sides
/// already switched (opponent to move), so we swap our results to match before
/// taking the Position ID.
fn ours(b: &Board, d1: u8, d2: u8) -> HashSet<String> {
    genmoves(b, &Dice::new(d1, d2))
        .iter()
        .map(|m| to_wildbg(&m.result.swap_perspective()).position_id())
        .collect()
}

/// Canonical set of resulting positions from the wildbg reference engine.
fn theirs(b: &Board, d1: u8, d2: u8) -> HashSet<String> {
    to_wildbg(b)
        .all_positions_after_moving(&WDice::new(d1 as usize, d2 as usize))
        .iter()
        .map(|p| p.position_id())
        .collect()
}

/// Draw `n` legal, non-terminal positions (mover to move) from random self-play.
fn sample_positions(n: usize, seed: u64) -> Vec<Board> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut board = Board::starting_position();
        for _ in 0..1000 {
            out.push(board.clone());
            let moves = genmoves(&board, &rng.roll());
            let idx = (rng.next_u64() % moves.len() as u64) as usize;
            board = moves[idx].result.clone();
            if board.has_won(MOVER) {
                break; // game over; start a fresh game
            }
            board = board.swap_perspective();
            if out.len() >= n {
                break;
            }
        }
    }
    out.truncate(n);
    out
}

/// Compare both engines over every position×dice; returns (checks, mismatches).
fn run_sweep(positions: &[Board], max_reports: usize) -> (usize, usize) {
    let mut checks = 0usize;
    let mut mismatches = 0usize;
    for b in positions {
        for d1 in 1..=6u8 {
            for d2 in d1..=6u8 {
                checks += 1;
                let (o, t) = (ours(b, d1, d2), theirs(b, d1, d2));
                if o != t {
                    mismatches += 1;
                    if mismatches <= max_reports {
                        eprintln!(
                            "MISMATCH  pos={}  dice={}-{}",
                            to_wildbg(b).position_id(),
                            d1,
                            d2
                        );
                        let mut only_ours: Vec<_> = o.difference(&t).collect();
                        let mut only_theirs: Vec<_> = t.difference(&o).collect();
                        only_ours.sort();
                        only_theirs.sort();
                        eprintln!("  only in bgcore: {:?}", only_ours);
                        eprintln!("  only in wildbg: {:?}", only_theirs);
                    }
                }
            }
        }
    }
    (checks, mismatches)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);

    println!("Sampling {n} positions from random self-play...");
    let positions = sample_positions(n, 0xC0FFEE_1234);

    println!("Comparing bgcore vs wildbg over all 21 dice per position...");
    let (checks, mismatches) = run_sweep(&positions, 10);

    println!("\nChecked {checks} (position, dice) pairs across {} positions.", positions.len());
    if mismatches == 0 {
        println!("ALL MATCH  bgcore::genmoves == wildbg reference engine");
    } else {
        eprintln!("{mismatches} MISMATCHES");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_matches_for_all_dice() {
        let b = Board::starting_position();
        // Sanity: our Position ID must equal wildbg's canonical starting ID.
        assert_eq!(to_wildbg(&b).position_id(), "4HPwATDgc/ABMA");
        for d1 in 1..=6u8 {
            for d2 in d1..=6u8 {
                assert_eq!(ours(&b, d1, d2), theirs(&b, d1, d2), "dice {d1}-{d2}");
            }
        }
    }

    #[test]
    fn differential_sweep_small() {
        let positions = sample_positions(3_000, 0xABCDEF);
        let (checks, mismatches) = run_sweep(&positions, 5);
        assert!(checks > 0);
        assert_eq!(mismatches, 0, "{mismatches} mismatches over {checks} checks");
    }
}
