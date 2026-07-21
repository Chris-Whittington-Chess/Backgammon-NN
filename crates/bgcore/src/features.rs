//! Neural-network input encoding (SPEC §6).
//!
//! The classic Tesauro / TD-Gammon 198-input representation, computed directly
//! from a mover-relative [`Board`]. This is the **single source of truth** for
//! feature encoding: the Python trainer calls it through the PyO3 bindings so
//! training and inference can never disagree.
//!
//! Layout (198 floats):
//! - `[0..96)`   — mover's checkers: 4 units per point (points 1..=24)
//! - `[96..192)` — opponent's checkers: 4 units per point
//! - `192,193`   — mover / opponent checkers on the bar, each ÷ 2
//! - `194,195`   — mover / opponent checkers borne off, each ÷ 15
//! - `196,197`   — side to move (one-hot). Always `(1,0)` in this mover-relative
//!   frame; kept for architectural fidelity to the classic 198-input net.
//!
//! The 4-unit per-point code for `n` checkers is
//! `(n≥1, n≥2, n≥3, max(n-3,0)/2)`.

use crate::board::{Board, MOVER, NUM_POINTS, OPP};

/// Number of network input features.
pub const NUM_INPUTS: usize = 198;

/// Encode a position into a fresh `[f32; 198]`.
pub fn encode(board: &Board) -> [f32; NUM_INPUTS] {
    let mut out = [0.0f32; NUM_INPUTS];
    encode_into(board, &mut out);
    out
}

/// Encode a position into `out` (first [`NUM_INPUTS`] elements). Useful for
/// filling a preallocated batch buffer without per-position allocation.
pub fn encode_into(board: &Board, out: &mut [f32]) {
    assert!(
        out.len() >= NUM_INPUTS,
        "output buffer too small: {} < {NUM_INPUTS}",
        out.len()
    );
    for v in out[..NUM_INPUTS].iter_mut() {
        *v = 0.0;
    }

    for p in 1..=NUM_POINTS {
        let c = board.point(p);
        let base = (p - 1) * 4;
        write_point(&mut out[base..base + 4], c.max(0) as u32);
        write_point(&mut out[96 + base..96 + base + 4], (-c).max(0) as u32);
    }

    out[192] = board.bar(MOVER) as f32 / 2.0;
    out[193] = board.bar(OPP) as f32 / 2.0;
    out[194] = board.off(MOVER) as f32 / 15.0;
    out[195] = board.off(OPP) as f32 / 15.0;
    out[196] = 1.0; // mover is, by construction, the side to move
    out[197] = 0.0;
}

/// Write the 4-unit code for `n` checkers into `slot` (length 4).
#[inline]
fn write_point(slot: &mut [f32], n: u32) {
    if n >= 1 {
        slot[0] = 1.0;
    }
    if n >= 2 {
        slot[1] = 1.0;
    }
    if n >= 3 {
        slot[2] = 1.0;
    }
    if n > 3 {
        slot[3] = (n - 3) as f32 / 2.0;
    }
}

// --- Strategic add-on features (task #7: features on clean labels) -------------
//
// 14 hand-crafted signals (7 per side) the raw 198 encoding forces the net to
// infer from checker positions. Kept as a SEPARATE block — the base encoding
// stays 198 — so a 212-input candidate and the 198-input champion can coexist in
// one process (Python concatenates `features()` + `strategic()` for the
// candidate; the champion just uses `features()`). Ported from the richer-features
// branch, where these HURT under noisy TD self-play; this tests them on clean
// (low-variance) rollout/distillation labels instead.

/// Strategic features per side.
pub const STRAT_PER_SIDE: usize = 7;
/// Total strategic add-on features (mover's block, then opponent's).
pub const STRAT_INPUTS: usize = 2 * STRAT_PER_SIDE;

/// The 14 strategic add-on features: the mover's 7, then the opponent's 7 (the
/// same block computed on the turn-passed board).
pub fn strategic(board: &Board) -> [f32; STRAT_INPUTS] {
    let mut out = [0.0f32; STRAT_INPUTS];
    strategic_block(board, &mut out[..STRAT_PER_SIDE]);
    let swapped = board.swap_perspective();
    strategic_block(&swapped, &mut out[STRAT_PER_SIDE..]);
    out
}

/// The 7 strategic features for the **mover** of `b` (`out` length 7):
/// 0 blot exposure (opponent rolls of 36 that hit a mover blot, ÷36); 1 blot
/// count ÷6; 2 home-board points made ÷6; 3 total made points ÷8; 4 rearmost
/// pip (bar = 25) ÷25; 5 checkers trapped in the opponent home + bar ÷15;
/// 6 pip count ÷167.
fn strategic_block(b: &Board, out: &mut [f32]) {
    let mut home_made = 0i32;
    let mut made = 0i32;
    let mut blots = 0i32;
    let mut back_trapped = b.bar(MOVER) as i32;
    for p in 1..=NUM_POINTS {
        let c = b.point(p);
        if c >= 2 {
            made += 1;
            if p <= 6 {
                home_made += 1;
            }
        } else if c == 1 {
            blots += 1;
        }
        if (19..=24).contains(&p) && c > 0 {
            back_trapped += c as i32;
        }
    }
    let rearmost = if b.bar(MOVER) > 0 { 25 } else { b.highest_mover_point() as i32 };

    out[0] = mover_blot_shots(b);
    out[1] = blots as f32 / 6.0;
    out[2] = home_made as f32 / 6.0;
    out[3] = made as f32 / 8.0;
    out[4] = rearmost as f32 / 25.0;
    out[5] = back_trapped as f32 / 15.0;
    out[6] = b.pip_count(MOVER) as f32 / 167.0;
}

/// Fraction of the 36 dice rolls that let the **opponent** hit at least one mover
/// blot (direct + two-dice combos through a landable intermediate + doubles),
/// unioned over every (blot, opponent-checker) pair.
fn mover_blot_shots(b: &Board) -> f32 {
    let mut hit = [[false; 6]; 6];
    let landable = |idx: i32| idx <= NUM_POINTS as i32 && b.point(idx as usize) < 2;

    for bi in 1..=NUM_POINTS {
        if b.point(bi) != 1 {
            continue; // only a mover blot can be hit
        }
        for p in 1..bi {
            if b.point(p) >= 0 {
                continue; // need an opponent checker behind the blot
            }
            let dist = (bi - p) as i32;

            if (1..=6).contains(&dist) {
                let d = (dist - 1) as usize;
                for k in 0..6 {
                    hit[d][k] = true;
                    hit[k][d] = true;
                }
            }

            if (2..=12).contains(&dist) {
                let lo = 1.max(dist - 6);
                let hi = 6.min(dist - 1);
                for d1 in lo..=hi {
                    let d2 = dist - d1;
                    let p = p as i32;
                    if landable(p + d1) || landable(p + d2) {
                        hit[(d1 - 1) as usize][(d2 - 1) as usize] = true;
                        hit[(d2 - 1) as usize][(d1 - 1) as usize] = true;
                    }
                }
            }

            for d in 1..=6i32 {
                for k in 2..=4i32 {
                    if k * d != dist {
                        continue;
                    }
                    let mut ok = true;
                    for j in 1..k {
                        if !landable(p as i32 + j * d) {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        hit[(d - 1) as usize][(d - 1) as usize] = true;
                    }
                }
            }
        }
    }

    let count: usize = hit.iter().flatten().filter(|&&h| h).count();
    count as f32 / 36.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_has_correct_dimension() {
        let v = encode(&Board::starting_position());
        assert_eq!(v.len(), 198);
    }

    #[test]
    fn starting_position_key_slots() {
        let v = encode(&Board::starting_position());
        // Mover: point 6 has 5 checkers -> (1,1,1,(5-3)/2=1.0).
        assert_eq!(&v[20..24], &[1.0, 1.0, 1.0, 1.0]);
        // Mover: point 8 has 3 -> (1,1,1,0).
        assert_eq!(&v[28..32], &[1.0, 1.0, 1.0, 0.0]);
        // Mover: point 24 has 2 -> (1,1,0,0).
        assert_eq!(&v[92..96], &[1.0, 1.0, 0.0, 0.0]);
        // Opponent: point 12 has 5 -> (1,1,1,1.0) at 96 + (12-1)*4 = 140.
        assert_eq!(&v[140..144], &[1.0, 1.0, 1.0, 1.0]);
        // Bar and off all zero at the start; turn one-hot.
        assert_eq!(&v[192..198], &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn bar_and_off_are_scaled() {
        let mut b = Board::empty();
        b.set_bar(MOVER, 2);
        b.set_point(1, 13); // 13 + 2 on bar = 15
        b.set_off(OPP, 15); // opponent all off
        let v = encode(&b);
        assert_eq!(v[192], 1.0); // 2 / 2
        assert_eq!(v[195], 1.0); // 15 / 15
    }

    #[test]
    fn encode_into_matches_encode() {
        let b = Board::starting_position();
        let mut buf = vec![9.0f32; NUM_INPUTS + 5]; // oversized, pre-dirtied
        encode_into(&b, &mut buf);
        assert_eq!(&buf[..NUM_INPUTS], &encode(&b)[..]);
    }
}
