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
