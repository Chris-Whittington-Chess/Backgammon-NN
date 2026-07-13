//! GnuBG-compatible Position ID (SPEC §3).
//!
//! A 10-byte key packed into a 14-character base64 string, interoperable with
//! GNU Backgammon and wildbg. The mover ("player on roll") is encoded as GnuBG's
//! "player on roll", so the starting position is `"4HPwATDgc/ABMA"`.
//!
//! Format: for the opponent, then the mover, walk the points appending one `1`
//! bit per checker on each point followed by a `0` separator; the opponent is
//! scanned from point 24 down to its bar, the mover from point 1 up to its bar.
//! Off checkers are implied (15 minus those on the board), so the ID is a
//! complete, canonical key for a position.

use crate::board::{Board, CHECKERS_PER_SIDE, MOVER, NUM_POINTS, OPP};

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

impl Board {
    /// The 14-character GnuBG Position ID for this position.
    pub fn position_id(&self) -> String {
        base64_encode(&self.position_key())
    }

    /// Parse a GnuBG Position ID back into a board. Returns `Err` on malformed
    /// input or an illegal (non-15-checker) position.
    pub fn from_position_id(id: &str) -> Result<Board, String> {
        let bytes = base64_decode(id).ok_or_else(|| format!("invalid base64: {id:?}"))?;
        if bytes.len() < 10 {
            return Err(format!("position id too short: {id:?}"));
        }
        let mut key = [0u8; 10];
        key.copy_from_slice(&bytes[..10]);
        Board::from_position_key(&key)
    }

    /// Pack the position into the 10-byte GnuBG key.
    fn position_key(&self) -> [u8; 10] {
        let mut key = [0u8; 10];
        let mut bit = 0usize;
        let push = |ones: u32, key: &mut [u8; 10], bit: &mut usize| {
            for _ in 0..ones {
                key[*bit / 8] |= 1 << (*bit % 8);
                *bit += 1;
            }
            *bit += 1; // trailing 0 separator
        };

        // Opponent (not on roll): points 24..=1, then the bar.
        for p in (1..=NUM_POINTS).rev() {
            push((-self.point(p)).max(0) as u32, &mut key, &mut bit);
        }
        push(self.bar(OPP) as u32, &mut key, &mut bit);

        // Mover (on roll): points 1..=24, then the bar.
        for p in 1..=NUM_POINTS {
            push(self.point(p).max(0) as u32, &mut key, &mut bit);
        }
        push(self.bar(MOVER) as u32, &mut key, &mut bit);

        key
    }

    /// Reconstruct a board from the 10-byte GnuBG key.
    fn from_position_key(key: &[u8; 10]) -> Result<Board, String> {
        let mut bit = 0usize;
        let read = |key: &[u8; 10], bit: &mut usize| -> u8 {
            let mut n = 0u8;
            while (key[*bit / 8] >> (*bit % 8)) & 1 == 1 {
                n += 1;
                *bit += 1;
            }
            *bit += 1; // skip the 0 separator
            n
        };

        let mut b = Board::empty();
        // Opponent: points 24..=1, then bar.
        for p in (1..=NUM_POINTS).rev() {
            let n = read(key, &mut bit);
            if n > 0 {
                b.set_point(p, -(n as i8));
            }
        }
        b.set_bar(OPP, read(key, &mut bit));
        // Mover: points 1..=24, then bar.
        for p in 1..=NUM_POINTS {
            let n = read(key, &mut bit);
            if n > 0 {
                b.set_point(p, n as i8);
            }
        }
        b.set_bar(MOVER, read(key, &mut bit));

        // Off counts are whatever is missing from 15 on each side.
        let on = |side: usize| b.checker_total(side); // off is still 0 here
        let mover_off = CHECKERS_PER_SIDE - on(MOVER);
        let opp_off = CHECKERS_PER_SIDE - on(OPP);
        if mover_off < 0 || opp_off < 0 {
            return Err("decoded more than 15 checkers for a side".into());
        }
        b.set_off(MOVER, mover_off as u8);
        b.set_off(OPP, opp_off as u8);

        b.validate().map_err(|e| format!("decoded illegal position: {e}"))?;
        Ok(b)
    }
}

fn base64_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        let out_chars = chunk.len() + 1; // 3->4, 2->3, 1->2 (no padding)
        let idx = [(n >> 18) & 63, (n >> 12) & 63, (n >> 6) & 63, n & 63];
        for &i in idx.iter().take(out_chars) {
            s.push(B64[i as usize] as char);
        }
    }
    s
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let mut vals = [0u32; 4];
        for (i, &c) in chunk.iter().enumerate() {
            vals[i] = b64_val(c)? as u32;
        }
        let n = (vals[0] << 18) | (vals[1] << 12) | (vals[2] << 6) | vals[3];
        let nbytes = chunk.len().checked_sub(1)?; // 4->3, 3->2, 2->1
        if nbytes >= 1 {
            out.push(((n >> 16) & 0xFF) as u8);
        }
        if nbytes >= 2 {
            out.push(((n >> 8) & 0xFF) as u8);
        }
        if nbytes >= 3 {
            out.push((n & 0xFF) as u8);
        }
    }
    Some(out)
}

fn b64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_has_canonical_id() {
        // The well-known GnuBG starting Position ID.
        assert_eq!(Board::starting_position().position_id(), "4HPwATDgc/ABMA");
    }

    #[test]
    fn round_trips_starting_position() {
        let b = Board::starting_position();
        let id = b.position_id();
        assert_eq!(Board::from_position_id(&id).unwrap(), b);
    }

    #[test]
    fn round_trips_positions_with_bar_and_off() {
        let mut b = Board::empty();
        b.set_bar(MOVER, 1);
        b.set_point(3, 5);
        b.set_point(6, 9); // 1 + 5 + 9 = 15
        b.set_point(20, -2);
        b.set_off(OPP, 13); // 2 + 13 = 15
        assert!(b.validate().is_ok());
        let id = b.position_id();
        assert_eq!(Board::from_position_id(&id).unwrap(), b);
    }

    #[test]
    fn decodes_known_wildbg_ids() {
        // IDs taken from wildbg's own round-trip tests; must parse and re-encode.
        for id in ["4HPwATDgc/ABMA", "jGfkASjg8wcBMA", "zGbiIQgxH/AAWA", "zGbiIYCYD3gALA"] {
            let b = Board::from_position_id(id).expect("valid id");
            assert_eq!(b.position_id(), id);
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(Board::from_position_id("!!!!not base64").is_err());
    }
}
