//! Board representation and invariants (SPEC §3).
//!
//! The board is stored **mover-relative**: points `1..=24` are from the side to
//! move's view (1 = mover's ace point, 24 = mover's furthest-back point).
//! `points[i] > 0` are the mover's checkers on point `i`; `points[i] < 0` are
//! the opponent's. Index `0` is unused padding so points are 1-based.

/// Number of playable points on the board.
pub const NUM_POINTS: usize = 24;
/// Checkers each side owns (always conserved).
pub const CHECKERS_PER_SIDE: i32 = 15;
/// Pip distance assigned to a checker sitting on the bar.
pub const BAR_PIPS: i32 = 25;

/// Index into the two-element `bar` / `off` arrays.
pub const MOVER: usize = 0;
pub const OPP: usize = 1;

/// A backgammon position, stored from the side-to-move's perspective.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Board {
    /// `points[1..=24]`; `points[0]` unused. `+` = mover, `-` = opponent.
    points: [i8; NUM_POINTS + 1],
    /// `[mover_on_bar, opp_on_bar]`.
    bar: [u8; 2],
    /// `[mover_borne_off, opp_borne_off]`.
    off: [u8; 2],
}

impl Board {
    /// An empty board (no checkers anywhere). Not a legal game position on its
    /// own — used as a builder base and in tests.
    pub fn empty() -> Self {
        Board {
            points: [0; NUM_POINTS + 1],
            bar: [0, 0],
            off: [0, 0],
        }
    }

    /// The standard opening position, from the mover's perspective.
    ///
    /// Mover: 2 on 24, 5 on 13, 3 on 8, 5 on 6. The opponent's checkers are the
    /// mirror image (their point `p` is the mover's point `25 - p`).
    pub fn starting_position() -> Self {
        let mut b = Board::empty();
        // Mover (positive).
        b.points[24] = 2;
        b.points[13] = 5;
        b.points[8] = 3;
        b.points[6] = 5;
        // Opponent (negative), mirrored across 25 - p.
        b.points[1] = -2; // opp's 24
        b.points[12] = -5; // opp's 13
        b.points[17] = -3; // opp's 8
        b.points[19] = -5; // opp's 6
        debug_assert!(b.validate().is_ok());
        b
    }

    /// Checkers on point `p` (`1..=24`): positive = mover, negative = opponent.
    #[inline]
    pub fn point(&self, p: usize) -> i8 {
        self.points[p]
    }

    /// Set point `p` (`1..=24`) directly. Positive = mover, negative = opponent.
    /// Intended for constructing test/fixture positions.
    pub fn set_point(&mut self, p: usize, v: i8) {
        assert!((1..=NUM_POINTS).contains(&p), "point index out of range: {p}");
        self.points[p] = v;
    }

    /// Checkers of the given side (`MOVER`/`OPP`) sitting on the bar.
    #[inline]
    pub fn bar(&self, side: usize) -> u8 {
        self.bar[side]
    }

    /// Set the bar count for a side. For building fixtures.
    pub fn set_bar(&mut self, side: usize, n: u8) {
        self.bar[side] = n;
    }

    /// Checkers of the given side already borne off.
    #[inline]
    pub fn off(&self, side: usize) -> u8 {
        self.off[side]
    }

    /// Set the borne-off count for a side. For building fixtures.
    pub fn set_off(&mut self, side: usize, n: u8) {
        self.off[side] = n;
    }

    /// Total checkers accounted for by a side (points + bar + off). Should
    /// always equal [`CHECKERS_PER_SIDE`] for a valid position.
    pub fn checker_total(&self, side: usize) -> i32 {
        let on_points: i32 = self
            .points
            .iter()
            .map(|&c| {
                let c = c as i32;
                match side {
                    MOVER if c > 0 => c,
                    OPP if c < 0 => -c,
                    _ => 0,
                }
            })
            .sum();
        on_points + self.bar[side] as i32 + self.off[side] as i32
    }

    /// Validate structural invariants (SPEC §3):
    /// - each side has exactly 15 checkers (points + bar + off),
    /// - no point holds both colours (guaranteed by the `i8` encoding, but the
    ///   count check catches malformed fixtures).
    pub fn validate(&self) -> Result<(), String> {
        for &side in &[MOVER, OPP] {
            let total = self.checker_total(side);
            if total != CHECKERS_PER_SIDE {
                return Err(format!(
                    "side {side} has {total} checkers, expected {CHECKERS_PER_SIDE}"
                ));
            }
        }
        Ok(())
    }

    /// Return the same position viewed from the opponent's side, i.e. after the
    /// turn passes. Point `i` maps to `25 - i` with sign flipped; bar and off
    /// arrays swap. This is an involution: `b.swap_perspective().swap_perspective() == b`.
    #[must_use]
    pub fn swap_perspective(&self) -> Board {
        let mut points = [0i8; NUM_POINTS + 1];
        // Reverse-index into a second array; enumerate() doesn't apply cleanly.
        #[allow(clippy::needless_range_loop)]
        for i in 1..=NUM_POINTS {
            points[i] = -self.points[NUM_POINTS + 1 - i];
        }
        Board {
            points,
            bar: [self.bar[OPP], self.bar[MOVER]],
            off: [self.off[OPP], self.off[MOVER]],
        }
    }

    /// Pip count for a side: total distance all its checkers must travel to bear
    /// off. A checker on the bar counts as [`BAR_PIPS`].
    pub fn pip_count(&self, side: usize) -> i32 {
        let from_points: i32 = (1..=NUM_POINTS)
            .map(|p| {
                let c = self.points[p] as i32;
                match side {
                    // Mover on point p is p pips from bearing off.
                    MOVER if c > 0 => c * p as i32,
                    // Opponent on point p (mover-relative) is on their own point
                    // 25 - p, so that many pips from bearing off.
                    OPP if c < 0 => (-c) * (BAR_PIPS - p as i32),
                    _ => 0,
                }
            })
            .sum();
        from_points + self.bar[side] as i32 * BAR_PIPS
    }

    // --- Queries used by move generation (SPEC §4) ---

    /// True if all of the mover's checkers are in their home board (points
    /// `1..=6`) — the precondition for bearing off. Checkers on the bar count as
    /// outside the home board.
    pub fn mover_all_home(&self) -> bool {
        if self.bar[MOVER] > 0 {
            return false;
        }
        (7..=NUM_POINTS).all(|p| self.points[p] <= 0)
    }

    /// The highest-numbered point (`1..=24`) holding a mover checker, or `0` if
    /// the mover has none on the board. Used for the bear-off overflow rule.
    pub fn highest_mover_point(&self) -> usize {
        (1..=NUM_POINTS).rev().find(|&p| self.points[p] > 0).unwrap_or(0)
    }

    /// True if point `to` (`1..=24`) is blocked for the mover, i.e. held by two
    /// or more opponent checkers.
    #[inline]
    pub fn is_blocked_for_mover(&self, to: usize) -> bool {
        self.points[to] <= -2
    }

    // --- Low-level mechanics (SPEC §4). Callers must ensure legality. ---

    /// Move a mover checker from point `from` to point `to` (both `1..=24`),
    /// hitting an opponent blot on `to` if present (sending it to the bar).
    pub fn move_checker(&mut self, from: usize, to: usize) {
        self.points[from] -= 1;
        if self.points[to] == -1 {
            self.points[to] = 0;
            self.bar[OPP] += 1;
        }
        self.points[to] += 1;
    }

    /// Enter a mover checker from the bar onto point `to`, hitting a blot there.
    pub fn enter_from_bar(&mut self, to: usize) {
        self.bar[MOVER] -= 1;
        if self.points[to] == -1 {
            self.points[to] = 0;
            self.bar[OPP] += 1;
        }
        self.points[to] += 1;
    }

    /// Bear a mover checker off from point `from`.
    pub fn bear_off_checker(&mut self, from: usize) {
        self.points[from] -= 1;
        self.off[MOVER] += 1;
    }

    /// True once a side has borne off all 15 checkers.
    pub fn has_won(&self, side: usize) -> bool {
        self.off[side] as i32 == CHECKERS_PER_SIDE
    }

    /// True if either side has borne off all checkers.
    pub fn is_terminal(&self) -> bool {
        self.has_won(MOVER) || self.has_won(OPP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_position_is_valid() {
        let b = Board::starting_position();
        assert!(b.validate().is_ok(), "{:?}", b.validate());
        assert_eq!(b.checker_total(MOVER), CHECKERS_PER_SIDE);
        assert_eq!(b.checker_total(OPP), CHECKERS_PER_SIDE);
    }

    #[test]
    fn starting_pip_count_is_167_each() {
        // The well-known opening pip count for both sides.
        let b = Board::starting_position();
        assert_eq!(b.pip_count(MOVER), 167);
        assert_eq!(b.pip_count(OPP), 167);
    }

    #[test]
    fn swap_perspective_is_an_involution() {
        let b = Board::starting_position();
        assert_eq!(b.swap_perspective().swap_perspective(), b);
    }

    #[test]
    fn swap_perspective_swaps_pip_counts() {
        let b = Board::starting_position();
        let s = b.swap_perspective();
        assert_eq!(b.pip_count(MOVER), s.pip_count(OPP));
        assert_eq!(b.pip_count(OPP), s.pip_count(MOVER));
    }

    #[test]
    fn empty_board_fails_validation() {
        assert!(Board::empty().validate().is_err());
    }

    #[test]
    fn validate_rejects_wrong_checker_count() {
        let mut b = Board::starting_position();
        b.set_point(6, 4); // was 5 -> mover now has 14
        assert!(b.validate().is_err());
    }

    #[test]
    fn win_detection() {
        let mut b = Board::empty();
        b.set_off(MOVER, 15);
        assert!(b.has_won(MOVER));
        assert!(b.is_terminal());
        assert!(!b.has_won(OPP));
    }

    #[test]
    fn pip_count_counts_bar_as_25() {
        let mut b = Board::empty();
        b.set_bar(MOVER, 1);
        b.set_point(1, 14); // 14 checkers on the ace point
        assert_eq!(b.checker_total(MOVER), CHECKERS_PER_SIDE);
        assert_eq!(b.pip_count(MOVER), 14 + BAR_PIPS);
    }
}
