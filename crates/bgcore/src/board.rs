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

// --- Class-aware output-head routing (SPEC: the class-bucketed value net) -----
// Positions are split by gnubg's classes — race / crashed / contact — then into
// total-pip sub-buckets calibrated for even population (trainer/measure_classes.py).
// [`Board::route_bucket`] maps a position to one of `N_ROUTE_HEADS` heads; the net
// emits a 6-outcome softmax per head and the engine slices the selected one. Head
// order: race `[0..3)`, crashed `[3..5)`, contact `[5..12)`. Keep these edges and
// the layout identical to `route_bucket` / `N_HEADS` in `trainer/model.py`.
/// Number of output heads for the class-aware bucketed net.
pub const N_ROUTE_HEADS: usize = 12;
/// Within-race total-pip edges (3 race heads).
const RACE_EDGES: [i32; 2] = [57, 96];
/// Within-crashed total-pip edge (2 crashed heads).
const CRASHED_EDGES: [i32; 1] = [127];
/// Within-contact total-pip edges (7 contact heads).
const CONTACT_EDGES: [i32; 6] = [168, 202, 232, 259, 284, 309];
const CRASHED_BASE: usize = RACE_EDGES.len() + 1; // 3
const CONTACT_BASE: usize = CRASHED_BASE + CRASHED_EDGES.len() + 1; // 5

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

    /// True once the two sides have passed each other and can no longer
    /// interact — a pure race. The mover runs high→low (24→off), the opponent
    /// low→high, so contact is impossible exactly when the mover's rearmost
    /// checker sits below the opponent's rearmost. Checkers on the bar always
    /// mean contact.
    pub fn no_contact(&self) -> bool {
        if self.bar[MOVER] > 0 || self.bar[OPP] > 0 {
            return false;
        }
        let mover_max = (1..=NUM_POINTS).rev().find(|&p| self.points[p] > 0);
        let opp_min = (1..=NUM_POINTS).find(|&p| self.points[p] < 0);
        match (mover_max, opp_min) {
            (Some(m), Some(o)) => m < o,
            _ => true, // a side already off the board can't be contacted
        }
    }

    /// gnubg's "crashed" classification (either side): a side with at most 6
    /// checkers not buried on its own 1- and 2-points. Verbatim port of gnubg's
    /// `ClassifyPosition` test (`N = 6`). Non-cyclic — every successor of a crashed
    /// position is also crashed. This is meaningful only for contact positions; a
    /// pure race is never "crashed" (see [`Board::route_bucket`], which gates it on
    /// contact). Perspective-invariant, since it checks *both* sides.
    pub fn crashed(&self) -> bool {
        const N: i32 = 6;
        for side in [MOVER, OPP] {
            // Checkers still on the board (not borne off), including the bar.
            let tot = CHECKERS_PER_SIDE - self.off[side] as i32;
            // That side's ace(1)- and two(2)-point counts. The opponent is stored
            // negated in the mover's frame, so their ace/2 are the high points 24/23.
            let (ace, two) = if side == MOVER {
                (self.points[1].max(0) as i32, self.points[2].max(0) as i32)
            } else {
                ((-self.points[24]).max(0) as i32, (-self.points[23]).max(0) as i32)
            };
            let side_crashed = if tot <= N {
                true
            } else if ace > 1 {
                tot <= N + ace || (two > 1 && (1 + tot - ace - two) <= N)
            } else {
                tot <= N + (two - 1)
            };
            if side_crashed {
                return true;
            }
        }
        false
    }

    /// The output head for this position under class-aware routing
    /// (`0..N_ROUTE_HEADS`). Race first (armies passed), else crashed, else
    /// contact; within each class, a total-pip sub-bucket. Total pips are
    /// perspective-invariant, so a board and its swap share a head and equity stays
    /// antisymmetric. Must match `route_bucket` in `trainer/model.py`.
    pub fn route_bucket(&self) -> usize {
        let total = self.pip_count(MOVER) + self.pip_count(OPP);
        let sub = |edges: &[i32]| edges.iter().filter(|&&e| total >= e).count();
        if self.no_contact() {
            sub(&RACE_EDGES) // 0..3
        } else if self.crashed() {
            CRASHED_BASE + sub(&CRASHED_EDGES) // 3..5
        } else {
            CONTACT_BASE + sub(&CONTACT_EDGES) // 5..12
        }
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
    fn starting_position_routes_to_top_contact_head() {
        // Opening: 334 total pips, in contact, not crashed -> top contact head (11).
        let b = Board::starting_position();
        assert!(!b.crashed());
        assert!(!b.no_contact());
        assert_eq!(b.route_bucket(), N_ROUTE_HEADS - 1);
    }

    #[test]
    fn crashed_position_routes_to_crashed_head() {
        // Mover has borne off 9 (6 left on the 6-point => tot=6 <= N, crashed), and
        // an opponent checker sits deep enough to keep contact.
        let mut b = Board::empty();
        b.set_point(6, 6); // mover: 6 checkers on the 6-point
        b.set_off(MOVER, 9);
        b.set_point(3, -2); // opponent back checker: opp_min(3) <= mover_max(6) => contact
        b.set_off(OPP, 13);
        assert!(b.crashed(), "tot=6 should be crashed");
        assert!(!b.no_contact(), "should still be in contact");
        let h = b.route_bucket();
        assert!((CRASHED_BASE..CONTACT_BASE).contains(&h), "crashed head range, got {h}");
    }

    #[test]
    fn raced_position_routes_to_race_head() {
        // Armies passed (mover rearmost below opponent rearmost) -> race, low heads.
        let mut b = Board::empty();
        b.set_point(2, 2); // mover_max = 2
        b.set_point(20, -2); // opp_min = 20, so 2 < 20 => no contact
        assert!(b.no_contact());
        assert!(b.route_bucket() < CRASHED_BASE, "race heads are [0, {CRASHED_BASE})");
    }

    #[test]
    fn pip_count_counts_bar_as_25() {
        let mut b = Board::empty();
        b.set_bar(MOVER, 1);
        b.set_point(1, 14); // 14 checkers on the ace point
        assert_eq!(b.checker_total(MOVER), CHECKERS_PER_SIDE);
        assert_eq!(b.pip_count(MOVER), 14 + BAR_PIPS);
    }

    #[test]
    fn no_contact_detects_races() {
        // Starting position: fully engaged.
        assert!(!Board::starting_position().no_contact());

        // Mover all on point 5, opponent all on point 8 — mover's rearmost (5)
        // is below the opponent's rearmost (8), so they have passed: a race.
        let mut race = Board::empty();
        race.set_point(5, 15);
        race.set_point(8, -15);
        assert!(race.no_contact());

        // Overlap: mover has a straggler on 20, above the opponent on 8.
        let mut touching = Board::empty();
        touching.set_point(5, 14);
        touching.set_point(20, 1);
        touching.set_point(8, -15);
        assert!(!touching.no_contact());

        // A checker on the bar is always contact.
        let mut barred = race.clone();
        barred.set_bar(MOVER, 1);
        barred.set_point(5, 14);
        assert!(!barred.no_contact());
    }
}
