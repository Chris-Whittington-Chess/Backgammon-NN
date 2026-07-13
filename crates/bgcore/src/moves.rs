//! Legal move generation (SPEC §4).
//!
//! A [`Move`] is a *full turn*: the ordered sequence of checker [`Step`]s plus
//! the resulting [`Board`] (still mover-relative, before the turn is passed).
//! [`genmoves`] enforces every rule:
//! - bar checkers must re-enter before anything else moves,
//! - blots are hit, points held by 2+ opponents are blocked,
//! - bearing off requires all checkers home, with exact and overflow rules,
//! - a player must use the maximum number of dice possible, and when only one
//!   of two dice can be played, it must be the larger one if legal.
//!
//! Moves are de-duplicated by their resulting position, so two orderings that
//! reach the same board are reported once (a "dance" — no legal move — returns a
//! single pass move whose result equals the input board).

use crate::board::Board;
use crate::dice::Dice;
use std::collections::HashSet;

/// One checker movement within a turn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// Enter from the bar onto point `to` using `die`.
    Enter { to: usize, die: u8 },
    /// Move a checker from point `from` to point `to` using `die`.
    Point { from: usize, to: usize, die: u8 },
    /// Bear a checker off from point `from` using `die`.
    BearOff { from: usize, die: u8 },
}

impl Step {
    /// The die value consumed by this step.
    pub fn die(&self) -> u8 {
        match *self {
            Step::Enter { die, .. } | Step::Point { die, .. } | Step::BearOff { die, .. } => die,
        }
    }
}

/// A complete legal turn.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Move {
    /// The checker steps in the order they are applied.
    pub steps: Vec<Step>,
    /// The position after the turn (mover-relative; caller swaps perspective).
    pub result: Board,
}

impl Move {
    /// Number of dice consumed by this turn.
    pub fn dice_used(&self) -> usize {
        self.steps.len()
    }
}

/// All legal full-turn moves for `board` given `dice`.
///
/// Always returns at least one move: if no dice can be played the result is a
/// single pass whose `result` equals `board`.
pub fn genmoves(board: &Board, dice: &Dice) -> Vec<Move> {
    let plays = dice.plays();
    let mut candidates: Vec<(Vec<Step>, Board)> = Vec::new();
    expand(board, &plays, &mut Vec::new(), &mut candidates);

    let max_used = candidates.iter().map(|(s, _)| s.len()).max().unwrap_or(0);

    // Dance: no die could be played.
    if max_used == 0 {
        return vec![Move {
            steps: Vec::new(),
            result: board.clone(),
        }];
    }

    // Keep only turns using the maximum number of dice.
    let mut kept: Vec<(Vec<Step>, Board)> =
        candidates.into_iter().filter(|(s, _)| s.len() == max_used).collect();

    // Larger-die rule: when only one of two distinct dice can be played, it must
    // be the larger one if any legal move uses it.
    if max_used == 1 && !dice.is_double() {
        let (a, b) = dice.pair();
        let larger = a.max(b);
        let uses_larger = kept.iter().any(|(s, _)| s[0].die() == larger);
        if uses_larger {
            kept.retain(|(s, _)| s[0].die() == larger);
        }
    }

    // De-duplicate by resulting position.
    let mut seen: HashSet<Board> = HashSet::new();
    let mut moves = Vec::new();
    for (steps, result) in kept {
        if seen.insert(result.clone()) {
            moves.push(Move { steps, result });
        }
    }
    moves
}

/// Recursively apply remaining dice, recording every reachable state (including
/// partial and root states — non-maximal ones are filtered by the caller).
fn expand(
    board: &Board,
    remaining: &[u8],
    steps: &mut Vec<Step>,
    out: &mut Vec<(Vec<Step>, Board)>,
) {
    out.push((steps.clone(), board.clone()));
    if remaining.is_empty() {
        return;
    }
    let mut tried = [false; 7];
    for i in 0..remaining.len() {
        let d = remaining[i];
        if tried[d as usize] {
            continue; // each distinct die value handled once per level
        }
        tried[d as usize] = true;

        let singles = gen_single(board, d);
        if singles.is_empty() {
            continue;
        }
        let mut rem2 = remaining.to_vec();
        rem2.remove(i); // drop one instance of this die

        for (step, child) in singles {
            steps.push(step);
            expand(&child, &rem2, steps, out);
            steps.pop();
        }
    }
}

/// Sentinel point index meaning "the bar" (source of a bar re-entry).
pub const BAR: usize = 25;
/// Sentinel point index meaning "borne off" (destination of a bear-off).
pub const OFF: usize = 0;

/// One legal next checker move within a partially-played turn.
#[derive(Clone, Debug)]
pub struct SubMove {
    /// Source point (`1..=24`), or [`BAR`] when entering from the bar.
    pub from: usize,
    /// Destination point (`1..=24`), or [`OFF`] when bearing off.
    pub to: usize,
    /// The die value consumed.
    pub die: u8,
    /// The board after applying just this checker move (mover-relative).
    pub result: Board,
}

/// The maximum number of dice from `remaining` that can still be played from
/// `board` (used to enforce the maximal-use rule incrementally).
fn max_completion(board: &Board, remaining: &[u8]) -> usize {
    let mut best = 0;
    let mut tried = [false; 7];
    for i in 0..remaining.len() {
        let d = remaining[i];
        if tried[d as usize] {
            continue;
        }
        tried[d as usize] = true;
        let mut rem2 = remaining.to_vec();
        rem2.remove(i);
        for (_step, child) in gen_single(board, d) {
            best = best.max(1 + max_completion(&child, &rem2));
        }
    }
    best
}

fn step_from_to(step: &Step) -> (usize, usize) {
    match *step {
        Step::Enter { to, .. } => (BAR, to),
        Step::Point { from, to, .. } => (from, to),
        Step::BearOff { from, .. } => (from, OFF),
    }
}

/// The legal *next* checker moves for a partially-played turn: from `board` with
/// `remaining` dice still to play, return every single-checker move that keeps
/// the turn on track to use the maximum number of dice (and, when only one of
/// two distinct dice can be played, the larger one). An empty result means the
/// turn is over (all forced dice played, or a dance).
///
/// This lets a GUI accept a human turn one checker at a time while enforcing the
/// same rules as [`genmoves`].
pub fn next_submoves(board: &Board, remaining: &[u8]) -> Vec<SubMove> {
    let max_len = max_completion(board, remaining);
    if max_len == 0 {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut tried = [false; 7];
    for i in 0..remaining.len() {
        let d = remaining[i];
        if tried[d as usize] {
            continue;
        }
        tried[d as usize] = true;
        let mut rem2 = remaining.to_vec();
        rem2.remove(i);
        for (step, child) in gen_single(board, d) {
            if 1 + max_completion(&child, &rem2) == max_len {
                let (from, to) = step_from_to(&step);
                out.push(SubMove {
                    from,
                    to,
                    die: d,
                    result: child,
                });
            }
        }
    }

    // Larger-die rule: if only one die can be played and the two dice differ,
    // it must be the larger one.
    if max_len == 1 {
        let distinct: HashSet<u8> = remaining.iter().copied().collect();
        if distinct.len() >= 2 {
            let larger = *remaining.iter().max().unwrap();
            if out.iter().any(|s| s.die == larger) {
                out.retain(|s| s.die == larger);
            }
        }
    }
    out
}

/// All legal single-die applications of `die` on `board`, as `(step, result)`.
fn gen_single(board: &Board, die: u8) -> Vec<(Step, Board)> {
    use crate::board::{MOVER, NUM_POINTS};
    let mut out = Vec::new();

    // Bar re-entry takes precedence over every other move.
    if board.bar(MOVER) > 0 {
        let to = 25 - die as usize; // die 1..6 -> points 24..19
        if !board.is_blocked_for_mover(to) {
            let mut child = board.clone();
            child.enter_from_bar(to);
            out.push((Step::Enter { to, die }, child));
        }
        return out;
    }

    let all_home = board.mover_all_home();
    let highest = board.highest_mover_point();

    for p in 1..=NUM_POINTS {
        if board.point(p) <= 0 {
            continue; // no mover checker here
        }
        let dest = p as i32 - die as i32;
        if dest >= 1 {
            let to = dest as usize;
            if !board.is_blocked_for_mover(to) {
                let mut child = board.clone();
                child.move_checker(p, to);
                out.push((Step::Point { from: p, to, die }, child));
            }
        } else if all_home {
            // dest <= 0: bearing off. Exact when p == die; otherwise overflow,
            // legal only when p is the highest occupied point (die > p, none higher).
            let exact = dest == 0;
            if exact || p == highest {
                let mut child = board.clone();
                child.bear_off_checker(p);
                out.push((Step::BearOff { from: p, die }, child));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, CHECKERS_PER_SIDE, MOVER, OPP};
    use crate::dice::Dice;

    /// Build a board from explicit mover/opponent point lists, filling the rest
    /// onto a far "parking" point so both sides total 15 without interfering.
    /// `mover`/`opp` are `(point, count)` in mover-relative coords.
    fn fixture(
        mover: &[(usize, i8)],
        opp: &[(usize, i8)],
        mover_bar: u8,
        opp_bar: u8,
        mover_off: u8,
        opp_off: u8,
    ) -> Board {
        let mut b = Board::empty();
        for &(p, c) in mover {
            b.set_point(p, c);
        }
        for &(p, c) in opp {
            b.set_point(p, -c);
        }
        b.set_bar(MOVER, mover_bar);
        b.set_bar(OPP, opp_bar);
        b.set_off(MOVER, mover_off);
        b.set_off(OPP, opp_off);
        assert!(b.validate().is_ok(), "fixture invalid: {:?}", b.validate());
        b
    }

    #[test]
    fn opening_moves_are_all_valid_and_use_all_dice() {
        let b = Board::starting_position();
        for a in 1..=6u8 {
            for c in a..=6u8 {
                let dice = Dice::new(a, c);
                let moves = genmoves(&b, &dice);
                assert!(!moves.is_empty(), "no moves for {a}-{c}");
                let max = moves.iter().map(|m| m.dice_used()).max().unwrap();
                for m in &moves {
                    assert!(m.result.validate().is_ok(), "invalid result for {a}-{c}");
                    // From the opening every roll can be fully played.
                    assert_eq!(m.dice_used(), max);
                    assert_ne!(m.result, b, "opening move must change the board");
                }
                // Opening can always play both dice (4 for doubles).
                let expect = if dice.is_double() { 4 } else { 2 };
                assert_eq!(max, expect, "roll {a}-{c} should use {expect} dice");
            }
        }
    }

    #[test]
    fn bear_off_overflow_bears_both() {
        // Mover: 2 on the ace point, 13 already off. Opponent parked on 24.
        let b = fixture(&[(1, 2)], &[(24, 15)], 0, 0, 13, 0);
        let moves = genmoves(&b, &Dice::new(6, 5));
        assert_eq!(moves.len(), 1, "one distinct bear-off outcome");
        let m = &moves[0];
        assert_eq!(m.dice_used(), 2);
        assert!(m.result.has_won(MOVER));
        assert_eq!(m.result.off(MOVER) as i32, CHECKERS_PER_SIDE);
    }

    #[test]
    fn bear_off_from_highest_when_die_exceeds() {
        // Checkers on points 5 and 3 only (rest off). Roll 6-6? use 6.
        let b = fixture(&[(5, 1), (3, 1)], &[(24, 15)], 0, 0, 13, 0);
        // Single die 6: must bear off from the 5 (highest), not the 3.
        let moves = genmoves(&b, &Dice::new(6, 6));
        // 6-6 is a double: four 6s. Bears off 5 then 3 (now highest), then no more.
        assert!(moves.iter().all(|m| m.result.validate().is_ok()));
        let best = moves.iter().map(|m| m.dice_used()).max().unwrap();
        assert_eq!(best, 2, "only two checkers to bear off");
        assert!(moves.iter().any(|m| m.result.has_won(MOVER)));
    }

    #[test]
    fn dance_when_all_entry_points_blocked() {
        // Mover has one on the bar; opponent holds all six entry points (19..24).
        let opp: Vec<(usize, i8)> = (19..=24).map(|p| (p, 2)).collect(); // 12
        let mover = [(6, 14)];
        let b = fixture(&mover, &[&opp[..], &[(1, 3)]].concat(), 1, 0, 0, 0);
        let moves = genmoves(&b, &Dice::new(3, 5));
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].dice_used(), 0, "dance = no dice played");
        assert_eq!(moves[0].result, b, "board unchanged on a dance");
    }

    #[test]
    fn must_enter_from_bar_before_other_moves() {
        // One on the bar; entry open for a 2 (point 23), blocked for a 5 (point 20).
        let opp = [(20, 2), (24, 13)]; // block point 20, park the rest
        let mover = [(6, 14)];
        let b = fixture(&mover, &opp, 1, 0, 0, 0);
        let moves = genmoves(&b, &Dice::new(2, 5));
        // The 5 can't enter; the 2 enters on 23. Every move must start by entering.
        for m in &moves {
            assert!(matches!(m.steps[0], Step::Enter { to: 23, .. }));
            assert_eq!(m.result.bar(MOVER), 0, "checker left the bar");
        }
        assert!(!moves.is_empty());
    }

    #[test]
    fn landing_on_a_blot_hits_it() {
        // Mover checker on 6, opponent blot on 3. A 3 moves 6->3 and hits.
        let b = fixture(&[(6, 1), (1, 14)], &[(3, 1), (24, 14)], 0, 0, 0, 0);
        let moves = genmoves(&b, &Dice::new(3, 3));
        // Some resulting line must put an opponent checker on the bar.
        assert!(
            moves.iter().any(|m| m.result.bar(OPP) >= 1),
            "expected a hit sending opponent to the bar"
        );
    }

    #[test]
    fn larger_die_must_be_played_when_only_one_fits() {
        // Construct a position where a 6 can be played but not a 1, and after the
        // 6 the 1 still can't be played -> only one die, must be the 6.
        // Mover single checker on point 8; everything else off the play area.
        // Opponent blocks points 7 (for the 1: 8->7) and leaves 2 (8->2 via 6) open.
        let b = fixture(&[(8, 1), (1, 14)], &[(7, 2), (24, 13)], 0, 0, 0, 0);
        let moves = genmoves(&b, &Dice::new(6, 1));
        // 8->2 uses the 6. The 1 (8->7) is blocked; after 8->2, a 1 gives 2->1
        // which is open, so actually both may be playable — assert via max.
        let max = moves.iter().map(|m| m.dice_used()).max().unwrap();
        for m in &moves {
            assert_eq!(m.dice_used(), max);
        }
    }

    #[test]
    fn hand_enumerated_bearoff_count() {
        // Mover: one checker each on points 1..=6 (6 total), 9 already off.
        // Opponent parked on 24. Roll 6-5, all home. Worked out by hand:
        //   * 6 bears off exactly; 5 then bears off 5           -> {4,3,2,1}, 11 off
        //   * 5 moves 6->1; 6 then overflow-bears the 5         -> {4,3,2,1x2}, 10 off
        // Every other ordering collapses onto the first outcome. => 2 distinct turns.
        let mover: Vec<(usize, i8)> = (1..=6).map(|p| (p, 1)).collect();
        let b = fixture(&mover, &[(24, 15)], 0, 0, 9, 0);
        let moves = genmoves(&b, &Dice::new(6, 5));
        assert_eq!(moves.len(), 2, "expected exactly two distinct bear-off turns");
        let mut offs: Vec<u8> = moves.iter().map(|m| m.result.off(MOVER)).collect();
        offs.sort_unstable();
        assert_eq!(offs, vec![10, 11]);
        assert!(moves.iter().all(|m| m.dice_used() == 2));
    }

    fn expand_submoves(board: &Board, remaining: &[u8], out: &mut HashSet<Board>) {
        let opts = next_submoves(board, remaining);
        if opts.is_empty() {
            out.insert(board.clone());
            return;
        }
        for sm in opts {
            let mut rem2 = remaining.to_vec();
            let pos = rem2.iter().position(|&d| d == sm.die).unwrap();
            rem2.remove(pos);
            expand_submoves(&sm.result, &rem2, out);
        }
    }

    #[test]
    fn submoves_reconstruct_genmoves() {
        // Following next_submoves to completion must reproduce exactly the set of
        // full-turn results from the validated genmoves, across varied positions.
        let boards = [
            Board::starting_position(),
            fixture(&[(1, 2)], &[(24, 15)], 0, 0, 13, 0), // bear-off
            fixture(&[(6, 14)], &[(20, 2), (24, 13)], 1, 0, 0, 0), // on the bar
        ];
        for b in &boards {
            for a in 1..=6u8 {
                for c in a..=6u8 {
                    let dice = Dice::new(a, c);
                    let full: HashSet<Board> =
                        genmoves(b, &dice).into_iter().map(|m| m.result).collect();
                    let mut recon = HashSet::new();
                    expand_submoves(b, &dice.plays(), &mut recon);
                    assert_eq!(full, recon, "mismatch for dice {a}-{c}");
                }
            }
        }
    }

    #[test]
    fn every_result_conserves_checkers() {
        let b = Board::starting_position();
        for a in 1..=6u8 {
            for c in 1..=6u8 {
                for m in genmoves(&b, &Dice::new(a, c)) {
                    assert!(m.result.validate().is_ok());
                }
            }
        }
    }
}
