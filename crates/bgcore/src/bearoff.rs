//! Exact one-sided bear-off database.
//!
//! When all of a side's 15 checkers are in its home board (points 1..=6), the
//! number of rolls to bear off is a *solved* quantity — no rollouts. Backward
//! dynamic programming over the 54,264 home-board configurations gives the exact
//! distribution of rolls-to-bear-off under the expected-roll-minimising play,
//! plus the rolls-to-first-checker distribution (for gammons). Two sides'
//! distributions convolve into exact race equities — win and gammon; a backgammon
//! is impossible in a home-vs-home race (that would require contact).
//!
//! The graph is a DAG (every legal play strictly reduces the pip count), so a
//! single pass in increasing-pip order solves it — no iteration to convergence.

use crate::board::{Board, MOVER, OPP};
use crate::dice::Dice;
use crate::eval::Value;
use crate::moves::genmoves;
use std::collections::HashMap;
use std::sync::OnceLock;

const NP: usize = 6; // home-board points
const CHK: u8 = 15; // checkers per side
const MAXR: usize = 48; // rolls tracked per distribution (mass beyond this is ~0)

/// Checkers on points 1..=6 (index i -> point i+1); borne off = 15 - sum.
type Config = [u8; NP];

fn pip(c: &Config) -> u32 {
    (0..NP).map(|i| c[i] as u32 * (i as u32 + 1)).sum()
}
fn total(c: &Config) -> u8 {
    c.iter().sum()
}
/// Pack a config into a u32 key (4 bits per point; counts are 0..=15).
fn pack(c: &Config) -> u32 {
    (0..NP).fold(0u32, |k, i| k | ((c[i] as u32) << (4 * i)))
}

pub struct Bearoff {
    index: HashMap<u32, u32>,
    /// Per config: P(all checkers off in exactly n rolls).
    dist_all: Vec<[f32; MAXR]>,
    /// Per config: P(first checker off on exactly roll n). For a config that
    /// already has a checker off (sum < 15) this is a sentinel [1,0,0,..] so the
    /// side reads as "un-gammonable".
    dist_first: Vec<[f32; MAXR]>,
    /// Per config: expected rolls to bear off (under the same policy).
    expected: Vec<f32>,
}

fn enumerate() -> Vec<Config> {
    let mut out = Vec::with_capacity(54_264);
    for a in 0..=CHK {
        for b in 0..=CHK - a {
            for c in 0..=CHK - a - b {
                for d in 0..=CHK - a - b - c {
                    for e in 0..=CHK - a - b - c - d {
                        for f in 0..=CHK - a - b - c - d - e {
                            out.push([a, b, c, d, e, f]);
                        }
                    }
                }
            }
        }
    }
    out
}

impl Bearoff {
    pub fn build() -> Self {
        let configs = enumerate();
        let n = configs.len();
        let mut index = HashMap::with_capacity(n);
        for (i, c) in configs.iter().enumerate() {
            index.insert(pack(c), i as u32);
        }
        let mut dist_all = vec![[0.0f32; MAXR]; n];
        let mut dist_first = vec![[0.0f32; MAXR]; n];
        let mut expected = vec![0.0f32; n];

        // Increasing pip order: every child has strictly fewer pips, so it is
        // already solved when its parent is reached.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| pip(&configs[i]));

        // The 21 distinct rolls, weighted out of 36 (doubles x1, others x2).
        let mut rolls = Vec::with_capacity(21);
        for a in 1..=6u8 {
            for c in a..=6u8 {
                rolls.push((a, c, if a == c { 1.0f32 } else { 2.0 }));
            }
        }

        for &pi in &order {
            let cfg = configs[pi];
            let s = total(&cfg);
            if s == 0 {
                dist_all[pi][0] = 1.0; // all borne off
                dist_first[pi][0] = 1.0; // sentinel (already off => not gammonable)
                expected[pi] = 0.0;
                continue;
            }
            if s < CHK {
                dist_first[pi][0] = 1.0; // already borne off >=1 => un-gammonable
            }

            // Mover = this config on points 1..6, opponent parked on 24 (no contact,
            // does not affect the bear-off moves).
            let mut board = Board::empty();
            for i in 0..NP {
                if cfg[i] > 0 {
                    board.set_point(i + 1, cfg[i] as i8);
                }
            }
            board.set_off(MOVER, CHK - s);
            board.set_point(24, -(CHK as i8));

            let parent_pip = pip(&cfg);
            let mut exp_acc = 0.0f32;
            for &(a, c, w) in &rolls {
                // Expected-roll-minimising play: the child with the fewest expected
                // remaining rolls.
                let mut best = usize::MAX;
                let mut best_e = f32::INFINITY;
                for mv in &genmoves(&board, &Dice::new(a, c)) {
                    let child: Config = std::array::from_fn(|i| mv.result.point(i + 1).max(0) as u8);
                    if pip(&child) >= parent_pip {
                        continue; // never happens for a real bear-off (paranoia vs a pass)
                    }
                    let ci = index[&pack(&child)] as usize;
                    if expected[ci] < best_e {
                        best_e = expected[ci];
                        best = ci;
                    }
                }
                exp_acc += w * best_e;
                for m in 1..MAXR {
                    dist_all[pi][m] += w * dist_all[best][m - 1];
                }
                if s == CHK {
                    if total(&configs[best]) < CHK {
                        dist_first[pi][1] += w; // first checker comes off this roll
                    } else {
                        for m in 1..MAXR {
                            dist_first[pi][m] += w * dist_first[best][m - 1];
                        }
                    }
                }
            }
            expected[pi] = 1.0 + exp_acc / 36.0;
            for m in 0..MAXR {
                dist_all[pi][m] /= 36.0;
            }
            if s == CHK {
                for m in 0..MAXR {
                    dist_first[pi][m] /= 36.0;
                }
            }
        }

        Bearoff { index, dist_all, dist_first, expected }
    }

    /// Expected rolls to bear off a mover home-board `config` (points 1..=6).
    pub fn expected_rolls(&self, config: &Config) -> f32 {
        self.expected[self.index[&pack(config)] as usize]
    }

    /// Exact race value for the side to move at `board`, which MUST be a
    /// home-vs-home race ([`is_home_race`]). Win / gammon only.
    pub fn value(&self, board: &Board) -> Value {
        let mi = self.index[&pack(&mover_config(board))] as usize;
        let oi = self.index[&pack(&opp_config(board))] as usize;
        let dm = &self.dist_all[mi];
        let do_ = &self.dist_all[oi];
        let cdf = |d: &[f32; MAXR]| {
            let mut c = [0.0f32; MAXR];
            let mut s = 0.0;
            for k in 0..MAXR {
                s += d[k];
                c[k] = s;
            }
            c
        };
        let cdf_o = cdf(do_);
        let cdf_fo = cdf(&self.dist_first[oi]);
        let cdf_fm = cdf(&self.dist_first[mi]);

        // Turns alternate M, O, M, O, ... with the mover first. If the mover
        // finishes on roll a, the opponent has had a-1 rolls; the mover wins iff
        // the opponent needs >= a rolls, and it is a gammon iff the opponent has
        // borne off nothing in those a-1 rolls.
        let mut win = 0.0f32;
        let mut win_g = 0.0f32;
        for a in 1..MAXR {
            let pm = dm[a];
            if pm == 0.0 {
                continue;
            }
            win += pm * (1.0 - cdf_o[a - 1]);
            win_g += pm * (1.0 - cdf_fo[a - 1]);
        }
        // If the opponent finishes on roll b the mover has had b rolls; the mover
        // is gammoned iff it has borne off nothing in them.
        let mut lose_g = 0.0f32;
        for b in 1..MAXR {
            let po = do_[b];
            if po == 0.0 {
                continue;
            }
            lose_g += po * (1.0 - cdf_fm[b]);
        }
        Value { win, win_g, win_bg: 0.0, lose_g, lose_bg: 0.0 }
    }
}

fn mover_config(board: &Board) -> Config {
    std::array::from_fn(|i| board.point(i + 1).max(0) as u8)
}

/// The opponent's home config: their point j (1..=6) is mover-frame point 25-j,
/// stored negated.
fn opp_config(board: &Board) -> Config {
    std::array::from_fn(|i| (-board.point(24 - i)).max(0) as u8)
}

/// True if both sides have every checker in their own home board and nothing on
/// the bar — i.e. a pure race the one-sided table solves exactly.
pub fn is_home_race(board: &Board) -> bool {
    if board.bar(MOVER) > 0 || board.bar(OPP) > 0 {
        return false;
    }
    for p in 7..=24 {
        if board.point(p) > 0 {
            return false; // a mover checker outside its home board
        }
    }
    for p in 1..=18 {
        if board.point(p) < 0 {
            return false; // an opponent checker outside its home board
        }
    }
    true
}

/// The process-wide table, built once on first use (~a few seconds).
pub fn table() -> &'static Bearoff {
    static TABLE: OnceLock<Bearoff> = OnceLock::new();
    TABLE.get_or_init(Bearoff::build)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(pts: &[(usize, u8)]) -> Config {
        let mut c = [0u8; NP];
        for &(p, n) in pts {
            c[p - 1] = n;
        }
        c
    }

    #[test]
    fn one_on_ace_is_exactly_one_roll() {
        let bo = Bearoff::build();
        let i = bo.index[&pack(&cfg(&[(1, 1)]))] as usize;
        assert!((bo.expected[i] - 1.0).abs() < 1e-5);
        assert!((bo.dist_all[i][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn three_on_ace_matches_hand_computation() {
        // 3 checkers on the 1-point: a double (6/36) clears all three in one roll;
        // any non-double (30/36) bears off two, then one more roll finishes it.
        let bo = Bearoff::build();
        let i = bo.index[&pack(&cfg(&[(1, 3)]))] as usize;
        assert!((bo.dist_all[i][1] - 6.0 / 36.0).abs() < 1e-5, "{}", bo.dist_all[i][1]);
        assert!((bo.dist_all[i][2] - 30.0 / 36.0).abs() < 1e-5, "{}", bo.dist_all[i][2]);
        assert!((bo.expected[i] - 11.0 / 6.0).abs() < 1e-4, "{}", bo.expected[i]);
    }

    #[test]
    fn count_and_distributions_are_well_formed() {
        let bo = Bearoff::build();
        assert_eq!(bo.expected.len(), 54_264);
        for &i in &[0usize, 1, 100, 5000, 30000, 54_263] {
            let s: f32 = bo.dist_all[i].iter().sum();
            assert!((s - 1.0).abs() < 1e-3, "config {i} dist sums to {s}");
            assert!(bo.expected[i].is_finite());
        }
    }

    #[test]
    fn more_checkers_take_longer() {
        let bo = Bearoff::build();
        let a = bo.expected_rolls(&cfg(&[(6, 5)]));
        let b = bo.expected_rolls(&cfg(&[(6, 10)]));
        assert!(b > a, "{b} !> {a}");
    }

    #[test]
    fn symmetric_race_gives_on_roll_advantage() {
        // Both sides 3 on each of their 1/2/3 points (9 on board, 6 off), mirrored.
        // The side on roll should win clearly above 50% but not runaway.
        let bo = Bearoff::build();
        let mut b = Board::empty();
        for p in 1..=3 {
            b.set_point(p, 3);
            b.set_point(25 - p, -3);
        }
        b.set_off(MOVER, 6);
        b.set_off(OPP, 6);
        assert!(is_home_race(&b));
        let v = bo.value(&b);
        // On-roll advantage: win = (1 + P(tie))/2 > 0.5; a low-pip race ties often
        // so it lands well above 0.5 (here ~0.72), but not a certainty.
        assert!(v.win > 0.5 && v.win < 0.85, "on-roll win {}", v.win);
        assert!((0.0..=1.0).contains(&v.win_g));
        assert!(v.win_g <= v.win && v.lose_g <= 1.0 - v.win);
    }

    #[test]
    fn lopsided_race_is_near_certain() {
        // Mover almost off (1 on the ace), opponent a full home board.
        let bo = Bearoff::build();
        let mut b = Board::empty();
        b.set_point(1, 1);
        b.set_off(MOVER, 14);
        for p in 1..=5 {
            b.set_point(25 - p, -3); // opp: 3 on each of its 1..5 points = 15, all home
        }
        b.set_off(OPP, 0);
        assert!(is_home_race(&b));
        let v = bo.value(&b);
        assert!(v.win > 0.99, "win {}", v.win);
        assert!(v.win_g > 0.9, "gammon {}", v.win_g); // opp can't bear off before we finish
    }
}
