//! Hand-crafted evaluator (SPEC §7).
//!
//! A transparent, racing-biased heuristic: it converts a pip-count lead (plus a
//! few positional nudges) into a win probability via a logistic. It is not
//! meant to be world-class — only clearly stronger and more stable than random,
//! enough to bootstrap self-play and serve as a fixed benchmark rung. Gammon and
//! backgammon terms are left at zero for now (a later refinement).

use super::{Evaluator, Value};
use crate::board::{Board, MOVER, OPP};

/// Tunable weights for [`HceEval`]. Units are "pip-equivalents" fed into the
/// logistic, so they are directly comparable to a pip-count lead.
#[derive(Clone, Copy, Debug)]
pub struct HceWeights {
    /// Effective pip advantage of being on roll (~half an average roll).
    pub on_roll: f32,
    /// Extra cost of a checker sitting on the bar, beyond its 25-pip distance.
    pub bar_penalty: f32,
    /// Value of each home-board point made (net of the opponent's).
    pub home_point: f32,
    /// Penalty for each of our exposed blots (bonus for the opponent's).
    pub blot: f32,
    /// Logistic slope: win-prob sensitivity to the pip-equivalent score.
    pub slope: f32,
}

impl Default for HceWeights {
    fn default() -> Self {
        HceWeights {
            on_roll: 4.0,
            bar_penalty: 4.0,
            home_point: 1.5,
            blot: 0.5,
            slope: 0.09,
        }
    }
}

/// Hand-crafted, pip-race-based evaluator.
#[derive(Clone, Copy, Debug, Default)]
pub struct HceEval {
    w: HceWeights,
}

impl HceEval {
    pub fn new() -> Self {
        HceEval::default()
    }

    pub fn with_weights(w: HceWeights) -> Self {
        HceEval { w }
    }
}

impl Evaluator for HceEval {
    fn evaluate(&self, board: &Board) -> Value {
        let w = &self.w;

        // Racing term: positive when the mover is ahead in the pip count.
        let mut score = board.pip_count(OPP) as f32 - board.pip_count(MOVER) as f32;
        score += w.on_roll;

        // Checkers on the bar hurt (their 25 pips are already counted; this is
        // the extra positional cost of being off the board).
        score -= w.bar_penalty * board.bar(MOVER) as f32;
        score += w.bar_penalty * board.bar(OPP) as f32;

        // Home-board strength: made points in each player's home (mover 1..=6,
        // opponent 19..=24 in mover-relative coordinates).
        let mover_home = (1..=6).filter(|&p| board.point(p) >= 2).count() as i32;
        let opp_home = (19..=24).filter(|&p| board.point(p) <= -2).count() as i32;
        score += w.home_point * (mover_home - opp_home) as f32;

        // Exposed blots: single checkers that can be hit.
        let mut mover_blots = 0i32;
        let mut opp_blots = 0i32;
        for p in 1..=24 {
            match board.point(p) {
                1 => mover_blots += 1,
                -1 => opp_blots += 1,
                _ => {}
            }
        }
        score -= w.blot * mover_blots as f32;
        score += w.blot * opp_blots as f32;

        Value::from_win_prob(sigmoid(w.slope * score))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_is_roughly_even_but_favors_mover() {
        // Both sides at 167 pips; only the on-roll advantage tilts it.
        let v = HceEval::new().evaluate(&Board::starting_position());
        assert!(v.win > 0.5 && v.win < 0.6, "win prob {}", v.win);
    }

    #[test]
    fn big_pip_lead_is_winning() {
        // Mover about to bear off (pip 15); opponent stuck far back near their
        // own 24 point (mover-relative point 2 -> 23 pips each, ~345 total).
        let mut b = Board::empty();
        b.set_point(1, 15); // mover all home on the ace point
        b.set_point(2, -15); // opponent stuck deep in its own back field
        let v = HceEval::new().evaluate(&b);
        assert!(v.win > 0.9, "win prob {}", v.win);
        assert!(v.equity() > 0.8);
    }
}
