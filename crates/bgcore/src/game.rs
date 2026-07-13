//! Game loop, engines, and the engine-vs-engine match runner (SPEC §5, §13).
//!
//! The [`Engine`] trait is the interchangeable-opponent abstraction — our own
//! evaluators wrap into an [`EvalEngine`], and external engines (gnubg, wildbg)
//! can implement it later. [`run_match`] pits any two engines against each other
//! with **duplicate (mirrored) dice** to cancel most of the dice luck, the
//! backgammon analogue of a cutechess/fastchess match — there is no standard
//! protocol, so this is our harness.

use crate::board::{Board, MOVER, OPP};
use crate::dice::{Dice, Rng};
use crate::eval::Evaluator;
use crate::moves::{genmoves, Move};

/// The status of a position after a move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    InProgress,
    /// The side on roll has borne off all checkers, winning this many points.
    MoverWins(u8),
    /// The side *not* on roll has won (only arises in constructed positions).
    OppWins(u8),
}

/// Points won given a terminal board: 1 = single, 2 = gammon, 3 = backgammon.
fn win_points(board: &Board, winner: usize) -> u8 {
    let loser = 1 - winner;
    if board.off(loser) > 0 {
        return 1; // loser bore at least one checker off -> single game
    }
    // Loser bore off nothing: gammon, or backgammon if a loser checker is still
    // on the bar or in the winner's home board.
    let backgammon = if winner == MOVER {
        // Loser = opponent: their checkers are negative; winner's home = 1..=6.
        board.bar(OPP) > 0 || (1..=6).any(|p| board.point(p) < 0)
    } else {
        // Loser = mover: their checkers are positive; winner's home = 19..=24.
        board.bar(MOVER) > 0 || (19..=24).any(|p| board.point(p) > 0)
    };
    if backgammon {
        3
    } else {
        2
    }
}

/// Classify a (mover-relative) position.
pub fn result(board: &Board) -> GameResult {
    if board.has_won(MOVER) {
        GameResult::MoverWins(win_points(board, MOVER))
    } else if board.has_won(OPP) {
        GameResult::OppWins(win_points(board, OPP))
    } else {
        GameResult::InProgress
    }
}

/// An interchangeable opponent: given the on-roll position and dice, choose a
/// full legal turn. `&mut self` allows stateful engines (e.g. external sockets).
pub trait Engine {
    fn choose(&mut self, board: &Board, dice: &Dice) -> Move;
    fn name(&self) -> &str;
}

/// Wraps any [`Evaluator`] into an [`Engine`] using 0-ply negamax move
/// selection: rank each legal move by the equity of the resulting position from
/// the *opponent's* perspective, negated (SPEC §5).
pub struct EvalEngine<E: Evaluator> {
    eval: E,
    name: String,
}

impl<E: Evaluator> EvalEngine<E> {
    pub fn new(eval: E, name: impl Into<String>) -> Self {
        EvalEngine {
            eval,
            name: name.into(),
        }
    }

    /// The scalar score this engine assigns to a resulting position (higher is
    /// better for the side that just moved). Exposed for hints/analysis.
    pub fn score_result(&self, result_board: &Board) -> f32 {
        match result(result_board) {
            GameResult::MoverWins(pts) => pts as f32, // we just won outright
            _ => -self.eval.evaluate(&result_board.swap_perspective()).equity(),
        }
    }
}

impl<E: Evaluator> Engine for EvalEngine<E> {
    fn choose(&mut self, board: &Board, dice: &Dice) -> Move {
        let mut moves = genmoves(board, dice);
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, m) in moves.iter().enumerate() {
            let s = self.score_result(&m.result);
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        moves.swap_remove(best)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// The result of a single game.
#[derive(Clone, Copy, Debug)]
pub struct GameOutcome {
    /// Seat index (0 or 1) of the winner.
    pub winner: usize,
    /// Points won: 1 single, 2 gammon, 3 backgammon.
    pub points: u8,
    /// Number of half-moves (turns) played.
    pub plies: u32,
}

/// Play one game between the seat-0 and seat-1 engines, consuming dice from
/// `rng`. Seat 0 is on roll first. Returns when a side bears off all checkers.
pub fn play_game(p0: &mut dyn Engine, p1: &mut dyn Engine, rng: &mut Rng) -> GameOutcome {
    const MAX_PLIES: u32 = 4000; // safety net; real games are far shorter
    let mut board = Board::starting_position();
    let mut on_roll = 0usize;

    for ply in 0..MAX_PLIES {
        let dice = rng.roll();
        let engine: &mut dyn Engine = if on_roll == 0 { p0 } else { p1 };
        let mv = engine.choose(&board, &dice);
        board = mv.result;

        if let GameResult::MoverWins(points) = result(&board) {
            return GameOutcome {
                winner: on_roll,
                points,
                plies: ply + 1,
            };
        }
        board = board.swap_perspective();
        on_roll ^= 1;
    }

    // Unreachable in practice; decide by pip count so the API stays total.
    let winner = if board.pip_count(MOVER) <= board.pip_count(OPP) {
        on_roll
    } else {
        1 - on_roll
    };
    GameOutcome {
        winner,
        points: 1,
        plies: MAX_PLIES,
    }
}

/// Aggregate results of a match, always from engine A's point of view.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatchStats {
    pub games: u32,
    /// Net points for A (win adds its point value, loss subtracts it).
    pub a_net_points: i32,
    pub a_wins: u32,
    pub a_losses: u32,
    /// Wins by [single, gammon, backgammon].
    pub a_win_kind: [u32; 3],
    /// Losses by [single, gammon, backgammon].
    pub a_loss_kind: [u32; 3],
}

impl MatchStats {
    fn record(&mut self, a_seat: usize, o: GameOutcome) {
        self.games += 1;
        let kind = (o.points - 1) as usize; // 0/1/2
        if o.winner == a_seat {
            self.a_wins += 1;
            self.a_net_points += o.points as i32;
            self.a_win_kind[kind] += 1;
        } else {
            self.a_losses += 1;
            self.a_net_points -= o.points as i32;
            self.a_loss_kind[kind] += 1;
        }
    }

    /// Fraction of games A won.
    pub fn a_win_rate(&self) -> f64 {
        self.a_wins as f64 / self.games as f64
    }

    /// Average points A scored per game (its skill margin; can be negative).
    pub fn a_ppg(&self) -> f64 {
        self.a_net_points as f64 / self.games as f64
    }

    /// 95% confidence half-width for the win rate (naive; mirrored dice make the
    /// true error smaller, so this is conservative).
    pub fn win_rate_ci95(&self) -> f64 {
        if self.games == 0 {
            return f64::NAN;
        }
        let p = self.a_win_rate();
        1.96 * (p * (1.0 - p) / self.games as f64).sqrt()
    }

    /// A one-line human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "{} games | A win rate {:.1}% ±{:.1} | A PPG {:+.3} | A W(s/g/bg) {:?} L(s/g/bg) {:?}",
            self.games,
            100.0 * self.a_win_rate(),
            100.0 * self.win_rate_ci95(),
            self.a_ppg(),
            self.a_win_kind,
            self.a_loss_kind,
        )
    }
}

/// Run a match of `pairs` game-pairs (so `2 * pairs` games) between engines
/// `a` and `b`, using **mirrored dice**: within each pair both games replay the
/// identical dice stream, once with A on seat 0 and once with B on seat 0. This
/// cancels most dice variance and the first-move edge, isolating skill.
pub fn run_match(a: &mut dyn Engine, b: &mut dyn Engine, pairs: u32, base_seed: u64) -> MatchStats {
    let mut stats = MatchStats::default();
    for i in 0..pairs {
        let seed = base_seed
            .wrapping_add(i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);

        // Game 1: A on seat 0, B on seat 1.
        let mut rng = Rng::new(seed);
        let g1 = play_game(a, b, &mut rng);
        stats.record(0, g1);

        // Game 2: same dice, seats swapped — B on seat 0, A on seat 1.
        let mut rng = Rng::new(seed);
        let g2 = play_game(b, a, &mut rng);
        stats.record(1, g2);
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{HceEval, RandomEval};

    fn hce() -> EvalEngine<HceEval> {
        EvalEngine::new(HceEval::new(), "HCE")
    }
    fn random(seed: u64) -> EvalEngine<RandomEval> {
        EvalEngine::new(RandomEval::new(seed), "Random")
    }

    #[test]
    fn a_game_terminates_with_a_valid_result() {
        let mut a = hce();
        let mut b = random(1);
        let mut rng = Rng::new(7);
        let o = play_game(&mut a, &mut b, &mut rng);
        assert!(o.winner == 0 || o.winner == 1);
        assert!((1..=3).contains(&o.points));
        assert!(o.plies > 0 && o.plies < 4000);
    }

    #[test]
    fn games_are_deterministic_for_a_fixed_seed() {
        let run = || {
            let mut a = hce();
            let mut b = random(1);
            let mut rng = Rng::new(999);
            play_game(&mut a, &mut b, &mut rng)
        };
        let o1 = run();
        let o2 = run();
        assert_eq!(o1.winner, o2.winner);
        assert_eq!(o1.points, o2.points);
        assert_eq!(o1.plies, o2.plies);
    }

    #[test]
    fn hce_beats_random_by_a_wide_margin() {
        // The M2 milestone gate: a pip-racing heuristic should crush random play.
        let mut a = hce();
        let mut b = random(0xABCD);
        let stats = run_match(&mut a, &mut b, 60, 12345); // 120 games, mirrored
        assert!(
            stats.a_win_rate() > 0.75,
            "HCE only won {:.1}% ({})",
            100.0 * stats.a_win_rate(),
            stats.summary()
        );
        assert!(stats.a_ppg() > 0.5, "HCE PPG {:.3}", stats.a_ppg());
    }
}
