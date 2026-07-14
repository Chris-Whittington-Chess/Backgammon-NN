//! Python (PyO3) bindings for `bgcore` (SPEC §2, milestone M3).
//!
//! Exposes the engine primitives the PyTorch trainer needs: build/parse
//! positions, generate legal moves, encode the 198-input features (from the same
//! Rust encoder used everywhere else), and drive HCE-bootstrapped self-play.
//!
//! The Python module is named `bgcore`. Example:
//! ```python
//! import bgcore
//! b = bgcore.Board.starting()
//! kids = bgcore.legal_moves(b, 3, 1)          # resulting positions
//! feats = bgcore.children_features(b, 3, 1)    # aligned [n, 198] features
//! ```

use bgengine::{
    genmoves, next_submoves, Board, Dice, Engine, EvalEngine, Evaluator, GameResult, HceEval, Step,
    BAR, OFF,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A backgammon position, always from the side-to-move's perspective.
#[pyclass(name = "Board")]
#[derive(Clone)]
struct PyBoard {
    inner: Board,
}

#[pymethods]
impl PyBoard {
    /// The standard opening position.
    #[staticmethod]
    fn starting() -> PyBoard {
        PyBoard {
            inner: Board::starting_position(),
        }
    }

    /// Parse a GnuBG Position ID (e.g. `"4HPwATDgc/ABMA"`).
    #[staticmethod]
    fn from_id(id: &str) -> PyResult<PyBoard> {
        Board::from_position_id(id)
            .map(|inner| PyBoard { inner })
            .map_err(PyValueError::new_err)
    }

    /// The GnuBG Position ID for this position.
    fn position_id(&self) -> String {
        self.inner.position_id()
    }

    /// The 198 neural-network input features (SPEC §6).
    fn features(&self) -> Vec<f32> {
        bgengine::encode(&self.inner).to_vec()
    }

    /// Pip count for a side: `0` = mover, `1` = opponent.
    fn pip_count(&self, side: usize) -> i32 {
        self.inner.pip_count(side)
    }

    /// Checkers on point `p` (`1..=24`): `+` mover, `-` opponent. For rendering.
    fn point(&self, p: usize) -> i8 {
        self.inner.point(p)
    }

    /// Checkers on the bar for a side: `0` = mover, `1` = opponent.
    fn bar(&self, side: usize) -> u8 {
        self.inner.bar(side)
    }

    /// Checkers borne off for a side: `0` = mover, `1` = opponent.
    fn off(&self, side: usize) -> u8 {
        self.inner.off(side)
    }

    /// True once the sides have passed each other — a pure race (no hits or
    /// blocking possible). Used to focus gammon-saving training on positions
    /// whose rollout labels are trustworthy.
    fn no_contact(&self) -> bool {
        self.inner.no_contact()
    }

    /// This position viewed from the opponent's side (turn passed).
    fn swap_perspective(&self) -> PyBoard {
        PyBoard {
            inner: self.inner.swap_perspective(),
        }
    }

    /// True once either side has borne off all checkers.
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// Signed game result: `+points` if the mover has won, `-points` if the
    /// opponent has won (1 single, 2 gammon, 3 backgammon), `None` if ongoing.
    fn winner_points(&self) -> Option<i32> {
        match bgengine::result(&self.inner) {
            GameResult::MoverWins(p) => Some(p as i32),
            GameResult::OppWins(p) => Some(-(p as i32)),
            GameResult::InProgress => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("Board(\"{}\")", self.inner.position_id())
    }

    fn __eq__(&self, other: &PyBoard) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }
}

/// All legal resulting positions after playing `d1, d2` from `board`.
/// A "dance" (no legal move) returns a single unchanged position.
#[pyfunction]
fn legal_moves(board: &PyBoard, d1: u8, d2: u8) -> Vec<PyBoard> {
    genmoves(&board.inner, &Dice::new(d1, d2))
        .into_iter()
        .map(|m| PyBoard { inner: m.result })
        .collect()
}

/// The 198-input features of every legal resulting position, aligned index-for-
/// index with [`legal_moves`]. Returns an `[n, 198]`-shaped nested list.
#[pyfunction]
fn children_features(board: &PyBoard, d1: u8, d2: u8) -> Vec<Vec<f32>> {
    genmoves(&board.inner, &Dice::new(d1, d2))
        .into_iter()
        .map(|m| bgengine::encode(&m.result).to_vec())
        .collect()
}

/// Features of every legal *next state* (resulting position with the turn
/// passed, i.e. opponent to move), aligned index-for-index with [`legal_moves`].
/// This is the batch the trainer evaluates for move selection and TD targets:
/// `net(next_state_features(s, d1, d2))` gives each child's value from the
/// opponent's perspective in one forward pass.
#[pyfunction]
fn next_state_features(board: &PyBoard, d1: u8, d2: u8) -> Vec<Vec<f32>> {
    genmoves(&board.inner, &Dice::new(d1, d2))
        .into_iter()
        .map(|m| bgengine::encode(&m.result.swap_perspective()).to_vec())
        .collect()
}

/// Apply a single checker step to a board, returning the new board. `from ==
/// BAR` enters from the bar; `to == OFF` bears off; otherwise a point-to-point
/// move (hitting a blot if present). Used to reconstruct intermediate positions
/// for animating a multi-step move. Caller must pass a legal step.
#[pyfunction]
fn apply_step(board: &PyBoard, from: usize, to: usize) -> PyBoard {
    let mut b = board.inner.clone();
    if from == BAR {
        b.enter_from_bar(to);
    } else if to == OFF {
        b.bear_off_checker(from);
    } else {
        b.move_checker(from, to);
    }
    PyBoard { inner: b }
}

/// All legal full turns as `(steps, result_board)`, where `steps` is a list of
/// `(from, to, die)` checker moves (`from == BAR` for a bar entry, `to == OFF`
/// for a bear-off). Lets the GUI show engine hints in standard move notation.
#[pyfunction]
fn legal_moves_with_steps(
    board: &PyBoard,
    d1: u8,
    d2: u8,
) -> Vec<(Vec<(usize, usize, u8)>, PyBoard)> {
    genmoves(&board.inner, &Dice::new(d1, d2))
        .into_iter()
        .map(|m| {
            let steps = m
                .steps
                .iter()
                .map(|s| {
                    let (from, to) = match *s {
                        Step::Enter { to, .. } => (BAR, to),
                        Step::Point { from, to, .. } => (from, to),
                        Step::BearOff { from, .. } => (from, OFF),
                    };
                    (from, to, s.die())
                })
                .collect();
            (steps, PyBoard { inner: m.result })
        })
        .collect()
}

/// The legal *next* checker moves for a partially-played turn, given the dice
/// still remaining. Each entry is `(from, to, die, result_board)`, where `from`
/// is `BAR` (25) for a bar entry and `to` is `OFF` (0) for a bear-off. An empty
/// result means the turn is over. Lets a GUI accept a human turn one checker at
/// a time while enforcing all move rules.
#[pyfunction]
fn submoves(board: &PyBoard, remaining: Vec<u8>) -> Vec<(usize, usize, u8, PyBoard)> {
    next_submoves(&board.inner, &remaining)
        .into_iter()
        .map(|s| (s.from, s.to, s.die, PyBoard { inner: s.result }))
        .collect()
}

/// The hand-crafted evaluator's cubeless equity for a position.
#[pyfunction]
fn hce_equity(board: &PyBoard) -> f32 {
    HceEval::new().evaluate(&board.inner).equity()
}

/// The move the HCE engine would play (used to bootstrap early self-play).
#[pyfunction]
fn hce_move(board: &PyBoard, d1: u8, d2: u8) -> PyBoard {
    let mut engine = EvalEngine::new(HceEval::new(), "HCE");
    let mv = engine.choose(&board.inner, &Dice::new(d1, d2));
    PyBoard { inner: mv.result }
}

/// Parallel Monte-Carlo rollout engine (requires the `onnx` build feature).
/// Loads an exported ONNX net once and rolls out positions natively in Rust.
#[cfg(feature = "onnx")]
#[pyclass]
struct Rollouts {
    nn: bgengine::eval::NnEval,
    cfg: bgengine::RolloutConfig,
    pool: Option<rayon::ThreadPool>,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl Rollouts {
    #[new]
    #[pyo3(signature = (onnx_path, trials = 180, truncate_plies = 11, candidates = 6, seed = 0x5EED, movetime_ms = 0, threads = 0))]
    fn new(
        onnx_path: &str,
        trials: usize,
        truncate_plies: usize,
        candidates: usize,
        seed: u64,
        movetime_ms: u64,
        threads: usize,
    ) -> PyResult<Self> {
        let nn = bgengine::eval::NnEval::from_path(onnx_path).map_err(PyValueError::new_err)?;
        let cfg = bgengine::RolloutConfig {
            trials,
            truncate_plies,
            candidates,
            seed,
            movetime_ms,
            threads,
        };
        let pool = bgengine::build_pool(threads);
        Ok(Rollouts { nn, cfg, pool })
    }

    /// Rollout equity for the side to move at `board`.
    fn equity(&self, board: &PyBoard) -> f32 {
        match &self.pool {
            Some(p) => p.install(|| bgengine::rollout_equity(&board.inner, &self.nn, &self.cfg)),
            None => bgengine::rollout_equity(&board.inner, &self.nn, &self.cfg),
        }
    }

    /// Rollout outcome distribution for the side to move at `board`, as
    /// `[win, win_g, win_bg, lose_g, lose_bg]` — the 5 training targets.
    fn dist(&self, board: &PyBoard) -> Vec<f32> {
        let f = || bgengine::rollout_dist(&board.inner, &self.nn, &self.cfg);
        match &self.pool {
            Some(p) => p.install(f).to_vec(),
            None => f().to_vec(),
        }
    }

    /// The rollout engine's move for dice `d1, d2` as `(result_board, equity)`,
    /// where equity is from the mover's perspective.
    fn best_move(&self, board: &PyBoard, d1: u8, d2: u8) -> (PyBoard, f32) {
        let dice = Dice::new(d1, d2);
        let f = || bgengine::rollout_best_scored(&board.inner, &dice, &self.nn, &self.cfg);
        let (mv, eq) = match &self.pool {
            Some(p) => p.install(f),
            None => f(),
        };
        (PyBoard { inner: mv.result }, eq)
    }
}

#[pymodule]
fn bgcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "onnx")]
    m.add_class::<Rollouts>()?;
    m.add_class::<PyBoard>()?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(children_features, m)?)?;
    m.add_function(wrap_pyfunction!(next_state_features, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves_with_steps, m)?)?;
    m.add_function(wrap_pyfunction!(apply_step, m)?)?;
    m.add_function(wrap_pyfunction!(submoves, m)?)?;
    m.add_function(wrap_pyfunction!(hce_equity, m)?)?;
    m.add_function(wrap_pyfunction!(hce_move, m)?)?;
    m.add("NUM_INPUTS", bgengine::NUM_INPUTS)?;
    m.add("BAR", BAR)?;
    m.add("OFF", OFF)?;
    Ok(())
}
