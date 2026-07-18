//! Neural-net evaluator via ONNX (SPEC §9, milestone M5).
//!
//! Loads a network exported from PyTorch (`trainer/export_onnx.py`) and runs it
//! with `tract-onnx` — pure Rust, no external ONNX Runtime binary. This lets the
//! trained net drive self-play and play in the GUI at native speed, using the
//! exact same 198-input encoding as training ([`crate::features`]).
//!
//! Enabled by the `onnx` cargo feature.

use crate::board::{Board, MOVER, OPP};
use crate::eval::{Evaluator, Value};
use crate::features;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::shapefactoid;

type TractModel = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Pip-count output buckets (must match `trainer/model.py`): a `[N, N_BUCKETS*6]`
/// net emits one 6-outcome softmax per total-pip bucket, and the engine picks the
/// bucket for the position it is scoring. The edges are calibrated octiles of
/// champion self-play (`trainer/calibrate_buckets.py`) for even population — keep
/// this array identical to `PIP_BUCKET_EDGES` in model.py.
const PIP_BUCKET_EDGES: [i32; 7] = [85, 131, 169, 205, 238, 271, 305];
const N_BUCKETS: usize = PIP_BUCKET_EDGES.len() + 1; // 8

/// Output heads of the newer **class-aware** bucketed net: race / crashed /
/// contact, each split into total-pip sub-buckets. Routing lives on the board
/// ([`Board::route_bucket`]); a net of width `N_HEADS*6` selects that head.
const N_HEADS: usize = crate::board::N_ROUTE_HEADS; // 12

/// Total-pip bucket for `board` (both sides): the number of edges it meets or
/// exceeds. Perspective-invariant, so a board and its swap share a bucket.
fn pip_bucket(board: &Board) -> usize {
    let total = board.pip_count(MOVER) + board.pip_count(OPP);
    PIP_BUCKET_EDGES.iter().filter(|&&e| total >= e).count()
}

/// An [`Evaluator`] backed by an ONNX value network, optimized for **any** batch
/// size so many positions can be scored in one forward pass.
///
/// Handles both output conventions: the original **5** nested sigmoids
/// (`win, win_g, win_bg, lose_g, lose_bg`) and the phase-split **6** outcome
/// softmax (`win s/g/bg, lose s/g/bg`). Both fold into the same [`Value`] — the
/// six outcomes are just the un-nested form of the five — so everything
/// downstream (equity, search, rollouts) is identical.
pub struct NnEval {
    model: TractModel,
    outputs: usize,
}

/// Fold a model output row into a [`Value`], by output width.
fn row_to_value(row: &[f32]) -> Value {
    match row.len() {
        5 => Value {
            win: row[0], win_g: row[1], win_bg: row[2], lose_g: row[3], lose_bg: row[4],
        },
        // 6-outcome softmax [ws, wg, wbg, ls, lg, lbg] -> nested probabilities.
        6 => {
            let (ws, wg, wbg, _ls, lg, lbg) = (row[0], row[1], row[2], row[3], row[4], row[5]);
            Value {
                win: ws + wg + wbg,
                win_g: wg + wbg,
                win_bg: wbg,
                lose_g: lg + lbg,
                lose_bg: lbg,
            }
        }
        n => panic!("unexpected net output width {n} (want 5 or 6)"),
    }
}

impl NnEval {
    /// Load and optimize an ONNX model from a file path. The model takes a
    /// `[N, 198]` float input and produces `[N, 5]` or `[N, 6]` outputs; the batch
    /// axis is left symbolic so a single optimized plan serves any batch size.
    pub fn from_path(path: &str) -> Result<Self, String> {
        let model = onnx().model_for_path(path).map_err(|e| format!("load {path}: {e}"))?;
        let batch = model.sym("N");
        let model = model
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), shapefactoid![batch, (features::NUM_INPUTS)]),
            )
            .map_err(|e| format!("input fact: {e}"))?
            .into_optimized()
            .map_err(|e| format!("optimize: {e}"))?
            .into_runnable()
            .map_err(|e| format!("runnable: {e}"))?;
        let mut eval = NnEval { model, outputs: 0 };
        // Probe the output width once, so evaluate() need not re-derive it.
        eval.outputs = eval.run_batch(&[0.0f32; features::NUM_INPUTS], 1).len();
        Ok(eval)
    }

    /// The number of raw outputs per position (5, 6, or `N_BUCKETS*6` bucketed).
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Fold one position's raw output row into a [`Value`], selecting the routed
    /// head's 6 outputs first for a bucketed net. The routing is chosen by output
    /// width: `N_HEADS*6` is the class-aware net (race/crashed/contact via
    /// [`Board::route_bucket`]); `N_BUCKETS*6` is the older total-pip net; anything
    /// else (5 or 6) is a single head folded directly.
    fn fold(&self, row: &[f32], board: &Board) -> Value {
        match self.outputs {
            n if n == N_HEADS * 6 => {
                let b = board.route_bucket();
                row_to_value(&row[b * 6..b * 6 + 6])
            }
            n if n == N_BUCKETS * 6 => {
                let b = pip_bucket(board);
                row_to_value(&row[b * 6..b * 6 + 6])
            }
            _ => row_to_value(row),
        }
    }

    /// Run the net on `n` concatenated 198-feature vectors, returning `n *
    /// outputs()` values (row-major). One `[n, 198]` matmul, not `n` of them.
    pub fn run_batch(&self, feats: &[f32], n: usize) -> Vec<f32> {
        debug_assert_eq!(feats.len(), n * features::NUM_INPUTS);
        let input =
            tract_ndarray::Array::from_shape_vec((n, features::NUM_INPUTS), feats.to_vec())
                .expect("feature shape");
        let out = self.model.run(tvec!(input.into_tvalue())).expect("tract inference");
        out[0].to_array_view::<f32>().expect("f32 output").iter().copied().collect()
    }
}

impl Evaluator for NnEval {
    fn evaluate(&self, board: &Board) -> Value {
        self.fold(&self.run_batch(&features::encode(board), 1), board)
    }

    fn evaluate_batch(&self, boards: &[Board]) -> Vec<Value> {
        if boards.is_empty() {
            return Vec::new();
        }
        let mut feats = Vec::with_capacity(boards.len() * features::NUM_INPUTS);
        for b in boards {
            feats.extend_from_slice(&features::encode(b));
        }
        let out = self.run_batch(&feats, boards.len());
        out.chunks_exact(self.outputs)
            .zip(boards)
            .map(|(row, b)| self.fold(row, b))
            .collect()
    }
}

/// A phase-routing [`Evaluator`]: a contact net for positions still in contact,
/// a race net once the armies have passed ([`Board::no_contact`]). The two nets
/// may differ in output width (e.g. a 5-output contact champion + a 6-output race
/// net) — [`row_to_value`] normalises both. Plugs into search and rollouts like
/// any other evaluator.
pub struct PhaseEval {
    contact: NnEval,
    race: NnEval,
}

impl PhaseEval {
    pub fn from_paths(contact: &str, race: &str) -> Result<Self, String> {
        Ok(PhaseEval {
            contact: NnEval::from_path(contact)?,
            race: NnEval::from_path(race)?,
        })
    }
}

impl Evaluator for PhaseEval {
    fn evaluate(&self, board: &Board) -> Value {
        if board.no_contact() {
            self.race.evaluate(board)
        } else {
            self.contact.evaluate(board)
        }
    }

    fn evaluate_batch(&self, boards: &[Board]) -> Vec<Value> {
        let zero = Value { win: 0.0, win_g: 0.0, win_bg: 0.0, lose_g: 0.0, lose_bg: 0.0 };
        let mut out = vec![zero; boards.len()];
        // Score each phase in one batched pass, then scatter back in order.
        for race in [false, true] {
            let idx: Vec<usize> = boards
                .iter()
                .enumerate()
                .filter(|(_, b)| b.no_contact() == race)
                .map(|(i, _)| i)
                .collect();
            if idx.is_empty() {
                continue;
            }
            let subset: Vec<Board> = idx.iter().map(|&i| boards[i].clone()).collect();
            let ev = if race { &self.race } else { &self.contact };
            for (k, v) in ev.evaluate_batch(&subset).into_iter().enumerate() {
                out[idx[k]] = v;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");

    #[test]
    fn tract_matches_pytorch_on_fixture() {
        // Live-net cross-language check: tract's folded Value for the starting
        // position must match parity.json's expected_output, whatever architecture
        // td.onnx currently is (5-output, 6-output, or bucketed — the export
        // scripts write the folded Value). Skips gracefully if artifacts absent.
        let onnx_path = format!("{MODELS}/td.onnx");
        let fixture_path = format!("{MODELS}/parity.json");
        if !std::path::Path::new(&onnx_path).exists()
            || !std::path::Path::new(&fixture_path).exists()
        {
            eprintln!("skipping: run trainer/export_onnx.py first");
            return;
        }

        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
        let expected: Vec<f32> = fixture["expected_output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let nn = NnEval::from_path(&onnx_path).unwrap();
        let got = nn.evaluate(&Board::starting_position());
        let got = [got.win, got.win_g, got.win_bg, got.lose_g, got.lose_bg];

        for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "output {i}: tract {g} vs pytorch {e}"
            );
        }
    }

    #[test]
    fn race_net_6output_folds_to_correct_equity() {
        // The 6-outcome race net (trainer/export_race.py) must load, report width
        // 6, and its folded Value must carry the same equity as the raw six
        // softmax probabilities dotted with the outcome points.
        let onnx_path = format!("{MODELS}/td_race.onnx");
        let fixture_path = format!("{MODELS}/parity_race.json");
        if !std::path::Path::new(&onnx_path).exists()
            || !std::path::Path::new(&fixture_path).exists()
        {
            eprintln!("skipping: run trainer/export_race.py first");
            return;
        }
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
        let probs: Vec<f32> = fixture["expected_output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let points = [1.0f32, 2.0, 3.0, -1.0, -2.0, -3.0];
        let expected_equity: f32 = probs.iter().zip(points).map(|(p, pt)| p * pt).sum();

        let nn = NnEval::from_path(&onnx_path).unwrap();
        assert_eq!(nn.outputs(), 6, "race net should have 6 outputs");
        let v = nn.evaluate(&Board::starting_position());
        assert!(
            (v.equity() - expected_equity).abs() < 1e-4,
            "folded equity {} vs fixture {expected_equity}",
            v.equity()
        );
    }

    #[test]
    fn bucketed_net_selects_the_pip_bucket() {
        // The bucketed net (trainer/export_bucketed.py) must load with width
        // N_BUCKETS*6, and evaluating a position must fold the *selected* pip
        // bucket's 6 probabilities — matching the fixture's chosen bucket.
        let onnx_path = format!("{MODELS}/td_bucket.onnx");
        let fixture_path = format!("{MODELS}/parity_bucket.json");
        if !std::path::Path::new(&onnx_path).exists()
            || !std::path::Path::new(&fixture_path).exists()
        {
            eprintln!("skipping: run trainer/export_bucketed.py first");
            return;
        }
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
        let probs: Vec<f32> = fixture["expected_bucket_output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let points = [1.0f32, 2.0, 3.0, -1.0, -2.0, -3.0];
        let expected_equity: f32 = probs.iter().zip(points).map(|(p, pt)| p * pt).sum();

        let nn = NnEval::from_path(&onnx_path).unwrap();
        assert_eq!(nn.outputs(), N_BUCKETS * 6, "bucketed net width");
        let start = Board::starting_position();
        // The starting position sits in the top bucket (334 total pips).
        assert_eq!(pip_bucket(&start), fixture["bucket"].as_u64().unwrap() as usize);
        let v = nn.evaluate(&start);
        assert!(
            (v.equity() - expected_equity).abs() < 1e-4,
            "folded equity {} vs fixture {expected_equity}",
            v.equity()
        );
    }
}
