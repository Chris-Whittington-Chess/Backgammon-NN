//! Neural-net evaluator via ONNX (SPEC §9, milestone M5).
//!
//! Loads a network exported from PyTorch (`trainer/export_onnx.py`) and runs it
//! with `tract-onnx` — pure Rust, no external ONNX Runtime binary. This lets the
//! trained net drive self-play and play in the GUI at native speed, using the
//! exact same 198-input encoding as training ([`crate::features`]).
//!
//! Enabled by the `onnx` cargo feature.

use crate::board::Board;
use crate::eval::{Evaluator, Value};
use crate::features;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::shapefactoid;

type TractModel = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// An [`Evaluator`] backed by an ONNX value network, optimized for **any** batch
/// size so many positions can be scored in one forward pass.
pub struct NnEval {
    model: TractModel,
}

fn out_to_value(row: &[f32]) -> Value {
    Value { win: row[0], win_g: row[1], win_bg: row[2], lose_g: row[3], lose_bg: row[4] }
}

impl NnEval {
    /// Load and optimize an ONNX model from a file path. The model must take a
    /// `[N, 198]` float input and produce `[N, 5]` probability outputs. The batch
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
        Ok(NnEval { model })
    }

    /// Run the net on a single 198-feature vector, returning the raw 5 outputs.
    pub fn run(&self, feats: &[f32; features::NUM_INPUTS]) -> [f32; 5] {
        let out = self.run_batch(feats, 1);
        [out[0], out[1], out[2], out[3], out[4]]
    }

    /// Run the net on `n` concatenated 198-feature vectors, returning `n * 5`
    /// outputs (row-major). One `[n, 198]` matmul instead of `n` `[1, 198]` ones.
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
        out_to_value(&self.run(&features::encode(board)))
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
        out.chunks_exact(5).map(out_to_value).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");

    #[test]
    fn tract_matches_pytorch_on_fixture() {
        // Requires `trainer/export_onnx.py` to have produced td.onnx + parity.json.
        // Skips gracefully if the artifacts are absent.
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
}
