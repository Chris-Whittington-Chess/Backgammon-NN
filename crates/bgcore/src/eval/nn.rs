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

/// An [`Evaluator`] backed by an ONNX value network (batch size 1).
pub struct NnEval {
    model: TractModel,
}

impl NnEval {
    /// Load and optimize an ONNX model from a file path. The model must take a
    /// `[N, 198]` float input and produce `[N, 5]` probability outputs.
    pub fn from_path(path: &str) -> Result<Self, String> {
        let model = onnx()
            .model_for_path(path)
            .map_err(|e| format!("load {path}: {e}"))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), shapefactoid![1, (features::NUM_INPUTS)]),
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
        let input = tract_ndarray::Array::from_shape_vec(
            (1, features::NUM_INPUTS),
            feats.to_vec(),
        )
        .expect("feature shape");
        let out = self
            .model
            .run(tvec!(input.into_tvalue()))
            .expect("tract inference");
        let view = out[0].to_array_view::<f32>().expect("f32 output");
        [
            view[[0, 0]],
            view[[0, 1]],
            view[[0, 2]],
            view[[0, 3]],
            view[[0, 4]],
        ]
    }
}

impl Evaluator for NnEval {
    fn evaluate(&self, board: &Board) -> Value {
        let o = self.run(&features::encode(board));
        Value {
            win: o[0],
            win_g: o[1],
            win_bg: o[2],
            lose_g: o[3],
            lose_bg: o[4],
        }
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
