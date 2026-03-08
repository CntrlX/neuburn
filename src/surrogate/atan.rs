//! Arctangent-like surrogate: smooth, bounded gradient.
//! Uses tanh as a differentiable stand-in for (1/π)*arctan (similar S-shape, no atan in Burn).

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::Surrogate;

/// ATan-style surrogate: 0.5 + 0.5 * tanh(α * x), in [0,1]. Smooth, bounded gradient.
#[derive(Debug, Clone)]
pub struct ATan {
    /// Steepness α.
    pub alpha: f64,
}

impl Default for ATan {
    fn default() -> Self {
        Self { alpha: 2.0 }
    }
}

impl ATan {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Surrogate for ATan {
    fn surrogate_forward<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let scaled = x.mul_scalar(self.alpha as f32);
        scaled.clone().tanh().add_scalar(1.0).mul_scalar(0.5)
    }
}
