//! Standard sigmoid surrogate: sigmoid(α * x).

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use super::Surrogate;

/// Sigmoid surrogate with configurable alpha (steepness).
#[derive(Debug, Clone)]
pub struct SigmoidSurrogate {
    /// Steepness parameter α.
    pub alpha: f64,
}

impl Default for SigmoidSurrogate {
    fn default() -> Self {
        Self { alpha: 10.0 }
    }
}

impl SigmoidSurrogate {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Surrogate for SigmoidSurrogate {
    fn surrogate_forward<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        sigmoid(x.mul_scalar(self.alpha as f32))
    }
}
