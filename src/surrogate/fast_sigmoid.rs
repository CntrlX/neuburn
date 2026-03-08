//! Fast sigmoid surrogate: gradient ∝ 1/(1 + k|x|)² style (snntorch fast_sigmoid).

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use super::Surrogate;

/// Fast sigmoid surrogate with configurable slope.
/// Forward: smooth approximation to step; backward: gradient ∝ 1/(1 + slope*|x|)².
#[derive(Debug, Clone)]
pub struct FastSigmoid {
    /// Slope (sharpness); higher = steeper. Typical: 25.
    pub slope: f64,
}

impl Default for FastSigmoid {
    fn default() -> Self {
        Self { slope: 25.0 }
    }
}

impl FastSigmoid {
    pub fn new(slope: f64) -> Self {
        Self { slope }
    }
}

impl Surrogate for FastSigmoid {
    fn surrogate_forward<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // snntorch-style: smooth forward sigmoid(slope * x) so backward gets smooth gradient.
        sigmoid(x.mul_scalar(self.slope as f32))
    }
}
